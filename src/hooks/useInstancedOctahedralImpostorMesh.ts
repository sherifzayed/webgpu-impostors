import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three/webgpu";
import { sampleOctahedralDirection } from "../utils/octahedralImpostorMath";
import { InstancedOctahedralImpostorMaterial } from "../utils/InstancedOctahedralImpostorMaterial";

// Constant yaw applied to the sampling direction to correct the fixed alignment
// offset between the baked atlas azimuth and the sampling convention. Negative =
// clockwise seen from above. Adjust in 90° (Math.PI / 2) steps if needed.
const ALIGNMENT_OFFSET_RADIANS = -Math.PI / 2;

/**
 * Builds a single InstancedMesh + single shared material for a whole field
 * of octahedral impostors (one draw call instead of one mesh per instance).
 *
 * Billboard rotation is handled entirely in the material's vertex shader
 * (see InstancedOctahedralImpostorMaterial), so the only per-frame CPU work
 * left is picking which atlas cell each instance should show, based on its
 * direction to the camera - exactly the same math OctahedralImpostor.jsx
 * used per-instance, just batched into one loop over all instances instead
 * of N separate useFrame callbacks.
 */
export function useInstancedOctahedralImpostorMesh({
  instances,
  atlas,
  gridSize,
  octType = 0,
  samplingCache,
  geometryArgs = [2, 2],
  atlasCoverage = 1.0,
  alphaTest = 0.5,
  useDither = false,
  showWireframe = false,
  directionThresholdRadians = 0.0872665,
  maxUpdatesPerFrame = 20000,
}) {
  const directionThresholdDot = useMemo(() => {
    const clamped = Math.min(Math.max(directionThresholdRadians, 0), Math.PI);
    return Math.cos(clamped);
  }, [directionThresholdRadians]);

  const stateRef = useRef(null);

  const instancedMesh = useMemo(() => {
    if (!atlas || !atlas.texture || instances.length === 0) {
      return null;
    }

    const count = instances.length;

    // WebGPU allows at most 8 vertex buffers per pipeline. A THREE.InstancedMesh
    // would spend one of those on its built-in instanceMatrix (unused here - the
    // billboard is positioned from instanceOffset, not a per-instance matrix),
    // and PlaneGeometry's normal is dead weight for this unlit view-shaded
    // material. Both are dropped by building a plain InstancedBufferGeometry with
    // only position + uv and rendering it with a regular Mesh, leaving room for
    // the 5 per-instance attributes below. LOD visibility is additionally packed
    // into instanceScale.w to keep the per-instance buffer count at 5.
    const baseGeometry = new THREE.PlaneGeometry(...geometryArgs);
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.index = baseGeometry.index;
    geometry.setAttribute("position", baseGeometry.getAttribute("position"));
    geometry.setAttribute("uv", baseGeometry.getAttribute("uv"));
    geometry.instanceCount = count;

    const instanceOffset = new Float32Array(count * 3);
    const instanceScale = new Float32Array(count * 4);
    const instanceYawSinCos = new Float32Array(count * 2);
    const instanceFaceIndices = new Float32Array(count * 3);
    const instanceFaceWeights = new Float32Array(count * 3);

    // Seed face indices/weights with the first valid triangle so every
    // instance renders something before the first updateFrame() call.
    let initialA = 0;
    let initialB = 1;
    let initialC = 2;
    const index = atlas.octahedralData?.geometry?.index;
    if (index && index.count > 0) {
      const maxIndex = atlas.octahedralData.geometry.attributes.position.count - 1;
      initialA = Math.max(0, Math.min(index.getX(0), maxIndex));
      initialB = Math.max(0, Math.min(index.getY(0), maxIndex));
      initialC = Math.max(0, Math.min(index.getZ(0), maxIndex));
    }

    const positions = new Float32Array(count * 3);
    const rotationsY = new Float32Array(count);
    const yawSinCos = new Float32Array(count * 2);

    for (let i = 0; i < count; i += 1) {
      const instance = instances[i];
      positions[i * 3] = instance.position[0];
      positions[i * 3 + 1] = instance.position[1];
      positions[i * 3 + 2] = instance.position[2];
      rotationsY[i] = instance.rotationY || 0;
      const cosYaw = Math.cos(rotationsY[i]);
      const sinYaw = Math.sin(rotationsY[i]);
      yawSinCos[i * 2] = sinYaw;
      yawSinCos[i * 2 + 1] = cosYaw;

      instanceOffset[i * 3] = instance.position[0];
      instanceOffset[i * 3 + 1] = instance.position[1];
      instanceOffset[i * 3 + 2] = instance.position[2];

      instanceScale[i * 4] = instance.scale[0];
      instanceScale[i * 4 + 1] = instance.scale[1];
      instanceScale[i * 4 + 2] = instance.scale[2];
      instanceScale[i * 4 + 3] = 1; // .w = visibility (1 = shown, 0 = hidden)

      instanceYawSinCos[i * 2] = sinYaw;
      instanceYawSinCos[i * 2 + 1] = cosYaw;

      instanceFaceIndices[i * 3] = initialA;
      instanceFaceIndices[i * 3 + 1] = initialB;
      instanceFaceIndices[i * 3 + 2] = initialC;

      instanceFaceWeights[i * 3] = 1 / 3;
      instanceFaceWeights[i * 3 + 1] = 1 / 3;
      instanceFaceWeights[i * 3 + 2] = 1 / 3;
    }

    geometry.setAttribute(
      "instanceOffset",
      new THREE.InstancedBufferAttribute(instanceOffset, 3)
    );
    geometry.setAttribute(
      "instanceScale",
      new THREE.InstancedBufferAttribute(instanceScale, 4)
    );
    geometry.setAttribute(
      "instanceYawSinCos",
      new THREE.InstancedBufferAttribute(instanceYawSinCos, 2)
    );
    geometry.setAttribute(
      "instanceFaceIndices",
      new THREE.InstancedBufferAttribute(instanceFaceIndices, 3)
    );
    geometry.setAttribute(
      "instanceFaceWeights",
      new THREE.InstancedBufferAttribute(instanceFaceWeights, 3)
    );

    const material = new InstancedOctahedralImpostorMaterial({
      atlasTexture: atlas.texture,
      gridSize,
      octType,
      atlasCoverage,
      alphaTest,
      useDither,
      showWireframe,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    // Impostor billboards don't shadow-map correctly (the quad re-orients to
    // face the shadow camera) and the depth pass over the whole field is
    // expensive; shadows come from TreeImpostorShadowDecals instead.
    mesh.castShadow = false;
    mesh.receiveShadow = false;

    // Conservative bounding-sphere radius per instance for frustum culling:
    // half the billboard diagonal at that instance's scale. The billboard can
    // rotate to face any direction, so the sphere must cover the worst case.
    const [planeWidth, planeHeight] = geometryArgs;
    const boundingRadii = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      const instance = instances[i];
      const halfW =
        0.5 * planeWidth * Math.max(instance.scale[0], instance.scale[2]);
      const halfH = 0.5 * planeHeight * instance.scale[1];
      boundingRadii[i] = Math.sqrt(halfW * halfW + halfH * halfH);
    }

    stateRef.current = {
      positions,
      rotationsY,
      yawSinCos,
      lastDirections: new Float32Array(count * 3),
      hasLastDirection: new Uint8Array(count),
      updateCursor: 0,
      tempDirection: new THREE.Vector3(),
      tempLocalDirection: new THREE.Vector3(),
      tempIndices: new THREE.Vector3(),
      tempWeights: new THREE.Vector3(),
      octahedralData: atlas.octahedralData,
      boundingRadii,
      // Visibility is the AND of two independent inputs, both packed into
      // instanceScale.w: LOD (the field hides impostors swapped for real
      // meshes) and frustum culling (instances outside the camera's view).
      lodVisible: new Uint8Array(count).fill(1),
      frustumCulled: new Uint8Array(count),
      tempFrustum: new THREE.Frustum(),
      tempProjScreenMatrix: new THREE.Matrix4(),
      tempSphere: new THREE.Sphere(),
    };

    return mesh;
  }, [
    atlas,
    gridSize,
    octType,
    instances,
    geometryArgs.join(","),
    atlasCoverage,
    alphaTest,
    useDither,
    showWireframe,
    maxUpdatesPerFrame,
  ]);

  useEffect(() => {
    return () => {
      if (instancedMesh) {
        instancedMesh.geometry.dispose();
        instancedMesh.material.dispose();
      }
    };
  }, [instancedMesh]);

  const updateFrame = (camera) => {
    if (!instancedMesh || !stateRef.current || !samplingCache) {
      return;
    }

    const {
      positions,
      yawSinCos,
      lastDirections,
      hasLastDirection,
      tempDirection,
      tempLocalDirection,
      tempIndices,
      tempWeights,
      octahedralData,
      boundingRadii,
      lodVisible,
      frustumCulled,
      tempFrustum,
      tempProjScreenMatrix,
      tempSphere,
    } = stateRef.current;
    const instanceCount = positions.length / 3;
    const updatesThisFrame = Math.min(maxUpdatesPerFrame, instanceCount);
    let cursor = stateRef.current.updateCursor % instanceCount;
    const startCursor = cursor;

    const faceIndicesAttr = instancedMesh.geometry.getAttribute(
      "instanceFaceIndices"
    );
    const faceWeightsAttr = instancedMesh.geometry.getAttribute(
      "instanceFaceWeights"
    );
    const scaleAttr = instancedMesh.geometry.getAttribute("instanceScale");

    let changed = false;
    let visibilityChanged = false;

    tempProjScreenMatrix.multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse
    );
    tempFrustum.setFromProjectionMatrix(tempProjScreenMatrix);

    // Static alignment offset: the baked atlas azimuth is rotated ~90° relative
    // to the sampling convention, so every impostor's default facing is turned.
    // Rotate the sampling direction by this constant yaw to line the impostor up
    // with the real mesh. Flip the sign if it aligns the wrong way.
    const alignCos = Math.cos(ALIGNMENT_OFFSET_RADIANS);
    const alignSin = Math.sin(ALIGNMENT_OFFSET_RADIANS);

    for (let updated = 0; updated < updatesThisFrame; updated += 1) {
      const i = cursor;
      cursor = (cursor + 1) % instanceCount;
      const positionOffset = i * 3;
      const yawOffset = i * 2;

      // Frustum cull: instances behind the camera or outside the view cone
      // are collapsed via instanceScale.w (the vertex shader shrinks them to
      // a point, so they cost no fill) and skip the atlas-cell sampling work.
      tempSphere.center.set(
        positions[positionOffset],
        positions[positionOffset + 1],
        positions[positionOffset + 2]
      );
      tempSphere.radius = boundingRadii[i];
      const inView = tempFrustum.intersectsSphere(tempSphere);

      if (!inView) {
        if (!frustumCulled[i]) {
          frustumCulled[i] = 1;
          scaleAttr.setW(i, 0);
          visibilityChanged = true;
        }
        continue;
      }
      if (frustumCulled[i]) {
        frustumCulled[i] = 0;
        scaleAttr.setW(i, lodVisible[i]);
        visibilityChanged = true;
      }

      // Direction from the instance toward the camera. The atlas is baked
      // indexed by the camera-position direction (camera placed at pntOct,
      // looking at the origin), so the lookup must use camera - object, not
      // object - camera, otherwise every impostor shows its opposite side.
      const viewDir = tempDirection.set(
        camera.position.x - positions[positionOffset],
        camera.position.y - positions[positionOffset + 1],
        camera.position.z - positions[positionOffset + 2]
      ).normalize();
      const sinYaw = yawSinCos[yawOffset];
      const cosYaw = yawSinCos[yawOffset + 1];

      tempLocalDirection.set(
        viewDir.x * cosYaw - viewDir.z * sinYaw,
        viewDir.y,
        viewDir.x * sinYaw + viewDir.z * cosYaw
      );

      // Apply the constant alignment offset (horizontal only).
      const alignedX =
        tempLocalDirection.x * alignCos - tempLocalDirection.z * alignSin;
      const alignedZ =
        tempLocalDirection.x * alignSin + tempLocalDirection.z * alignCos;
      tempLocalDirection.x = alignedX;
      tempLocalDirection.z = alignedZ;

      const lastOffset = i * 3;
      if (
        hasLastDirection[i] &&
        lastDirections[lastOffset] * tempLocalDirection.x +
          lastDirections[lastOffset + 1] * tempLocalDirection.y +
          lastDirections[lastOffset + 2] * tempLocalDirection.z >=
          directionThresholdDot
      ) {
        continue;
      }

      const samplingSuccess = sampleOctahedralDirection({
        direction: tempLocalDirection,
        cache: samplingCache,
        indicesTarget: tempIndices,
        weightsTarget: tempWeights,
      });

      if (!samplingSuccess) {
        if (octahedralData?.geometry?.index && octahedralData.geometry.index.count > 0) {
          const indices = octahedralData.geometry.index;
          const maxIndex = octahedralData.geometry.attributes.position.count - 1;
          tempIndices.set(
            Math.max(0, Math.min(indices.getX(0), maxIndex)),
            Math.max(0, Math.min(indices.getY(0), maxIndex)),
            Math.max(0, Math.min(indices.getZ(0), maxIndex))
          );
        } else {
          tempIndices.set(0, 1, 2);
        }
        tempWeights.set(1 / 3, 1 / 3, 1 / 3);
      }

      faceIndicesAttr.setXYZ(i, tempIndices.x, tempIndices.y, tempIndices.z);
      faceWeightsAttr.setXYZ(i, tempWeights.x, tempWeights.y, tempWeights.z);

      lastDirections[lastOffset] = tempLocalDirection.x;
      lastDirections[lastOffset + 1] = tempLocalDirection.y;
      lastDirections[lastOffset + 2] = tempLocalDirection.z;
      hasLastDirection[i] = 1;
      changed = true;
    }
    stateRef.current.updateCursor = cursor;

    if (changed) {
      // Only the contiguous window [startCursor, startCursor + updatesThisFrame)
      // was touched this frame. Upload just that window (each face attribute has
      // itemSize 3) instead of the whole buffer, so per-frame GPU transfer is
      // O(updatesThisFrame) rather than O(instanceCount). The window can wrap
      // past the end of the buffer, in which case it splits into two ranges.
      if (updatesThisFrame >= instanceCount) {
        faceIndicesAttr.addUpdateRange(0, instanceCount * 3);
        faceWeightsAttr.addUpdateRange(0, instanceCount * 3);
      } else {
        const firstCount = Math.min(updatesThisFrame, instanceCount - startCursor);
        faceIndicesAttr.addUpdateRange(startCursor * 3, firstCount * 3);
        faceWeightsAttr.addUpdateRange(startCursor * 3, firstCount * 3);
        const wrapCount = updatesThisFrame - firstCount;
        if (wrapCount > 0) {
          faceIndicesAttr.addUpdateRange(0, wrapCount * 3);
          faceWeightsAttr.addUpdateRange(0, wrapCount * 3);
        }
      }
      faceIndicesAttr.needsUpdate = true;
      faceWeightsAttr.needsUpdate = true;
    }

    if (visibilityChanged) {
      scaleAttr.needsUpdate = true;
    }
  };

  // LOD visibility (owned by the field) and frustum culling (owned by
  // updateFrame) share instanceScale.w; this setter records the LOD side and
  // only writes the attribute when the instance isn't already frustum-culled.
  const setLodVisibility = (index, visible) => {
    if (!instancedMesh || !stateRef.current) {
      return;
    }
    const { lodVisible, frustumCulled } = stateRef.current;
    lodVisible[index] = visible ? 1 : 0;
    if (!frustumCulled[index]) {
      const scaleAttr = instancedMesh.geometry.getAttribute("instanceScale");
      scaleAttr.setW(index, lodVisible[index]);
      scaleAttr.needsUpdate = true;
    }
  };

  return { instancedMesh, updateFrame, setLodVisibility };
}
