import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three/webgpu";
import { LitInstancedOctahedralImpostorMaterial } from "../utils/InstancedOctahedralImpostorMaterial";

/**
 * Builds a single InstancedMesh + single shared material for a whole field
 * of octahedral impostors (one draw call instead of one mesh per instance).
 *
 * Everything that changes with the camera runs on the GPU: the material's
 * vertex shader turns each billboard to face the camera and picks the three
 * baked frames nearest the view direction (selectOctahedralFrames). The
 * per-instance buffers are written once and only touched again when the LOD
 * system swaps an instance for its real mesh.
 *
 * Off-screen quads need no culling of their own (the rasterizer clips all
 * four vertices), so frustum culling here is only an on-demand query for
 * the LOD system.
 */
export function useInstancedOctahedralImpostorMesh({
  instances,
  atlas,
  gridSize,
  octType = 0,
  geometryArgs = [2, 2],
  alphaTest = 0.5,
  materialOptions = null,
}) {
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
    // only position + uv and rendering it with a regular Mesh.
    const baseGeometry = new THREE.PlaneGeometry(...geometryArgs);
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.index = baseGeometry.index;
    geometry.setAttribute("position", baseGeometry.getAttribute("position"));
    geometry.setAttribute("uv", baseGeometry.getAttribute("uv"));
    geometry.instanceCount = count;

    const instanceOffset = new Float32Array(count * 3);
    // xyz = scale, w = LOD visibility (1 = shown, 0 = swapped for the real mesh).
    const instanceScale = new Float32Array(count * 4);
    // xy = sin/cos of the instance yaw, z = a per-instance random seed in [0, 1)
    // (instance.seed) for material variation; packed together to save a buffer.
    const instanceYawSinCos = new Float32Array(count * 3);

    // Conservative bounding-sphere radius per instance for frustum queries:
    // half the billboard diagonal at that instance's scale. The billboard can
    // rotate to face any direction, so the sphere must cover the worst case.
    const [planeWidth, planeHeight] = geometryArgs;
    const boundingRadii = new Float32Array(count);

    for (let i = 0; i < count; i += 1) {
      const instance = instances[i];
      const yaw = instance.rotationY || 0;

      instanceOffset[i * 3] = instance.position[0];
      instanceOffset[i * 3 + 1] = instance.position[1];
      instanceOffset[i * 3 + 2] = instance.position[2];

      instanceScale[i * 4] = instance.scale[0];
      instanceScale[i * 4 + 1] = instance.scale[1];
      instanceScale[i * 4 + 2] = instance.scale[2];
      instanceScale[i * 4 + 3] = 1;

      instanceYawSinCos[i * 3] = Math.sin(yaw);
      instanceYawSinCos[i * 3 + 1] = Math.cos(yaw);
      instanceYawSinCos[i * 3 + 2] = instance.seed ?? 0;

      const halfW =
        0.5 * planeWidth * Math.max(instance.scale[0], instance.scale[2]);
      const halfH = 0.5 * planeHeight * instance.scale[1];
      boundingRadii[i] = Math.sqrt(halfW * halfW + halfH * halfH);
    }

    geometry.setAttribute(
      "instanceOffset",
      new THREE.InstancedBufferAttribute(instanceOffset, 3),
    );
    geometry.setAttribute(
      "instanceScale",
      new THREE.InstancedBufferAttribute(instanceScale, 4),
    );
    geometry.setAttribute(
      "instanceYawSinCos",
      new THREE.InstancedBufferAttribute(instanceYawSinCos, 3),
    );

    // "surface" atlases carry normals + depth, so the impostor is lit by the
    // scene's lights. three's vertex-stage normal accessors read a geometry
    // normal even though lit impostors shade with the baked normals.
    geometry.setAttribute("normal", baseGeometry.getAttribute("normal"));
    const material = new LitInstancedOctahedralImpostorMaterial({
      atlasTexture: atlas.texture,
      normalDepthTexture: atlas.normalDepthTexture,
      gridSize,
      octType,
      alphaTest,
      planeHeight: geometryArgs[1],
      ...(materialOptions ?? {}),
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    // Impostor billboards don't cast into shadow maps correctly (the quad
    // re-orients to face the shadow camera) and the depth pass over the whole
    // field is expensive, so they never cast. Lit impostors do receive: their
    // baked depth puts each texel's shadow lookup on the leaf it shows.
    mesh.castShadow = false;
    mesh.receiveShadow = true;

    stateRef.current = {
      positions: instanceOffset,
      boundingRadii,
      frustum: new THREE.Frustum(),
      tempProjScreenMatrix: new THREE.Matrix4(),
      tempViewMatrix: new THREE.Matrix4(),
      tempSphere: new THREE.Sphere(),
    };

    return mesh;
  }, [
    atlas,
    gridSize,
    octType,
    instances,
    geometryArgs.join(","),
    alphaTest,
    materialOptions,
  ]);

  useEffect(() => {
    return () => {
      if (instancedMesh) {
        instancedMesh.geometry.dispose();
        instancedMesh.material.dispose();
      }
    };
  }, [instancedMesh]);

  // Refreshes the frustum isFrustumCulled tests against. O(1): no per-instance work.
  const updateFrame = (camera) => {
    if (!stateRef.current) {
      return;
    }
    const { frustum, tempProjScreenMatrix, tempViewMatrix } = stateRef.current;

    // camera.matrixWorldInverse is only refreshed by renderer.render(), which
    // runs after this frame callback - it lags the camera by one frame. At a
    // telephoto FOV a single fast-drag frame can rotate past the whole view,
    // so culling with the stale inverse blinks visible trees out. Rebuild the
    // inverse from the camera's up-to-date transform instead (CameraControls
    // updates at useFrame priority -1, before this).
    camera.updateMatrixWorld();
    tempViewMatrix.copy(camera.matrixWorld).invert();
    tempProjScreenMatrix.multiplyMatrices(
      camera.projectionMatrix,
      tempViewMatrix,
    );
    // Match the renderer's clip conventions (WebGPU z in [0,1], reversed-Z),
    // otherwise the extracted near/far planes are wrong.
    frustum.setFromProjectionMatrix(
      tempProjScreenMatrix,
      camera.coordinateSystem,
      camera.reversedDepth,
    );
  };

  // LOD visibility lives in instanceScale.w. Only this instance's 4 floats are
  // uploaded, not the whole buffer.
  const setLodVisibility = (index, visible) => {
    if (!instancedMesh) {
      return;
    }
    const scaleAttr = instancedMesh.geometry.getAttribute("instanceScale");
    scaleAttr.setW(index, visible ? 1 : 0);
    scaleAttr.addUpdateRange(index * 4, 4);
    scaleAttr.needsUpdate = true;
  };

  // Whether this instance's bounding sphere is outside the frustum from the last updateFrame().
  const isFrustumCulled = (index) => {
    const state = stateRef.current;
    if (!state) {
      return false;
    }
    const { positions, boundingRadii, frustum, tempSphere } = state;
    tempSphere.center.set(
      positions[index * 3],
      positions[index * 3 + 1],
      positions[index * 3 + 2],
    );
    tempSphere.radius = boundingRadii[index];
    return !frustum.intersectsSphere(tempSphere);
  };

  return { instancedMesh, updateFrame, setLodVisibility, isFrustumCulled };
}
