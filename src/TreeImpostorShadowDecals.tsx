import { useEffect, useMemo } from "react";
import * as THREE from "three/webgpu";
import {
  texture,
  attribute,
  uniform,
  uv,
  vec2,
  vec3,
  float,
  smoothstep,
} from "three/tsl";
import { sampleOctahedralDirection } from "./utils/octahedralImpostorMath";

// Must match ALIGNMENT_OFFSET_RADIANS in useInstancedOctahedralImpostorMesh -
// the baked atlas azimuth is rotated 90 degrees relative to the sampling
// convention, and the shadow silhouette lookup goes through the same table.
const ALIGNMENT_OFFSET_RADIANS = -Math.PI / 2;

const tempObject = new THREE.Object3D();
tempObject.rotation.order = "YXZ";
const tempDirection = new THREE.Vector3();
const tempIndices = new THREE.Vector3();
const tempWeights = new THREE.Vector3();

/**
 * Projected-silhouette shadows for an impostor field, at blob-shadow cost.
 *
 * Instead of a shadow map (a whole extra render pass of every instance) or a
 * generic dark blob, this reuses the octahedral atlas that was already baked
 * for the impostors: for each instance it picks the atlas cell whose view
 * direction matches the sun, and stamps that cell's alpha channel onto a
 * ground quad stretched along the sun azimuth by height / tan(elevation).
 * The result is an actual tree-shaped shadow that respects each instance's
 * yaw and scale.
 *
 * Everything is static (sun and instances don't move), so the whole field is
 * one InstancedMesh built once: a single draw call and zero per-frame work.
 * The fragment shader fakes a penumbra by blurring and fading the silhouette
 * the farther it gets from the trunk, plus a soft contact patch at the base.
 */
export default function TreeImpostorShadowDecals({
  instances,
  atlas,
  samplingCache,
  gridSize,
  atlasCoverage = 1.0,
  geometryArgs = [2, 2],
  groundY = -2,
  sunPosition = [35, 55, 35],
  opacity = 0.45,
  minElevation = 0.35,
}) {
  const shadowMesh = useMemo(() => {
    if (!atlas?.texture || !samplingCache || !instances.length) {
      return null;
    }

    const count = instances.length;
    const [planeWidth, planeHeight] = geometryArgs;

    const sunDir = new THREE.Vector3(...sunPosition).normalize();
    const horizontal = Math.hypot(sunDir.x, sunDir.z);
    // Clamp elevation so a low sun can't stretch shadows to the horizon.
    const elevation = Math.max(
      Math.atan2(sunDir.y, horizontal),
      minElevation
    );
    const stretch = 1 / Math.tan(elevation);
    // Ground direction the shadow falls along (away from the sun).
    const shadowDirX = horizontal > 1e-6 ? -sunDir.x / horizontal : 0;
    const shadowDirZ = horizontal > 1e-6 ? -sunDir.z / horizontal : -1;
    // Yaw that points the quad's local +v axis along (shadowDirX, shadowDirZ)
    // once the plane is laid flat (local up maps to world -Z before the yaw).
    const quadYaw = Math.atan2(-shadowDirX, -shadowDirZ);

    const geometry = new THREE.PlaneGeometry(1, 1);
    const faceIndicesArray = new Float32Array(count * 3);
    const faceWeightsArray = new Float32Array(count * 3);
    geometry.setAttribute(
      "shadowFaceIndices",
      new THREE.InstancedBufferAttribute(faceIndicesArray, 3)
    );
    geometry.setAttribute(
      "shadowFaceWeights",
      new THREE.InstancedBufferAttribute(faceWeightsArray, 3)
    );

    const material = createShadowDecalMaterial({
      atlasTexture: atlas.texture,
      gridSize,
      atlasCoverage,
      opacity,
    });

    const mesh = new THREE.InstancedMesh(geometry, material, count);
    mesh.frustumCulled = false;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.renderOrder = 1;

    const alignCos = Math.cos(ALIGNMENT_OFFSET_RADIANS);
    const alignSin = Math.sin(ALIGNMENT_OFFSET_RADIANS);

    for (let i = 0; i < count; i += 1) {
      const instance = instances[i];
      const yaw = instance.rotationY || 0;
      const cosYaw = Math.cos(yaw);
      const sinYaw = Math.sin(yaw);

      // Same lookup as the impostor view selection, but pointed at the sun
      // instead of the camera: rotate the sun direction into the instance's
      // local frame, then apply the constant atlas alignment offset.
      const localX = sunDir.x * cosYaw - sunDir.z * sinYaw;
      const localZ = sunDir.x * sinYaw + sunDir.z * cosYaw;
      tempDirection.set(
        localX * alignCos - localZ * alignSin,
        sunDir.y,
        localX * alignSin + localZ * alignCos
      );

      const sampled = sampleOctahedralDirection({
        direction: tempDirection,
        cache: samplingCache,
        indicesTarget: tempIndices,
        weightsTarget: tempWeights,
      });
      if (!sampled) {
        tempIndices.set(0, 1, 2);
        tempWeights.set(1 / 3, 1 / 3, 1 / 3);
      }

      faceIndicesArray[i * 3] = tempIndices.x;
      faceIndicesArray[i * 3 + 1] = tempIndices.y;
      faceIndicesArray[i * 3 + 2] = tempIndices.z;
      faceWeightsArray[i * 3] = tempWeights.x;
      faceWeightsArray[i * 3 + 1] = tempWeights.y;
      faceWeightsArray[i * 3 + 2] = tempWeights.z;

      // The impostor quad is centered on instance.position, so the visible
      // treetop sits half a quad above it; shadow length follows from the
      // height of that top edge above the ground plane.
      const halfHeight = (planeHeight * instance.scale[1]) / 2;
      const heightAboveGround = Math.max(
        instance.position[1] + halfHeight - groundY,
        0.5
      );
      const length = heightAboveGround * stretch;
      // Slightly wider than the tree so the blurred edge has room.
      const width = planeWidth * instance.scale[0] * 1.15;

      tempObject.position.set(
        instance.position[0] + shadowDirX * length * 0.5,
        groundY + 0.05,
        instance.position[2] + shadowDirZ * length * 0.5
      );
      tempObject.rotation.set(-Math.PI / 2, quadYaw, 0);
      tempObject.scale.set(width, length, 1);
      tempObject.updateMatrix();
      mesh.setMatrixAt(i, tempObject.matrix);
    }

    mesh.instanceMatrix.needsUpdate = true;
    return mesh;
  }, [
    instances,
    atlas,
    samplingCache,
    gridSize,
    atlasCoverage,
    geometryArgs.join(","),
    groundY,
    sunPosition.join(","),
    opacity,
    minElevation,
  ]);

  useEffect(() => {
    return () => {
      if (!shadowMesh) return;
      shadowMesh.geometry.dispose();
      shadowMesh.material.dispose();
      shadowMesh.dispose();
    };
  }, [shadowMesh]);

  if (!shadowMesh) return null;

  return <primitive object={shadowMesh} />;
}

function createShadowDecalMaterial({
  atlasTexture,
  gridSize,
  atlasCoverage,
  opacity,
}) {
  const material = new THREE.MeshBasicNodeMaterial();
  material.transparent = true;
  material.depthWrite = false;
  material.toneMapped = false;
  material.side = THREE.DoubleSide;

  // Same cell layout as InstancedOctahedralImpostorMaterial: the atlas is
  // (gridSize + 1) x (gridSize + 1) cells and face indices use that stride.
  const frameCount = gridSize + 1;
  const gridSizeNode = float(frameCount);
  const invGridSize = float(1 / frameCount);
  const maxCell = float(frameCount - 1);
  const atlasNode = texture(atlasTexture);

  const faceIndices = attribute("shadowFaceIndices", "vec3");
  const faceWeights = attribute("shadowFaceWeights", "vec3");
  const vUv = uv();

  const cellFromIndex = (flatIndex) => {
    const row = flatIndex.div(gridSizeNode).floor();
    const col = flatIndex.sub(row.mul(gridSizeNode));
    return vec2(col.clamp(0.0, maxCell), row.clamp(0.0, maxCell));
  };

  const cellA = cellFromIndex(float(faceIndices.x));
  const cellB = cellFromIndex(float(faceIndices.y));
  const cellC = cellFromIndex(float(faceIndices.z));

  const weightSum = faceWeights.x.add(faceWeights.y).add(faceWeights.z);
  const weightA = faceWeights.x.div(weightSum.max(0.0001));
  const weightB = faceWeights.y.div(weightSum.max(0.0001));
  const weightC = faceWeights.z.div(weightSum.max(0.0001));

  const coverageOffset = float((1 - atlasCoverage) * 0.5);
  const coverageScale = float(atlasCoverage);
  const epsilon = float(0.002);

  // Silhouette alpha at a quad-local uv: same coverage remap, horizontal
  // flip and 3-cell barycentric blend as the impostor color lookup, but only
  // the alpha channel is needed.
  const silhouetteAlpha = (localUv) => {
    const scaledU = localUv.x.mul(coverageScale).add(coverageOffset);
    const scaledV = localUv.y.mul(coverageScale).add(coverageOffset);
    const flippedU = float(1.0).sub(scaledU.clamp(0.0, 1.0));
    const safeUv = vec2(
      flippedU.mul(float(1.0).sub(epsilon.mul(2.0))).add(epsilon),
      scaledV.clamp(0.0, 1.0).mul(float(1.0).sub(epsilon.mul(2.0))).add(epsilon)
    );

    const alphaA = atlasNode
      .sample(cellA.add(safeUv).mul(invGridSize).clamp(0.0, 1.0))
      .a.mul(weightA);
    const alphaB = atlasNode
      .sample(cellB.add(safeUv).mul(invGridSize).clamp(0.0, 1.0))
      .a.mul(weightB);
    const alphaC = atlasNode
      .sample(cellC.add(safeUv).mul(invGridSize).clamp(0.0, 1.0))
      .a.mul(weightC);

    return alphaA.add(alphaB).add(alphaC);
  };

  // Fake penumbra: blur radius grows from the trunk (v = 0) to the treetop
  // end of the shadow (v = 1), like a real shadow softening with distance
  // from the occluder.
  const blurRadius = float(0.006).add(vUv.y.mul(0.045));
  const blurOffset = vec2(blurRadius, 0.0);
  const blurred = silhouetteAlpha(vUv)
    .add(silhouetteAlpha(vUv.add(blurOffset)))
    .add(silhouetteAlpha(vUv.sub(blurOffset)))
    .div(3.0);

  const shaped = smoothstep(float(0.12), float(0.7), blurred);
  // Distance fade: sky light fills in the far end of real shadows.
  const distanceFade = float(1.0).sub(vUv.y.mul(0.5));

  // Soft contact patch at the trunk so every tree reads as grounded even
  // where the silhouette is only a thin stem.
  const contactDistance = vUv
    .sub(vec2(0.5, 0.05))
    .mul(vec2(2.8, 10.0))
    .length();
  const contact = float(1.0)
    .sub(contactDistance.clamp(0.0, 1.0))
    .pow(1.5)
    .mul(0.55);

  const opacityUniform = uniform(float(opacity));
  material.colorNode = vec3(0.016, 0.02, 0.014);
  material.opacityNode = shaped
    .mul(distanceFade)
    .max(contact)
    .mul(opacityUniform);

  return material;
}
