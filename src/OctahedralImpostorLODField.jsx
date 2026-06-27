import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three/webgpu";
import { useFrame, useThree } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { getSamplingCache } from "./utils/octahedralImpostorMath";
import { generateFieldInstances } from "./utils/generateFieldInstances";
import { buildLodModelParts } from "./utils/buildLodModelParts";
import { useOctahedralAtlasCompute } from "./hooks/useOctahedralAtlasCompute";
import { useInstancedOctahedralImpostorMesh } from "./hooks/useInstancedOctahedralImpostorMesh";
import TreeInstanceShadows from "./TreeInstanceShadows";

// Unused pool slots are parked far below the field and scaled to zero so a
// stray normal/depth read can't pick them up at the origin.
const PARKED_MATRIX = new THREE.Matrix4().compose(
  new THREE.Vector3(0, -100000, 0),
  new THREE.Quaternion(),
  new THREE.Vector3(0, 0, 0)
);

const tempObject = new THREE.Object3D();

/**
 * Renders a field of octahedral impostors (one InstancedMesh, one draw call -
 * see useInstancedOctahedralImpostorMesh) and swaps individual instances over
 * to the real GLTF mesh once the camera gets within `lodDistance`, swapping
 * back once it moves past `lodDistance + lodHysteresis`. The hysteresis gap
 * avoids flicker for instances sitting right at the boundary.
 *
 * The real-mesh side is drawn through a small fixed-size pool of InstancedMesh
 * (one per GLTF sub-mesh, to support multi-material models), reused across
 * whichever data instances are currently "near".
 *
 * Crucially, the real geometry is normalized into the SAME space the atlas was
 * baked in (see buildLodModelParts), then scaled by geometryArgs * instanceScale
 * so it lands exactly where the billboard it replaces was - otherwise the raw
 * model would be the wrong size/offset and appear to vanish on zoom-in.
 */
export default function OctahedralImpostorLODField({
  modelPath = "/tree.glb",
  position = [0, 0, 0],
  count = 150,
  areaSize = [60, 60],
  minHeight = 0,
  maxHeight = 0,
  minScale = 0.7,
  maxScale = 1.4,
  heightVariation = 0,
  widthVariation = 0,
  baseScale = [1, 1, 1],
  avoidRadius = 0,
  seed = 2024,
  randomYaw = true,
  shadowGroundY = -2,
  showInstanceShadows = true,
  gridSize = 16,
  atlasSize = 2048,
  octType = 0,
  geometryArgs = [2, 2],
  alphaTest = 0.5,
  usePostProcessing = true,
  brightness = 1.0,
  contrast = 1.0,
  optimizeSize = false,
  atlasCoverage = 1.0,
  usePostDilatation = false,
  dilationRadius = 1,
  directionThresholdRadians = 0.0872665,
  useDither = false,
  showWireframe = false,
  lodDistance = 15,
  lodHysteresis = 3,
  maxNearInstances = 32,
  maxImpostorUpdatesPerFrame = 20000,
  lodCheckBatchSize = 10000,
}) {
  const { camera } = useThree();
  const gltf = useGLTF(modelPath);

  const sourceMesh = useMemo(() => {
    if (!gltf?.scene) return null;

    let foundMesh = null;
    gltf.scene.traverse((child) => {
      if (!foundMesh && child.isMesh) {
        foundMesh = child;
      }
    });

    if (foundMesh) {
      foundMesh.userData.__impostorSourceId = modelPath;
    }

    return foundMesh;
  }, [gltf, modelPath]);

  const instances = useMemo(
    () =>
      generateFieldInstances({
        count,
        areaSize,
        position,
        baseScale,
        minHeight,
        maxHeight,
        minScale,
        maxScale,
        heightVariation,
        widthVariation,
        avoidRadius,
        seed,
        randomYaw,
      }),
    [
      count,
      areaSize,
      position,
      baseScale,
      minHeight,
      maxHeight,
      minScale,
      maxScale,
      heightVariation,
      widthVariation,
      avoidRadius,
      seed,
      randomYaw,
    ]
  );

  const { atlas } = useOctahedralAtlasCompute({
    mesh: sourceMesh,
    gridSize,
    atlasSize,
    octType,
    enabled: !!sourceMesh,
    usePostProcessing,
    brightness,
    contrast,
    optimizeSize,
    atlasCoverage,
    usePostDilatation,
    dilationRadius,
  });

  const samplingCache = useMemo(
    () => getSamplingCache(octType, gridSize),
    [octType, gridSize]
  );

  const { instancedMesh, updateFrame } = useInstancedOctahedralImpostorMesh({
    instances,
    atlas,
    gridSize,
    octType,
    samplingCache,
    geometryArgs,
    atlasCoverage,
    alphaTest,
    useDither,
    showWireframe,
    directionThresholdRadians,
    maxUpdatesPerFrame: maxImpostorUpdatesPerFrame,
  });

  // Real GLTF sub-meshes, normalized into the atlas baking space so they
  // align with the impostor billboards they replace.
  const lodParts = useMemo(
    () => (gltf?.scene ? buildLodModelParts(gltf.scene) : []),
    [gltf]
  );

  // lodParts owns the cloned geometry (materials are shared with the GLTF cache
  // and must not be disposed here). Dispose only when the parts themselves change.
  useEffect(() => {
    return () => {
      lodParts.forEach((part) => part.geometry.dispose());
    };
  }, [lodParts]);

  // One pooled InstancedMesh per sub-mesh, each holding maxNearInstances slots.
  const poolMeshes = useMemo(() => {
    if (lodParts.length === 0) return [];

    return lodParts.map((part) => {
      const mesh = new THREE.InstancedMesh(
        part.geometry,
        part.material,
        maxNearInstances
      );
      mesh.frustumCulled = false;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      for (let slot = 0; slot < maxNearInstances; slot += 1) {
        mesh.setMatrixAt(slot, PARKED_MATRIX);
      }
      mesh.instanceMatrix.needsUpdate = true;
      return mesh;
    });
  }, [lodParts, maxNearInstances]);

  useEffect(() => {
    // Dispose only the InstancedMesh (its instanceMatrix buffer); the geometry
    // is owned by lodParts and may be reused by a rebuilt pool.
    return () => {
      poolMeshes.forEach((mesh) => mesh.dispose());
    };
  }, [poolMeshes]);

  // nearAssignment[dataIndex] = pool slot currently showing that instance, or -1.
  // slotOccupant[slot] = data index currently occupying that slot, or -1.
  const lodState = useMemo(
    () => ({
      nearAssignment: new Int32Array(instances.length).fill(-1),
      slotOccupant: new Int32Array(maxNearInstances).fill(-1),
      checkCursor: 0,
    }),
    [instances, maxNearInstances]
  );

  // The impostor quad spans `geometryArgs` world units per atlas cell, so the
  // normalized real mesh must be scaled by geometryArgs * instanceScale to
  // occupy the same footprint. Width/depth use geometryArgs[0], height [1].
  const [planeWidth, planeHeight] = geometryArgs;

  useFrame(() => {
    updateFrame(camera);

    if (instances.length === 0 || poolMeshes.length === 0) {
      return;
    }

    const { nearAssignment, slotOccupant } = lodState;
    const impostorScaleAttr =
      instancedMesh?.geometry.getAttribute("instanceScale");
    let impostorScaleChanged = false;
    let matricesChanged = false;

    const enterThresholdSq = lodDistance * lodDistance;
    const leaveThresholdSq = (lodDistance + lodHysteresis) ** 2;
    const checksThisFrame = Math.min(lodCheckBatchSize, instances.length);
    let checkCursor = lodState.checkCursor % instances.length;

    for (let checked = 0; checked < checksThisFrame; checked += 1) {
      const i = checkCursor;
      checkCursor = (checkCursor + 1) % instances.length;
      const instance = instances[i];
      const dx = instance.position[0] - camera.position.x;
      const dy = instance.position[1] - camera.position.y;
      const dz = instance.position[2] - camera.position.z;
      const distSq = dx * dx + dy * dy + dz * dz;

      const isNear = nearAssignment[i] !== -1;

      if (!isNear && distSq < enterThresholdSq) {
        const freeSlot = slotOccupant.indexOf(-1);
        if (freeSlot === -1) {
          // Pool is full - this instance just stays an impostor for now.
          continue;
        }

        slotOccupant[freeSlot] = i;
        nearAssignment[i] = freeSlot;

        tempObject.position.set(...instance.position);
        tempObject.rotation.set(0, instance.rotationY || 0, 0);
        tempObject.scale.set(
          planeWidth * instance.scale[0],
          planeHeight * instance.scale[1],
          planeWidth * instance.scale[2]
        );
        tempObject.updateMatrix();
        for (const mesh of poolMeshes) {
          mesh.setMatrixAt(freeSlot, tempObject.matrix);
        }
        matricesChanged = true;

        if (impostorScaleAttr) {
          impostorScaleAttr.setXYZ(i, 0, 0, 0);
          impostorScaleChanged = true;
        }
      } else if (isNear && distSq > leaveThresholdSq) {
        const slot = nearAssignment[i];
        slotOccupant[slot] = -1;
        nearAssignment[i] = -1;

        for (const mesh of poolMeshes) {
          mesh.setMatrixAt(slot, PARKED_MATRIX);
        }
        matricesChanged = true;

        if (impostorScaleAttr) {
          impostorScaleAttr.setXYZ(
            i,
            instance.scale[0],
            instance.scale[1],
            instance.scale[2]
          );
          impostorScaleChanged = true;
        }
      }
    }
    lodState.checkCursor = checkCursor;

    if (matricesChanged) {
      for (const mesh of poolMeshes) {
        mesh.instanceMatrix.needsUpdate = true;
      }
    }

    if (impostorScaleChanged) {
      impostorScaleAttr.needsUpdate = true;
    }
  });

  if (!sourceMesh || instances.length === 0) {
    return null;
  }

  return (
    <>
      {instancedMesh && <primitive object={instancedMesh} />}
      {showInstanceShadows && (
        <TreeInstanceShadows instances={instances} groundY={shadowGroundY} />
      )}
      {poolMeshes.map((mesh, partIndex) => (
        <primitive key={partIndex} object={mesh} />
      ))}
    </>
  );
}

useGLTF.preload("/tree.glb");
