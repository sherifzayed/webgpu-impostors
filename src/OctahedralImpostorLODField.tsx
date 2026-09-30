import { useEffect, useMemo } from "react";
import * as THREE from "three/webgpu";
import { useFrame, useThree } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { getSamplingCache } from "./utils/octahedralImpostorMath";
import { generateFieldInstances } from "./utils/generateFieldInstances";
import {
  buildLocalModelParts,
  buildLodModelParts,
  computeBakeNormalization,
} from "./utils/buildLodModelParts";
import { useOctahedralAtlasComputeFromObject } from "./hooks/useOctahedralAtlasCompute";
import { useInstancedOctahedralImpostorMesh } from "./hooks/useInstancedOctahedralImpostorMesh";
import TreeImpostorShadowDecals from "./TreeImpostorShadowDecals";

const PARKED_MATRIX = new THREE.Matrix4().compose(
  new THREE.Vector3(0, -100000, 0),
  new THREE.Quaternion(),
  new THREE.Vector3(0, 0, 0),
);

const UP = new THREE.Vector3(0, 1, 0);
const tempObject = new THREE.Object3D();
const tempMatrix = new THREE.Matrix4();
const tempPosition = new THREE.Vector3();
const tempQuaternion = new THREE.Quaternion();
const tempScale = new THREE.Vector3();
const tempOffset = new THREE.Vector3();

/** At most this many near-pool evictions per frame, each of which rescans the pool for its new weakest slot. */
const MAX_EVICTIONS_PER_FRAME = 16;
/** A candidate must look this much bigger on screen than the pool's weakest occupant to evict it (squared-size ratio). */
const EVICTION_MARGIN = 1.2;
/** Max LOD distance checks per frame; larger fields amortize over several frames. */
const LOD_CHECK_BATCH_SIZE = 10000;

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
  geometryArgs = [2, 2],
  ...coreProps
}) {
  const gltf = useGLTF(modelPath);

  const sourceObject = useMemo(() => {
    if (!gltf?.scene) return null;
    gltf.scene.userData.__impostorSourceId = modelPath;
    return gltf.scene;
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
    ],
  );

  // Real GLTF sub-meshes, normalized into the atlas baking space so they
  // align with the impostor billboards they replace.
  const nearParts = useMemo(
    () => (gltf?.scene ? buildLodModelParts(gltf.scene) : []),
    [gltf],
  );

  // The impostor quad spans `geometryArgs` world units per atlas cell, so the
  // normalized real mesh must be scaled by geometryArgs * instanceScale to
  // occupy the same footprint. Width/depth use geometryArgs[0], height [1].
  const [planeWidth, planeHeight] = geometryArgs;
  const nearMatrixAt = useMemo(
    () => (instance, target) => {
      tempObject.position.set(...instance.position);
      tempObject.rotation.set(0, instance.rotationY || 0, 0);
      tempObject.scale.set(
        planeWidth * instance.scale[0],
        planeHeight * instance.scale[1],
        planeWidth * instance.scale[2],
      );
      tempObject.updateMatrix();
      return target.copy(tempObject.matrix);
    },
    [planeWidth, planeHeight],
  );

  if (!sourceObject) return null;

  return (
    <ImpostorLODFieldCore
      source={sourceObject}
      instances={instances}
      nearParts={nearParts}
      nearMatrixAt={nearMatrixAt}
      geometryArgs={geometryArgs}
      {...coreProps}
    />
  );
}

/**
 * An impostor field over explicit placements of an already-loaded model, e.g.
 * trees from a community GLB. Each placement is the model's world transform:
 * `position` is the model origin (a tree's trunk base), `rotationY` its yaw,
 * `scale` a uniform model → world scale, `seed` an optional [0, 1) random
 * value for per-instance variation (impostor hue, and the near meshes'
 * `nearSeedAttribute`).
 *
 * The billboards are derived from the bake's own normalization (see
 * computeBakeNormalization), and near meshes draw the model's geometry in its
 * own space with the placement's matrix, so both LODs land on the same spot
 * and materials that read model-space positions keep working.
 *
 * `modelKey` names the bake in the shared atlas cache.
 */
export function PlacedImpostorLODField({
  model,
  modelKey,
  placements,
  nearMaterial = null,
  nearSeedAttribute = null,
  ...coreProps
}) {
  const source = useMemo(() => {
    model.userData.__impostorSourceId = modelKey;
    return model;
  }, [model, modelKey]);

  const instances = useMemo(() => {
    const { center, scaleFactor } = computeBakeNormalization(model);
    return placements.map((placement) => {
      const s = placement.scale;
      tempOffset
        .copy(center)
        .multiplyScalar(s)
        .applyAxisAngle(UP, placement.rotationY);
      const size = s / scaleFactor;
      return {
        position: [
          placement.position[0] + tempOffset.x,
          placement.position[1] + tempOffset.y,
          placement.position[2] + tempOffset.z,
        ],
        scale: [size, size, size],
        rotationY: placement.rotationY,
        seed: placement.seed ?? 0,
        placement,
      };
    });
  }, [model, placements]);

  // `nearMaterial` lets the near meshes use an instancing-aware variant of the
  // material the atlas was baked with (e.g. one reading `nearSeedAttribute`).
  const nearParts = useMemo(
    () =>
      buildLocalModelParts(model).map((part) => ({
        ...part,
        material: nearMaterial ?? part.material,
      })),
    [model, nearMaterial],
  );

  const nearMatrixAt = useMemo(
    () => (instance, target) => {
      const { position, rotationY, scale } = instance.placement;
      tempPosition.set(position[0], position[1], position[2]);
      tempQuaternion.setFromAxisAngle(UP, rotationY);
      tempScale.setScalar(scale);
      return target.compose(tempPosition, tempQuaternion, tempScale);
    },
    [],
  );

  return (
    <ImpostorLODFieldCore
      source={source}
      instances={instances}
      nearParts={nearParts}
      nearMatrixAt={nearMatrixAt}
      nearSeedAttribute={nearSeedAttribute}
      geometryArgs={[1, 1]}
      {...coreProps}
    />
  );
}

function ImpostorLODFieldCore({
  source,
  instances,
  nearParts,
  nearMatrixAt,
  nearSeedAttribute = null,
  shadowGroundY = -2,
  showInstanceShadows = true,
  shadowOpacity = 0.45,
  sunPosition = [35, 55, 35],
  gridSize = 16,
  atlasSize = 2048,
  octType = 0,
  geometryArgs = [2, 2],
  alphaTest = 0.5,
  lodDistance = 15,
  lodHysteresis = 3,
  maxNearInstances = 32,
  materialOptions = null,
}) {
  const camera = useThree((state) => state.camera);

  const { atlas } = useOctahedralAtlasComputeFromObject({
    source,
    gridSize,
    atlasSize,
    octType,
    enabled: !!source,
  });

  const samplingCache = useMemo(
    () => getSamplingCache(octType, gridSize),
    [octType, gridSize],
  );

  const { instancedMesh, updateFrame, setLodVisibility, isFrustumCulled } =
    useInstancedOctahedralImpostorMesh({
      instances,
      atlas,
      gridSize,
      octType,
      geometryArgs,
      alphaTest,
      materialOptions,
    });

  // nearParts owns its cloned geometry (materials are shared with the caller
  // and must not be disposed here). Dispose only when the parts change.
  useEffect(
    () => () => nearParts.forEach((part) => part.geometry.dispose()),
    [nearParts],
  );

  // One pooled InstancedMesh per sub-mesh, each holding maxNearInstances slots.
  const poolMeshes = useMemo(() => {
    return nearParts.map((part) => {
      if (nearSeedAttribute) {
        part.geometry.setAttribute(
          nearSeedAttribute,
          new THREE.InstancedBufferAttribute(
            new Float32Array(maxNearInstances),
            1,
          ),
        );
      }
      const mesh = new THREE.InstancedMesh(
        part.geometry,
        part.material,
        maxNearInstances,
      );
      mesh.frustumCulled = false;
      // Shadows come from elsewhere (projected decals, or a static shadow
      // map of stand-ins); casting here would double them up.
      mesh.castShadow = false;
      mesh.receiveShadow = true;
      for (let slot = 0; slot < maxNearInstances; slot += 1) {
        mesh.setMatrixAt(slot, PARKED_MATRIX);
      }
      mesh.instanceMatrix.needsUpdate = true;
      // Slots are allocated compactly (see the frame loop), so only the first
      // `activeCount` instances are ever live; keep the drawn range at that
      // size instead of paying vertex work for every parked slot.
      mesh.count = 0;
      return mesh;
    });
  }, [nearParts, maxNearInstances, nearSeedAttribute]);

  // Dispose only the InstancedMesh (its instanceMatrix buffer); the geometry
  // is owned by nearParts and may be reused by a rebuilt pool.
  useEffect(
    () => () => poolMeshes.forEach((mesh) => mesh.dispose()),
    [poolMeshes],
  );

  // nearAssignment[dataIndex] = pool slot currently showing that instance, or -1.
  // slotOccupant[slot] = data index currently occupying that slot, or -1.
  // Slots [0, activeCount) are always the occupied ones: releases back-fill the
  // freed slot with the last active occupant, so the pool meshes can draw just
  // the first activeCount instances. slotScore[slot] is the occupant's inverse
  // squared distance (bigger = closer = more deserving of a real mesh).
  const lodState = useMemo(() => {
    return {
      nearAssignment: new Int32Array(instances.length).fill(-1),
      slotOccupant: new Int32Array(maxNearInstances).fill(-1),
      slotScore: new Float32Array(maxNearInstances),
      activeCount: 0,
      checkCursor: 0,
    };
  }, [instances, maxNearInstances]);

  // A fresh lodState invalidates every slot; reset the drawn range so stale
  // matrices from the previous instance set can't linger as ghost trees.
  useEffect(() => {
    for (const mesh of poolMeshes) {
      mesh.count = 0;
    }
  }, [poolMeshes, lodState]);

  useFrame(() => {
    updateFrame(camera);

    if (instances.length === 0 || poolMeshes.length === 0) {
      return;
    }

    const { nearAssignment, slotOccupant, slotScore } = lodState;
    const seedAttributes = nearSeedAttribute
      ? poolMeshes.map((mesh) => mesh.geometry.getAttribute(nearSeedAttribute))
      : [];
    let matricesChanged = false;

    const writeSlot = (slot, dataIndex) => {
      nearMatrixAt(instances[dataIndex], tempMatrix);
      for (const mesh of poolMeshes) {
        mesh.setMatrixAt(slot, tempMatrix);
      }
      for (const attribute of seedAttributes) {
        attribute.setX(slot, instances[dataIndex].seed ?? 0);
      }
      matricesChanged = true;
    };

    const assign = (dataIndex, score) => {
      const slot = lodState.activeCount;
      lodState.activeCount += 1;
      slotOccupant[slot] = dataIndex;
      slotScore[slot] = score;
      nearAssignment[dataIndex] = slot;
      writeSlot(slot, dataIndex);
      setLodVisibility(dataIndex, false);
    };

    // Keep the live range dense: move the last active occupant into the freed
    // slot, then shrink the range by one.
    const release = (dataIndex) => {
      const slot = nearAssignment[dataIndex];
      const lastSlot = lodState.activeCount - 1;
      if (slot !== lastSlot) {
        const movedIndex = slotOccupant[lastSlot];
        slotOccupant[slot] = movedIndex;
        slotScore[slot] = slotScore[lastSlot];
        nearAssignment[movedIndex] = slot;
        writeSlot(slot, movedIndex);
      }
      slotOccupant[lastSlot] = -1;
      nearAssignment[dataIndex] = -1;
      lodState.activeCount = lastSlot;
      matricesChanged = true;
      setLodVisibility(dataIndex, true);
    };

    // Distance LOD: compare squared distances to avoid a sqrt per instance.
    const enterThresholdSq = lodDistance * lodDistance;
    const leaveThresholdSq = (lodDistance + lodHysteresis) ** 2;

    let weakestSlot = -1;
    const findWeakest = () => {
      weakestSlot = -1;
      let weakest = Infinity;
      for (let slot = 0; slot < lodState.activeCount; slot += 1) {
        if (slotScore[slot] < weakest) {
          weakest = slotScore[slot];
          weakestSlot = slot;
        }
      }
    };
    let evictions = 0;

    const checksThisFrame = Math.min(LOD_CHECK_BATCH_SIZE, instances.length);
    let checkCursor = lodState.checkCursor % instances.length;

    for (let checked = 0; checked < checksThisFrame; checked += 1) {
      const i = checkCursor;
      checkCursor = (checkCursor + 1) % instances.length;
      const instance = instances[i];
      const dx = instance.position[0] - camera.position.x;
      const dy = instance.position[1] - camera.position.y;
      const dz = instance.position[2] - camera.position.z;
      const distSq = Math.max(dx * dx + dy * dy + dz * dz, 1e-6);
      const isNear = nearAssignment[i] !== -1;
      const culled = isFrustumCulled(i);

      // Score = inverse squared distance, so closer instances outrank farther
      // ones when competing for pool slots.
      const score = 1 / distSq;
      const wantsNear = distSq < enterThresholdSq;
      const mustLeave = culled || distSq > leaveThresholdSq;

      if (isNear) {
        if (mustLeave) {
          release(i);
          weakestSlot = -1;
        } else {
          slotScore[nearAssignment[i]] = score;
        }
        continue;
      }
      if (!wantsNear || culled) continue;

      if (lodState.activeCount < maxNearInstances) {
        assign(i, score);
        continue;
      }
      // Pool is full: take the weakest occupant's slot if this one clearly outranks it.
      if (evictions >= MAX_EVICTIONS_PER_FRAME) continue;
      if (weakestSlot === -1) findWeakest();
      if (
        weakestSlot !== -1 &&
        score > slotScore[weakestSlot] * EVICTION_MARGIN
      ) {
        release(slotOccupant[weakestSlot]);
        assign(i, score);
        evictions += 1;
        weakestSlot = -1;
      }
    }
    lodState.checkCursor = checkCursor;

    if (matricesChanged) {
      for (const mesh of poolMeshes) {
        mesh.count = lodState.activeCount;
        mesh.instanceMatrix.needsUpdate = true;
      }
      for (const attribute of seedAttributes) {
        attribute.needsUpdate = true;
      }
    }
  });

  if (!source || instances.length === 0) {
    return null;
  }

  return (
    <>
      {instancedMesh && <primitive object={instancedMesh} />}
      {showInstanceShadows && atlas && (
        <TreeImpostorShadowDecals
          instances={instances}
          atlas={atlas}
          samplingCache={samplingCache}
          gridSize={gridSize}
          geometryArgs={geometryArgs}
          groundY={shadowGroundY}
          sunPosition={sunPosition}
          opacity={shadowOpacity}
        />
      )}
      {poolMeshes.map((mesh, partIndex) => (
        <primitive key={partIndex} object={mesh} />
      ))}
    </>
  );
}
