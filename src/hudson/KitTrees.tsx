import { useEffect, useMemo } from "react";
import * as THREE from "three/webgpu";
import type { TreePlacement } from "./communityData";
import { SHADOW_CASTER_LAYER, requestShadowUpdate } from "./staticShadows";
import {
  TREE_SEED_ATTRIBUTE,
  treeSeed,
  type TreeVariant,
} from "./vegetationKit";

const _position = new THREE.Vector3();
const _quaternion = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _matrix = new THREE.Matrix4();
const UP = new THREE.Vector3(0, 1, 0);

export function placementMatrix(
  placement: TreePlacement,
  variant: TreeVariant,
  target = _matrix,
) {
  _position.set(...placement.position);
  _quaternion.setFromAxisAngle(UP, placement.rotationY);
  _scale.setScalar(placement.height / variant.height);
  return target.compose(_position, _quaternion, _scale);
}

function createInstances(
  geometry: THREE.BufferGeometry,
  variant: TreeVariant,
  placements: TreePlacement[],
) {
  const seeds = new Float32Array(
    placements.map((p) => treeSeed(p.position[0], p.position[2])),
  );
  geometry.setAttribute(
    TREE_SEED_ATTRIBUTE,
    new THREE.InstancedBufferAttribute(seeds, 1),
  );
  const mesh = new THREE.InstancedMesh(
    geometry,
    variant.material,
    placements.length,
  );
  placements.forEach((placement, i) =>
    mesh.setMatrixAt(i, placementMatrix(placement, variant)),
  );
  mesh.instanceMatrix.needsUpdate = true;
  mesh.computeBoundingSphere();
  return mesh;
}

/**
 * Every tree as a real kit mesh, the way Hawk draws them. It is the ground truth the impostor field
 * is judged against, and the shadow casters both modes share.
 */
export function KitTreeShadowCasters({
  variants,
  placements,
  debugVisible = false,
}: {
  variants: TreeVariant[];
  placements: TreePlacement[][];
  /** Also draw the casters with the main camera, to inspect what the shadow map sees. */
  debugVisible?: boolean;
}) {
  const casters = useMemo(() => {
    const group = new THREE.Group();
    group.name = "tree-shadow-casters";
    variants.forEach((variant, i) => {
      if (!placements[i]?.length) return;
      const mesh = createInstances(
        variant.shadowGeometry,
        variant,
        placements[i],
      );
      mesh.name = `tree-shadow-${variant.key}`;
      mesh.castShadow = true;
      mesh.receiveShadow = false;
      mesh.layers.set(SHADOW_CASTER_LAYER);
      group.add(mesh);
    });
    return group;
  }, [variants, placements]);

  useEffect(() => {
    casters.children.forEach((mesh) => {
      mesh.layers.set(SHADOW_CASTER_LAYER);
      if (debugVisible) mesh.layers.enable(0);
    });
  }, [casters, debugVisible]);

  useEffect(() => {
    requestShadowUpdate();
    return () =>
      casters.children.forEach((mesh) =>
        (mesh as THREE.InstancedMesh).dispose(),
      );
  }, [casters]);

  return <primitive object={casters} />;
}
