import * as THREE from "three/webgpu";

/**
 * Reproduces the exact model normalization the atlas baker applies in
 * useOctahedralAtlasCompute (generateAtlasWithCompute) so the real mesh used
 * for the close-up LOD lives in the same space the impostor billboard
 * represents.
 *
 * The baker: clones every sub-mesh (local transform only, ancestor transforms
 * are intentionally dropped), recenters each geometry on its own bounding
 * sphere, computes the combined bounding sphere, then scales the whole group
 * by 0.5 / (radius * 1.5) about the group origin. We mirror that, then bake the
 * resulting world matrices straight into the geometries so each part can be
 * drawn with a single InstancedMesh.
 *
 * Returns one entry per sub-mesh: { geometry, material } already in normalized
 * (atlas) space. To place an instance in the field, scale these by
 * geometryArgs * instanceScale (the impostor quad spans `geometryArgs` world
 * units per atlas cell) and translate to the instance position.
 */
export function buildLodModelParts(gltfScene) {
  if (!gltfScene) return [];

  const group = new THREE.Group();
  const meshes = [];

  gltfScene.traverse((child) => {
    if (child.isMesh && child.geometry) {
      const mesh = new THREE.Mesh(child.geometry.clone(), child.material);
      mesh.position.copy(child.position);
      mesh.quaternion.copy(child.quaternion);
      mesh.scale.copy(child.scale);
      group.add(mesh);
      meshes.push(mesh);
    }
  });

  if (meshes.length === 0) return [];

  // Recenter each geometry on its bounding-sphere center, compensating with the
  // node position so world placement is unchanged (matches the baker).
  meshes.forEach((mesh) => {
    mesh.geometry.computeBoundingSphere();
    const center = mesh.geometry.boundingSphere.center.clone();
    mesh.geometry.translate(-center.x, -center.y, -center.z);
    mesh.position.add(center);
  });

  group.updateMatrixWorld(true);

  // Combined bounding sphere across all parts, in group space.
  const boundingSphere = new THREE.Sphere();
  meshes.forEach((mesh) => {
    mesh.geometry.computeBoundingSphere();
    const partSphere = mesh.geometry.boundingSphere.clone();
    partSphere.applyMatrix4(mesh.matrixWorld);
    boundingSphere.union(partSphere);
  });

  const radius = boundingSphere.radius * 1.5;
  const scaleFactor = radius > 0 ? 0.5 / radius : 1;
  group.scale.setScalar(scaleFactor);
  group.position.set(0, 0, 0);
  group.updateMatrixWorld(true);

  // Bake the final world matrix into each geometry so a plain InstancedMesh can
  // draw it with only a per-instance transform.
  return meshes.map((mesh) => {
    const geometry = mesh.geometry.clone();
    geometry.applyMatrix4(mesh.matrixWorld);
    geometry.computeVertexNormals?.();
    return { geometry, material: mesh.material };
  });
}
