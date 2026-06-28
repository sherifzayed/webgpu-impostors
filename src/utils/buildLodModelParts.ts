import * as THREE from "three/webgpu";

export function bakeMorphTargetsIntoGeometry(sourceMesh) {
  const geometry = sourceMesh.geometry.clone();
  const influences = sourceMesh.morphTargetInfluences;
  const morphAttributes = geometry.morphAttributes;

  if (!influences || !morphAttributes?.position?.length) {
    return geometry;
  }

  const position = geometry.getAttribute("position");
  const normal = geometry.getAttribute("normal");
  const morphPositions = morphAttributes.position || [];
  const morphNormals = morphAttributes.normal || [];
  const relative = geometry.morphTargetsRelative === true;

  for (let vertex = 0; vertex < position.count; vertex += 1) {
    const baseX = position.getX(vertex);
    const baseY = position.getY(vertex);
    const baseZ = position.getZ(vertex);
    let x = baseX;
    let y = baseY;
    let z = baseZ;

    let normalBaseX = 0;
    let normalBaseY = 0;
    let normalBaseZ = 0;
    let nx = 0;
    let ny = 0;
    let nz = 0;

    if (normal) {
      normalBaseX = normal.getX(vertex);
      normalBaseY = normal.getY(vertex);
      normalBaseZ = normal.getZ(vertex);
      nx = normalBaseX;
      ny = normalBaseY;
      nz = normalBaseZ;
    }

    for (let target = 0; target < morphPositions.length; target += 1) {
      const influence = influences[target] || 0;
      if (influence === 0) continue;

      const morphPosition = morphPositions[target];
      if (relative) {
        x += morphPosition.getX(vertex) * influence;
        y += morphPosition.getY(vertex) * influence;
        z += morphPosition.getZ(vertex) * influence;
      } else {
        x += (morphPosition.getX(vertex) - baseX) * influence;
        y += (morphPosition.getY(vertex) - baseY) * influence;
        z += (morphPosition.getZ(vertex) - baseZ) * influence;
      }

      const morphNormal = morphNormals[target];
      if (normal && morphNormal) {
        if (relative) {
          nx += morphNormal.getX(vertex) * influence;
          ny += morphNormal.getY(vertex) * influence;
          nz += morphNormal.getZ(vertex) * influence;
        } else {
          nx += (morphNormal.getX(vertex) - normalBaseX) * influence;
          ny += (morphNormal.getY(vertex) - normalBaseY) * influence;
          nz += (morphNormal.getZ(vertex) - normalBaseZ) * influence;
        }
      }
    }

    position.setXYZ(vertex, x, y, z);
    if (normal) {
      normal.setXYZ(vertex, nx, ny, nz);
    }
  }

  position.needsUpdate = true;
  if (normal) {
    normal.needsUpdate = true;
  }

  geometry.morphAttributes = {};
  return geometry;
}

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
      const mesh = new THREE.Mesh(
        bakeMorphTargetsIntoGeometry(child),
        child.material
      );
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
