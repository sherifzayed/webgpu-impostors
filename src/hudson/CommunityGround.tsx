import { useThree } from "@react-three/fiber";
import { use, useEffect, useMemo } from "react";
import * as THREE from "three/webgpu";
import type { CommunityFile } from "./communityData";
import { configureSurfaceTexture, createSurfaceMaterials, type SurfaceTextures } from "./groundMaterials";
import { getGltfLoader, getKtx2Loader } from "./loaders";

const cache = new Map<string, Promise<{ scene: THREE.Object3D; textures: SurfaceTextures }>>();

async function loadSurfaces(renderer, baseUrl: string, community: CommunityFile) {
  const [gltfLoader, ktx2Loader] = await Promise.all([getGltfLoader(renderer), getKtx2Loader(renderer)]);
  const textureJobs = Object.entries(community.textures).flatMap(([materialName, nodes]) =>
    Object.entries(nodes).map(async ([nodeName, file]) => {
      const texture = await ktx2Loader.loadAsync(`${baseUrl}/${file}`);
      return [materialName, nodeName, configureSurfaceTexture(texture, nodeName, materialName)] as const;
    }),
  );
  const [gltf, ...loaded] = await Promise.all([gltfLoader.loadAsync(`${baseUrl}/ground.glb`), ...textureJobs]);
  const textures: SurfaceTextures = {};
  for (const [materialName, nodeName, texture] of loaded) (textures[materialName] ??= {})[nodeName] = texture;
  return { scene: gltf.scene, textures };
}

/** Terrain, roads, curbs and lakes from the community GLB, re-shaded with TSL rebuilds of their Blender materials. */
type CommunityGroundProps = {
  baseUrl: string;
  community: CommunityFile;
  normalStrength?: number;
  /** true = standard (lit, shadowed) materials, false = unlit albedo-only materials. */
  lit?: boolean;
  receiveShadows?: boolean;
  wireframe?: boolean;
  visible?: boolean;
};

export function CommunityGround({
  baseUrl,
  community,
  normalStrength = 1.5,
  lit = true,
  receiveShadows = true,
  wireframe = false,
  visible = true,
}: CommunityGroundProps) {
  const renderer = useThree((state) => state.gl);
  const key = `${baseUrl}`;
  if (!cache.has(key)) cache.set(key, loadSurfaces(renderer, baseUrl, community));
  const { scene, textures } = use(cache.get(key));

  const materials = useMemo(
    () => createSurfaceMaterials(textures, { normalStrength, lit, wireframe }),
    [textures, normalStrength, lit, wireframe],
  );

  useEffect(() => {
    scene.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (!mesh.isMesh) return;
      // Remember the glTF material name: the materials get swapped again whenever they are rebuilt.
      mesh.userData.sourceMaterial ??= (mesh.material as THREE.Material).name;
      mesh.material = materials[mesh.userData.sourceMaterial] ?? materials.fallback;
      mesh.receiveShadow = receiveShadows;
      mesh.castShadow = false;
    });
    return () => Object.values(materials).forEach((material) => material.dispose());
  }, [scene, materials, receiveShadows]);

  return <primitive object={scene} visible={visible} />;
}
