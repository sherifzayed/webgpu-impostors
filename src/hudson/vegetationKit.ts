import * as THREE from "three/webgpu";
import { attribute, cameraViewMatrix, float, hue, max, mix, normalize, normalView, positionGeometry, smoothstep, texture, vec4 } from "three/tsl";
import { getGltfLoader, getKtx2Loader } from "./loaders";

/**
 * The vegetation kit Hawk re-plants communities with (the Primland kit): one Draco GLB holding every
 * species mesh plus a KTX2 leaf atlas per species. Ported from
 * hawk/src/.../v3d-components/trees/tree-variants.ts so both apps render the same trees.
 */
const TREE_ASSETS_URL = "/vegetation/trees";

/** Share of lone leaf triangles the shadow-casting stand-in keeps; shadows are blurred, so they need far fewer. */
const SHADOW_LEAF_KEEP_RATIO = 0.2;
/** How far leaf normals are bent towards straight up for lighting (0 = the cards' own normals, 1 = all up). */
const LEAF_NORMAL_UP_BLEND = 0.6;
/** Light the base of a crown keeps relative to its top: a crude self-shadow, since the inside of a canopy sees little sky. */
export const CROWN_BASE_LIGHT = 0.75;
/** Leaf cards are cut out of their atlas at this alpha; lower keeps more fringe, higher thins the canopy. */
export const TREE_ALPHA_TEST = 0.28;

export type TreeTextureQuality = "desktop" | "mobile";

/** Species tuning, in the same order and with the same weights scripts/extract-community.mjs picks species with. */
export const TREE_SPECIES = [
  { key: "treeA", meshName: "hd_treeA", hueShift: [-0.2, 0.1] },
  { key: "treeB", meshName: "hd_treeB", hueShift: [-0.05, 0.08] },
  { key: "treeC", meshName: "hd_treeC", hueShift: [-0.25, 0.12] },
] as const;

/** Name of the per-instance [0, 1) random attribute tree materials read their hue jitter from. */
export const TREE_SEED_ATTRIBUTE = "treeSeed";

/**
 * A tree's stable random value from its ground position (Hawk's position hash), so a tree keeps its
 * hue whichever mesh draws it: the baseline instances, the near-LOD pool or its impostor.
 */
export function treeSeed(x: number, z: number) {
  const value = Math.sin(x * 12.9898 + z * 78.233 + 91.7) * 43758.5453;
  return value - Math.floor(value);
}

export type TreeVariant = {
  key: string;
  /** Kit-scale geometry, trunk base at the origin. */
  geometry: THREE.BufferGeometry;
  /** Thinned canopy that only ever renders into the shadow map. */
  shadowGeometry: THREE.BufferGeometry;
  /** Reads its hue jitter from the `treeSeed` instance attribute. */
  material: THREE.MeshLambertNodeMaterial;
  /** The species with no hue jitter, as the octahedral atlas bakes it; impostors add the jitter back per instance. */
  bakeModel: THREE.Mesh;
  hueShift: readonly [number, number];
  leafAtlas: THREE.Texture;
  /** Height of `geometry` in kit units; a placement's scale is `height / variant.height`. */
  height: number;
};

/**
 * Leaf material for one species: alpha-tested double-sided cards, diffuse only, hue-jittered per
 * instance (green texels only, bark shares the atlas) and darkened towards the base of the crown.
 */
export function createTreeMaterial(leafAtlas: THREE.Texture, hueShift: readonly [number, number], height: number, seed = attribute(TREE_SEED_ATTRIBUTE, "float")) {
  const material = new THREE.MeshLambertNodeMaterial({ map: leafAtlas, alphaTest: TREE_ALPHA_TEST, side: THREE.DoubleSide });

  // `map` stays set although `colorNode` replaces it: the shadow pass cuts leaf cards out of the shadow map from `material.map`'s alpha.
  const leaves = texture(leafAtlas).toVar();
  const greenness = smoothstep(0, 0.08, leaves.g.sub(max(leaves.r, leaves.b)));
  const instanceHue = mix(float(hueShift[0]), float(hueShift[1]), seed).mul(greenness);
  const crownLight = mix(float(CROWN_BASE_LIGHT), float(1), smoothstep(0, 1, positionGeometry.y.div(height)));
  material.colorNode = vec4(hue(leaves.rgb, instanceHue).mul(crownLight), leaves.a);
  // Sideways and downward leaf cards see only the dark lower sky; bending normals up gives the shaded canopy the ground's sky fill.
  const upView = cameraViewMatrix.mul(vec4(0, 1, 0, 0)).xyz;
  material.normalNode = normalize(mix(normalView, upView, LEAF_NORMAL_UP_BLEND));
  material.name = "vegetation-tree-leaves";
  return material;
}

/**
 * Thins a kit canopy: branches (connected pieces) stay whole, only every n-th lone leaf triangle is
 * kept, grown about its centroid by √n so the canopy covers about the same area.
 */
export function thinTreeLeaves(geometry: THREE.BufferGeometry, keepRatio: number): THREE.BufferGeometry {
  const index = geometry.index;
  if (!index || keepRatio >= 1) return geometry.clone();

  const vertexCount = geometry.attributes.position.count;
  const parent = Int32Array.from({ length: vertexCount }, (_, vertex) => vertex);
  const find = (vertex: number): number => {
    let root = vertex;
    while (parent[root] !== root) {
      parent[root] = parent[parent[root]];
      root = parent[root];
    }
    return root;
  };
  for (let offset = 0; offset < index.count; offset += 3) {
    const root = find(index.getX(offset));
    parent[find(index.getX(offset + 1))] = root;
    parent[find(index.getX(offset + 2))] = root;
  }
  const trianglesPerPiece = new Map<number, number>();
  for (let offset = 0; offset < index.count; offset += 3) {
    const root = find(index.getX(offset));
    trianglesPerPiece.set(root, (trianglesPerPiece.get(root) ?? 0) + 1);
  }

  const thinned = geometry.clone();
  const position = thinned.attributes.position;
  const keepEvery = Math.max(1, Math.round(1 / keepRatio));
  const grow = Math.sqrt(keepEvery);
  const keptIndices: number[] = [];
  let leafCount = 0;
  for (let offset = 0; offset < index.count; offset += 3) {
    const a = index.getX(offset);
    const b = index.getX(offset + 1);
    const c = index.getX(offset + 2);
    if (trianglesPerPiece.get(find(a)) !== 1) {
      keptIndices.push(a, b, c);
      continue;
    }
    if (leafCount++ % keepEvery !== 0) continue;
    for (const axis of [0, 1, 2]) {
      const centroid = (position.getComponent(a, axis) + position.getComponent(b, axis) + position.getComponent(c, axis)) / 3;
      for (const vertex of [a, b, c]) {
        position.setComponent(vertex, axis, centroid + (position.getComponent(vertex, axis) - centroid) * grow);
      }
    }
    keptIndices.push(a, b, c);
  }
  const IndexArray = vertexCount > 65535 ? Uint32Array : Uint16Array;
  thinned.setIndex(new THREE.BufferAttribute(new IndexArray(keptIndices), 1));
  thinned.computeBoundingBox();
  thinned.computeBoundingSphere();
  return thinned;
}

async function loadLeafAtlas(renderer, key: string, quality: TreeTextureQuality) {
  const loader = await getKtx2Loader(renderer);
  const atlas = await loader.loadAsync(`${TREE_ASSETS_URL}/summer/${key}-${quality}.ktx2`);
  atlas.colorSpace = THREE.SRGBColorSpace;
  atlas.flipY = false;
  atlas.name = `vegetation-${key}-leaves`;
  return atlas;
}

/** Loads every kit species; resolves once meshes and leaf atlases are ready. */
export async function loadTreeVariants(renderer, quality: TreeTextureQuality = "desktop"): Promise<TreeVariant[]> {
  const gltfLoader = await getGltfLoader(renderer);
  const [gltf, ...atlases] = await Promise.all([
    gltfLoader.loadAsync(`${TREE_ASSETS_URL}/model.glb`),
    ...TREE_SPECIES.map((species) => loadLeafAtlas(renderer, species.key, quality)),
  ]);

  return TREE_SPECIES.map((species, index) => {
    const mesh = gltf.scene.getObjectByName(species.meshName) as THREE.Mesh | undefined;
    if (!mesh?.isMesh) throw new Error(`vegetation kit has no mesh named ${species.meshName}`);
    const geometry = mesh.geometry;
    geometry.computeBoundingBox();
    const height = geometry.boundingBox.max.y - geometry.boundingBox.min.y;
    // Its own geometry clone: instancing adds per-instance attributes to `geometry` that the bake must not see.
    const bakeModel = new THREE.Mesh(geometry.clone(), createTreeMaterial(atlases[index], [0, 0], height, float(0)));
    bakeModel.name = `${species.key}-bake`;
    return {
      key: species.key,
      geometry,
      shadowGeometry: thinTreeLeaves(geometry, SHADOW_LEAF_KEEP_RATIO),
      material: createTreeMaterial(atlases[index], species.hueShift, height),
      bakeModel,
      hueShift: species.hueShift,
      leafAtlas: atlases[index],
      height,
    };
  });
}
