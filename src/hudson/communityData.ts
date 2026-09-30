/** Shapes of the files scripts/extract-community.mjs writes to public/<community>/. */

export type TreesFile = {
  version: 1;
  /** Kit species keys, indexed by `trees.species`. */
  species: string[];
  /** Blender tree templates the placements came from, indexed by `trees.template`. */
  templates: { key: string; height: number }[];
  count: number;
  trees: {
    species: number[];
    template: number[];
    /** Flat xyz, three.js Y-up metres; y is the trunk base on the terrain. */
    position: number[];
    /** Radians about +Y. */
    yaw: number[];
    /** Final tree height in metres (template height × Blender scale × Hawk's size jitter). */
    height: number[];
  };
};

export type CommunityFile = {
  name: string | null;
  areaId: string | null;
  latitude: number | null;
  longitude: number | null;
  srid: number | null;
  /** Site width × depth in metres. */
  dimensions: number[];
  /** material name → Blender node name → texture path, relative to the community folder. */
  textures: Record<string, Record<string, string>>;
};

/** One tree ready to place, in the form OctahedralImpostorLODField instances use. */
export type TreePlacement = {
  position: [number, number, number];
  rotationY: number;
  height: number;
};

export async function loadCommunity(baseUrl: string): Promise<{ community: CommunityFile; trees: TreesFile }> {
  const [community, trees] = await Promise.all([
    fetch(`${baseUrl}/community.json`).then((r) => r.json()),
    fetch(`${baseUrl}/trees.json`).then((r) => r.json()),
  ]);
  return { community, trees };
}

/** Splits the flat tree arrays into one placement list per kit species. */
export function placementsBySpecies(trees: TreesFile): TreePlacement[][] {
  const bySpecies: TreePlacement[][] = trees.species.map(() => []);
  const { species, position, yaw, height } = trees.trees;
  for (let i = 0; i < species.length; i++) {
    bySpecies[species[i]].push({
      position: [position[i * 3], position[i * 3 + 1], position[i * 3 + 2]],
      rotationY: yaw[i],
      height: height[i],
    });
  }
  return bySpecies;
}
