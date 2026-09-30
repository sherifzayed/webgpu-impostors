import { useMemo } from "react";
import { PlacedImpostorLODField } from "../OctahedralImpostorLODField";
import type { TreePlacement } from "./communityData";
import {
  TREE_SEED_ATTRIBUTE,
  treeSeed,
  type TreeVariant,
} from "./vegetationKit";
import { sunDirection } from "./Sun";

// Must match the <Sun azimuth/elevation> in App.tsx; decals only use the direction.
const SUN_POSITION = sunDirection(150, 34).toArray() as [number, number, number];

type ImpostorTreesProps = {
  variants: TreeVariant[];
  placements: TreePlacement[][];
  /** Real-mesh pool size per species; the biggest trees on screen get the slots. */
  maxNearInstances: number;
  atlasSize: number;
  /** Multiplier on the baked atlas colour, to match the impostors to the real meshes' lighting. */
  exposure: number;
  decalShadows?: boolean;
};

/**
 * The forest as octahedral impostors: one PlacedImpostorLODField per kit species, each one draw call
 * for its far trees plus a pool of real kit meshes for the ones big enough on screen to need them.
 * Atlases are "surface" bakes (colour + normal + depth), so impostors are lit by the scene's sun and
 * sky like the real trees, and receive the shared static shadow map (cast by KitTreeShadowCasters).
 */
export function ImpostorTrees({
  variants,
  placements,
  maxNearInstances,
  atlasSize,
  exposure,
  decalShadows = true,
}: ImpostorTreesProps) {
  const fieldPlacements = useMemo(
    () =>
      variants.map((variant, i) =>
        (placements[i] ?? []).map((p) => ({
          position: p.position,
          rotationY: p.rotationY,
          scale: p.height / variant.height,
          seed: treeSeed(p.position[0], p.position[2]),
        })),
      ),
    [variants, placements],
  );

  const materialOptions = useMemo(
    () =>
      variants.map((variant) => ({
        colorScale: exposure,
        hueShift: variant.hueShift,
      })),
    [variants, exposure],
  );

  return (
    <>
      {variants.map(
        (variant, i) =>
          fieldPlacements[i].length && (
            <PlacedImpostorLODField
              key={variant.key}
              model={variant.bakeModel}
              modelKey={`kit:${variant.key}`}
              placements={fieldPlacements[i]}
              nearMaterial={variant.material}
              nearSeedAttribute={TREE_SEED_ATTRIBUTE}
              materialOptions={materialOptions[i]}
              atlasSize={atlasSize}
              gridSize={16}
              octType={0}
              alphaTest={0.5}
              showInstanceShadows={decalShadows}
              sunPosition={SUN_POSITION}
              maxNearInstances={maxNearInstances}
            />
          ),
      )}
    </>
  );
}
