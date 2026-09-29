import { loadCommunity } from "./communityData";
import { loadTreeVariants, type TreeVariant } from "./vegetationKit";

/**
 * Loads kept outside App.tsx on purpose: hot-reloading App.tsx re-runs that module, and promises
 * created there would re-download the community and rebuild the tree kit on every save.
 */

/** Output folder of `node scripts/extract-community.mjs --out hudson`. */
export const COMMUNITY_URL = "/hudson";

export const communityPromise = loadCommunity(COMMUNITY_URL);

let treeVariantsPromise: Promise<TreeVariant[]> | undefined;

export function getTreeVariants(renderer): Promise<TreeVariant[]> {
  treeVariantsPromise ??= loadTreeVariants(renderer, "desktop");
  return treeVariantsPromise;
}
