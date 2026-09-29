import * as THREE from "three/webgpu";
import { Fn, cos, dFdx, dFdy, dot, float, floor, fract, hue, length, mix, normalMap, positionGeometry, saturation, sin, texture, uv, vec2, vec3 } from "three/tsl";

/**
 * TSL rebuilds of the community surfaces' Blender/Verge3D node graphs. The exported glTF carries only
 * empty core materials; the real shading lives in S8S_v3d_materials node graphs, which this mirrors
 * closely enough for an aerial view: same textures, same world tiling, same HSV grading.
 */

/** The ground graph scales its UVs by this "scale_factor" (half the site's depth) to tile in near-metres. */
const GROUND_UV_SCALE = 597.08447265625;
/** Voronoi scale of the ground graph's anti-tiling: each cell gets its own texture rotation. */
const ANTI_TILE_CELLS = 1.98;

export type SurfaceTextures = Record<string, Record<string, THREE.Texture>>;

export type SurfaceMaterialOptions = {
  /** Strength of the ground's detail normal map. */
  normalStrength?: number;
  /** true = MeshStandardNodeMaterial (lights + shadows), false = MeshBasicNodeMaterial (albedo only). */
  lit?: boolean;
  wireframe?: boolean;
};

/** Same node graph, switchable shading model: standard when lit, basic (unlit albedo) otherwise. */
function makeSurfaceMaterial(name: string, lit: boolean, params: Record<string, unknown> = {}) {
  return lit
    ? new THREE.MeshStandardNodeMaterial({ name, ...params })
    : new THREE.MeshBasicNodeMaterial({ name, ...params });
}

/** Blender's Hue/Saturation/Value node: hue 0.5 is neutral, saturation and value multiply. */
function hsv(color, h: number, s: number, v: number) {
  const shifted = h === 0.5 ? color : hue(color, float((h - 0.5) * Math.PI * 2));
  return saturation(shifted, float(s)).mul(v);
}

const hash21 = Fn(([p]) => fract(sin(dot(p, vec2(127.1, 311.7))).mul(43758.5453)));
const hash22 = Fn(([p]) => vec2(hash21(p), hash21(p.add(vec2(19.19, 7.31)))));

/** Voronoi F1 cell id: a stable random value per cell of the nearest jittered feature point. */
const voronoiCellRandom = Fn(([p]) => {
  const cell = floor(p);
  const local = fract(p);
  const bestDistance = float(8).toVar();
  const bestCell = vec2(0).toVar();
  for (let y = -1; y <= 1; y++) {
    for (let x = -1; x <= 1; x++) {
      const neighbour = cell.add(vec2(x, y));
      const distance = length(vec2(x, y).add(hash22(neighbour)).sub(local));
      const closer = distance.lessThan(bestDistance);
      bestCell.assign(closer.select(neighbour, bestCell));
      bestDistance.assign(closer.select(distance, bestDistance));
    }
  }
  return hash21(bestCell.add(vec2(3.7, 1.3)));
});

function rotate2D(p, angle) {
  const c = cos(angle);
  const s = sin(angle);
  return vec2(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)));
}

/**
 * Samples a tiling texture rotated by a per-Voronoi-cell angle. Derivatives come from the unrotated
 * coordinates (rotated alongside), so cell borders do not break mip selection into seam lines.
 */
function antiTiledSample(map: THREE.Texture, coords, scale: number, cellAngle) {
  const scaled = coords.mul(scale);
  const rotated = rotate2D(scaled, cellAngle);
  const dx = rotate2D(dFdx(scaled), cellAngle);
  const dy = rotate2D(dFdy(scaled), cellAngle);
  return texture(map, rotated).grad(dx, dy);
}

/** Blender "Object" texture coordinates: Blender local XY is three's local X and -Z. */
const objectCoords = vec2(positionGeometry.x, positionGeometry.z.negate());

function groundMaterial(t: Record<string, THREE.Texture>, normalStrength: number, lit: boolean) {
  const worldUV = uv().mul(GROUND_UV_SCALE);
  const cellAngle = voronoiCellRandom(worldUV.mul(ANTI_TILE_CELLS)).mul(Math.PI * 2);

  const forestFloor = hsv(antiTiledSample(t["Image Texture.001"], worldUV, 0.25, cellAngle).rgb, 0.55, 1.3, 0.6);
  const grassRaw = hsv(antiTiledSample(t["Image Texture.005"], worldUV, 0.5, cellAngle).rgb, 0.54, 1.1, 1.6);
  const urban = texture(t["urban_area_mask"], uv()).r;
  const grass = mix(grassRaw.mul(0.8), grassRaw, urban);
  const wood = texture(t["wood_mask"], uv()).r;

  const material = makeSurfaceMaterial("ground", lit);
  material.colorNode = mix(grass, forestFloor, wood);
  if (material instanceof THREE.MeshStandardNodeMaterial) {
    material.roughnessNode = mix(float(0.9), float(0.95), wood);
    material.normalNode = normalMap(texture(t["Image Texture.003"], worldUV), vec2(normalStrength));
  }
  return material;
}

function tiledColorMaterial(name: string, map: THREE.Texture, coords, scale: number, grade: [number, number, number], roughness: number, lit: boolean) {
  const material = makeSurfaceMaterial(name, lit);
  material.colorNode = hsv(texture(map, coords.mul(scale)).rgb, ...grade);
  if (material instanceof THREE.MeshStandardNodeMaterial) material.roughness = roughness;
  return material;
}

/** Builds one material per community surface material name. Unknown names fall back to a neutral grey. */
export function createSurfaceMaterials(textures: SurfaceTextures, { normalStrength = 1.5, lit = true, wireframe = false }: SurfaceMaterialOptions = {}) {
  const t = (material: string, node: string) => textures[material]?.[node];
  const materials: Record<string, THREE.Material> = {};

  if (textures.ground) materials.ground = groundMaterial(textures.ground, normalStrength, lit);
  if (t("way_surface_asphalt", "Node.002"))
    materials.way_surface_asphalt = tiledColorMaterial("asphalt", t("way_surface_asphalt", "Node.002"), objectCoords, 10, [0.5, 1, 0.8], 0.86, lit);
  if (t("way_surface_asphalt_light", "Node.002"))
    materials.way_surface_asphalt_light = tiledColorMaterial("asphalt-light", t("way_surface_asphalt_light", "Node.002"), objectCoords, 10, [0.5, 1, 1.6], 0.86, lit);
  if (t("way_surface_concrete", "CURB_BASE_COLOR"))
    materials.way_surface_concrete = tiledColorMaterial("concrete", t("way_surface_concrete", "CURB_BASE_COLOR"), objectCoords, 0.5, [0.5, 1, 0.5], 0.9, lit);
  if (t("way_surface_gravel", "Image Texture.001"))
    materials.way_surface_gravel = tiledColorMaterial("gravel", t("way_surface_gravel", "Image Texture.001"), objectCoords, 0.25, [0.52, 1.2, 1.4], 1, lit);
  if (t("way_curb", "CURB_BASE_COLOR"))
    materials.way_curb = tiledColorMaterial("curb", t("way_curb", "CURB_BASE_COLOR"), objectCoords, 0.5, [0.5, 1, 1], 0.8, lit);
  if (t("lake_bed", "Image Texture"))
    materials.lake_bed = tiledColorMaterial("lake-bed", t("lake_bed", "Image Texture"), objectCoords, 1, [0.55, 1.4, 0.53], 1, lit);

  const water = makeSurfaceMaterial("lake-water", lit, { transparent: true, opacity: 0.85 });
  water.colorNode = vec3(0.174, 0.257, 0.263);
  if (water instanceof THREE.MeshStandardNodeMaterial) {
    water.roughness = 0.08;
    water.metalness = 0;
  }
  materials.lake_water = water;

  materials["lane markings"] = makeSurfaceMaterial("lane-markings", lit, { color: new THREE.Color(0.653, 0.653, 0.653), ...(lit && { roughness: 1 }) });
  materials.fallback = makeSurfaceMaterial("surface-fallback", lit, { color: 0x777777, ...(lit && { roughness: 1 }) });
  for (const material of Object.values(materials)) (material as THREE.MeshStandardNodeMaterial).wireframe = wireframe;
  return materials;
}

/** Colour textures are sRGB; masks and normal maps are data. Keyed by Blender node name. */
const DATA_TEXTURE_NODES = new Set(["wood_mask", "urban_area_mask", "Image Texture.003", "Node", "Node.001", "Image Texture.002"]);

export function configureSurfaceTexture(texture: THREE.Texture, nodeName: string, materialName: string) {
  texture.colorSpace = DATA_TEXTURE_NODES.has(nodeName) ? THREE.NoColorSpace : THREE.SRGBColorSpace;
  const isMask = nodeName.endsWith("_mask");
  texture.wrapS = texture.wrapT = isMask ? THREE.ClampToEdgeWrapping : THREE.RepeatWrapping;
  texture.anisotropy = 8;
  texture.flipY = false;
  texture.name = `${materialName}/${nodeName}`;
  texture.needsUpdate = true;
  return texture;
}
