const hashSeed = (value) => {
  const stringValue =
    typeof value === "number" ? value.toString() : String(value);
  let hash = 0;
  for (let index = 0; index < stringValue.length; index += 1) {
    hash = (hash << 5) - hash + stringValue.charCodeAt(index);
    hash |= 0;
  }
  return hash >>> 0;
};

const createSeededRandom = (seedValue) => {
  // Mulberry32 generator ensures reproducible distributions.
  let seed = hashSeed(seedValue);
  return () => {
    seed += 0x6d2b79f5;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/**
 * Scatters `count` instances within an area using a seeded RNG, so the same
 * seed always produces the same layout. Shared between the instanced field
 * renderer and the LOD field renderer.
 */
export function generateFieldInstances({
  count,
  areaSize,
  position,
  baseScale,
  minHeight = 0,
  maxHeight = 0,
  minScale = 0.7,
  maxScale = 1.4,
  avoidRadius = 0,
  seed = 2024,
  randomYaw = true,
  heightVariation = 0,
  widthVariation = 0,
}) {
  if (count <= 0) {
    return [];
  }

  const [width, depth] = areaSize;
  const [originX, originY, originZ] = position;
  const [baseScaleX, baseScaleY, baseScaleZ] = baseScale;
  const random = createSeededRandom(seed);

  const generated = [];
  let attempts = 0;

  while (generated.length < count && attempts < count * 10) {
    attempts += 1;

    const offsetX = (random() - 0.5) * width;
    const offsetZ = (random() - 0.5) * depth;

    const candidateX = originX + offsetX;
    const candidateZ = originZ + offsetZ;

    if (
      avoidRadius > 0 &&
      Math.hypot(candidateX - originX, candidateZ - originZ) < avoidRadius
    ) {
      continue;
    }

    const heightOffset =
      minHeight === maxHeight
        ? minHeight
        : minHeight + random() * (maxHeight - minHeight);

    const uniformScale =
      minScale === maxScale
        ? Math.max(0.0001, minScale)
        : Math.max(0.0001, minScale + random() * (maxScale - minScale));

    const yaw = randomYaw ? random() * Math.PI * 2 : 0;
    const widthScale =
      widthVariation > 0
        ? Math.max(0.0001, 1 + (random() * 2 - 1) * widthVariation)
        : 1;
    const heightScale =
      heightVariation > 0
        ? Math.max(0.0001, 1 + (random() * 2 - 1) * heightVariation)
        : 1;

    generated.push({
      position: [candidateX, originY + heightOffset, candidateZ],
      scale: [
        Math.abs(baseScaleX * uniformScale * widthScale),
        Math.abs(baseScaleY * uniformScale * heightScale),
        Math.abs(baseScaleZ * uniformScale * widthScale),
      ],
      rotationY: yaw,
    });
  }

  return generated;
}
