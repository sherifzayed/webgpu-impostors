import { buildOctahedralMesh } from "./octahedralHelper";

const samplingCache = new Map();

// Reused across every sampleOctahedralDirection call so the per-frame hot path
// (up to tens of thousands of calls) allocates nothing and never triggers GC.
const _scratchUV = { u: 0, v: 0 };

function encodeDirectionToOctUV(direction, octType, out) {
  // `direction` points from the object toward the camera (camera - object,
  // normalized). The atlas is baked by placing the render camera at each
  // pntOct point and looking at the origin, so pntOct *is* this same
  // object-to-camera direction - encode it directly, no axis flips.

  const x = direction.x;
  const y = direction.y;
  const z = direction.z;

  if (octType === 0) {
    // HEMI mode: the atlas only stores the upper hemisphere (pntOct.y >= 0),
    // which doubles resolution for the side views that matter for foliage.
    // When the camera dips below the horizon (y < 0) there is no baked data,
    // so clamp to the horizon ring instead of mirroring (which would show a
    // wrong-handed view). X/Z are preserved exactly so azimuth stays correct.
    let mappedX = x;
    let mappedY = y < 0 ? 0 : y;
    let mappedZ = z;

    // L1-normalize (sum of absolutes) to match octHemi's encoding.
    const sum = Math.abs(mappedX) + Math.abs(mappedY) + Math.abs(mappedZ);

    if (sum < 1e-9) {
      out.u = 0.5;
      out.v = 0.5;
      return out;
    }

    const nx = mappedX / sum;
    const ny = mappedY / sum;
    const nz = mappedZ / sum;

    // Inverse of octHemi: it builds x = ox - oy, z = -1 + ox + oy,
    // y = 1 - |x| - |z|. Solving for the plane UV gives:
    //   ox = (x + z + 1) / 2,  oy = (z + 1 - x) / 2
    const ox = (nx + nz + 1) / 2;
    const oy = (nz + 1 - nx) / 2;

    out.u = Math.max(0, Math.min(1, ox));
    out.v = Math.max(0, Math.min(1, oy));
    return out;
  } else {
    // FULL mode: standard octahedral encoding (both hemispheres)
    // Use L1 normalization to match octFull encoding
    const absX = Math.abs(x);
    const absY = Math.abs(y);
    const absZ = Math.abs(z);
    const sum = absX + absY + absZ;

    if (sum < 1e-9) {
      out.u = 0.5;
      out.v = 0.5;
      return out;
    }

    let nx = x / sum;
    let ny = y / sum;
    let nz = z / sum;

    // Handle lower hemisphere (y < 0)
    if (ny < 0) {
      const signX = nx >= 0 ? 1 : -1;
      const signZ = nz >= 0 ? 1 : -1;
      const newX = (1 - Math.abs(nz)) * signX;
      const newZ = (1 - Math.abs(nx)) * signZ;
      nx = newX;
      nz = newZ;
    }

    // Convert to UV: map [-1, 1] to [0, 1]
    out.u = nx * 0.5 + 0.5;
    out.v = nz * 0.5 + 0.5;
    return out;
  }
}

function computeBarycentric2D(px, py, triangleUV, target) {
  const ax = triangleUV[0][0];
  const ay = triangleUV[0][1];
  const bx = triangleUV[1][0];
  const by = triangleUV[1][1];
  const cx = triangleUV[2][0];
  const cy = triangleUV[2][1];

  const v0x = bx - ax;
  const v0y = by - ay;
  const v1x = cx - ax;
  const v1y = cy - ay;
  const v2x = px - ax;
  const v2y = py - ay;

  const denom = v0x * v1y - v1x * v0y;

  if (Math.abs(denom) < 1e-9) {
    target.set(1 / 3, 1 / 3, 1 / 3);
    return target;
  }

  const invDenom = 1 / denom;
  const v = (v2x * v1y - v1x * v2y) * invDenom;
  const w = (v0x * v2y - v2x * v0y) * invDenom;
  const u = 1 - v - w;

  const clampedU = Math.max(u, 0);
  const clampedV = Math.max(v, 0);
  const clampedW = Math.max(w, 0);
  const sum = clampedU + clampedV + clampedW;

  if (sum <= 0) {
    target.set(1 / 3, 1 / 3, 1 / 3);
  } else {
    target.set(clampedU / sum, clampedV / sum, clampedW / sum);
  }

  return target;
}

function buildSamplingCache(octType, gridSize) {
  const key = `${octType}_${gridSize}`;
  if (samplingCache.has(key)) {
    return samplingCache.get(key);
  }

  const octahedralData = buildOctahedralMesh(octType, gridSize);
  const geometry = octahedralData.geometry;
  const indexAttr = geometry.getIndex();

  if (!indexAttr) {
    samplingCache.set(key, null);
    return null;
  }

  const indexArray = indexAttr.array;
  const stride = gridSize + 1;
  const cells = new Array(gridSize * gridSize);
  let cursor = 0;

  for (let row = 0; row < gridSize; row += 1) {
    for (let col = 0; col < gridSize; col += 1) {
      const tri1 = [
        indexArray[cursor],
        indexArray[cursor + 1],
        indexArray[cursor + 2],
      ];
      const tri2 = [
        indexArray[cursor + 3],
        indexArray[cursor + 4],
        indexArray[cursor + 5],
      ];
      cursor += 6;

      const isBackslash =
        tri1[1] - tri1[0] === stride && tri1[2] - tri1[1] === 1;

      const triangles = isBackslash
        ? [
            {
              indices: tri1,
              uv: [
                [0, 0],
                [0, 1],
                [1, 1],
              ],
            },
            {
              indices: tri2,
              uv: [
                [1, 1],
                [1, 0],
                [0, 0],
              ],
            },
          ]
        : [
            {
              indices: tri1,
              uv: [
                [1, 0],
                [0, 0],
                [0, 1],
              ],
            },
            {
              indices: tri2,
              uv: [
                [0, 1],
                [1, 1],
                [1, 0],
              ],
            },
          ];

      cells[row * gridSize + col] = {
        isBackslash,
        triangles,
      };
    }
  }

  geometry.dispose();

  const cache = { gridSize, octType, cells };
  samplingCache.set(key, cache);
  return cache;
}

export function getSamplingCache(octType, gridSize) {
  return buildSamplingCache(octType, gridSize);
}

export function sampleOctahedralDirection({
  direction,
  cache,
  indicesTarget,
  weightsTarget,
}) {
  if (!cache) {
    console.warn("sampleOctahedralDirection: No cache provided");
    return false;
  }

  if (!direction || !direction.isVector3) {
    console.warn("sampleOctahedralDirection: Invalid direction");
    return false;
  }

  // Ensure direction is normalized. Normalize in place (the caller passes a
  // scratch vector) so the hot path allocates nothing.
  if (direction.lengthSq() < 0.01) {
    console.warn(
      "sampleOctahedralDirection: Direction too small after normalization"
    );
    return false;
  }
  direction.normalize();

  const uv = encodeDirectionToOctUV(direction, cache.octType, _scratchUV);

  // Validate UV coordinates
  if (
    isNaN(uv.u) ||
    isNaN(uv.v) ||
    uv.u < 0 ||
    uv.u > 1 ||
    uv.v < 0 ||
    uv.v > 1
  ) {
    console.warn("sampleOctahedralDirection: Invalid UV coordinates", uv);
    return false;
  }

  const gridSize = cache.gridSize;

  const scaledU = uv.u * gridSize;
  const scaledV = uv.v * gridSize;

  const col = Math.min(Math.max(Math.floor(scaledU), 0), gridSize - 1);
  const row = Math.min(Math.max(Math.floor(scaledV), 0), gridSize - 1);

  const localU = Math.min(Math.max(scaledU - col, 0), 0.999999);
  const localV = Math.min(Math.max(scaledV - row, 0), 0.999999);

  const cellIndex = row * gridSize + col;
  const cell = cache.cells[cellIndex];

  if (!cell) {
    console.warn(
      `sampleOctahedralDirection: No cell found at [${row}, ${col}] (index ${cellIndex})`
    );
    return false;
  }

  const triangle =
    cell.isBackslash && localU > localV
      ? cell.triangles[1]
      : !cell.isBackslash && localU + localV > 1
      ? cell.triangles[1]
      : cell.triangles[0];

  if (!triangle || !triangle.indices || triangle.indices.length !== 3) {
    console.warn("sampleOctahedralDirection: Invalid triangle");
    return false;
  }

  computeBarycentric2D(localU, localV, triangle.uv, weightsTarget);

  // Validate/clamp indices inline (no array allocation on the hot path).
  const maxIndex = (gridSize + 1) * (gridSize + 1) - 1;
  indicesTarget.set(
    Math.max(0, Math.min(triangle.indices[0], maxIndex)),
    Math.max(0, Math.min(triangle.indices[1], maxIndex)),
    Math.max(0, Math.min(triangle.indices[2], maxIndex))
  );

  return true;
}
