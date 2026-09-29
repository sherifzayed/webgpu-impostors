import * as THREE from "three/webgpu";
import {
  texture,
  attribute,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
  float,
  Fn,
  positionGeometry,
  modelViewMatrix,
  hue,
  max,
  mix,
  smoothstep,
  normalize,
  varying,
  cameraWorldMatrix,
  cameraPosition,
  cameraProjectionMatrix,
  modelWorldMatrix,
} from "three/tsl";

export const ALIGNMENT_OFFSET_RADIANS = -Math.PI / 2;

/**
 * Single shared material for an entire field of octahedral impostors.
 *
 * Billboard orientation follows the same trick three/webgpu's own
 * SpriteNodeMaterial uses (setupPositionView.js): the quad corner offset is
 * added directly in view space, which is automatically camera-facing
 * because view-space X/Y are the camera's right/up axes. That makes the
 * billboard rotation entirely GPU-side with no per-instance JS quaternion
 * math, and it composes naturally with instancing since the anchor position
 * comes from a per-instance attribute instead of a single object transform.
 *
 * Atlas cell blending logic (flat index -> grid cell -> UV -> 3-tap
 * barycentric blend) is ported from OctahedralImpostor.jsx; the face
 * indices/weights come from selectOctahedralFrames in the vertex shader
 * instead of per-object uniforms.
 */
/** Per-instance hue rotation in [min, max] radians by `seed`, on green texels only so bark keeps its colour. */
export function applyInstanceHue(color, hueShift, seed) {
  const greenness = smoothstep(0, 0.08, color.g.sub(max(color.r, color.b)));
  return hue(
    color,
    mix(float(hueShift[0]), float(hueShift[1]), seed).mul(greenness),
  );
}

/**
 * View-space position of this billboard vertex: the instance anchor
 * (instanceOffset) projected to view space, plus the quad corner offset in
 * view-space X/Y so the quad always faces the camera. Multiplying by the
 * visibility flag (instanceScale.w) collapses hidden or frustum-culled
 * instances to a degenerate point, so they rasterize no fragments.
 */
export function billboardPositionView() {
  const instanceOffset = attribute("instanceOffset", "vec3");
  const instanceScale = attribute("instanceScale", "vec4");
  const mvPosition = modelViewMatrix.mul(vec3(instanceOffset));
  const alignedPosition = positionGeometry.xy
    .mul(instanceScale.xy)
    .mul(instanceScale.w);
  return vec4(mvPosition.xy.add(alignedPosition), mvPosition.zw);
}

/**
 * Picks the three baked frames nearest this instance's view direction and
 * their barycentric weights, on the GPU, every frame. This is a closed-form
 * port of sampleOctahedralDirection (octahedralImpostorMath): encode the
 * instance-local camera direction to octahedral UV, find its grid cell, split
 * the cell along the same diagonal octPlaneIndices uses, and weight the
 * triangle's corners. It runs in the vertex stage (4 vertices per instance,
 * all computing the same answer), so a field costs no per-frame CPU work or
 * attribute uploads for view selection, however many instances it has.
 */
export function selectOctahedralFrames({ gridSize, octType = 0 }) {
  const instanceOffset = attribute("instanceOffset", "vec3");
  const yawSinCos = attribute("instanceYawSinCos", "vec3");

  // Instance -> camera, the direction the atlas is indexed by. No normalize:
  // the octahedral encoding L1-normalizes, and rotations are linear.
  const worldPosition = modelWorldMatrix.mul(vec4(instanceOffset, 1)).xyz;
  const viewDir = cameraPosition.sub(worldPosition);
  const sinYaw = yawSinCos.x;
  const cosYaw = yawSinCos.y;
  const localX = viewDir.x.mul(cosYaw).sub(viewDir.z.mul(sinYaw));
  const localZ = viewDir.x.mul(sinYaw).add(viewDir.z.mul(cosYaw));
  const alignCos = Math.cos(ALIGNMENT_OFFSET_RADIANS);
  const alignSin = Math.sin(ALIGNMENT_OFFSET_RADIANS);
  const x = localX.mul(alignCos).sub(localZ.mul(alignSin));
  const z = localX.mul(alignSin).add(localZ.mul(alignCos));
  const y = viewDir.y;

  let octU;
  let octV;
  if (octType === 1) {
    const sum = x.abs().add(y.abs()).add(z.abs()).max(1e-6);
    const nx = x.div(sum);
    const ny = y.div(sum);
    const nz = z.div(sum);
    const below = ny.lessThan(0);
    const foldedX = below.select(
      float(1).sub(nz.abs()).mul(nx.greaterThanEqual(0).select(1, -1)),
      nx,
    );
    const foldedZ = below.select(
      float(1).sub(nx.abs()).mul(nz.greaterThanEqual(0).select(1, -1)),
      nz,
    );
    octU = foldedX.mul(0.5).add(0.5);
    octV = foldedZ.mul(0.5).add(0.5);
  } else {
    // HEMI: below the horizon there is no baked data; clamp to the horizon ring.
    const sum = x.abs().add(y.max(0)).add(z.abs()).max(1e-6);
    const nx = x.div(sum);
    const nz = z.div(sum);
    octU = nx.add(nz).add(1).mul(0.5).clamp(0, 1);
    octV = nz.sub(nx).add(1).mul(0.5).clamp(0, 1);
  }

  const stride = gridSize + 1;
  const scaledU = octU.mul(gridSize);
  const scaledV = octV.mul(gridSize);
  const col = scaledU.floor().clamp(0, gridSize - 1);
  const row = scaledV.floor().clamp(0, gridSize - 1);
  const u = scaledU.sub(col).clamp(0, 0.999999);
  const v = scaledV.sub(row).clamp(0, 0.999999);

  // Corners: a = (0,0), b = (0,1), c = (1,1), d = (1,0) in cell (u, v).
  const a = row.mul(stride).add(col);
  const b = a.add(stride);
  const c = b.add(1);
  const d = a.add(1);

  // Quadrants alternate the split diagonal (octPlaneIndices).
  const half = Math.max(1, Math.floor(gridSize / 2));
  const alt = col.div(half).floor().add(row.div(half).floor()).mod(2);
  const backslash = octType === 1 ? alt.greaterThan(0.5) : alt.lessThan(0.5);

  const backSecond = u.greaterThan(v);
  const backIndices = backSecond.select(vec3(c, d, a), vec3(a, b, c));
  const backWeights = backSecond.select(
    vec3(v, u.sub(v), float(1).sub(u)),
    vec3(float(1).sub(v), v.sub(u), u),
  );
  const forwardSecond = u.add(v).greaterThan(1);
  const forwardIndices = forwardSecond.select(vec3(b, c, d), vec3(d, a, b));
  const forwardWeights = forwardSecond.select(
    vec3(float(1).sub(u), u.add(v).sub(1), float(1).sub(v)),
    vec3(u, float(1).sub(u).sub(v), v),
  );

  return {
    faceIndices: varying(
      backslash.select(backIndices, forwardIndices),
      "vImpostorFaceIndices",
    ),
    faceWeights: varying(
      backslash.select(backWeights, forwardWeights).max(0),
      "vImpostorFaceWeights",
    ),
  };
}

/**
 * The per-fragment atlas lookup every instanced impostor material shares:
 * turns the instance's three nearest baked frames (selectOctahedralFrames)
 * and their barycentric weights into atlas UVs for this quad texel, and
 * blends any atlas baked with the same layout.
 */
export function buildImpostorAtlasSampler({ gridSize, octType = 0 }) {
  // The octahedral grid has (gridSize + 1) viewpoints per side, so the atlas
  // is laid out as (gridSize + 1) x (gridSize + 1) cells. Sampling returns
  // true vertex indices using this same stride, so the divisor here must be
  // gridSize + 1 (not gridSize) or the column shears as elevation changes.
  const frameCount = gridSize + 1;
  const gridSizeUniform = uniform(float(frameCount));

  const { faceIndices: interpolatedIndices, faceWeights } =
    selectOctahedralFrames({ gridSize, octType });
  // Every vertex of the quad carries the same indices; round away any
  // interpolation drift before they're split into row / column.
  const faceIndices = interpolatedIndices.round();
  // LOD visibility rides in instanceScale.w (see useInstancedOctahedralImpostorMesh).
  const visibility = attribute("instanceScale", "vec4").w;
  const yawSinCos = attribute("instanceYawSinCos", "vec3");

  const vUv = uv();

  const flatIndexA = float(faceIndices.x);
  const flatIndexB = float(faceIndices.y);
  const flatIndexC = float(faceIndices.z);

  const maxRow = float(frameCount - 1);
  const maxCol = float(frameCount - 1);

  const rowA = flatIndexA.div(gridSizeUniform).floor();
  const colA = flatIndexA.sub(rowA.mul(gridSizeUniform));
  const cellIndexA = vec2(colA.clamp(0.0, maxCol), rowA.clamp(0.0, maxRow));

  const rowB = flatIndexB.div(gridSizeUniform).floor();
  const colB = flatIndexB.sub(rowB.mul(gridSizeUniform));
  const cellIndexB = vec2(colB.clamp(0.0, maxCol), rowB.clamp(0.0, maxRow));

  const rowC = flatIndexC.div(gridSizeUniform).floor();
  const colC = flatIndexC.sub(rowC.mul(gridSizeUniform));
  const cellIndexC = vec2(colC.clamp(0.0, maxCol), rowC.clamp(0.0, maxRow));

  const weightSum = faceWeights.x.add(faceWeights.y).add(faceWeights.z);
  const normalizedWeights = vec3(
    faceWeights.x.div(weightSum.max(0.0001)),
    faceWeights.y.div(weightSum.max(0.0001)),
    faceWeights.z.div(weightSum.max(0.0001)),
  );

  const directionFromCell = (cellIndex) => {
    const planeUv = cellIndex.div(float(Math.max(frameCount - 1, 1)));

    if (octType === 1) {
      const x = planeUv.x.mul(2.0).sub(1.0);
      const z = planeUv.y.mul(2.0).sub(1.0);
      return vec2(x, z);
    }

    const x = planeUv.x.sub(planeUv.y);
    const z = planeUv.x.add(planeUv.y).sub(1.0);
    return vec2(x, z);
  };

  const localViewXZ = directionFromCell(cellIndexA)
    .mul(normalizedWeights.x)
    .add(directionFromCell(cellIndexB).mul(normalizedWeights.y))
    .add(directionFromCell(cellIndexC).mul(normalizedWeights.z));

  const invGridSize = float(1.0).div(gridSizeUniform);

  const clampedUv = vec2(vUv.x.clamp(0.0, 1.0), vUv.y.clamp(0.0, 1.0));

  const epsilon = float(0.0001);
  // Mirror the sampled frame horizontally (flip u within the cell). Without
  // this, the displayed frame is left-right flipped relative to the viewpoint
  // it was selected for, so its effective view azimuth reads as -phi instead
  // of +phi and the impostor appears to spin at ~2x the camera as you orbit
  // horizontally. This single flip cancels that stray mirror; it only touches
  // the horizontal (azimuth) axis, so elevation is unaffected.
  const flippedClampedX = float(1.0).sub(clampedUv.x);
  const safeUv = vec2(
    flippedClampedX.mul(float(1.0).sub(epsilon.mul(2.0))).add(epsilon),
    clampedUv.y.mul(float(1.0).sub(epsilon.mul(2.0))).add(epsilon),
  );

  const finalUVA = cellIndexA.add(safeUv).mul(invGridSize).clamp(0.0, 1.0);
  const finalUVB = cellIndexB.add(safeUv).mul(invGridSize).clamp(0.0, 1.0);
  const finalUVC = cellIndexC.add(safeUv).mul(invGridSize).clamp(0.0, 1.0);

  /** The three frames of `atlas` blended by the barycentric weights. */
  const sampleBlended = (atlas) => {
    const node = texture(atlas);
    return node
      .sample(finalUVA)
      .mul(normalizedWeights.x)
      .add(node.sample(finalUVB).mul(normalizedWeights.y))
      .add(node.sample(finalUVC).mul(normalizedWeights.z));
  };

  return {
    sampleBlended,
    localViewXZ,
    yawSinCos,
    visibility,
    vUv,
  };
}

/**
 * The billboard's clip-space position, for `material.vertexNode`. Setting
 * vertexNode (instead of subclassing setupPositionView) keeps the material a
 * stock node material; three reconstructs the fragment-stage positionView
 * from clip space whenever vertexNode is set (Position.js), so lighting and
 * fog still see the billboarded position.
 */
export function billboardVertexNode() {
  return cameraProjectionMatrix.mul(billboardPositionView());
}

/**
 * Every runtime-tunable knob of the unlit impostor material, as TSL uniform
 * nodes. Create once, wire into createImpostorNodes, then write `.value` to
 * tune live - no shader recompile.
 */
export function createImpostorUniforms({
  alphaTest = 0.5,
  useDither = false,
  // Exposure applied to the baked colour, and the fake two-tone sun shading.
  colorScale = 1.5,
  lightDirectionXZ = [0.70710678, 0.70710678],
  sideShadeAmount = 0.225,
} = {}) {
  return {
    alphaTest: uniform(float(alphaTest)),
    useDither: uniform(float(useDither ? 1.0 : 0.0)),
    colorScale: uniform(float(colorScale)),
    lightDirectionXZ: uniform(vec2(...lightDirectionXZ)),
    sideShadeAmount: uniform(float(sideShadeAmount)),
  };
}

/**
 * Node graph of the unlit (baked-lighting) impostor: assign the result onto
 * any MeshBasicNodeMaterial. Structural options (atlas, grid layout,
 * wireframe, hueShift range) are baked into the graph; everything tunable
 * lives in `uniforms` (createImpostorUniforms).
 */
export function createImpostorNodes({
  atlasTexture,
  gridSize,
  octType = 0,
  showWireframe = false,
  // [min, max] radians of per-instance hue rotation (seed in
  // instanceYawSinCos.z), applied to green texels only so bark keeps its
  // colour. null disables it.
  hueShift = null,
  uniforms,
}) {
  const { sampleBlended, localViewXZ, yawSinCos, visibility, vUv } =
    buildImpostorAtlasSampler({ gridSize, octType });

  const sinYaw = yawSinCos.x;
  const cosYaw = yawSinCos.y;
  const rawWorldViewXZ = vec2(
    localViewXZ.x.mul(cosYaw).add(localViewXZ.y.mul(sinYaw)),
    localViewXZ.y.mul(cosYaw).sub(localViewXZ.x.mul(sinYaw)),
  );
  const worldViewXZ = rawWorldViewXZ.div(rawWorldViewXZ.length().max(0.0001));
  const lightSide = worldViewXZ.dot(uniforms.lightDirectionXZ).clamp(-1.0, 1.0);
  const backsideShade = lightSide
    .mul(uniforms.sideShadeAmount)
    .add(float(1).sub(uniforms.sideShadeAmount));

  const blended = sampleBlended(atlasTexture);
  const blendedColor = blended.rgb;

  let variedColor = blendedColor;
  if (hueShift) {
    variedColor = applyInstanceHue(blendedColor, hueShift, yawSinCos.z);
  }

  const finalColor = variedColor.mul(uniforms.colorScale).mul(backsideShade);

  const finalAlpha = blended.a;

  const ditherPattern = Fn(({ uv }) => {
    const x = uv.x.mul(4.0).floor().mod(4.0);
    const y = uv.y.mul(4.0).floor().mod(4.0);
    const index = x.add(y.mul(4.0));
    return index.div(15.0).sub(0.5);
  });

  const flippedUv = vec2(vUv.x, float(1.0).sub(vUv.y));
  const ditherValue = ditherPattern({ uv: flippedUv });
  const ditherThreshold = uniforms.alphaTest.add(ditherValue.mul(0.1));

  const ditheredAlpha = finalAlpha
    .sub(ditherThreshold)
    .step(0.0)
    .mul(finalAlpha);
  const cutAlpha = finalAlpha.step(uniforms.alphaTest).mul(finalAlpha);
  const processedAlpha = uniforms.useDither
    .mul(ditheredAlpha)
    .add(float(1.0).sub(uniforms.useDither).mul(cutAlpha));

  return {
    vertexNode: billboardVertexNode(),
    colorNode: finalColor,
    opacityNode: showWireframe ? visibility : processedAlpha.mul(visibility),
  };
}

/** Runtime-tunable knobs of the lit impostor material (see createImpostorUniforms). */
export function createLitImpostorUniforms({ colorScale = 1.0 } = {}) {
  return {
    colorScale: uniform(float(colorScale)),
  };
}

/**
 * Node graph of the lit (surface atlas: albedo + normal/depth) impostor:
 * assign the result onto any MeshLambertNodeMaterial.
 */
export function createLitImpostorNodes({
  atlasTexture,
  normalDepthTexture,
  gridSize,
  octType = 0,
  hueShift = null,
  // World height of the quad per unit of instanceScale.y (the field's geometryArgs[1]).
  planeHeight = 1,
  uniforms,
}) {
  const { sampleBlended, yawSinCos, visibility } = buildImpostorAtlasSampler({
    gridSize,
    octType,
  });

  // Both surface atlases are premultiplied by coverage (albedo.a): divide
  // the filtered, frame-blended values by the filtered coverage.
  const albedo = sampleBlended(atlasTexture);
  const coverage = albedo.a;
  const unpremultiply = float(1).div(coverage.max(0.001));
  let color = albedo.rgb.mul(unpremultiply);
  if (hueShift) color = applyInstanceHue(color, hueShift, yawSinCos.z);

  const normalDepth = sampleBlended(normalDepthTexture).mul(unpremultiply);
  const bakedNormal = normalDepth.xyz.mul(2).sub(1);

  // The bake camera sat 0.5 bake units from the billboard plane, and a
  // bake unit is the whole quad, so (0.5 - depth) * quad height is how far
  // this texel's surface sits in front of (+) or behind the quad.
  const quadHeight = attribute("instanceScale", "vec4").y.mul(planeHeight);
  const towardCamera = float(0.5).sub(normalDepth.w).mul(quadHeight);
  const quadView = varying(billboardPositionView().xyz, "vImpostorView");
  const surfaceView = vec3(quadView.xy, quadView.z.add(towardCamera));
  const surfaceWorld = cameraWorldMatrix.mul(vec4(surfaceView, 1)).xyz;

  return {
    vertexNode: billboardVertexNode(),
    colorNode: color.mul(uniforms.colorScale),
    opacityNode: coverage.mul(visibility),
    normalNode: normalize(bakedNormal),
    // three adds normalWorld * shadow.normalBias on top of this, and
    // normalWorld follows normalNode, so the bias already runs along the
    // baked leaf normal like it does on the real mesh.
    receivedShadowPositionNode: surfaceWorld,
  };
}

export class InstancedOctahedralImpostorMaterial
  extends THREE.MeshBasicNodeMaterial
{
  constructor({
    atlasTexture,
    gridSize,
    octType = 0,
    alphaTest = 0.5,
    useDither = false,
    showWireframe = false,
    // Blended edges suit a handful of impostors; a dense forest in one draw
    // call is better as an opaque alpha cutout (depth-correct, order-free).
    transparent = true,
    colorScale = 1.5,
    lightDirectionXZ = [0.70710678, 0.70710678],
    sideShadeAmount = 0.225,
    hueShift = null,
  }) {
    super();

    this.transparent = transparent;
    this.side = THREE.FrontSide;
    this.wireframe = showWireframe;
    this.alphaTest = useDither ? 0.001 : alphaTest;
    this.toneMapped = true;

    // Live-updatable TSL uniform nodes: set `.value` (no shader rebuild).
    this.uniforms = createImpostorUniforms({
      alphaTest,
      useDither,
      colorScale,
      lightDirectionXZ,
      sideShadeAmount,
    });
    Object.assign(
      this,
      createImpostorNodes({
        atlasTexture,
        gridSize,
        octType,
        showWireframe,
        hueShift,
        uniforms: this.uniforms,
      }),
    );
  }
}

export class LitInstancedOctahedralImpostorMaterial
  extends THREE.MeshLambertNodeMaterial
{
  constructor({
    atlasTexture,
    normalDepthTexture,
    gridSize,
    octType = 0,
    alphaTest = 0.5,
    colorScale = 1.0,
    hueShift = null,
    planeHeight = 1,
  }) {
    super();
    this.transparent = false;
    this.side = THREE.FrontSide;
    this.alphaTest = alphaTest;

    // Live-updatable TSL uniform nodes: set `.value` (no shader rebuild).
    this.uniforms = createLitImpostorUniforms({ colorScale });
    Object.assign(
      this,
      createLitImpostorNodes({
        atlasTexture,
        normalDepthTexture,
        gridSize,
        octType,
        hueShift,
        planeHeight,
        uniforms: this.uniforms,
      }),
    );
  }
}
