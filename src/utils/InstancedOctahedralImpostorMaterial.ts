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
} from "three/tsl";

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
 * barycentric blend) is ported from OctahedralImpostor.jsx, only swapping
 * the face indices/weights source from per-object uniforms to per-instance
 * attributes.
 */
export class InstancedOctahedralImpostorMaterial extends THREE.MeshBasicNodeMaterial {
  constructor({
    atlasTexture,
    gridSize,
    octType = 0,
    atlasCoverage = 1.0,
    alphaTest = 0.5,
    useDither = false,
    showWireframe = false,
  }) {
    super();

    this.transparent = true;
    this.side = THREE.FrontSide;
    this.wireframe = showWireframe;

    // The octahedral grid has (gridSize + 1) viewpoints per side, so the atlas
    // is laid out as (gridSize + 1) x (gridSize + 1) cells. Sampling returns
    // true vertex indices using this same stride, so the divisor here must be
    // gridSize + 1 (not gridSize) or the column shears as elevation changes.
    const frameCount = gridSize + 1;
    const gridSizeUniform = uniform(float(frameCount));
    const atlasTextureNode = texture(atlasTexture);

    const faceIndices = attribute("instanceFaceIndices", "vec3");
    const faceWeights = attribute("instanceFaceWeights", "vec3");
    // LOD visibility is packed into instanceScale.w to stay under WebGPU's
    // 8-vertex-buffer limit (see useInstancedOctahedralImpostorMesh).
    const visibility = attribute("instanceScale", "vec4").w;
    const yawSinCos = attribute("instanceYawSinCos", "vec2");

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
      faceWeights.z.div(weightSum.max(0.0001))
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

    const sinYaw = yawSinCos.x;
    const cosYaw = yawSinCos.y;
    const rawWorldViewXZ = vec2(
      localViewXZ.x.mul(cosYaw).add(localViewXZ.y.mul(sinYaw)),
      localViewXZ.y.mul(cosYaw).sub(localViewXZ.x.mul(sinYaw))
    );
    const worldViewXZ = rawWorldViewXZ.div(rawWorldViewXZ.length().max(0.0001));
    const lightXZ = vec2(0.70710678, 0.70710678);
    const lightSide = worldViewXZ.dot(lightXZ).clamp(-1.0, 1.0);
    const backsideShade = lightSide.mul(0.225).add(0.775);

    const atlasCoverageUniform = uniform(float(atlasCoverage));
    const invGridSize = float(1.0).div(gridSizeUniform);

    const coverageOffset = float(1.0).sub(atlasCoverageUniform).mul(0.5);
    const scaledUv = vUv.mul(atlasCoverageUniform).add(coverageOffset);
    const clampedUv = vec2(scaledUv.x.clamp(0.0, 1.0), scaledUv.y.clamp(0.0, 1.0));

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
      clampedUv.y.mul(float(1.0).sub(epsilon.mul(2.0))).add(epsilon)
    );

    const finalUVA = cellIndexA.add(safeUv).mul(invGridSize).clamp(0.0, 1.0);
    const finalUVB = cellIndexB.add(safeUv).mul(invGridSize).clamp(0.0, 1.0);
    const finalUVC = cellIndexC.add(safeUv).mul(invGridSize).clamp(0.0, 1.0);

    const colorA = atlasTextureNode.sample(finalUVA);
    const colorB = atlasTextureNode.sample(finalUVB);
    const colorC = atlasTextureNode.sample(finalUVC);

    const finalColor = colorA.rgb
      .mul(normalizedWeights.x)
      .add(colorB.rgb.mul(normalizedWeights.y))
      .add(colorC.rgb.mul(normalizedWeights.z))
      .mul(float(1.5))
      .mul(backsideShade);

    const finalAlpha = colorA.a
      .mul(normalizedWeights.x)
      .add(colorB.a.mul(normalizedWeights.y))
      .add(colorC.a.mul(normalizedWeights.z));

    const alphaTestUniform = uniform(float(alphaTest));
    const useDitherUniform = uniform(float(useDither ? 1.0 : 0.0));

    const ditherPattern = Fn(({ uv }) => {
      const x = uv.x.mul(4.0).floor().mod(4.0);
      const y = uv.y.mul(4.0).floor().mod(4.0);
      const index = x.add(y.mul(4.0));
      return index.div(15.0).sub(0.5);
    });

    const flippedUv = vec2(vUv.x, float(1.0).sub(vUv.y));
    const ditherValue = ditherPattern({ uv: flippedUv });
    const ditherThreshold = alphaTestUniform.add(ditherValue.mul(0.1));

    const ditheredAlpha = finalAlpha.sub(ditherThreshold).step(0.0).mul(finalAlpha);
    const cutAlpha = finalAlpha.step(alphaTestUniform).mul(finalAlpha);
    const processedAlpha = useDitherUniform
      .mul(ditheredAlpha)
      .add(float(1.0).sub(useDitherUniform).mul(cutAlpha));

    this.colorNode = finalColor;
    this.opacityNode = showWireframe ? visibility : processedAlpha.mul(visibility);
    this.alphaTest = useDither ? 0.001 : alphaTest;
    this.toneMapped = true;
  }

  setupPositionView(/* builder */) {
    const instanceOffset = attribute("instanceOffset", "vec3");
    const instanceScale = attribute("instanceScale", "vec4");

    const mvPosition = modelViewMatrix.mul(vec3(instanceOffset));
    const alignedPosition = positionGeometry.xy.mul(instanceScale.xy);

    return vec4(mvPosition.xy.add(alignedPosition), mvPosition.zw);
  }
}
