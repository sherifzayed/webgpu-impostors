import { useMemo } from "react";
import { extend, type ThreeElement } from "@react-three/fiber";
import * as THREE from "three/webgpu";
import {
  createImpostorNodes,
  createImpostorUniforms,
  createLitImpostorNodes,
  createLitImpostorUniforms,
} from "../utils/InstancedOctahedralImpostorMaterial";

declare module "@react-three/fiber" {
  interface ThreeElements {
    meshBasicNodeMaterial: ThreeElement<typeof THREE.MeshBasicNodeMaterial>;
    meshLambertNodeMaterial: ThreeElement<typeof THREE.MeshLambertNodeMaterial>;
  }
}

extend({
  MeshBasicNodeMaterial: THREE.MeshBasicNodeMaterial,
  MeshLambertNodeMaterial: THREE.MeshLambertNodeMaterial,
});

type HueShift = [number, number] | null;

export interface OctahedralImpostorMaterialProps {
  atlasTexture: THREE.Texture;
  gridSize: number;
  octType?: number;
  alphaTest?: number;
  useDither?: boolean;
  showWireframe?: boolean;
  transparent?: boolean;
  colorScale?: number;
  lightDirectionXZ?: [number, number];
  sideShadeAmount?: number;
  hueShift?: HueShift;
}

export function OctahedralImpostorMaterial({
  atlasTexture,
  gridSize,
  octType = 0,
  alphaTest = 0.5,
  useDither = false,
  showWireframe = false,
  transparent = true,
  colorScale = 1.5,
  lightDirectionXZ = [0.70710678, 0.70710678],
  sideShadeAmount = 0.225,
  hueShift = null,
}: OctahedralImpostorMaterialProps) {
  const uniforms = useMemo(() => createImpostorUniforms(), []);

  // Uniform writes are idempotent per render; nothing recompiles.
  uniforms.alphaTest.value = alphaTest;
  uniforms.useDither.value = useDither ? 1 : 0;
  uniforms.colorScale.value = colorScale;
  uniforms.lightDirectionXZ.value.set(lightDirectionXZ[0], lightDirectionXZ[1]);
  uniforms.sideShadeAmount.value = sideShadeAmount;

  // hueShift is usually an inline array literal; key on its contents so a
  // new-but-equal array doesn't rebuild the node graph.
  const hueShiftKey = hueShift ? hueShift.join(",") : "";
  const nodes = useMemo(
    () =>
      createImpostorNodes({
        atlasTexture,
        gridSize,
        octType,
        showWireframe,
        hueShift,
        uniforms,
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [atlasTexture, gridSize, octType, showWireframe, hueShiftKey, uniforms],
  );

  return (
    <meshBasicNodeMaterial
      // Structural change -> fresh material, compiled once, old one disposed by R3F.
      key={`${atlasTexture.uuid}/${gridSize}/${octType}/${showWireframe}/${hueShiftKey}`}
      vertexNode={nodes.vertexNode}
      colorNode={nodes.colorNode}
      opacityNode={nodes.opacityNode}
      // Blended edges suit a handful of impostors; a dense forest in one draw
      // call is better as an opaque alpha cutout (depth-correct, order-free).
      transparent={transparent}
      side={THREE.FrontSide}
      wireframe={showWireframe}
      alphaTest={useDither ? 0.001 : alphaTest}
      toneMapped
    />
  );
}

export interface LitOctahedralImpostorMaterialProps {
  atlasTexture: THREE.Texture;
  normalDepthTexture: THREE.Texture;
  gridSize: number;
  octType?: number;
  alphaTest?: number;
  colorScale?: number;
  hueShift?: HueShift;
  planeHeight?: number;
}

export function LitOctahedralImpostorMaterial({
  atlasTexture,
  normalDepthTexture,
  gridSize,
  octType = 0,
  alphaTest = 0.5,
  colorScale = 1.0,
  hueShift = null,
  planeHeight = 1,
}: LitOctahedralImpostorMaterialProps) {
  const uniforms = useMemo(() => createLitImpostorUniforms(), []);

  uniforms.colorScale.value = colorScale;

  const hueShiftKey = hueShift ? hueShift.join(",") : "";
  const nodes = useMemo(
    () =>
      createLitImpostorNodes({
        atlasTexture,
        normalDepthTexture,
        gridSize,
        octType,
        hueShift,
        planeHeight,
        uniforms,
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      atlasTexture,
      normalDepthTexture,
      gridSize,
      octType,
      hueShiftKey,
      planeHeight,
      uniforms,
    ],
  );

  return (
    <meshLambertNodeMaterial
      key={`${atlasTexture.uuid}/${normalDepthTexture.uuid}/${gridSize}/${octType}/${hueShiftKey}/${planeHeight}`}
      vertexNode={nodes.vertexNode}
      colorNode={nodes.colorNode}
      opacityNode={nodes.opacityNode}
      normalNode={nodes.normalNode}
      receivedShadowPositionNode={nodes.receivedShadowPositionNode}
      transparent={false}
      side={THREE.FrontSide}
      alphaTest={alphaTest}
    />
  );
}
