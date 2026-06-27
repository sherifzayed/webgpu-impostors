import { useMemo } from "react";
import * as THREE from "three/webgpu";
import { useFrame, useThree } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { getSamplingCache } from "./utils/octahedralImpostorMath";
import { generateFieldInstances } from "./utils/generateFieldInstances";
import { useOctahedralAtlasCompute } from "./hooks/useOctahedralAtlasCompute";
import { useInstancedOctahedralImpostorMesh } from "./hooks/useInstancedOctahedralImpostorMesh";
import TreeInstanceShadows from "./TreeInstanceShadows";

const DEFAULT_MODEL_PATH = "/tree.glb";

/**
 * TreeOctahedralImpostorFieldCompute - Renders hundreds of impostor instances
 * as a single InstancedMesh sharing one atlas texture and one material, so
 * the whole field costs exactly one draw call regardless of `count`.
 */
export default function TreeOctahedralImpostorFieldCompute({
  showWireframe = false,
  count = 150,
  modelPath = DEFAULT_MODEL_PATH,
  position = [0, 0, 0],
  areaSize = [60, 60],
  minHeight = 0,
  maxHeight = 0,
  minScale = 0.7,
  maxScale = 1.4,
  heightVariation = 0,
  widthVariation = 0,
  baseScale = [1, 1, 1],
  avoidRadius = 0,
  seed = 2024,
  randomYaw = true,
  shadowGroundY = -2,
  showInstanceShadows = true,
  gridSize = 16,
  atlasSize = 2048,
  octType = 0,
  geometryArgs = [2, 2],
  alphaTest = 0.5,
  // WebGPU Compute specific parameters
  usePostProcessing = true,
  brightness = 1.0,
  contrast = 1.0,
  optimizeSize = false,
  atlasCoverage = 1.0,
  usePostDilatation = false,
  dilationRadius = 1,
  directionThresholdRadians = 0.0872665,
  useDither = false,
}) {
  const { camera } = useThree();
  const gltf = useGLTF(modelPath);

  const sourceMesh = useMemo(() => {
    if (!gltf?.scene) return null;

    let foundMesh = null;
    gltf.scene.traverse((child) => {
      if (!foundMesh && child.isMesh) {
        foundMesh = child;
      }
    });

    if (foundMesh) {
      foundMesh.userData.__impostorSourceId = modelPath;
    }

    return foundMesh;
  }, [gltf, modelPath]);

  const instances = useMemo(
    () =>
      generateFieldInstances({
        count,
        areaSize,
        position,
        baseScale,
        minHeight,
        maxHeight,
        minScale,
        maxScale,
        heightVariation,
        widthVariation,
        avoidRadius,
        seed,
        randomYaw,
      }),
    [
      count,
      areaSize,
      position,
      baseScale,
      minHeight,
      maxHeight,
      minScale,
      maxScale,
      heightVariation,
      widthVariation,
      avoidRadius,
      seed,
      randomYaw,
    ]
  );

  const { atlas } = useOctahedralAtlasCompute({
    mesh: sourceMesh,
    gridSize,
    atlasSize,
    octType,
    enabled: !!sourceMesh,
    usePostProcessing,
    brightness,
    contrast,
    optimizeSize,
    atlasCoverage,
    usePostDilatation,
    dilationRadius,
  });

  const samplingCache = useMemo(
    () => getSamplingCache(octType, gridSize),
    [octType, gridSize]
  );

  const { instancedMesh, updateFrame } = useInstancedOctahedralImpostorMesh({
    instances,
    atlas,
    gridSize,
    octType,
    samplingCache,
    geometryArgs,
    atlasCoverage,
    alphaTest,
    useDither,
    showWireframe,
    directionThresholdRadians,
  });

  useFrame(() => {
    updateFrame(camera);
  });

  if (!sourceMesh || instances.length === 0 || !instancedMesh) {
    return null;
  }

  return (
    <>
      <primitive object={instancedMesh} />
      {showInstanceShadows && (
        <TreeInstanceShadows instances={instances} groundY={shadowGroundY} />
      )}
    </>
  );
}

// Preload GLTF models
useGLTF.preload(DEFAULT_MODEL_PATH);
