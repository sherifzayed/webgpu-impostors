import * as THREE from "three/webgpu";
import { Canvas, extend } from "@react-three/fiber";
import SceneLight from "./SceneLight";
import { Suspense } from "react";
import TreeOctahedralImpostorField from "./TreeOctahedralImpostorField";
import TreeOctahedralImpostor from "./TreeOctahedralImpostor";
import TreeOctahedralImpostorCompute from "./TreeOctahedralImpostorCompute"; // New: WebGPU Compute-based
import TreeOctahedralImpostorFieldCompute from "./TreeOctahedralImpostorFieldCompute"; // New: WebGPU Compute-based field with atlas caching
import OctahedralImpostorLODField from "./OctahedralImpostorLODField"; // New: single-draw-call field with close-up real-mesh LOD swap
import { Gltf, Loader, OrbitControls, Stats } from "@react-three/drei";
import GridWrapper from "./GridWrapper";
import { useControls } from "leva";

export default function App() {
  const { count, lodDistance, maxNearInstances } = useControls({
    count: {
      min: 100,
      max: 50000,
      value: 1000,
      step: 100,
    },
    lodDistance: {
      min: 0,
      max: 80,
      value: 15,
    },
    maxNearInstances: {
      min: 1,
      max: 1000,
      value: 100,
      step: 10,
    },
  });

  return (
    <>
      <Canvas
        gl={async (props) => {
          extend(THREE);
          const renderer = new THREE.WebGPURenderer(props);
          renderer.shadowMap.enabled = false;
          renderer.shadowMap.type = THREE.PCFSoftShadowMap;

          await renderer.init();
          return renderer;
        }}
        camera={{
          position: [7, 8, 15],
          fov: 30,
          near: 0.5,
          far: 1000,
        }}
      >
        <Suspense fallback={null}>
          <SceneLight />
          <OrbitControls maxPolarAngle={Math.PI / 2} />
          {/* 🔥 NEW: WebGPU Compute-based Field with Atlas Caching */}
          {/* Uncomment to render hundreds of instances sharing a single atlas */}
          {/* Atlas is generated once and automatically cached for all instances */}
          <Stats />
          <OctahedralImpostorLODField
            // modelPath="/car.glb"
            modelPath="/tree.glb"
            position={[0, -2, 0]}
            count={count} // Hundreds of instances sharing the same atlas
            areaSize={[250, 250]}
            minHeight={0}
            maxHeight={0}
            minScale={0.55}
            maxScale={1.45}
            widthVariation={0.22}
            heightVariation={0.28}
            baseScale={[1.8, 1.8, 1.8]}
            avoidRadius={6}
            seed={2024}
            randomYaw={true}
            shadowGroundY={-1.2}
            showInstanceShadows={false}
            gridSize={16}
            atlasSize={512}
            octType={0} // 0 = HEMI, 1 = FULL
            geometryArgs={[4, 4]}
            roughness={1}
            metalness={0}
            alphaTest={0.35}
            envMapIntensity={1}
            // WebGPU Compute specific options
            usePostProcessing={true}
            brightness={1.0}
            contrast={1.0}
            optimizeSize={true}
            atlasCoverage={1.0}
            usePostDilatation={false}
            dilationRadius={0}
            showWireframe={false}
            directionThresholdRadians={0.0872665}
            // LOD: swap impostors for the real instanced mesh up close
            lodDistance={lodDistance}
            lodHysteresis={3}
            maxNearInstances={maxNearInstances}
          />
          <mesh
            receiveShadow
            position={[0, -1.225, 0]}
            rotation={[-Math.PI / 2, 0, 0]}
          >
            <planeGeometry args={[320, 320]} />
            <meshStandardMaterial color="#73766d" roughness={0.95} />
          </mesh>
        </Suspense>
      </Canvas>

      <Loader />
    </>
  );
}
