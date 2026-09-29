import * as THREE from "three/webgpu";
import { Canvas, extend } from "@react-three/fiber";
import SceneLight from "./SceneLight";
import { Suspense } from "react";
import OctahedralImpostorLODField from "./OctahedralImpostorLODField"; // single-draw-call field with close-up real-mesh LOD swap
import { Gltf, Loader, OrbitControls } from "@react-three/drei";
import GridWrapper from "./GridWrapper";
import { useControls } from "leva";
import { Perf } from "r3f-webgpu-perf";

export default function DemoApp() {
  const { count, lodDistance, maxNearInstances } = useControls({
    count: {
      min: 5000,
      max: 50000,
      value: 500,
      step: 100,
    },
    lodDistance: {
      min: 0,
      max: 250,
      value: 175,
    },
    maxNearInstances: {
      min: 0,
      max: 10000,
      value: 3000,
      step: 100,
    },
  });

  return (
    <>
      <Canvas
        gl={async (props) => {
          extend(THREE);
          const renderer = new THREE.WebGPURenderer(props);
          renderer.shadowMap.enabled = true;
          renderer.shadowMap.type = THREE.PCFSoftShadowMap;

          await renderer.init();
          return renderer;
        }}
        camera={{
          position: [7, 8, 15],
          fov: 30,
          near: 0.5,
          far: 10000,
        }}
      >
        <Suspense fallback={null}>
          <SceneLight />
          <OrbitControls maxPolarAngle={Math.PI / 2} />
          {/* 🔥 NEW: WebGPU Compute-based Field with Atlas Caching */}
          {/* Uncomment to render hundreds of instances sharing a single atlas */}
          {/* Atlas is generated once and automatically cached for all instances */}

          <Perf />

          <OctahedralImpostorLODField
            // modelPath="/car.glb"
            modelPath="/tree.glb"
            position={[0, 13, 0]}
            count={count} // Hundreds of instances sharing the same atlas
            areaSize={[3000, 3000]}
            minHeight={0}
            maxHeight={0}
            minScale={4}
            maxScale={6}
            widthVariation={0.22}
            heightVariation={0.28}
            baseScale={[1.8, 1.8, 1.8]}
            avoidRadius={6}
            seed={2024}
            randomYaw={true}
            shadowGroundY={-1.2}
            showInstanceShadows={true}
            shadowOpacity={0.45}
            sunPosition={[35, 55, 35]} // keep in sync with SceneLight's directionalLight
            gridSize={16}
            atlasSize={2048}
            octType={0} // 0 = HEMI, 1 = FULL
            geometryArgs={[4, 4]}
            alphaTest={0.03}
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
            <planeGeometry args={[3200, 3200]} />
            <meshStandardMaterial color="#73766d" roughness={0.95} />
          </mesh>
          <Gltf src="/vegetation/trees/model.glb"/>
        </Suspense>
      </Canvas>

      <Loader />
    </>
  );
}
