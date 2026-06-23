import { Suspense } from "react";
import { Environment } from "@react-three/drei";

export default function SceneLight() {
  return (
    <>
      <color attach="background" args={["#666971"]} />
      <directionalLight
        castShadow
        intensity={3.4}
        position={[35, 55, 35]}
        shadow-mapSize={4096}
        shadow-camera-near={0.5}
        shadow-camera-far={180}
        shadow-camera-left={-145}
        shadow-camera-right={145}
        shadow-camera-top={145}
        shadow-camera-bottom={-145}
        shadow-bias={-0.0001}
        shadow-normalBias={0.017}
      />

      <Suspense fallback={null}>
        <Environment preset="city" />
      </Suspense>
    </>
  );
}
