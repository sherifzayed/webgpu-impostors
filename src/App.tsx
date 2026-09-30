import { Loader, OrbitControls } from "@react-three/drei";
import { Canvas, extend, useThree } from "@react-three/fiber";
import { Leva, useControls } from "leva";
import { Perf } from "r3f-webgpu-perf";
import { Suspense, use, useMemo } from "react";
import * as THREE from "three/webgpu";
import { placementsBySpecies } from "./hudson/communityData";
import { CommunityGround } from "./hudson/CommunityGround";
import { ImpostorTrees } from "./hudson/ImpostorTrees";
import { KitTreeShadowCasters } from "./hudson/KitTrees";
import { Sun } from "./hudson/Sun";
import {
  communityPromise,
  COMMUNITY_URL,
  getTreeVariants,
} from "./hudson/sceneAssets";

type ImpostorControls = {
  atlasSize: number;
  exposure: number;
  maxNearInstances: number;
  showShadowCasters: boolean;
  decalShadows: boolean;
};

type GroundControls = {
  visible: boolean;
  lit: boolean;
  receiveShadows: boolean;
  normalStrength: number;
  wireframe: boolean;
};

function useImpostorControls(): ImpostorControls {
  const surface = useControls("Surface bake", {
    atlasSize: { value: 2048, options: [1024, 2048, 4096] },
    exposure: { value: 1, min: 0.25, max: 3, step: 0.05 },
  });

  const lod = useControls("LOD & shadows", {
    maxNearInstances: { value: 0, min: 0, max: 2000, step: 50 },
    showShadowCasters: { value: false, label: "show shadow casters" },
    decalShadows: { value: false, label: "decal shadows" },
  });

  return { ...surface, ...lod } as ImpostorControls;
}

function useGroundControls(): GroundControls {
  return useControls("Ground", {
    visible: true,
    lit: { value: true, label: "lit shading" },
    receiveShadows: { value: true, label: "receive shadows" },
    normalStrength: { value: 1.5, min: 0, max: 5, step: 0.1, label: "normal strength" },
    wireframe: false,
  }) as GroundControls;
}

function HudsonScene({ controls, ground }: { controls: ImpostorControls; ground: GroundControls }) {
  const renderer = useThree((state) => state.gl);
  const { community, trees } = use(communityPromise);
  const variants = use(getTreeVariants(renderer));
  const placements = useMemo(() => placementsBySpecies(trees), [trees]);
  const { atlasSize, exposure, maxNearInstances } = controls;

  return (
    <>
      <CommunityGround
        baseUrl={COMMUNITY_URL}
        community={community}
        visible={ground.visible}
        lit={ground.lit}
        receiveShadows={ground.receiveShadows}
        normalStrength={ground.normalStrength}
        wireframe={ground.wireframe}
      />
      <KitTreeShadowCasters
        variants={variants}
        placements={placements}
        debugVisible={controls.showShadowCasters}
      />
      <ImpostorTrees
        variants={variants}
        placements={placements}
        maxNearInstances={maxNearInstances}
        atlasSize={atlasSize}
        exposure={exposure}
        decalShadows={controls.decalShadows}
      />
    </>
  );
}

export default function App() {
  const controls = useImpostorControls();
  const ground = useGroundControls();
  return (
    <>
      {/* Explicit mount: leva's auto-mounted panel is flaky under React 19.
          Only theme keys leva 0.10 has defaults for may be passed - unknown
          keys (e.g. zIndices) crash its mergeTheme. */}
      <Leva collapsed={false} theme={{ sizes: { rootWidth: "320px" } }} />
      <Canvas
        shadows
        gl={async (props) => {
          extend(THREE as any);
          // Reversed-Z keeps depth precision across the kilometres a telephoto camera spans.
          const renderer = new THREE.WebGPURenderer({
            ...(props as any),
            antialias: true,
            reversedDepthBuffer: true,
          });
          renderer.shadowMap.enabled = true;
          renderer.shadowMap.type = THREE.PCFSoftShadowMap;
          renderer.toneMapping = THREE.ACESFilmicToneMapping;
          await renderer.init();
          return renderer;
        }}
        camera={{ fov: 5, near: 10, far: 30000, position: [0, 3000, 7000] }}
      >
        <Suspense fallback={null}>
          <color attach="background" args={["#c9d6de"]} />
          <hemisphereLight args={["#cfe0ee", "#5d5b3f", 1.3]} />
          <OrbitControls />
          <HudsonScene controls={controls} ground={ground} />
          <Sun
            azimuth={150}
            elevation={34}
            intensity={3.2}
            color="#fff1dc"
            shadowExtent={780}
            target={[0, 60, 0]}
          />
          <Perf position="bottom-left" />
        </Suspense>
      </Canvas>
      <Loader />
    </>
  );
}
