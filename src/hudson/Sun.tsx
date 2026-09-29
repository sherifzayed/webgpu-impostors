import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three/webgpu";
import { SHADOW_CASTER_LAYER, consumeShadowUpdate, requestShadowUpdate } from "./staticShadows";

type SunProps = {
  /** Degrees from +X towards +Z, like a compass heading. */
  azimuth: number;
  /** Degrees above the horizon. */
  elevation: number;
  intensity: number;
  color: string;
  /** Half-extent of the square the shadow map covers, in metres; the whole site plus a margin. */
  shadowExtent: number;
  target: [number, number, number];
  mapSize?: number;
};

/** World-space direction towards the sun. */
export function sunDirection(azimuth: number, elevation: number, target = new THREE.Vector3()) {
  const az = THREE.MathUtils.degToRad(azimuth);
  const el = THREE.MathUtils.degToRad(elevation);
  return target.set(Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az));
}

/**
 * The one directional light. Its shadow map covers the whole site and is rendered on demand only
 * (see staticShadows.ts): the forest is static, so a 4K map costs one render, not one per frame.
 */
export function Sun({ azimuth, elevation, intensity, color, shadowExtent, target, mapSize = 4096 }: SunProps) {
  const light = useMemo(() => {
    const sun = new THREE.DirectionalLight();
    sun.name = "sun";
    sun.castShadow = true;
    sun.shadow.autoUpdate = false;
    sun.shadow.mapSize.set(mapSize, mapSize);
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.6;
    sun.shadow.radius = 3;
    sun.shadow.camera.layers.enable(SHADOW_CASTER_LAYER);
    return sun;
  }, [mapSize]);

  useEffect(() => {
    const distance = shadowExtent * 3;
    light.target.position.set(...target);
    light.position.copy(sunDirection(azimuth, elevation).multiplyScalar(distance)).add(light.target.position);
    const camera = light.shadow.camera;
    camera.left = camera.bottom = -shadowExtent;
    camera.right = camera.top = shadowExtent;
    camera.near = distance * 0.25;
    camera.far = distance * 1.75;
    camera.updateProjectionMatrix();
    light.target.updateMatrixWorld();
    requestShadowUpdate();
  }, [light, azimuth, elevation, shadowExtent, target]);

  useEffect(() => {
    light.color.set(color);
    light.intensity = intensity;
  }, [light, color, intensity]);

  useFrame(() => {
    if (consumeShadowUpdate()) light.shadow.needsUpdate = true;
  });

  return (
    <>
      <primitive object={light} />
      <primitive object={light.target} />
    </>
  );
}
