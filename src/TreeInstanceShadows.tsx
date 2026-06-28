import { useEffect, useMemo } from "react";
import * as THREE from "three/webgpu";

const tempObject = new THREE.Object3D();
const DEFAULT_LIGHT_OFFSET = [0.6, 0, 1.1];

function createShadowTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;

  const context = canvas.getContext("2d");
  const gradient = context.createRadialGradient(64, 64, 8, 64, 64, 62);
  gradient.addColorStop(0, "rgba(255, 255, 255, 0.9)");
  gradient.addColorStop(0.45, "rgba(255, 255, 255, 0.35)");
  gradient.addColorStop(1, "rgba(255, 255, 255, 0)");

  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;
  return texture;
}

export default function TreeInstanceShadows({
  instances,
  groundY = -2,
  opacity = 0.38,
  lightOffset = DEFAULT_LIGHT_OFFSET,
}) {
  const shadowMesh = useMemo(() => {
    if (!instances.length) return null;

    const texture = createShadowTexture();
    const geometry = new THREE.PlaneGeometry(1, 1);
    const material = new THREE.MeshBasicMaterial({
      color: "#050505",
      alphaMap: texture,
      transparent: true,
      opacity,
      depthWrite: false,
      toneMapped: false,
    });

    const mesh = new THREE.InstancedMesh(geometry, material, instances.length);
    mesh.frustumCulled = false;
    mesh.receiveShadow = false;
    mesh.castShadow = false;
    mesh.renderOrder = 1;

    for (let index = 0; index < instances.length; index += 1) {
      const instance = instances[index];
      const width = Math.max(instance.scale[0], instance.scale[2]);
      const height = instance.scale[1];
      const yaw = instance.rotationY || 0;

      tempObject.position.set(
        instance.position[0] + lightOffset[0] * width,
        groundY + 0.015,
        instance.position[2] + lightOffset[2] * width
      );
      tempObject.rotation.set(-Math.PI / 2, 0, yaw + 0.35);
      tempObject.scale.set(width * 1.25, height * 0.55, 1);
      tempObject.updateMatrix();
      mesh.setMatrixAt(index, tempObject.matrix);
    }

    mesh.instanceMatrix.needsUpdate = true;
    return mesh;
  }, [groundY, instances, lightOffset, opacity]);

  useEffect(() => {
    return () => {
      if (!shadowMesh) return;
      shadowMesh.geometry.dispose();
      shadowMesh.material.alphaMap?.dispose();
      shadowMesh.material.dispose();
    };
  }, [shadowMesh]);

  if (!shadowMesh) return null;

  return <primitive object={shadowMesh} />;
}
