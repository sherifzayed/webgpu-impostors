import * as THREE from "three/webgpu";
import {
  texture,
  textureStore,
  Fn,
  instanceIndex,
  uvec2,
  vec2,
  float,
  uniform,
} from "three/tsl";

/**
 * Creates a WebGPU compute shader to copy and process atlas cells.
 * This uses StorageTexture for direct GPU-based atlas generation.
 * Comments in English per project guidelines.
 */
export function createAtlasCopyComputeShader({
  sourceTexture,
  targetStorageTexture,
  cellSize,
  targetX,
  targetY,
  atlasSize,
}) {
  const width = cellSize;
  const height = cellSize;

  // Define compute function to copy one cell
  const copyCellShader = Fn(
    ({ storageTexture, sourceTexture, cellOffsetX, cellOffsetY }) => {
      // Get current pixel position within the cell
      const localX = instanceIndex.mod(width);
      const localY = instanceIndex.div(width);

      // Calculate source UV (normalized 0-1)
      const sourceU = float(localX).div(float(width));
      const sourceV = float(localY).div(float(height));
      const sourceUV = vec2(sourceU, sourceV);

      // Sample from source texture
      const color = texture(sourceTexture, sourceUV);

      // Calculate target position in atlas
      const targetPosX = cellOffsetX.add(localX);
      const targetPosY = cellOffsetY.add(localY);
      const targetPos = uvec2(targetPosX, targetPosY);

      // Write to storage texture
      textureStore(storageTexture, targetPos, color).toWriteOnly();
    }
  );

  // Create uniforms for cell offset
  const cellOffsetXUniform = uniform(float(targetX));
  const cellOffsetYUniform = uniform(float(targetY));

  // Create compute node
  const computeNode = copyCellShader({
    storageTexture: targetStorageTexture,
    sourceTexture: sourceTexture,
    cellOffsetX: cellOffsetXUniform,
    cellOffsetY: cellOffsetYUniform,
  }).compute(width * height);

  return {
    computeNode,
    cellOffsetXUniform,
    cellOffsetYUniform,
  };
}

/**
 * Helper function to create a StorageTexture for WebGPU compute shaders
 */
export function createStorageTexture(width, height) {
  return new THREE.StorageTexture(width, height);
}
