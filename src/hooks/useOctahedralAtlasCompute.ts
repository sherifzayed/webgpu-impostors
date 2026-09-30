import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three/webgpu";
import { useThree } from "@react-three/fiber";
import { buildOctahedralMesh, OCT_TYPE } from "../utils/octahedralHelper";
import { useEnvironment, useGLTF } from "@react-three/drei";
import {
  createAtlasCopyComputeShader,
  createStorageTexture,
} from "../utils/atlasComputeShader";
import { bakeMorphTargetsIntoGeometry } from "../utils/buildLodModelParts";
import {
  float,
  normalize,
  normalView,
  positionView,
  texture,
  vec4,
} from "three/tsl";

/**
 * Cache storage shared across impostor instances. (English comment)
 */
const atlasCache = new Map();
const pendingAtlasPromises = new Map();
const MAX_COMPUTE_ATLAS_SIZE = 4096;

/**
 * Builds a cache key for atlas generation. (English comment)
 */
const ATLAS_CACHE_VERSION = "centered-v8-premultiplied-surface";

function buildAtlasCacheKey(mesh, gridSize, atlasSize, octType) {
  if (!mesh) {
    return null;
  }

  if (!mesh.userData.__impostorSourceId) {
    // Persist an identifier to allow clones to reuse the same atlas. (English comment)
    mesh.userData.__impostorSourceId =
      mesh.name && mesh.name.length > 0
        ? mesh.name
        : THREE.MathUtils.generateUUID();
  }

  return `${ATLAS_CACHE_VERSION}|${mesh.userData.__impostorSourceId}|g${gridSize}|a${atlasSize}|o${octType}`;
}

/**
 * Hook to generate octahedral impostor atlas using WebGPU compute shaders.
 * This version uses StorageTexture for direct GPU-based atlas generation.
 *
 * `mesh.userData.__impostorSourceId` doubles as a model path: the whole GLTF
 * at that path is baked (callers pass one mesh found inside it). Callers that
 * already hold the object to bake should use useOctahedralAtlasComputeFromObject.
 */
export function useOctahedralAtlasCompute(options) {
  const modelPath = options.mesh?.userData?.__impostorSourceId || null;
  const gltfScene = useGLTF(modelPath || "/dummy.glb");
  return useOctahedralAtlasComputeFromObject({
    ...options,
    source: modelPath ? gltfScene?.scene : options.mesh,
  });
}

/**
 * Bakes `source` (any Object3D; every Mesh under it is included) into an
 * octahedral atlas. `mesh` is only the cache identity: its
 * `userData.__impostorSourceId` keys the shared atlas cache, and defaults to
 * `source` when omitted.
 *
 */
export function useOctahedralAtlasComputeFromObject({
  source = null,
  mesh = source,
  gridSize = 16,
  atlasSize = 2048,
  octType = OCT_TYPE.HEMI,
  enabled = true,
}) {
  const { gl, scene, camera } = useThree();
  const [atlas, setAtlas] = useState(null);
  const [error, setError] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const environment = useEnvironment({ files: "/potsdamer_platz_1k.hdr" });
  const octahedralDataRef = useRef(null);

  // Build octahedral mesh data
  const octahedralData = useMemo(() => {
    if (!enabled) return null;
    try {
      return buildOctahedralMesh(octType, gridSize);
    } catch (err) {
      console.error("Failed to build octahedral mesh:", err);
      return null;
    }
  }, [octType, gridSize, enabled]);

  octahedralDataRef.current = octahedralData;

  const effectiveAtlasSize = Math.min(atlasSize, MAX_COMPUTE_ATLAS_SIZE);

  // Generate atlas using WebGPU compute shaders
  useEffect(() => {
    if (!enabled || !source || !octahedralData || !gl) {
      setAtlas(null);
      return;
    }

    const cacheKey = buildAtlasCacheKey(
      mesh,
      gridSize,
      effectiveAtlasSize,
      octType,
    );

    if (cacheKey && atlasCache.has(cacheKey)) {
      const cachedAtlas = atlasCache.get(cacheKey);
      setAtlas(cachedAtlas);
      setIsGenerating(false);
      setError(null);
      return;
    }

    if (cacheKey && pendingAtlasPromises.has(cacheKey)) {
      setIsGenerating(true);
      setError(null);
      const pendingPromise = pendingAtlasPromises.get(cacheKey);
      pendingPromise
        .then((cachedAtlas) => {
          setAtlas(cachedAtlas);
          setIsGenerating(false);
        })
        .catch((err) => {
          console.error("Failed to generate atlas:", err);
          setError(err);
          setIsGenerating(false);
        });
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      const atlasPromise = generateAtlasWithCompute({
        environment,
        source,
        octahedralData,
        gridSize,
        atlasSize: effectiveAtlasSize,
        gl,
        camera,
      }).then(
        ({ texture, normalDepthTexture }) => {
          const atlasPayload = {
            texture,
            normalDepthTexture,
            gridSize,
            octType,
            octahedralData,
          };

          if (cacheKey) {
            atlasCache.set(cacheKey, atlasPayload);
          }

          setAtlas(atlasPayload);
          setIsGenerating(false);

          return atlasPayload;
        },
        (err) => {
          console.error("Failed to generate atlas:", err);
          setError(err);
          setIsGenerating(false);
          throw err;
        },
      );

      if (cacheKey) {
        pendingAtlasPromises.set(cacheKey, atlasPromise);
      }

      atlasPromise.finally(() => {
        if (cacheKey) {
          pendingAtlasPromises.delete(cacheKey);
        }
      });
    } catch (err) {
      console.error("Error in atlas generation:", err);
      setError(err);
      setIsGenerating(false);
    }
  }, [
    mesh,
    source,
    octahedralData,
    gridSize,
    effectiveAtlasSize,
    enabled,
    gl,
    scene,
    camera,
    octType,
    environment,
  ]);

  return {
    atlas,
    error,
    isGenerating,
    octahedralData,
  };
}

/**
 * Bakes share the app's renderer and await GPU work between cells. Two bakes
 * interleaving would each capture the other's cell target as the "original"
 * render target and restore it at the end, leaving the canvas drawing into a
 * disposed offscreen target. Run them one at a time.
 */
let bakeQueue = Promise.resolve();

function generateAtlasWithCompute(params) {
  const run = bakeQueue.then(() => generateAtlasNow(params));
  bakeQueue = run.catch(() => {});
  return run;
}

/**
 * Generates the octahedral impostor atlas using WebGPU compute shaders.
 * This version uses StorageTexture for direct GPU-based processing.
 * @param {Object} params - Generation parameters
 */
async function generateAtlasNow({
  environment,
  source,
  octahedralData,
  gridSize,
  atlasSize,
  gl,
  camera,
}) {
  console.log("🚀 Starting WebGPU Compute-based atlas generation...");

  // Prepare render mesh (same as before)
  const renderGroup = new THREE.Group();
  let meshCount = 0;

  const sourceScene = source;

  if (sourceScene) {
    sourceScene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        const clonedMesh = child.clone();
        clonedMesh.geometry = bakeMorphTargetsIntoGeometry(child);
        if (child.material) {
          if (Array.isArray(child.material)) {
            clonedMesh.material = child.material.map((mat) => mat.clone());
          } else {
            clonedMesh.material = child.material.clone();
          }
        }
        renderGroup.add(clonedMesh);
        meshCount++;
      }
    });
  }

  console.log(`✓ Cloned ${meshCount} meshes for rendering`);

  const renderMesh = renderGroup;
  renderMesh.visible = true;

  // Create isolated scene for offscreen rendering
  const renderScene = new THREE.Scene();

  // Add lighting
  const directionalLight = new THREE.DirectionalLight(0xffffff, 3);
  directionalLight.position.set(0, 4, 8);
  renderScene.add(directionalLight);

  if (environment) {
    renderScene.environment = environment;
    console.log("✓ Using environment map");
  }

  renderScene.add(renderMesh);

  // Center geometry and configure materials
  let processedMeshCount = 0;
  renderMesh.traverse((node) => {
    if (node instanceof THREE.Mesh && node.geometry) {
      processedMeshCount++;
      node.geometry = node.geometry.clone();

      if (node.material) {
        if (Array.isArray(node.material)) {
          node.material = node.material.map((mat) => mat.clone());
        } else {
          node.material = node.material.clone();
        }
      }

      if (node.material) {
        const materials = Array.isArray(node.material)
          ? node.material
          : [node.material];
        materials.forEach((mat) => {
          if (!mat) return;

          if (mat.isMeshStandardMaterial) {
            if (environment) {
              mat.envMap = environment;
            }
            mat.toneMapped = true;

            // Force the bake to diffuse albedo. Any view-dependent reflection
            // is meaningless in a fixed-angle octahedral atlas (the impostor
            // samples one baked view, so a "reflection" would be frozen/wrong),
            // and several ways of introducing one all turn foliage into a gray
            // mirror of the environment instead of showing its leaf color:
            //   - glTF's default metallicFactor is 1.0, so foliage that omits it
            //     (e.g. tree3.glb's Leaf_01) loads fully metallic.
            //   - KHR_materials_specular + KHR_materials_ior can drive dielectric
            //     F0 to ~1 (e.g. tree.glb's Leaf: ior=1000, specularColor 0.82).
            // Zero out metalness and every specular/reflective channel so the
            // baked cell shows the leaf's baseColor/map, lit only by diffuse.
            mat.metalness = 0;

            if (mat.isMeshPhysicalMaterial) {
              mat.specularIntensity = 0; // kills dielectric F0 regardless of ior
              mat.ior = 1.5;
              mat.clearcoat = 0;
              mat.sheen = 0;
              mat.transmission = 0;
            }
          }

          // Alpha-textured foliage (GLTF "BLEND" materials) arrives here as
          // transparent. Rendering it blended against the transparent cell
          // background premultiplies its RGB by alpha; the impostor then blends
          // again at draw time, squaring the alpha and producing dark, fringed
          // leaves. Bake it as a hard alpha cutout instead so the atlas stores
          // un-premultiplied leaf color plus a clean coverage mask.
          if (mat.transparent) {
            mat.transparent = false;
            mat.depthWrite = true;
            if (!(mat.alphaTest > 0)) {
              mat.alphaTest = 0.5;
            }
          }

          mat.needsUpdate = true;
        });
      }

      const geometry = node.geometry;
      geometry.computeBoundingSphere();
    }
  });

  console.log(`✓ Processed ${processedMeshCount} meshes for centering`);

  // Refresh world matrices so the combined bounding sphere below reflects the
  // per-mesh recentering we just applied (node.position was mutated). Without
  // this the sphere is measured from the STALE pre-recenter world matrices, so
  // for models whose geometry is authored far from the origin (e.g. tree.glb,
  // whose vertices sit near x=-68 with a compensating +68 node translation) the
  // atlas camera frames empty space and bakes a blank/clipped atlas. Mirrors
  // buildLodModelParts, which already updates matrices before measuring.
  renderMesh.updateMatrixWorld(true);

  // Compute bounding sphere and scale
  const boundingSphere = new THREE.Sphere();
  renderMesh.traverse((node) => {
    if (node instanceof THREE.Mesh && node.geometry) {
      node.geometry.computeBoundingSphere();
      if (node.geometry.boundingSphere) {
        const tempSphere = node.geometry.boundingSphere.clone();
        tempSphere.applyMatrix4(node.matrixWorld);
        boundingSphere.union(tempSphere);
      }
    }
  });

  const radius = boundingSphere.radius * 1.5;
  const scaleFactor = radius > 0 ? 0.5 / radius : 1;
  const center = boundingSphere.center.clone();
  renderMesh.scale.setScalar(scaleFactor);
  renderMesh.position.copy(center).multiplyScalar(-scaleFactor);
  renderMesh.updateMatrixWorld(true);

  // "surface" bakes render every cell twice through per-mesh override
  // materials: unlit colour, then view-space normal + depth.
  const surfacePasses = createSurfacePassMaterials(renderMesh);

  // Set up orthographic camera
  const orthoSize = 0.5;
  const renderCam = new THREE.OrthographicCamera(
    -orthoSize,
    orthoSize,
    orthoSize,
    -orthoSize,
    0.001,
    100,
  );

  // Save original render state
  const originalRenderTarget = gl.getRenderTarget();
  const originalClearColor = new THREE.Color();
  gl.getClearColor(originalClearColor);
  const originalClearAlpha = gl.getClearAlpha();

  // The octahedral grid has (gridSize + 1) viewpoint vertices per side, so the
  // atlas stores (gridSize + 1) frames per side. The cell stride must match the
  // vertex stride used by the sampling/material code (gridSize + 1), otherwise
  // baked viewpoints land in the wrong cells and the view shears with elevation.
  const numFrames = gridSize + 1;
  const cellSize = Math.max(1, Math.floor(atlasSize / numFrames));

  const { pntOct } = octahedralData;

  // 🚀 CREATE STORAGE TEXTURE FOR ATLAS (WebGPU Compute)
  console.log("🚀 Creating StorageTexture for atlas...");
  const storageTexture = createStorageTexture(atlasSize, atlasSize);
  // Surface bakes keep linear colour and normals in half floats until the
  // final blit; 8-bit linear storage bands badly in the dark leaf tones.
  const normalStorageTexture = createStorageTexture(atlasSize, atlasSize);
  storageTexture.type = THREE.HalfFloatType;
  normalStorageTexture.type = THREE.HalfFloatType;

  // Create temporary render target for each cell
  // In WebGPU, we use THREE.RenderTarget (not WebGPURenderTarget)
  const cellRenderTarget = new THREE.RenderTarget(cellSize, cellSize, {
    format: THREE.RGBAFormat,
    type: THREE.FloatType,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    generateMipmaps: true,
  });

  console.log(`✓ Created ${cellSize}x${cellSize} render target for cells`);

  // One copy kernel per destination, built once and moved between cells by
  // its offset uniforms. Building a fresh node graph per cell made three
  // re-run node analysis and WGSL generation for every one of the
  // (gridSize + 1)^2 cells, even though the pipeline itself was cached.
  const copyKernels = new Map();
  const copyKernelFor = (target) => {
    if (!copyKernels.has(target)) {
      copyKernels.set(
        target,
        createAtlasCopyComputeShader({
          sourceTexture: cellRenderTarget.texture,
          targetStorageTexture: target,
          cellSize,
          targetX: 0,
          targetY: 0,
          atlasSize,
        }),
      );
    }
    return copyKernels.get(target);
  };

  // Renders the current materials into the cell target and copies the cell
  // into `target` at (pixelX, pixelY).
  const renderCell = async (target, pixelX, pixelY) => {
    gl.setRenderTarget(cellRenderTarget);
    gl.setClearColor(0x000000, 0);
    gl.clear();
    gl.render(renderScene, renderCam);
    // Hand the renderer back before awaiting: the app's own frame can run
    // during the await and must not draw into this offscreen target.
    gl.setRenderTarget(originalRenderTarget);
    gl.setClearColor(originalClearColor, originalClearAlpha);

    // 🚀 USE COMPUTE SHADER TO COPY CELL TO STORAGE TEXTURE
    const { computeNode, cellOffsetXUniform, cellOffsetYUniform } =
      copyKernelFor(target);
    cellOffsetXUniform.value = pixelX;
    cellOffsetYUniform.value = pixelY;
    await gl.computeAsync(computeNode);
  };

  // Render each cell and copy to StorageTexture using compute shader
  let renderedCells = 0;
  const startTime = performance.now();

  for (let rowIdx = 0; rowIdx < numFrames; rowIdx++) {
    for (let colIdx = 0; colIdx < numFrames; colIdx++) {
      const flatIdx = rowIdx * numFrames + colIdx;
      if (flatIdx * 3 + 2 >= pntOct.length) continue;

      const px = pntOct[flatIdx * 3];
      const py = pntOct[flatIdx * 3 + 1];
      const pz = pntOct[flatIdx * 3 + 2];

      const viewDir = new THREE.Vector3(px, py, pz).normalize();

      // Position camera
      const cameraDistance = 0.5;
      renderCam.position.copy(viewDir.multiplyScalar(cameraDistance));
      renderCam.lookAt(0, 0, 0);

      const pixelX = Math.floor((colIdx / numFrames) * atlasSize);
      const pixelY = Math.floor((rowIdx / numFrames) * atlasSize);

      surfacePasses.use("albedo");
      await renderCell(storageTexture, pixelX, pixelY);
      surfacePasses.use("normalDepth");
      await renderCell(normalStorageTexture, pixelX, pixelY);

      renderedCells++;

      // Progress logging
      if (renderedCells % 50 === 0 || renderedCells === numFrames ** 2) {
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
        console.log(
          `⏳ Progress: ${renderedCells}/${numFrames ** 2} cells (${elapsed}s)`,
        );
      }
    }
  }

  const renderTime = ((performance.now() - startTime) / 1000).toFixed(2);
  console.log(`✓ Rendered ${renderedCells} cells in ${renderTime}s`);

  // Cleanup temporary resources
  for (const { computeNode } of copyKernels.values()) computeNode.dispose?.();
  cellRenderTarget.dispose();
  surfacePasses.dispose();

  // StorageTextures are write targets for compute shaders, not meant to be
  // sampled repeatedly by a regular render-pass material long term: three's
  // WebGPU backend re-initializes a StorageTexture's GPU resource on every
  // binding validation pass instead of caching it like a normal texture,
  // which throws "Texture already initialized" once it's bound by an actual
  // mesh material across multiple frames. Blit the finished atlas into a
  // plain RenderTarget texture once here, and use that for sampling instead.
  //
  // Colour goes into a real sRGB target (encoded on write, decoded on read,
  // so the material samples the exact linear albedo), normals/depth into a
  // linear one.
  const atlasTexture = blitToSampleableTexture(
    gl,
    storageTexture,
    atlasSize,
    THREE.SRGBColorSpace,
  );
  const normalDepthTexture = blitToSampleableTexture(
    gl,
    normalStorageTexture,
    atlasSize,
    THREE.NoColorSpace,
  );

  storageTexture.dispose();
  normalStorageTexture.dispose();

  console.log("✅ Atlas generation complete!");
  console.log(`📊 Stats: ${renderedCells} cells, ${renderTime}s total`);

  // Restore original state
  gl.setRenderTarget(originalRenderTarget);
  gl.setClearColor(originalClearColor, originalClearAlpha);

  // Cleanup
  renderScene.remove(renderMesh);
  renderMesh.geometry?.dispose();
  renderMesh.material?.dispose();

  return { texture: atlasTexture, normalDepthTexture };
}

/**
 * Copies a finished storage atlas into a regular render-target texture that
 * materials can sample every frame. `colorSpace` is set on the target before
 * rendering, so an sRGB target encodes on write, and the copy is exact (no
 * blending, no tone mapping).
 */
function blitToSampleableTexture(gl, sourceTexture, atlasSize, colorSpace) {
  const target = new THREE.RenderTarget(atlasSize, atlasSize, {
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    generateMipmaps: false,
    colorSpace,
  });

  const blitMaterial = new THREE.MeshBasicNodeMaterial();
  blitMaterial.colorNode = texture(sourceTexture);
  // A straight copy: no blending into the cleared target, no tone mapping.
  blitMaterial.transparent = false;
  blitMaterial.blending = THREE.NoBlending;
  blitMaterial.toneMapped = false;

  const blitScene = new THREE.Scene();
  const blitGeometry = new THREE.PlaneGeometry(2, 2);
  blitScene.add(new THREE.Mesh(blitGeometry, blitMaterial));

  const blitCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  blitCamera.position.set(0, 0, 1);
  blitCamera.lookAt(0, 0, 0);

  const previousTarget = gl.getRenderTarget();
  gl.setRenderTarget(target);
  gl.render(blitScene, blitCamera);
  gl.setRenderTarget(previousTarget);

  blitGeometry.dispose();
  blitMaterial.dispose();

  const result = target.texture;
  result.flipY = false;
  result.minFilter = THREE.LinearFilter;
  result.magFilter = THREE.LinearFilter;
  result.wrapS = THREE.ClampToEdgeWrapping;
  result.wrapT = THREE.ClampToEdgeWrapping;
  result.colorSpace = colorSpace;
  return result;
}

/**
 * Per-mesh override materials for a "surface" bake. `albedo` renders each
 * material's unlit colour (its colorNode, else map × color) with its alpha
 * cutout; `normalDepth` renders its shading normal (its normalNode, else the
 * geometry normal) in bake-camera view space plus the distance from the bake
 * camera, masked by the same cutout so both atlases share one silhouette.
 */
function createSurfacePassMaterials(root) {
  const entries = [];
  root.traverse((node) => {
    if (!(node instanceof THREE.Mesh) || !node.material) return;
    const sources = Array.isArray(node.material)
      ? node.material
      : [node.material];
    const albedo = [];
    const normalDepth = [];
    for (const source of sources) {
      const baseColor = source.colorNode
        ? vec4(source.colorNode)
        : source.map
          ? texture(source.map).mul(
              vec4(source.color ?? new THREE.Color(1, 1, 1), 1),
            )
          : vec4(source.color ?? new THREE.Color(1, 1, 1), 1);
      const cutoff = source.alphaTest > 0 ? source.alphaTest : 0.5;

      // Coverage is binary: a fragment either survives the material's cutout
      // (coverage 1) or leaves the cleared background (0). Source alpha itself
      // is not coverage (foliage atlases often keep partial alpha inside the
      // leaf and rely on alphaTest), so both passes cut out via maskNode and
      // write full-coverage values; the cleared background makes the atlases
      // premultiplied (see the "surface" bake notes above).
      const kept = baseColor.a.greaterThan(float(cutoff));

      const albedoMaterial = new THREE.MeshBasicNodeMaterial({
        side: source.side,
      });
      albedoMaterial.colorNode = vec4(baseColor.rgb, 1);
      albedoMaterial.maskNode = kept;
      albedoMaterial.blending = THREE.NoBlending;
      albedoMaterial.toneMapped = false;
      albedo.push(albedoMaterial);

      const shadingNormal = normalize(source.normalNode ?? normalView);
      const normalMaterial = new THREE.MeshBasicNodeMaterial({
        side: source.side,
      });
      normalMaterial.colorNode = vec4(
        shadingNormal.mul(0.5).add(0.5),
        positionView.z.negate().clamp(0, 1),
      );
      // The alpha channel carries depth, so the cutout goes through maskNode instead of alphaTest.
      normalMaterial.maskNode = kept;
      normalMaterial.blending = THREE.NoBlending;
      normalMaterial.toneMapped = false;
      normalDepth.push(normalMaterial);
    }
    const single = !Array.isArray(node.material);
    entries.push({
      node,
      albedo: single ? albedo[0] : albedo,
      normalDepth: single ? normalDepth[0] : normalDepth,
    });
  });

  return {
    use(pass) {
      for (const entry of entries) entry.node.material = entry[pass];
    },
    dispose() {
      for (const entry of entries) {
        for (const material of [entry.albedo, entry.normalDepth].flat())
          material.dispose();
      }
    },
  };
}
