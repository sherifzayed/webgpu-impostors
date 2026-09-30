import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/examples/jsm/loaders/KTX2Loader.js";

/** Decoders are copied into public/ by scripts/extract-community.mjs, so nothing loads from a CDN. */
const BASIS_TRANSCODER_PATH = "/basis/";
const DRACO_DECODER_PATH = "/draco/";

let ktx2Loader: KTX2Loader | undefined;
let gltfLoader: GLTFLoader | undefined;

/** One KTX2 loader per page: each instance spins up its own transcoder workers. The renderer must already be initialised. */
export async function getKtx2Loader(renderer): Promise<KTX2Loader> {
  ktx2Loader ??= new KTX2Loader().setTranscoderPath(BASIS_TRANSCODER_PATH).detectSupport(renderer);
  return ktx2Loader;
}

/** GLTF loader that understands the community's KTX2 textures and the vegetation kit's Draco meshes. */
export async function getGltfLoader(renderer): Promise<GLTFLoader> {
  if (!gltfLoader) {
    gltfLoader = new GLTFLoader()
      .setDRACOLoader(new DRACOLoader().setDecoderPath(DRACO_DECODER_PATH))
      .setKTX2Loader(await getKtx2Loader(renderer));
  }
  return gltfLoader;
}
