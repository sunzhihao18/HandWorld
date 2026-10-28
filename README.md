# HandWorld: Hand-Centric Unified Video Action Generation (CVPR 2026)

## Installation

Linux, Python 3.10 and CUDA 12.8 are recommended. DiffSynth must stay at **1.1.8**.

```bash
conda create -n handworld python=3.10 -y
conda activate handworld
git clone https://github.com/sunzhihao18/HandWorld.git
cd HandWorld
python -m pip install 'setuptools<81' wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install .
```

The provided commands use two 32 GB GPUs: one for the generator and one for T5/VAE. Peak allocated memory is approximately 21.3 / 16.5 GiB. A single larger GPU can omit `--encoder-device`. No flash-attn or xformers is required.

## Weights

Download from [Zachary18/HandWorld](https://huggingface.co/Zachary18/HandWorld) into the empty `weights/` directory:

```bash
python scripts/download_weights.py --output weights --preprocess
```

This places `stage2.safetensors`, `stage3.safetensors` and the optional detector under `weights/`. Both stage checkpoints are complete; no earlier-stage checkpoint or original Wan video-DiT shards are needed. For a single model, add `--stage stage2` or `--stage stage3`. Omit `--preprocess` if you do not need the hand detector.

Download the shared assets separately from the official [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/main) repository. Place `models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.2_VAE.pth`, and the `google/umt5-xxl/` tokenizer directory under `weights/Wan2.2-TI2V-5B/`, preserving their relative paths. These official assets are not reuploaded to the HandWorld repository.

## Prepare data

For raw EgoDex data, use the following three scripts. RGB, actions and MANO renders must refer to the same frame indices.

Preprocessing additionally needs [HaMeR](https://github.com/geopavlakos/hamer), its pretrained checkpoint/config/mean parameters, and the licensed MANO model from the [MANO website](https://mano.is.tue.mpg.de/). Follow HaMeR's installation and `fetch_demo_data.sh` instructions in a separate preprocessing environment. Keep a CUDA 12.8 compatible PyTorch build when using RTX 5090. Then install the additional dependencies and our handedness detector:

```bash
python -m pip install -r requirements-preprocess.txt
```

The detector is `weights/preprocess/detector.pt` (class 0: left hand, class 1: right hand). HaMeR and MANO assets are obtained separately. A generic object detector is not interchangeable with this checkpoint.

1. Convert an EgoDex MP4 and its matching HDF5 annotation to RGB, 138-dimensional actions and `input.json`:

```bash
python scripts/prepare_egodex.py \
  --video /path/to/clip.mp4 --annotation /path/to/clip.hdf5 \
  --prompt 'charge uncharge device' --frames 64 --output data/clip
```

2. Estimate MANO parameters from RGB with the original detector and hand-crop preprocessing:

```bash
python scripts/estimate_mano.py \
  --video data/clip/rgb.mp4 --frames 64 --output data/clip/mano_params.pth \
  --detector weights/preprocess/detector.pt \
  --hamer-weights /path/to/hamer/_DATA/hamer_ckpts \
  --mano-model-dir /path/to/hamer/_DATA/data/mano \
  --mano-mean-params /path/to/hamer/_DATA/data/mano_mean_params.npz
```

3. Smooth the estimated meshes and render the aligned MANO video:

```bash
python scripts/render_mano.py \
  --params data/clip/mano_params.pth --video data/clip/rgb.mp4 \
  --mano-model-dir /path/to/hamer/_DATA/data/mano --output data/clip/mano.mp4
```

Rendering uses black backgrounds, cyan left hands and purple right hands. Missing detections remain absent. The default focal length is **736.6339 pixels**, matching the original 1920×1080 EgoDex processing; for another calibration, pass the same `--focal-length` to both MANO scripts. Headless rendering uses EGL.

For stage3 with only history available, use `--frames 13` in both preparation and estimation. The renderer follows the number of saved parameter frames. For stage2, provide at least 49 aligned MANO/action frames; the 64-frame preparation above preserves context for mesh smoothing.

`prepare_egodex.py` automatically creates `data/clip/input.json` for inference, with file paths relative to the JSON file:

```json
{
  "schema_version": 1,
  "prompt": "charge uncharge device",
  "rgb": "rgb.mp4",
  "mano": "mano.mp4",
  "actions": "actions.pth"
}
```

Actions are a tensor `(T,138)` or `(1,T,138)`: row-major camera rotation (9), camera translation (3), and camera-space hand XYZ positions (126; left hand first, 21 joints per hand). Retain EgoDex units and coordinates. Hand order is wrist, then thumb/index/middle/ring/little, four joints per finger. No temporal resampling or action normalization is applied.

## Inference

The Python package is `handworld` in `src/handworld/`. You can also use `python -m handworld` in place of the `handworld` command below.

```bash
CUDA_VISIBLE_DEVICES=0,1 handworld stage2 \
  --input data/clip/input.json --weights-root weights \
  --device cuda:0 --encoder-device cuda:1 --output outputs/stage2

CUDA_VISIBLE_DEVICES=0,1 handworld stage3 \
  --input data/clip/input.json --weights-root weights \
  --device cuda:0 --encoder-device cuda:1 --output outputs/stage3
```

Both generate 49 frames at 480×832, with 50 denoising steps and seed 0. CFG defaults to 5 for stage2 and 1 for stage3. Videos are saved at 8 fps. Options include `--steps`, `--cfg-scale`, `--seed` and `--tiled` for VAE tiling.

Stage2 writes `video.mp4` and uses the complete supplied MANO/action sequence as conditions. Stage3 writes `video.mp4` and `action.pth` shaped `(1,49,138)`; only the first 13 input frames are used, and future inputs are masked before encoding. Use stage2 for AC video generation and stage3 for joint forecasting.

## Citation

```bibtex
@InProceedings{Sun_2026_CVPR,
  author = {Sun, Zhihao and Du, Zhiying and Yang, Xitong and Wu, Zuxuan},
  title = {HandWorld: Hand-Centric Unified Video Action Generation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2026},
  pages = {15976-15985}
}
```

Built on [Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [HaMeR](https://github.com/geopavlakos/hamer).
