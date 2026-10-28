"""Estimate MANO parameters with the original handedness detector and HaMeR crops."""

import argparse
import copy
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output MANO parameter .pth")
    parser.add_argument(
        "--detector", type=Path, required=True, help="0=left, 1=right detector checkpoint"
    )
    parser.add_argument(
        "--hamer-weights",
        type=Path,
        required=True,
        help="Directory with model_config.yaml and checkpoints/hamer.ckpt",
    )
    parser.add_argument("--mano-model-dir", type=Path, required=True)
    parser.add_argument("--mano-mean-params", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--focal-length", type=float, default=736.6339)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.frames < 1 or args.batch_size < 1 or args.focal_length <= 0:
        parser.error("frames, batch-size and focal-length must be positive")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import cv2
    import torch
    from ultralytics import YOLO
    from hamer.configs import get_config
    from hamer.models import HAMER
    from hamer.utils import recursive_to
    from hamer.utils.renderer import cam_crop_to_full
    from _preprocess.crops import HamerDataset
    from _preprocess.tracking import track_hand

    torch.set_num_threads(8)
    cap = cv2.VideoCapture(str(args.video))
    frames = []
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Missing RGB frame {i}")
        frames.append(frame)
    cap.release()
    detections = track_hand(frames, YOLO(str(args.detector)), track_conf=0.4)
    crops = HamerDataset(frames, copy.deepcopy(detections))
    params = [{"lhandR": None, "rhand": None} for _ in frames]
    if len(crops):
        cfg = get_config(str(args.hamer_weights / "model_config.yaml"), update_cachedir=True)
        cfg.defrost()
        cfg.MANO.MEAN_PARAMS = str(args.mano_mean_params)
        cfg.MANO.MODEL_PATH = str(args.mano_model_dir)
        if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
            cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
        cfg.freeze()
        model = (
            HAMER.load_from_checkpoint(
                str(args.hamer_weights / "checkpoints/hamer.ckpt"),
                strict=False,
                cfg=cfg,
                init_renderer=False,
            )
            .eval()
            .to(args.device)
        )
        batches = torch.utils.data.DataLoader(
            crops, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        cursor = 0
        with torch.no_grad():
            for batch in batches:
                batch = recursive_to(batch, args.device)
                pred = model(batch)
                n = len(batch["img"])
                transl = cam_crop_to_full(
                    pred["pred_cam"],
                    batch["box_center"],
                    batch["box_size"],
                    batch["img_size"],
                    args.focal_length,
                ).cpu()
                betas = pred["pred_mano_params"]["betas"].cpu()
                pose = (
                    torch.cat(
                        [
                            pred["pred_mano_params"]["global_orient"].reshape(n, -1),
                            pred["pred_mano_params"]["hand_pose"].reshape(n, -1),
                        ],
                        dim=-1,
                    )
                    .reshape(n, 16, 3, 3)
                    .cpu()
                )
                for j in range(n):
                    key = "rhand" if crops.is_rhand[cursor + j] else "lhandR"
                    params[crops.frames_index[cursor + j]][key] = dict(
                        betas=betas[j], pose=pose[j], transl=transl[j]
                    )
                cursor += n
                print(f"Estimated {cursor}/{len(crops)} hand crops", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(params, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
