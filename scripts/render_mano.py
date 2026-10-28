"""Render estimated MANO parameters to aligned cyan/purple hands on black."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument(
        "--video", type=Path, required=True, help="Aligned RGB video for image size and frame rate"
    )
    parser.add_argument("--mano-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focal-length", type=float, default=736.6339)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.focal_length <= 0:
        parser.error("--focal-length must be positive")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import cv2
    import numpy as np
    import torch
    from _preprocess.mano import MANORotMatDecoder
    from _preprocess.renderer import MANORenderer, LHAND_COLOR, RHAND_COLOR
    from _preprocess.smoothing import remove_isolated, filter_outliers, interpolate, smooth_savgol

    params = torch.load(args.params, map_location="cpu", weights_only=True)
    if not isinstance(params, list) or not params:
        raise ValueError("Expected a nonempty list of per-frame MANO parameters")
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for i in range(len(params)):
        ok, _ = cap.read()
        if not ok:
            raise ValueError(f"RGB video missing frame {i}")
    cap.release()
    if fps <= 0 or min(width, height) <= 0:
        raise ValueError("Invalid RGB video properties")
    decoder = MANORotMatDecoder(model_path=str(args.mano_model_dir), is_rhand=True).to(args.device)
    temporal = {}
    with torch.no_grad():
        for key in ("lhandR", "rhand"):
            indices = [i for i, p in enumerate(params) if p[key] is not None]
            hand = [params[i][key] for i in indices]
            sequence = [None] * len(params)
            if hand:
                vertices = (
                    decoder(
                        betas=torch.stack([p["betas"] for p in hand]).to(args.device),
                        global_orient=torch.stack([p["pose"][:1] for p in hand]).to(args.device),
                        hand_pose=torch.stack([p["pose"][1:] for p in hand]).to(args.device),
                        transl=torch.stack([p["transl"] for p in hand]).to(args.device),
                    )["vertices"]
                    .cpu()
                    .numpy()
                )
                for i, v in zip(indices, vertices):
                    if key == "lhandR":
                        v[:, 0] *= -1
                    sequence[i] = v
            for fn in (remove_isolated, filter_outliers, interpolate, smooth_savgol):
                sequence = fn(sequence)
            temporal[key] = sequence
    renderer = MANORenderer(model_path=str(args.mano_model_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open MP4 video writer")
    for i in range(len(params)):
        present = [
            (key, right, color)
            for key, right, color in (("lhandR", False, LHAND_COLOR), ("rhand", True, RHAND_COLOR))
            if temporal[key][i] is not None
        ]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if present:
            rgba = renderer.render_mesh_RGBA(
                [temporal[key][i] for key, _, _ in present],
                is_rhand=[right for _, right, _ in present],
                render_res=[width, height],
                focal_length=[args.focal_length] * 2,
                camera_center=[width / 2, height / 2],
                mesh_base_color=[color for _, _, color in present],
            )
            canvas = (rgba[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)
        writer.write(canvas)
        print(f"Rendered {i+1}/{len(params)}", flush=True)
    writer.release()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
