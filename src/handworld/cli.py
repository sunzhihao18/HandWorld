"""Stage2 action-conditioned video and stage3 joint video/action prediction."""

import argparse
import os
from pathlib import Path

from .artifacts import check_weights, read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("stage2", "stage3"))
    parser.add_argument("--weights-root", type=Path, default=Path("weights"))
    parser.add_argument("--input", type=Path, required=True, help="Prepared input.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoder-device", help="Optional second GPU for T5/VAE")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, help="Default: stage2=5, stage3=1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--tiled", action="store_true", help="Enable VAE tiling")
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error("--steps and --threads must be positive")
    profile = read_config("profiles.json")["profiles"][args.task]
    if args.cfg_scale is None:
        args.cfg_scale = profile["cfg_scale"]
    if args.cfg_scale < 0:
        parser.error("--cfg-scale must be nonnegative")
    check_weights(args.weights_root, args.task)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from diffsynth import save_video
    from .devices import place_inference_modules
    from .inputs import load_inputs
    from .loading import load_pipeline

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    sample, rgb, mano, actions = load_inputs(args.input, profile)
    pipe = load_pipeline(args.task, args.weights_root)
    place_inference_modules(pipe, args.device, args.encoder_device)
    original_model_fn = pipe.model_fn

    def checked_model_fn(*fn_args, **fn_kwargs):
        result = original_model_fn(*fn_args, **fn_kwargs)
        if any(x is not None and not x.isfinite().all() for x in result):
            raise ValueError("Nonfinite denoising prediction")
        return result

    pipe.model_fn = checked_model_fn
    kwargs = dict(
        prompt=sample["prompt"],
        input_video=rgb,
        input_image=rgb[0],
        input_action_v=mano,
        input_action=actions,
        height=profile["height"],
        width=profile["width"],
        num_frames=profile["frames"],
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=args.tiled,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.task == "stage3":
        frames, predicted_actions = pipe.infer_stage3(**kwargs)
        if not predicted_actions.isfinite().all():
            raise ValueError("Nonfinite forecast actions")
        torch.save(predicted_actions.cpu(), args.output / "action.pth")
    else:
        frames = pipe.infer_stage2(**kwargs)
    save_video(frames, str(args.output / "video.mp4"), fps=profile["fps"], quality=9)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
