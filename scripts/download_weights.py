"""Download complete HandWorld checkpoints and the optional hand detector."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Zachary18/HandWorld")
    parser.add_argument("--output", type=Path, default=Path("weights"))
    parser.add_argument("--stage", choices=("stage2", "stage3", "both"), default="both")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--preprocess", action="store_true", help="Also download the handedness detector"
    )
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    names = ["stage2", "stage3"] if args.stage == "both" else [args.stage]
    patterns = [name + ".safetensors" for name in names]
    if args.preprocess:
        patterns += ["preprocess/detector.pt"]
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(args.output),
        allow_patterns=patterns,
    )
    print(f"Weights saved in {args.output}")


if __name__ == "__main__":
    main()
