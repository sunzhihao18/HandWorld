"""Convert an EgoDex MP4/HDF5 pair into aligned HandWorld inference inputs."""

import argparse
import json
import shutil
from pathlib import Path


JOINTS = [
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--annotation", type=Path, help="Defaults to the video path with .hdf5 suffix"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--frames", type=int, default=64, help="Use 13 for history-only stage3 inputs"
    )
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    import cv2
    import h5py
    import numpy as np
    import torch

    cap = cv2.VideoCapture(str(args.video))
    for i in range(args.frames):
        ok, _ = cap.read()
        if not ok:
            raise ValueError(f"Video has fewer than {args.frames} frames (missing frame {i})")
    cap.release()
    annotation = args.annotation or args.video.with_suffix(".hdf5")
    with h5py.File(annotation, "r") as file:
        transforms = file["transforms"]
        camera = transforms["camera"][: args.frames]
        hand = np.stack(
            [
                transforms[side + name][: args.frames]
                for side in ("left", "right")
                for name in JOINTS
            ],
            axis=1,
        )
    if camera.shape != (args.frames, 4, 4) or hand.shape != (args.frames, 42, 4, 4):
        raise ValueError("Annotations do not contain the requested aligned transforms")
    inverse = np.zeros_like(camera)
    inverse[:, :3, :3] = camera[:, :3, :3].transpose(0, 2, 1)
    inverse[:, :3, 3:] = -inverse[:, :3, :3] @ camera[:, :3, 3:]
    inverse[:, 3, 3] = 1
    hand_camera = inverse[:, None] @ hand
    actions = np.concatenate(
        [
            camera[:, :3, :3].reshape(-1, 9),
            camera[:, :3, 3].reshape(-1, 3),
            hand_camera[:, :, :3, 3].reshape(-1, 126),
        ],
        axis=1,
    )
    if not np.isfinite(actions).all():
        raise ValueError("Annotations contain nonfinite action values")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "rgb.mp4"
    if args.video.resolve() != destination.resolve():
        shutil.copyfile(args.video, destination)
    torch.save(torch.from_numpy(actions), args.output / "actions.pth")
    data = dict(
        schema_version=1, prompt=args.prompt, rgb="rgb.mp4", mano="mano.mp4", actions="actions.pth"
    )
    (args.output / "input.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"Saved RGB, actions and input.json to {args.output}; next estimate and render MANO.")


if __name__ == "__main__":
    main()
