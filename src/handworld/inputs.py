"""Read only the observed input frames, with explicit temporal alignment."""

import json
from pathlib import Path

import torch
from PIL import Image
from diffsynth import VideoData


def load_inputs(manifest_path, profile):
    path = Path(manifest_path).resolve()
    sample = json.loads(path.read_text())
    if sample.get("schema_version") != 1:
        raise ValueError("Expected input manifest schema_version=1")
    paths = {k: (path.parent / sample[k]).resolve() for k in ("rgb", "mano", "actions")}
    joints = torch.load(paths["actions"], map_location="cpu", weights_only=True)
    if joints.ndim == 2:
        joints = joints.unsqueeze(0)
    required = profile["observed_action_frames"]
    if (
        joints.ndim != 3
        or joints.shape[0] != 1
        or joints.shape[-1] != 138
        or joints.shape[1] < required
    ):
        raise ValueError(
            f"Expected actions (T,138) or (1,T,138), T>={required}; got {tuple(joints.shape)}"
        )
    # Validate only the observed prefix; future labels never enter the pipeline.
    if not joints[:, :required].isfinite().all():
        raise ValueError("Observed actions contain NaN or infinity")
    actions = torch.zeros((1, profile["frames"], 138), dtype=joints.dtype)
    actions[:, :required] = joints[:, :required]
    observed = {}
    for kind in ("rgb", "mano"):
        count = profile[f"observed_{kind}_frames"]
        if paths[kind].suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            if count != 1:
                raise ValueError(f"{kind} requires {count} frames; a still image is insufficient")
            with Image.open(paths[kind]) as image:
                from diffsynth.data.video import crop_and_resize

                frames = [
                    crop_and_resize(image.convert("RGB"), profile["height"], profile["width"])
                ]
        else:
            video = VideoData(
                video_file=str(paths[kind]), height=profile["height"], width=profile["width"]
            )
            if len(video) < count:
                raise ValueError(f"{kind} requires at least {count} observed frames")
            frames = [video[i] for i in range(count)]
        blank = Image.new("RGB", (profile["width"], profile["height"]))
        observed[kind] = frames + [blank] * (profile["frames"] - count)
    if not isinstance(sample.get("prompt"), str):
        raise ValueError("Input manifest requires a text prompt")
    return sample, observed["rgb"], observed["mano"], actions
