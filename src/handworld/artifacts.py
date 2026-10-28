"""Inference settings and local weight paths."""

import json
from importlib.resources import files
from pathlib import Path


def read_config(name):
    return json.loads(files("handworld").joinpath("configs", name).read_text())


def check_weights(weights_root, profile):
    root = Path(weights_root)
    config = read_config("profiles.json")
    names = [config["profiles"][profile]["checkpoint"]]
    names += [config["shared"][key] for key in ("text_encoder", "vae")]
    names += [
        config["shared"]["tokenizer"] + "/" + name
        for name in (
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
    ]
    missing = [str(root / name) for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing weights: "
            + ", ".join(missing)
            + ". Download Zachary18/HandWorld into --weights-root."
        )
