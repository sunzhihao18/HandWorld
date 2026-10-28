"""Strict loading of the two validated inference checkpoint combinations."""

from pathlib import Path

import torch
from diffsynth.models import ModelManager
from diffsynth.models.model_manager import init_weights_on_device
from safetensors.torch import load_file

from .artifacts import read_config
from .models.action_dit import ActionDit, adit_config
from .models.dual_vac_dit import DualVideoActionConditionDit, dual_vac_config
from .models.video_dit import VideoDit
from .pipeline import HandWorldPipeline


def load_pipeline(profile, weights_root):
    root = Path(weights_root).resolve()
    config = read_config("profiles.json")
    selected = config["profiles"][profile]
    dtype = torch.bfloat16
    pipe = HandWorldPipeline(device="cpu", torch_dtype=dtype)
    pipe.audio_encoder = None
    manager = ModelManager()
    for name in ("text_encoder", "vae"):
        manager.load_model(str(root / config["shared"][name]), device="cpu", torch_dtype=dtype)
    pipe.text_encoder = manager.fetch_model("wan_video_text_encoder")
    pipe.vae = manager.fetch_model("wan_video_vae")
    pipe.prompter.fetch_models(pipe.text_encoder)
    pipe.prompter.fetch_tokenizer(str(root / config["shared"]["tokenizer"]))
    # The explicit config is identical to the original Wan shard-header config.
    # Complete HandWorld checkpoints supply every DiT tensor; no base shards needed.
    with init_weights_on_device():
        pipe.vac = DualVideoActionConditionDit(**dual_vac_config)
        pipe.adit = ActionDit(**adit_config, vac_layers=pipe.vac.vac_layers)
        pipe.dit = VideoDit(**read_config("video_dit.json"), vac_layers=pipe.vac.vac_layers)
    print(f'Loading {selected["checkpoint"]}', flush=True)
    state = load_file(str(root / selected["checkpoint"]), device="cpu")
    for name in ("adit", "dit", "vac"):
        prefix = name + "."
        tensors = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        module = getattr(pipe, name)
        module.load_state_dict(tensors, strict=True, assign=True)
        module.to(dtype=dtype, device="cpu")
        print(f"  Loaded {name}", flush=True)
    del state
    pipe.height_division_factor = pipe.vae.upsampling_factor * 2
    pipe.width_division_factor = pipe.vae.upsampling_factor * 2
    pipe.eval()
    return pipe
