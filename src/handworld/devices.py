"""Place complete inference modules on separate devices without changing their math."""

from functools import wraps

import torch
from accelerate.hooks import AlignDevicesHook, add_hook_to_module


def place_inference_modules(pipe, generator_device, encoder_device=None):
    generator_device = torch.device(generator_device)
    encoder_device = torch.device(encoder_device or generator_device)
    pipe.device = generator_device
    for name in ("dit", "adit", "vac"):
        getattr(pipe, name).to(generator_device)
    pipe.text_encoder.to(encoder_device)
    pipe.vae.to(encoder_device)
    if encoder_device != generator_device:
        # Prompt tokens originate on the generator device. Return embeddings there,
        # while retaining the full, unmodified T5 encoder on the second device.
        add_hook_to_module(
            pipe.text_encoder,
            AlignDevicesHook(execution_device=encoder_device, io_same_device=True),
        )
        # VAE encode/decode are methods, not Module.forward, so forward hooks do
        # not route them. DiffSynth accepts the execution device explicitly.
        for method_name in ("encode", "decode"):
            original = getattr(pipe.vae, method_name)

            def bridge(method):
                @wraps(method)
                def call(tensors, device, *args, **kwargs):
                    return method(tensors, encoder_device, *args, **kwargs).to(device)

                return call

            setattr(pipe.vae, method_name, bridge(original))
    pipe.eval()
    print(f"Generator (DiT/ADiT/VAC): {generator_device}; T5/VAE: {encoder_device}", flush=True)
    return pipe
