from typing import Tuple, Optional

import torch
import torch.nn as nn

from diffsynth.models.wan_video_dit import WanModel, DiTBlock
from diffsynth.models.wan_video_dit import modulate
from diffsynth.models.wan_video_dit import precompute_freqs_cis


class ActionDiTBlock(DiTBlock):
    def __init__(
        self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6
    ):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)

    def _init_meta_parameters(self):
        pass


class ActionDiTHintBlock(DiTBlock):
    def __init__(
        self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6
    ):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)

    def forward(self, x, vac_cond, context, t_mod, freqs):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)

        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))

        x = x + self.cross_attn(self.norm3(x), context)

        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))

        x = x + vac_cond

        return x

    def _init_meta_parameters(self):
        pass


class ActionDit(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        num_layers: int,
        vac_layers: Tuple = None,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim

        self.has_image_input = False
        self.vac_layers = vac_layers

        self.input_embedding = nn.Linear(in_dim, dim)

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        blocks = []

        for block_id in range(num_layers):
            if block_id in vac_layers:
                blocks.append(
                    ActionDiTHintBlock(self.has_image_input, dim, num_heads, ffn_dim, eps)
                )
            else:
                blocks.append(ActionDiTBlock(self.has_image_input, dim, num_heads, ffn_dim, eps))

        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Linear(dim, out_dim)

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis(head_dim)

    def forward(
        self,
        **kwargs,
    ):
        pass

    def _init_meta_parameters(self):
        pass


adit_config = {
    "in_dim": 138,
    "dim": 1536,
    "num_heads": 12,
    "ffn_dim": 8960,
    "out_dim": 138,
    "freq_dim": 256,
    "eps": 1e-6,
    "num_layers": 30,
}
