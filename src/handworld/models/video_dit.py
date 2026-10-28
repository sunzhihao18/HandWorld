from typing import Tuple, Optional

import torch
import torch.nn as nn

from diffsynth.models.wan_video_dit import WanModel, DiTBlock
from diffsynth.models.wan_video_dit import modulate


class VideoDiTBlock(DiTBlock):
    def __init__(
        self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6
    ):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)

    def _init_meta_parameters(self):
        pass


class VideoDiTHintBlock(DiTBlock):
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


class VideoDit(WanModel):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        vac_layers: Tuple = None,
    ):
        super().__init__(
            dim=dim,
            in_dim=in_dim,
            ffn_dim=ffn_dim,
            out_dim=out_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            eps=eps,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            has_image_input=has_image_input,
            has_image_pos_emb=has_image_pos_emb,
            has_ref_conv=has_ref_conv,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            seperated_timestep=seperated_timestep,
            require_vae_embedding=require_vae_embedding,
            require_clip_embedding=require_clip_embedding,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )

        self.vac_layers = vac_layers

        blocks = []

        for block_id in range(num_layers):
            if block_id in vac_layers:
                blocks.append(VideoDiTHintBlock(has_image_input, dim, num_heads, ffn_dim, eps))
            else:
                blocks.append(VideoDiTBlock(has_image_input, dim, num_heads, ffn_dim, eps))

        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        **kwargs,
    ):
        pass

    def _init_meta_parameters(self):
        pass
