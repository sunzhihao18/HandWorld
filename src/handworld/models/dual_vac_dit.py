import torch
import torch.nn as nn
from diffsynth.models.wan_video_dit import SelfAttention, GateModule, modulate
from diffsynth.models.wan_video_dit import RMSNorm, AttentionModule, flash_attention


class CrossAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-06):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        return self.o(x)


class DualVACDitBlock(nn.Module):

    def __init__(
        self,
        video_vace_dim: int,
        video_vace_num_heads: int,
        video_vace_ffn_dim: int,
        action_vace_dim: int,
        action_vace_num_heads: int,
        action_vace_ffn_dim: int,
        eps: float = 1e-06,
        block_id: int = 0,
        last_block_id: int = -1,
    ):
        super().__init__()
        self.block_id = block_id
        self.last_block_id = last_block_id
        self.video_vace_dim = video_vace_dim
        self.video_vace_num_heads = video_vace_num_heads
        self.video_vace_ffn_dim = video_vace_ffn_dim
        self.video_vace_self_attn = SelfAttention(video_vace_dim, video_vace_num_heads, eps)
        self.video_vace_cross_attn = CrossAttention(video_vace_dim, video_vace_num_heads, eps)
        self.video_vace_norm1 = nn.LayerNorm(video_vace_dim, eps=eps, elementwise_affine=False)
        self.video_vace_norm3 = nn.LayerNorm(video_vace_dim, eps=eps)
        self.video_vace_norm2 = nn.LayerNorm(video_vace_dim, eps=eps, elementwise_affine=False)
        self.video_vace_ffn = nn.Sequential(
            nn.Linear(video_vace_dim, video_vace_ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(video_vace_ffn_dim, video_vace_dim),
        )
        self.video_vace_modulation = nn.Parameter(
            torch.randn(1, 6, video_vace_dim) / video_vace_dim**0.5
        )
        self.video_vace_gate = GateModule()
        self.action_vace_dim = action_vace_dim
        self.action_vace_num_heads = action_vace_num_heads
        self.action_vace_ffn_dim = action_vace_ffn_dim
        self.action_vace_self_attn = SelfAttention(action_vace_dim, action_vace_num_heads, eps)
        self.action_vace_cross_attn = CrossAttention(action_vace_dim, action_vace_num_heads, eps)
        self.action_vace_norm1 = nn.LayerNorm(action_vace_dim, eps=eps, elementwise_affine=False)
        self.action_vace_norm3 = nn.LayerNorm(action_vace_dim, eps=eps)
        self.action_vace_norm2 = nn.LayerNorm(action_vace_dim, eps=eps, elementwise_affine=False)
        self.action_vace_ffn = nn.Sequential(
            nn.Linear(action_vace_dim, action_vace_ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_vace_ffn_dim, action_vace_dim),
        )
        self.action_vace_modulation = nn.Parameter(
            torch.randn(1, 6, action_vace_dim) / action_vace_dim**0.5
        )
        self.action_vace_gate = GateModule()
        self.video2action_proj = nn.Linear(video_vace_dim, action_vace_dim)
        self.video2action_norm = nn.LayerNorm(action_vace_dim, eps=eps)
        self.video2action_cross_attn = CrossAttention(action_vace_dim, action_vace_num_heads, eps)
        self.action2video_proj = nn.Linear(action_vace_dim, video_vace_dim)
        self.action2video_norm = nn.LayerNorm(video_vace_dim, eps=eps)
        self.action2video_cross_attn = CrossAttention(video_vace_dim, video_vace_num_heads, eps)
        if block_id == 0:
            self.video_vace_before_proj = nn.Linear(video_vace_dim, video_vace_dim)
            nn.init.eye_(self.video_vace_before_proj.weight)
            nn.init.zeros_(self.video_vace_before_proj.bias)
            self.action_vace_before_proj = nn.Linear(action_vace_dim, action_vace_dim)
            nn.init.eye_(self.action_vace_before_proj.weight)
            nn.init.zeros_(self.action_vace_before_proj.bias)
        self.video_vace_after_proj = nn.Linear(video_vace_dim, video_vace_dim)
        nn.init.normal_(self.video_vace_after_proj.weight, mean=0, std=1e-05)
        nn.init.zeros_(self.video_vace_after_proj.bias)
        self.action_vace_after_proj = nn.Linear(action_vace_dim, action_vace_dim)
        nn.init.normal_(self.action_vace_after_proj.weight, mean=0, std=1e-05)
        nn.init.zeros_(self.action_vace_after_proj.bias)

    def forward(
        self,
        video_cond,
        x_video,
        context,
        t_mod_in_video,
        video_freqs,
        action_cond,
        x_action,
        context_in_action,
        t_mod_in_action,
        action_freqs,
    ):
        if self.block_id == 0:
            video_cond = self.video_vace_before_proj(video_cond)
            if x_video is not None:
                video_cond = video_cond + x_video
            all_video_cond = []
            action_cond = self.action_vace_before_proj(action_cond)
            if x_action is not None:
                action_cond = action_cond + x_action
            all_action_cond = []
        else:
            all_video_cond = list(torch.unbind(video_cond))
            video_cond = all_video_cond.pop(-1)
            all_action_cond = list(torch.unbind(action_cond))
            action_cond = all_action_cond.pop(-1)
        has_seq = len(t_mod_in_video.shape) == 4
        chunk_dim = 2 if has_seq else 1
        (
            video_shift_msa,
            video_scale_msa,
            video_gate_msa,
            video_shift_mlp,
            video_scale_mlp,
            video_gate_mlp,
        ) = (
            self.video_vace_modulation.to(dtype=t_mod_in_video.dtype, device=t_mod_in_video.device)
            + t_mod_in_video
        ).chunk(
            6, dim=chunk_dim
        )
        (
            action_shift_msa,
            action_scale_msa,
            action_gate_msa,
            action_shift_mlp,
            action_scale_mlp,
            action_gate_mlp,
        ) = (
            self.action_vace_modulation.to(
                dtype=t_mod_in_action.dtype, device=t_mod_in_action.device
            )
            + t_mod_in_action
        ).chunk(
            6, dim=chunk_dim
        )
        if has_seq:
            (
                video_shift_msa,
                video_scale_msa,
                video_gate_msa,
                video_shift_mlp,
                video_scale_mlp,
                video_gate_mlp,
            ) = (
                video_shift_msa.squeeze(2),
                video_scale_msa.squeeze(2),
                video_gate_msa.squeeze(2),
                video_shift_mlp.squeeze(2),
                video_scale_mlp.squeeze(2),
                video_gate_mlp.squeeze(2),
            )
            (
                action_shift_msa,
                action_scale_msa,
                action_gate_msa,
                action_shift_mlp,
                action_scale_mlp,
                action_gate_mlp,
            ) = (
                action_shift_msa.squeeze(2),
                action_scale_msa.squeeze(2),
                action_gate_msa.squeeze(2),
                action_shift_mlp.squeeze(2),
                action_scale_mlp.squeeze(2),
                action_gate_mlp.squeeze(2),
            )
        input_video_cond = modulate(
            self.video_vace_norm1(video_cond), video_shift_msa, video_scale_msa
        )
        video_cond = self.video_vace_gate(
            video_cond, video_gate_msa, self.video_vace_self_attn(input_video_cond, video_freqs)
        )
        video_cond = video_cond + self.video_vace_cross_attn(
            self.video_vace_norm3(video_cond), context
        )
        input_action_cond = modulate(
            self.action_vace_norm1(action_cond), action_shift_msa, action_scale_msa
        )
        action_cond = self.action_vace_gate(
            action_cond,
            action_gate_msa,
            self.action_vace_self_attn(input_action_cond, action_freqs),
        )
        action_cond = action_cond + self.action_vace_cross_attn(
            self.action_vace_norm3(action_cond), context_in_action
        )
        action2video_cond = self.action2video_proj(action_cond)
        video_cond = video_cond + self.action2video_cross_attn(
            self.action2video_norm(video_cond), action2video_cond
        )
        video2action_cond = self.video2action_proj(video_cond)
        action_cond = action_cond + self.video2action_cross_attn(
            self.video2action_norm(action_cond), video2action_cond
        )
        input_video_cond = modulate(
            self.video_vace_norm2(video_cond), video_shift_mlp, video_scale_mlp
        )
        video_cond = self.video_vace_gate(
            video_cond, video_gate_mlp, self.video_vace_ffn(input_video_cond)
        )
        input_action_cond = modulate(
            self.action_vace_norm2(action_cond), action_shift_mlp, action_scale_mlp
        )
        action_cond = self.action_vace_gate(
            action_cond, action_gate_mlp, self.action_vace_ffn(input_action_cond)
        )
        video_cond_skip = self.video_vace_after_proj(video_cond)
        all_video_cond += [video_cond_skip, video_cond]
        action_cond_skip = self.action_vace_after_proj(action_cond)
        all_action_cond += [action_cond_skip, action_cond]
        video_cond = torch.stack(all_video_cond)
        action_cond = torch.stack(all_action_cond)
        return (video_cond, action_cond)


class DualVideoActionConditionDit(nn.Module):

    def __init__(
        self,
        vac_layers,
        video_cond_in_dim=48,
        video_vace_patch_size=(1, 2, 2),
        video_vace_dim=3072,
        video_vace_num_heads=24,
        video_vace_ffn_dim=14336,
        action_cond_in_dim=138,
        action_vace_dim=1536,
        action_vace_num_heads=12,
        action_vace_ffn_dim=8960,
        eps=1e-06,
    ):
        super().__init__()
        self.vac_layers = vac_layers
        self.vac_layers_mapping = {i: n for n, i in enumerate(self.vac_layers)}
        self.video_cond_in_dim = video_cond_in_dim
        self.action_cond_in_dim = action_cond_in_dim
        self.action_vace_dim = action_vace_dim
        self.video_vace_patch_embedding = nn.Conv3d(
            video_cond_in_dim,
            video_vace_dim,
            kernel_size=video_vace_patch_size,
            stride=video_vace_patch_size,
        )
        self.action_vace_input_embedding = nn.Linear(action_cond_in_dim, action_vace_dim)
        self.context_in_action_proj = nn.Linear(video_vace_dim, action_vace_dim)
        self.dual_vac_blocks = nn.ModuleList(
            [
                DualVACDitBlock(
                    video_vace_dim,
                    video_vace_num_heads,
                    video_vace_ffn_dim,
                    action_vace_dim,
                    action_vace_num_heads,
                    action_vace_ffn_dim,
                    eps,
                    block_id=i,
                    last_block_id=self.vac_layers[-1],
                )
                for i in self.vac_layers
            ]
        )
        self.video_uncond = nn.Parameter(torch.randn(48, 1, 30, 52))
        self.action_uncond_v = nn.Parameter(torch.randn(48, 1, 30, 52))
        self.action_uncond = nn.Parameter(torch.randn(1, 138))

    def forward(
        self,
        video_cond,
        x_video,
        video_freqs,
        context_in_video,
        t_mod_in_video,
        action_cond,
        x_action,
        action_freqs,
        context_in_action,
        t_mod_in_action,
    ):
        video_cond = [self.video_vace_patch_embedding(u.unsqueeze(0)) for u in video_cond]
        video_cond = [u.flatten(2).transpose(1, 2) for u in video_cond]
        video_cond = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, x_video.shape[1] - u.size(1), u.size(2))], dim=1)
                for u in video_cond
            ]
        )
        action_cond = self.action_vace_input_embedding(action_cond)
        for block in self.dual_vac_blocks:
            video_cond, action_cond = block(
                video_cond,
                x_video,
                context_in_video,
                t_mod_in_video,
                video_freqs,
                action_cond,
                x_action,
                context_in_action,
                t_mod_in_action,
                action_freqs,
            )
        video_cond_hints = torch.unbind(video_cond)[:-1]
        action_cond_hints = torch.unbind(action_cond)[:-1]
        return (video_cond_hints, action_cond_hints)


dual_vac_config = {
    "vac_layers": (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
    "video_cond_in_dim": 96,
    "video_vace_patch_size": [1, 2, 2],
    "video_vace_dim": 3072,
    "video_vace_num_heads": 24,
    "video_vace_ffn_dim": 14336,
    "action_cond_in_dim": 138,
    "action_vace_dim": 1536,
    "action_vace_num_heads": 12,
    "action_vace_ffn_dim": 8960,
    "eps": 1e-06,
}
