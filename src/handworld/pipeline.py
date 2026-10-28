"""Stage-2 AC and stage-3 joint denoising; original architecture and inference math."""

from typing import Optional
from typing_extensions import Literal
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from diffsynth.utils import PipelineUnit
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, TeaCache, TemporalTiler_BCTHW
from .models.video_dit import VideoDit
from .models.action_dit import ActionDit
from .models.dual_vac_dit import DualVideoActionConditionDit


class HandWorldPipeline(WanVideoPipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        self.dit: VideoDit = None
        self.adit: ActionDit = None
        self.vac: DualVideoActionConditionDit = None
        self.in_iteration_models = ["adit", "dit", "vac"]
        self.units[4] = HandWorldUnit_VideoEmbedder()
        self.units += [HandWorldUnit_VideoConditionEmbedder(), HandWorldUnit_ActionEmbedder()]
        self.model_fn = model_fn_hand_world

    @torch.no_grad()
    def infer_stage2(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        input_image: Optional[Image.Image] = None,
        end_image: Optional[Image.Image] = None,
        input_video: Optional[list[Image.Image]] = None,
        input_action_v: Optional[list[Image.Image]] = None,
        input_action: Optional[torch.tensor] = None,
        denoising_strength: Optional[float] = 1.0,
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        camera_control_direction: Optional[
            Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]
        ] = None,
        camera_control_speed: Optional[float] = 1 / 54,
        camera_control_origin: Optional[tuple] = (
            0,
            0.532139961,
            0.946026558,
            0.5,
            0.5,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
        ),
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        switch_DiT_boundary: Optional[float] = 0.875,
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        motion_bucket_id: Optional[int] = None,
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        progress_bar_cmd=tqdm,
        load_model=True,
        joint_denoising=False,
    ):
        self.scheduler.set_timesteps(
            num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift
        )
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video,
            "denoising_strength": denoising_strength,
            "input_action_v": input_action_v,
            "input_action": input_action,
            "control_video": control_video,
            "reference_image": reference_image,
            "camera_control_direction": camera_control_direction,
            "camera_control_speed": camera_control_speed,
            "camera_control_origin": camera_control_origin,
            "vace_video": vace_video,
            "vace_video_mask": vace_video_mask,
            "vace_reference_image": vace_reference_image,
            "vace_scale": vace_scale,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio,
            "audio_sample_rate": audio_sample_rate,
            "s2v_pose_video": s2v_pose_video,
            "audio_embeds": audio_embeds,
            "s2v_pose_latents": s2v_pose_latents,
            "motion_video": motion_video,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )
        video_cond = inputs_shared["video_cond"]
        action_cond_v = inputs_shared["action_cond_v"]
        action_cond = inputs_shared["action_cond"]
        video_uncond = (
            self.vac.video_uncond[None]
            .expand(video_cond.shape)
            .to(dtype=action_cond.dtype, device=action_cond.device)
        )
        action_uncond_v = (
            self.vac.action_uncond_v[None]
            .expand(action_cond_v.shape)
            .to(dtype=action_cond.dtype, device=action_cond.device)
        )
        action_uncond = (
            self.vac.action_uncond[None]
            .expand(action_cond.shape)
            .to(dtype=action_cond.dtype, device=action_cond.device)
        )
        video_mask = torch.ones_like(video_cond).to(
            dtype=action_cond.dtype, device=action_cond.device
        )
        action_v_mask = torch.ones_like(action_cond_v).to(
            dtype=action_cond.dtype, device=action_cond.device
        )
        action_mask = torch.ones_like(action_cond).to(
            dtype=action_cond.dtype, device=action_cond.device
        )
        video_history = (num_frames - 1) // 4 // 4 + 1
        action_history = (video_history - 1) * 4 + 1
        if joint_denoising:
            video_mask[:, :, video_history:] = 0
            action_v_mask[:, :, video_history:] = 0
            action_mask[:, action_history:] = 0
        else:
            video_mask[:, :, 1:] = 0
        video_cond = video_mask * video_cond + (1 - video_mask) * video_uncond
        action_cond_v = action_v_mask * action_cond_v + (1 - action_v_mask) * action_uncond_v
        action_cond = action_mask * action_cond + (1 - action_mask) * action_uncond
        num_frames = inputs_shared["num_frames"]
        video_latent_frame_num = (num_frames - 1) // 4 + 1
        if video_cond.shape[2] < video_latent_frame_num:
            pad_shape = (
                *video_cond.shape[:2],
                video_latent_frame_num - video_cond.shape[2],
                *video_cond.shape[3:],
            )
            video_cond_pad = (
                self.vac.video_uncond[None]
                .expand(pad_shape)
                .to(dtype=video_cond.dtype, device=video_cond.device)
            )
            video_cond = torch.cat([video_cond, video_cond_pad], dim=2)
        inputs_shared["video_cond"] = torch.cat([video_cond, action_cond_v], dim=1)
        inputs_shared["action_cond"] = action_cond
        action_noise = inputs_shared["action_latents"].clone()
        video_noise = inputs_shared["noise"].clone()
        if load_model:
            self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=action_cond.dtype, device=action_cond.device)
            if joint_denoising:
                noisy_video = self.scheduler.add_noise(
                    inputs_shared["input_latents"], video_noise, timestep
                )
                noisy_action = self.scheduler.add_noise(
                    inputs_shared["first_frame_action_latents"], action_noise, timestep
                )
                inputs_shared["latents"][:, :, :video_history] = noisy_video[:, :, :video_history]
                inputs_shared["action_latents"][:, :action_history] = noisy_action[
                    :, :action_history
                ]
            model_stage = "3" if joint_denoising else "2"
            noise_pred_posi_video, noise_pred_posi_action = self.model_fn(
                **models, **inputs_shared, **inputs_posi, timestep=timestep, stage=model_stage
            )
            if cfg_scale != 1.0:
                noise_pred_nega_video, noise_pred_nega_action = self.model_fn(
                    **models, **inputs_shared, **inputs_nega, timestep=timestep, stage=model_stage
                )
                noise_pred_video = noise_pred_nega_video + cfg_scale * (
                    noise_pred_posi_video - noise_pred_nega_video
                )
            else:
                noise_pred_video = noise_pred_posi_video
            if joint_denoising:
                noise_pred_action = (
                    noise_pred_nega_action
                    + cfg_scale * (noise_pred_posi_action - noise_pred_nega_action)
                    if cfg_scale != 1.0
                    else noise_pred_posi_action
                )
                inputs_shared["action_latents"] = self.scheduler.step(
                    noise_pred_action,
                    self.scheduler.timesteps[progress_id],
                    inputs_shared["action_latents"],
                )
                inputs_shared["action_latents"][:, :action_history] = inputs_shared[
                    "first_frame_action_latents"
                ][:, :action_history]
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred_video, self.scheduler.timesteps[progress_id], inputs_shared["latents"]
            )
            if joint_denoising:
                inputs_shared["latents"][:, :, :video_history] = inputs_shared["input_latents"][
                    :, :, :video_history
                ]
            elif "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, :1] = inputs_shared["first_frame_latents"]
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )
        if load_model:
            self.load_models_to_device(["vae"])
        video = self.vae.decode(
            inputs_shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video = self.vae_output_to_video(video)
        return (video, inputs_shared["action_latents"]) if joint_denoising else video

    @torch.no_grad()
    def infer_stage3(self, **kwargs):
        """Joint video/action forecast, with the first quarter observed.

        Return (video_frames, action_tensor). Conditions and observed-branch
        noise levels match the recovered stage-3 observed-prefix schedule. No future labels are used.
        """
        return self.infer_stage2(**kwargs, joint_denoising=True)


def model_fn_hand_world(
    stage: int,
    dit: VideoDit,
    adit: ActionDit,
    vac: DualVideoActionConditionDit,
    latents: torch.Tensor,
    action_latents: torch.Tensor,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    video_cond: torch.Tensor = None,
    action_cond: torch.Tensor = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            video_cond=video_cond,
            action_cond=action_cond,
            y=y,
            tea_cache=tea_cache,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            sliding_window_size,
            sliding_window_stride,
            latents.device,
            latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1,
        )
    "\n    latents:\n    - i2v_a14b: torch.Size([1, 16, 13, 60, 104])\n    - ti2v_5b:  torch.Size([1, 48, 13, 30, 52])\n    y / dit.require_vae_embedding:\n    - i2v_a14b: torch.Size([1, 20, 13, 60, 104]) / True\n    - ti2v_5b:  None / False\n    "
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep_in_video = torch.concat(
            [
                torch.zeros(
                    (1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t_in_video = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep_in_video).unsqueeze(0)
        )
        t_mod_in_video = dit.time_projection(t_in_video).unflatten(2, (6, dit.dim))
        timestep_in_action = torch.concat(
            [
                torch.zeros(1, dtype=action_cond.dtype, device=action_cond.device),
                torch.ones(
                    action_cond.shape[1] - 1, dtype=action_cond.dtype, device=action_cond.device
                )
                * timestep,
            ]
        ).flatten()
        t_in_action = adit.time_embedding(
            sinusoidal_embedding_1d(adit.freq_dim, timestep_in_action).unsqueeze(0)
        )
        t_mod_in_action = adit.time_projection(t_in_action).unflatten(2, (6, adit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod_in_video = dit.time_projection(t).unflatten(1, (6, dit.dim))
        t_mod_in_action = adit.time_projection(t).unflatten(1, (6, adit.dim))
    context_in_video = dit.text_embedding(context)
    context_in_action = vac.context_in_action_proj(context_in_video)
    x_video = latents
    x_action = action_latents
    if x_video.shape[0] != context_in_video.shape[0]:
        x_video = torch.concat([x_video] * context_in_video.shape[0], dim=0)
    if timestep.shape[0] != context_in_video.shape[0]:
        timestep = torch.concat([timestep] * context_in_video.shape[0], dim=0)
    if y is not None and dit.require_vae_embedding:
        x_video = torch.cat([x_video, y], dim=1)
    x_video, (f, h, w) = dit.patchify(x_video)
    x_action = adit.input_embedding(x_action)
    video_freqs = (
        torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(video_cond.device)
    )
    action_freqs = (
        adit.freqs[: action_cond.shape[1]]
        .reshape(action_cond.shape[1], 1, -1)
        .to(action_cond.device)
    )
    video_hints, action_hints = vac(
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
    )
    if stage in ["2", "3"]:
        for block_id, block in enumerate(dit.blocks):
            if block_id in vac.vac_layers_mapping:
                current_video_hint = video_hints[vac.vac_layers_mapping[block_id]]
            if block_id in vac.vac_layers_mapping:
                x_video = block(
                    x_video, current_video_hint, context_in_video, t_mod_in_video, video_freqs
                )
            else:
                x_video = block(x_video, context_in_video, t_mod_in_video, video_freqs)
        x_video = dit.head(x_video, t_in_video)
        x_video = dit.unpatchify(x_video, (f, h, w))
    if stage == "3":
        for block_id, block in enumerate(adit.blocks):
            if block_id in vac.vac_layers_mapping:
                current_action_hint = action_hints[vac.vac_layers_mapping[block_id]]
            if block_id in vac.vac_layers_mapping:
                x_action = block(
                    x_action, current_action_hint, context_in_action, t_mod_in_action, action_freqs
                )
            else:
                x_action = block(x_action, context_in_action, t_mod_in_action, action_freqs)
        x_action = adit.head(x_action)
    return (x_video, x_action)


class HandWorldUnit_VideoEmbedder(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "noise",
                "num_frames",
                "tiled",
                "tile_size",
                "tile_stride",
                "vace_reference_image",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_video,
        noise,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
        vace_reference_image,
    ):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)[:, :, :num_frames]
        input_latents = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            vace_reference_image = pipe.preprocess_video([vace_reference_image])
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        return {"latents": noise, "input_latents": input_latents}


class HandWorldUnit_VideoConditionEmbedder(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=(
                "input_latents",
                "input_action_v",
                "num_frames",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_latents,
        input_action_v,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        pipe.load_models_to_device(self.onload_model_names)
        action_cond_v = pipe.preprocess_video(input_action_v)[:, :, :num_frames]
        action_cond_v = pipe.vae.encode(
            action_cond_v,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        action_cond_v = action_cond_v.to(dtype=pipe.torch_dtype, device=pipe.device)
        video_cond = input_latents.clone().detach()
        return {"video_cond": video_cond, "action_cond_v": action_cond_v}


class HandWorldUnit_ActionEmbedder(PipelineUnit):

    def __init__(self):
        super().__init__(input_params=("input_action", "num_frames", "seed", "rand_device"))

    def process(self, pipe: WanVideoPipeline, input_action, num_frames, seed, rand_device):
        input_action = input_action[:, :num_frames].to(dtype=pipe.torch_dtype, device=pipe.device)
        noise = pipe.generate_noise(input_action.shape, seed=seed, rand_device=rand_device)
        z = input_action.clone().detach().to(dtype=pipe.torch_dtype, device=pipe.device)
        action_cond = input_action.clone().detach().to(dtype=pipe.torch_dtype, device=pipe.device)
        return {
            "action_latents": noise,
            "action_cond": action_cond,
            "first_frame_action_latents": z,
        }
