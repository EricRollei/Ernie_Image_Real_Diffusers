"""
Eric ERNIE-Image UltraGen  -  nodes/node_ernie_ultragen.py
==========================================================
Two-stage hierarchical generation for ERNIE-Image.

Stage 1 (Composition): small latent, high CFG, full txt2img via pipeline.
Stage 2 (Detail):      upscaled latent re-noised at s2_denoise level,
                       partial denoise with low CFG via pipeline.

Uses pipeline.__call__() for both stages so the scheduler's trained
shift=4.0 is always applied correctly to timestep embeddings.

Author : Eric Hiss  (GitHub: EricRollei)
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flow-matching noise helper
# ---------------------------------------------------------------------------

def _add_noise_flowmatch(
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Apply flow-matching noising: x_noisy = (1 - σ) * x + σ * noise."""
    return (1.0 - sigma) * latents + sigma * noise


# ---------------------------------------------------------------------------
# Latent helpers - match ErnieImagePipeline exactly
# ---------------------------------------------------------------------------

def _patchify(x: torch.Tensor) -> torch.Tensor:
    """[B, 32, H, W] → [B, 128, H/2, W/2]"""
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * 4, h // 2, w // 2)


def _unpatchify(x: torch.Tensor) -> torch.Tensor:
    """[B, 128, H/2, W/2] → [B, 32, H, W]"""
    b, c, h, w = x.shape
    x = x.reshape(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c // 4, h * 2, w * 2)





# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _decode(pipe, latents: torch.Tensor, device: torch.device) -> Image.Image:
    """Apply BN unnorm, unpatchify, VAE decode - matches pipeline exactly."""
    with torch.no_grad():
        bn_mean   = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(device)
        bn_std    = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + 1e-5).to(device)
        unnormed  = latents.float() * bn_std.float() + bn_mean.float()
        raw       = _unpatchify(unnormed)
        images    = pipe.vae.decode(raw, return_dict=False)[0]
        images    = (images.clamp(-1, 1) + 1) / 2
        images    = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return Image.fromarray((images[0] * 255).astype("uint8"))


def _pil_to_comfy(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# Dimension helpers
# ---------------------------------------------------------------------------

_LATENT_ALIGN = 16


def _dims_at_mp(resolution: str, target_mp: float, presets: dict) -> tuple[int, int]:
    p = presets.get(resolution)
    if p is None:
        side = max(_LATENT_ALIGN,
                   round(math.sqrt(target_mp * 1_048_576) / _LATENT_ALIGN) * _LATENT_ALIGN)
        return side, side
    pw, ph = p
    px     = target_mp * 1_048_576
    h      = math.sqrt(px / (pw / ph))
    w      = h * (pw / ph)
    return (max(_LATENT_ALIGN, round(w / _LATENT_ALIGN) * _LATENT_ALIGN),
            max(_LATENT_ALIGN, round(h / _LATENT_ALIGN) * _LATENT_ALIGN))


def _scale_dims(w: int, h: int, scale: float, max_mp: float) -> tuple[int, int]:
    nw, nh = w * scale, h * scale
    cap    = max_mp * 1_048_576
    if nw * nh > cap:
        s = math.sqrt(cap / (nw * nh))
        nw, nh = nw * s, nh * s
    return (max(_LATENT_ALIGN, round(nw / _LATENT_ALIGN) * _LATENT_ALIGN),
            max(_LATENT_ALIGN, round(nh / _LATENT_ALIGN) * _LATENT_ALIGN))


# ---------------------------------------------------------------------------
# ErnieImageUltraGen node
# ---------------------------------------------------------------------------

class ErnieImageUltraGen:
    """
    Two-stage hierarchical generation for ERNIE-Image.

    Outputs both a Stage 1 preview (composition at small resolution) and
    the final Stage 2 image (detail at full resolution).  The preview is
    useful for tuning Stage 1 settings before committing to Stage 2.
    """

    @classmethod
    def INPUT_TYPES(cls):
        try:
            from .node_ernie_image import RESOLUTION_PRESETS, SIGMA_SCHEDULES
        except ImportError:
            RESOLUTION_PRESETS = {"1:1 square ★": (1024, 1024), "custom": None}
            SIGMA_SCHEDULES    = ["uniform", "fixed shift", "dynamic (resolution-based)"]

        presets = list(RESOLUTION_PRESETS.keys())
        return {
            "required": {
                "pipeline":   ("ERNIE_PIPELINE",),
                "prompt":     ("STRING", {"multiline": True, "default": "",
                                          "tooltip": "Ignored when prompt_embeds connected."}),
                "resolution": (presets, {"default": "5:4",
                                         "tooltip": "Aspect ratio for both stages."}),
                "seed":       ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "use_pe":     ("BOOLEAN", {"default": False}),

                # ---- Stage 1 ----------------------------------------- #
                "s1_initial_mp": ("FLOAT", {
                    "default": 0.40, "min": 0.2, "max": 1.0, "step": 0.05,
                    "tooltip": "Stage 1 megapixels. 0.3-0.5 MP recommended.",
                }),
                "s1_denoise": ("FLOAT", {
                    "default": 1.00, "min": 0.05, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Stage 1 denoise strength - fraction of σ=[1.0→0] covered.  "
                        "0.15 → stops at σ=0.85 (just lays in composition)."
                    ),
                }),
                "s1_steps": ("INT", {
                    "default": 6, "min": 5, "max": 100,
                    "tooltip": "Stage 1 steps. 6-8 for Turbo, 20-30 for SFT.",
                }),
                "s1_guidance_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Stage 1 CFG. 1.0 for Turbo, 6-9 for SFT.",
                }),
                "s1_sigma_schedule": (SIGMA_SCHEDULES, {
                    "default": "fixed shift",
                    "tooltip": (
                        "'fixed shift' with s1_shift_value 6-8 keeps all steps structural. "
                        "'karras'/'beta' add detail-range emphasis on top of shift."
                    ),
                }),
                "s1_shift_value": ("FLOAT", {
                    "default": 6.0, "min": 0.1, "max": 16.0, "step": 0.1,
                    "tooltip": "Stage 1 shift. 6-8 recommended. 1.0 = uniform.",
                }),

                # ---- Stage 2 ----------------------------------------- #
                "scale_factor": ("FLOAT", {
                    "default": 2.5, "min": 1.0, "max": 8.0, "step": 0.5,
                    "tooltip": "Linear upscale factor. 2.5× on 0.4 MP ≈ 1 MP. Higher risks body horror.",
                }),
                "s2_denoise": ("FLOAT", {
                    "default": 0.85, "min": 0.10, "max": 1.00, "step": 0.01,
                    "tooltip": (
                        "Noise level added to upscaled S1 latent before S2 denoise.  "
                        "0.45 = 45% noise (light refinement, preserves S1 composition).  "
                        "0.65+ = heavier re-generation with more new detail.  "
                        "1.0 = full re-generation (ignores S1)."
                    ),
                }),
                "s2_steps": ("INT", {
                    "default": 9, "min": 5, "max": 100,
                    "tooltip": "Stage 2 detail steps. 6-10 for Turbo, 20+ for SFT.",
                }),
                "s2_guidance_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Stage 2 CFG. 1.0 for Turbo, 2-3 for SFT.",
                }),
                "s2_sigma_schedule": (SIGMA_SCHEDULES, {
                    "default": "fixed shift",
                    "tooltip": (
                        "'fixed shift' with s2_shift_value for consistent noise schedule. "
                        "'beta' allocates more steps to fine-detail range. "
                        "'karras' emphasizes both extremes."
                    ),
                }),
                "s2_shift_value": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Stage 2 shift. 5.0 tested well with Turbo at 2.5× scale.",
                }),
                "max_final_mp": ("FLOAT", {
                    "default": 8.0, "min": 0.5, "max": 16.0, "step": 0.5,
                    "tooltip": "Megapixel cap on final output.",
                }),
                "upscale_method": (["bislerp", "bicubic", "bilinear", "nearest", "area"], {
                    "default": "bislerp",
                    "tooltip": (
                        "bislerp (slerp-interpolation) preserves latent vector norms "
                        "and angular relationships — sharper and more coherent than bicubic."
                    ),
                }),
                "guidance_rescale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Rescale CFG output toward unit variance. "
                        "Reduces saturation at high guidance_scale. "
                        "0.0 = off. 0.5-0.7 recommended for S1 cfg 6-9. "
                        "Applied to both stages."
                    ),
                }),
                "vae_decode_fp32": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Upcast VAE to float32 for decode. "
                        "Prevents bf16/fp16 quantization of fine textures. "
                        "Applies to both S1 preview and S2 final decode."
                    ),
                }),
            },
            "optional": {
                "s1_negative_prompt": ("STRING", {"multiline": True, "default": "",
                                                   "tooltip": "Stage 1 negative prompt."}),
                "s2_negative_prompt": ("STRING", {"multiline": True, "default": "",
                                                   "tooltip": "Stage 2 negative. Leave empty to reuse Stage 1."}),
                "prompt_embeds":       ("ERNIE_EMBEDS", {"tooltip": "Pre-computed positive embeddings."}),
                "s1_negative_embeds":  ("ERNIE_EMBEDS", {"tooltip": "Pre-computed Stage 1 negative embeddings."}),
                "s2_negative_embeds":  ("ERNIE_EMBEDS", {"tooltip": "Pre-computed Stage 2 negative embeddings."}),
            },
        }

    # Two outputs: stage1 preview + final image
    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("final_image", "stage1_preview")
    FUNCTION      = "generate"
    CATEGORY      = "Eric ERNIE Image"
    DESCRIPTION   = (
        "Two-stage hierarchical generation. "
        "Returns final image + Stage 1 composition preview."
    )

    def generate(
        self,
        pipeline,
        prompt: str,
        resolution: str,
        seed: int,
        use_pe: bool,
        s1_initial_mp: float,
        s1_denoise: float,
        s1_steps: int,
        s1_guidance_scale: float,
        s1_sigma_schedule: str,
        s1_shift_value: float,
        scale_factor: float,
        s2_denoise: float,
        s2_steps: int,
        s2_guidance_scale: float,
        s2_sigma_schedule: str,
        s2_shift_value: float,
        max_final_mp: float,
        upscale_method: str,
        guidance_rescale: float,
        vae_decode_fp32: bool,
        s1_negative_prompt: str = "",
        s2_negative_prompt:  str = "",
        prompt_embeds=None,
        s1_negative_embeds=None,
        s2_negative_embeds=None,
    ) -> tuple:

        from .node_ernie_image import (
            RESOLUTION_PRESETS, _apply_sigma_schedule_to_call,
            _enable_vae_tiling, _disable_vae_tiling, _patch_cosine_blend,
        )

        device = next(pipeline.transformer.parameters()).device
        dtype  = pipeline.transformer.dtype

        # PE fallback
        if use_pe and not getattr(pipeline, "_ernie_pe_loaded", False):
            logger.warning("ERNIE UltraGen: use_pe=True but PE not loaded.")
            use_pe = False

        # ---- Dimensions ----------------------------------------------- #
        s1_w, s1_h = _dims_at_mp(resolution, s1_initial_mp, RESOLUTION_PRESETS)
        s2_w, s2_h = _scale_dims(s1_w, s1_h, scale_factor, max_final_mp)

        logger.info(
            f"ERNIE UltraGen | "
            f"S1={s1_w}×{s1_h} ({s1_w*s1_h/1e6:.2f}MP) → "
            f"S2={s2_w}×{s2_h} ({s2_w*s2_h/1e6:.2f}MP)\n"
            f"  S1: steps={s1_steps} cfg={s1_guidance_scale} "
            f"schedule={s1_sigma_schedule}(shift={s1_shift_value})\n"
            f"  S2: denoise={s2_denoise:.2f} steps={s2_steps} cfg={s2_guidance_scale} "
            f"schedule={s2_sigma_schedule}(shift={s2_shift_value})\n"
            f"  guidance_rescale={guidance_rescale} vae_fp32={vae_decode_fp32}"
        )

        # ---- VAE float32 upcast for decode quality -------------------- #
        vae = pipeline.vae
        vae_orig_dtype = vae.dtype
        if vae_decode_fp32 and vae_orig_dtype != torch.float32:
            vae.to(torch.float32)
            logger.debug("ERNIE UltraGen: VAE upcast to float32 for decode.")

        # ---- Guidance rescale (reduce CFG saturation) ----------------- #
        _orig_sched_step = None
        if guidance_rescale > 0:
            _orig_sched_step = pipeline.scheduler.step
            _gr = guidance_rescale
            def _rescaled_step(model_output, timestep, sample, **kwargs):
                dims = list(range(1, model_output.ndim))
                std = model_output.std(dim=dims, keepdim=True).clamp(min=1e-6)
                rescaled = model_output / std
                model_output = (1.0 - _gr) * model_output + _gr * rescaled
                return _orig_sched_step(model_output, timestep, sample, **kwargs)
            pipeline.scheduler.step = _rescaled_step

        # ---- VAE tiling + cosine blend for S2 (high-res) -------------- #
        if s2_w * s2_h > 2 * 1_048_576:
            _enable_vae_tiling(pipeline)
            _patch_cosine_blend(pipeline)
        else:
            _disable_vae_tiling(pipeline)

        # ---- Progress bar --------------------------------------------- #
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(s1_steps + s2_steps)
        except ImportError:
            pbar = None

        def _step_callback(pipe, step_index, timestep, callback_kwargs):
            if pbar is not None:
                pbar.update(1)
            return callback_kwargs

        generator = torch.Generator("cpu").manual_seed(seed)

        # ---- Apply S1 sigma schedule ---------------------------------- #
        _apply_sigma_schedule_to_call(pipeline, s1_sigma_schedule, s1_shift_value)

        # ---- Build S1 call kwargs ------------------------------------- #
        s1_kwargs = dict(
            height=s1_h,
            width=s1_w,
            num_inference_steps=s1_steps,
            guidance_scale=s1_guidance_scale,
            generator=generator,
            use_pe=use_pe,
            output_type="latent",
            callback_on_step_end=_step_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        if prompt_embeds is not None:
            s1_kwargs["prompt_embeds"] = prompt_embeds
        else:
            s1_kwargs["prompt"] = prompt

        if s1_guidance_scale > 1.0:
            if s1_negative_embeds is not None:
                s1_kwargs["negative_prompt_embeds"] = s1_negative_embeds
            elif prompt_embeds is not None:
                s1_kwargs["negative_prompt_embeds"] = pipeline.encode_prompt(
                    s1_negative_prompt or "", device=device, num_images_per_prompt=1
                )
            else:
                s1_kwargs["negative_prompt"] = s1_negative_prompt or ""

        # ---- Stage 1: Composition (full txt2img) ---------------------- #
        logger.info("ERNIE UltraGen: Stage 1 - composition...")
        s1_latents = pipeline(**s1_kwargs)
        # output_type="latent" returns the raw patchified BN-normed tensor
        logger.info(
            f"ERNIE UltraGen: S1 done - "
            f"latent shape={list(s1_latents.shape)} "
            f"mean={s1_latents.mean().item():.4f} std={s1_latents.std().item():.4f}"
        )

        # ---- Stage 1 preview ------------------------------------------ #
        logger.info("ERNIE UltraGen: Decoding Stage 1 preview...")
        s1_preview_pil = _decode(pipeline, s1_latents, device)

        # ---- Remove S1 sigma schedule wrapper before S2 --------------- #
        # _apply_sigma_schedule_to_call may have wrapped pipeline.__call__;
        # clear it so we can set up S2's schedule cleanly.
        if hasattr(pipeline, "__call__") and hasattr(pipeline.__call__, "__wrapped_schedule__"):
            del pipeline.__call__

        # ---- Upscale latents ------------------------------------------ #
        logger.info(f"ERNIE UltraGen: Upscaling {s1_w}×{s1_h} → {s2_w}×{s2_h} via {upscale_method}")
        unpatched = _unpatchify(s1_latents)  # [1, 32, s1_h//8, s1_w//8]

        dst_h_lat = s2_h // 8
        dst_w_lat = s2_w // 8

        if upscale_method == "bislerp":
            import comfy.utils as comfy_utils
            scaled = comfy_utils.bislerp(unpatched.float(), dst_w_lat, dst_h_lat).to(dtype)
        else:
            kw = {"align_corners": False} if upscale_method in ("bicubic", "bilinear") else {}
            scaled = F.interpolate(
                unpatched.float(), size=(dst_h_lat, dst_w_lat),
                mode=upscale_method, **kw
            ).to(dtype)

        s2_clean = _patchify(scaled)  # [1, 128, s2_h//16, s2_w//16]

        # ---- Configure S2 scheduler BEFORE noise addition ------------- #
        # Must set shift/karras/beta first so the dry-run set_timesteps
        # produces the real sigma schedule; we read sigmas[0] for noising.
        _orig_shift  = pipeline.scheduler.shift
        _orig_karras = pipeline.scheduler.config.use_karras_sigmas
        _orig_beta   = pipeline.scheduler.config.use_beta_sigmas

        if s2_sigma_schedule in ("fixed shift", "karras", "beta"):
            pipeline.scheduler._shift = s2_shift_value
        if s2_sigma_schedule == "karras":
            pipeline.scheduler.config.use_karras_sigmas = True
        elif s2_sigma_schedule == "beta":
            pipeline.scheduler.config.use_beta_sigmas = True

        # ---- Build S2 partial sigma schedule (raw, pre-shift) --------- #
        # N raw sigmas from s2_denoise → σ_min.  set_timesteps applies
        # the (now-configured) shift and optionally karras/beta transform,
        # then appends terminal 0.
        sigma_min = 1.0 / s2_steps
        raw_sigmas_s2 = torch.linspace(s2_denoise, sigma_min, s2_steps)

        # Dry-run to get the actual processed sigma schedule
        pipeline.scheduler.set_timesteps(sigmas=raw_sigmas_s2, device=device)
        noise_sigma = pipeline.scheduler.sigmas[0].item()

        # Monkey-patch set_timesteps so the pipeline.__call__ reproduces
        # the same schedule when it calls set_timesteps internally
        _orig_set_ts = pipeline.scheduler.set_timesteps
        def _patched_set_ts(sigmas=None, device=None, **kw):
            return _orig_set_ts(sigmas=raw_sigmas_s2, device=device, **kw)

        # ---- Add noise for Stage 2 ----------------------------------- #
        # noise_sigma is the ACTUAL first sigma after shift+karras/beta,
        # so the noise level exactly matches the scheduler's expectation.
        noise = torch.randn(
            s2_clean.shape, generator=generator, dtype=dtype, device="cpu"
        ).to(device)
        s2_noised = _add_noise_flowmatch(s2_clean, noise, noise_sigma)

        logger.info(
            f"ERNIE UltraGen: S2 noise σ={noise_sigma:.4f} "
            f"(raw denoise={s2_denoise:.3f}, "
            f"effective shift={pipeline.scheduler.shift})\n"
            f"  clean  mean={s2_clean.mean().item():.4f} "
            f"std={s2_clean.std().item():.4f}\n"
            f"  noised mean={s2_noised.mean().item():.4f} "
            f"std={s2_noised.std().item():.4f}"
        )

        # ---- Build S2 call kwargs ------------------------------------- #
        s2_kwargs = dict(
            height=s2_h,
            width=s2_w,
            num_inference_steps=s2_steps,
            guidance_scale=s2_guidance_scale,
            latents=s2_noised,
            use_pe=use_pe,
            output_type="pil",
            callback_on_step_end=_step_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        if prompt_embeds is not None:
            s2_kwargs["prompt_embeds"] = prompt_embeds
        else:
            s2_kwargs["prompt"] = prompt

        if s2_guidance_scale > 1.0:
            if s2_negative_embeds is not None:
                s2_kwargs["negative_prompt_embeds"] = s2_negative_embeds
            elif prompt_embeds is not None:
                if s2_negative_prompt and s2_negative_prompt != s1_negative_prompt:
                    s2_kwargs["negative_prompt_embeds"] = pipeline.encode_prompt(
                        s2_negative_prompt, device=device, num_images_per_prompt=1
                    )
                elif s1_negative_embeds is not None:
                    s2_kwargs["negative_prompt_embeds"] = s1_negative_embeds
                else:
                    s2_kwargs["negative_prompt_embeds"] = pipeline.encode_prompt(
                        s1_negative_prompt or "", device=device, num_images_per_prompt=1
                    )
            else:
                s2_kwargs["negative_prompt"] = s2_negative_prompt or s1_negative_prompt or ""

        # ---- Stage 2: Detail (partial denoise) ------------------------ #
        logger.info("ERNIE UltraGen: Stage 2 - detail...")
        pipeline.scheduler.set_timesteps = _patched_set_ts
        try:
            s2_result = pipeline(**s2_kwargs)
        finally:
            pipeline.scheduler.set_timesteps = _orig_set_ts
            pipeline.scheduler._shift = _orig_shift
            pipeline.scheduler.config.use_karras_sigmas = _orig_karras
            pipeline.scheduler.config.use_beta_sigmas   = _orig_beta
            # Restore S1 schedule wrapper cleanup
            if hasattr(pipeline, "__call__") and hasattr(pipeline.__call__, "__wrapped_schedule__"):
                del pipeline.__call__
            # Restore VAE dtype
            if vae_decode_fp32 and vae_orig_dtype != torch.float32:
                vae.to(vae_orig_dtype)
            # Restore scheduler.step
            if _orig_sched_step is not None:
                pipeline.scheduler.step = _orig_sched_step

        final_pil = s2_result.images[0]
        logger.info(
            f"ERNIE UltraGen: Done - "
            f"{final_pil.size[0]}×{final_pil.size[1]} "
            f"({final_pil.size[0]*final_pil.size[1]/1e6:.2f} MP)"
        )

        try:
            pipeline.maybe_free_model_hooks()
        except Exception:
            pass

        return (_pil_to_comfy(final_pil), _pil_to_comfy(s1_preview_pil))
