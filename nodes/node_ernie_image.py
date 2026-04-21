"""
Eric ERNIE-Image Nodes  -  nodes/node_ernie_image.py
=====================================================
ComfyUI custom nodes for Baidu ERNIE-Image (SFT) and ERNIE-Image-Turbo generation.

Architecture summary
--------------------
- Transformer  : 8B single-stream DiT (36 layers, in_channels=128 after 2x2 patchify)
- Text encoder : Ministral 3B  (model_type='ministral3', flux2-style)
- PE model     : separate 3B generative LLM  (requires transformers 5.x to load)
- VAE          : AutoencoderKLFlux2  (32 latent channels, patch_size 2x2)
- Latent scale : 16x  (8x VAE x 2x patch)  -> dimensions must be multiples of 16
- Scheduler    : FlowMatchEulerDiscreteScheduler

Sigma schedule notes
--------------------
scheduler_config.json values:
  shift: 4.0              -- fixed shift when set_timesteps(num_inference_steps=N) called
  base_shift: 0.5  }      -- used only when use_dynamic_shifting=True (currently False)
  max_shift: 1.15  }
  use_dynamic_shifting: false

Pipeline default: passes sigmas=linspace(1,0,N+1) directly, bypassing all shift.
At 1024px with dynamic shift, mu would compute to ~1.15 (nearly uniform).
shift=4.0 is a strong deviation -- try "uniform" first.

sigma_schedule is exposed on ErnieImageGenerate (not the loader) so you can
A/B test different schedules without reloading the model.

Compatibility patches applied
------------------------------
1. TokenizersBackend -> PreTrainedTokenizerFast
2. MistralAttention projections at head_dim=128 (fixes all 104 attention weights)
3. model.rotary_emb.inv_freq [48]->[64] for head_dim=128

encode_prompt interface (confirmed by inspection)
-------------------------------------------------
pipeline.encode_prompt() returns List[Tensor] where each tensor is
outputs.hidden_states[-2][0] -- shape [T, 3072].
ErnieImageGenerate accepts pre-computed prompt_embeds to bypass this,
enabling embedding reuse and external embedding manipulation.

Author : Eric Hiss  (GitHub: EricRollei)
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch 1: TokenizersBackend + ministral3 CONFIG_MAPPING
# ---------------------------------------------------------------------------

def _patch_transformers_compat() -> None:
    import transformers

    if not hasattr(transformers, "TokenizersBackend"):
        from transformers import PreTrainedTokenizerFast as _TBClass
        try:
            transformers.TokenizersBackend = _TBClass
        except Exception:
            pass
        try:
            sys.modules["transformers"].__dict__["TokenizersBackend"] = _TBClass
        except Exception:
            pass
        _lazy_cls = type(sys.modules["transformers"])
        _orig_ga = _lazy_cls.__getattr__

        def _patched_getattr(self, name: str):
            if name == "TokenizersBackend":
                try:
                    object.__setattr__(self, "TokenizersBackend", _TBClass)
                except Exception:
                    pass
                return _TBClass
            return _orig_ga(self, name)

        try:
            _lazy_cls.__getattr__ = _patched_getattr
        except Exception:
            pass

    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        if "ministral3" not in CONFIG_MAPPING:
            try:
                config_cls = CONFIG_MAPPING["mistral3"]
            except (KeyError, Exception):
                from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
                config_cls = Mistral3Config
            CONFIG_MAPPING.register("ministral3", config_cls)
            try:
                from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
                if "ministral3" not in MODEL_FOR_CAUSAL_LM_MAPPING:
                    MODEL_FOR_CAUSAL_LM_MAPPING.register(
                        "ministral3", MODEL_FOR_CAUSAL_LM_MAPPING["mistral3"]
                    )
            except Exception:
                pass
    except Exception as exc:
        logger.warning(f"ERNIE-Image: ministral3 CONFIG_MAPPING patch failed - {exc}")


# ---------------------------------------------------------------------------
# Patch 2: MistralAttention projection shapes (head_dim=128)
# ---------------------------------------------------------------------------

def _patch_mistral_for_explicit_head_dim() -> bool:
    try:
        import transformers.models.mistral.modeling_mistral as _mm

        if getattr(_mm, "_ernie_head_dim_patched", False):
            return True

        _orig_init = _mm.MistralAttention.__init__

        def _patched_init(self, config, layer_idx=None):
            _orig_init(self, config, layer_idx)
            explicit_hd = getattr(config, "_explicit_head_dim", None)
            if explicit_hd is None or explicit_hd == self.head_dim:
                return
            import torch.nn as nn
            self.head_dim = explicit_hd
            hs  = config.hidden_size
            nh  = config.num_attention_heads
            nkv = config.num_key_value_heads
            self.q_proj = nn.Linear(hs, nh  * explicit_hd, bias=False)
            self.k_proj = nn.Linear(hs, nkv * explicit_hd, bias=False)
            self.v_proj = nn.Linear(hs, nkv * explicit_hd, bias=False)
            self.o_proj = nn.Linear(nh  * explicit_hd, hs,  bias=False)

        _mm.MistralAttention.__init__ = _patched_init
        _mm._ernie_head_dim_patched = True
        logger.info("ERNIE-Image: MistralAttention patched for head_dim=128.")
        return True

    except Exception as exc:
        logger.warning(f"ERNIE-Image: MistralAttention head_dim patch failed - {exc}")
        return False


# ---------------------------------------------------------------------------
# Sigma schedule patches
# ---------------------------------------------------------------------------

SIGMA_SCHEDULES = [
    "uniform",
    "fixed shift",
    "karras",
    "beta",
    "dynamic (resolution-based)",
]


def _apply_sigma_schedule_to_call(pipe, schedule: str, shift_value: float = 4.0):
    """
    Wraps pipe.__call__ to apply the selected sigma schedule.

    Schedules:
      uniform                 -- pipeline default (shift=4.0 from scheduler config)
      fixed shift             -- override scheduler.shift to shift_value
      karras                  -- shift override + karras sigma transform
      beta                    -- shift override + beta sigma transform
      dynamic (resolution-based) -- resolution-dependent shift (mu from image size)

    Called from ErnieImageGenerate.generate() on each run, so the user can
    switch schedules without reloading the model.
    """
    import types

    _pipeline_cls = type(pipe)
    _orig_call    = _pipeline_cls.__call__

    if schedule == "uniform":
        if hasattr(pipe, "__call__") and hasattr(pipe.__call__, "__wrapped_schedule__"):
            del pipe.__call__
        logger.debug("ERNIE-Image: sigma schedule = uniform (pipeline default).")
        return pipe

    if schedule in ("fixed shift", "karras", "beta"):
        def _wrapped_call(self, *args, **kwargs):
            _orig_shift  = self.scheduler.shift
            _orig_karras = self.scheduler.config.use_karras_sigmas
            _orig_beta   = self.scheduler.config.use_beta_sigmas

            self.scheduler._shift = shift_value
            if schedule == "karras":
                self.scheduler.config.use_karras_sigmas = True
            elif schedule == "beta":
                self.scheduler.config.use_beta_sigmas = True

            try:
                return _orig_call(self, *args, **kwargs)
            finally:
                self.scheduler._shift = _orig_shift
                self.scheduler.config.use_karras_sigmas = _orig_karras
                self.scheduler.config.use_beta_sigmas   = _orig_beta

        _wrapped_call.__wrapped_schedule__ = schedule
        pipe.__call__ = types.MethodType(_wrapped_call, pipe)
        logger.info(
            f"ERNIE-Image: sigma schedule = {schedule!r} shift={shift_value}."
        )
        return pipe

    if schedule == "dynamic (resolution-based)":
        BASE_SHIFT    = 0.5
        MAX_SHIFT     = 1.15
        BASE_TOKENS   = 256
        MAX_TOKENS    = 4096
        LATENT_FACTOR = 16

        def _wrapped_call(self, *args, **kwargs):
            _orig_sts = self.scheduler.set_timesteps

            h = kwargs.get("height", 1024)
            w = kwargs.get("width",  1024)
            n_tokens = (h // LATENT_FACTOR) * (w // LATENT_FACTOR)
            mu = (MAX_SHIFT - BASE_SHIFT) * (n_tokens - BASE_TOKENS) / \
                 (MAX_TOKENS - BASE_TOKENS) + BASE_SHIFT
            mu = max(BASE_SHIFT, min(MAX_SHIFT, mu))

            def _dyn_sts(sigmas=None, device=None, **kw):
                if sigmas is not None:
                    try:
                        return _orig_sts(
                            num_inference_steps=len(sigmas), device=device, mu=mu
                        )
                    except TypeError:
                        return _orig_sts(num_inference_steps=len(sigmas), device=device)
                return _orig_sts(sigmas=sigmas, device=device, **kw)
            self.scheduler.set_timesteps = _dyn_sts

            try:
                return _orig_call(self, *args, **kwargs)
            finally:
                self.scheduler.set_timesteps = _orig_sts

        _wrapped_call.__wrapped_schedule__ = schedule
        pipe.__call__ = types.MethodType(_wrapped_call, pipe)
        logger.info("ERNIE-Image: sigma schedule = dynamic (resolution-based).")
        return pipe

    logger.warning(f"ERNIE-Image: unknown sigma schedule {schedule!r}, using uniform.")
    return pipe
    return pipe


# ---------------------------------------------------------------------------
# Build correct text encoder (Patches 2 + 3)
# ---------------------------------------------------------------------------

def _build_text_encoder(model_path: str, dtype: torch.dtype):
    try:
        from transformers import MistralConfig, MistralModel
    except ImportError:
        logger.warning("ERNIE-Image: MistralModel not available.")
        return None

    text_enc_dir = os.path.join(model_path, "text_encoder")
    cfg_path = os.path.join(text_enc_dir, "config.json")
    sf_path  = os.path.join(text_enc_dir, "model.safetensors")

    if not os.path.exists(cfg_path) or not os.path.exists(sf_path):
        logger.warning("ERNIE-Image: text_encoder files missing.")
        return None

    try:
        with open(cfg_path) as f:
            outer_cfg = json.load(f)
        inner = outer_cfg.get("text_config", {})

        mc_kwargs: dict = dict(
            hidden_size              = inner.get("hidden_size",           3072),
            intermediate_size        = inner.get("intermediate_size",     9216),
            num_hidden_layers        = inner.get("num_hidden_layers",       26),
            num_attention_heads      = inner.get("num_attention_heads",     32),
            num_key_value_heads      = inner.get("num_key_value_heads",      8),
            max_position_embeddings  = inner.get("max_position_embeddings", 262144),
            vocab_size               = inner.get("vocab_size",          131072),
            rms_norm_eps             = inner.get("rms_norm_eps",           1e-5),
            hidden_act               = inner.get("hidden_act",           "silu"),
            tie_word_embeddings      = inner.get("tie_word_embeddings",   True),
        )
        rope_params = inner.get("rope_parameters", {})
        rope_theta  = float(rope_params.get("rope_theta", 1_000_000.0))
        mc_kwargs["rope_theta"] = rope_theta

        mistral_cfg = MistralConfig(**mc_kwargs)
        explicit_hd = inner.get("head_dim", 128)
        mistral_cfg._explicit_head_dim = explicit_hd

        logger.info(f"ERNIE-Image: building text encoder hidden=3072, head_dim={explicit_hd}...")

        text_model = MistralModel(mistral_cfg)

        # Patch 3: fix shared rotary_emb.inv_freq on model
        if hasattr(text_model, "rotary_emb") and hasattr(text_model.rotary_emb, "inv_freq"):
            new_inv_freq = 1.0 / (
                rope_theta ** (
                    torch.arange(0, explicit_hd, 2, dtype=torch.float32) / explicit_hd
                )
            )
            text_model.rotary_emb._buffers["inv_freq"] = new_inv_freq
            for _attr in ("cos_cached", "sin_cached", "_cos_cached", "_sin_cached"):
                if hasattr(text_model.rotary_emb, _attr):
                    setattr(text_model.rotary_emb, _attr, None)
            logger.info(f"ERNIE-Image: rotary inv_freq fixed [48]->[{explicit_hd//2}].")
        else:
            logger.warning("ERNIE-Image: model.rotary_emb not found - rotary fix skipped.")

        from safetensors.torch import load_file as sf_load
        state_dict = sf_load(sf_path)
        STRIP = "language_model.model."
        inner_sd = {
            k[len(STRIP):]: v
            for k, v in state_dict.items()
            if k.startswith(STRIP)
        }

        if not inner_sd:
            logger.warning("ERNIE-Image: no 'language_model.model.*' keys found.")
            return None

        missing, unexpected = text_model.load_state_dict(inner_sd, strict=False)
        n_loaded = len(inner_sd) - len(missing)
        logger.info(
            f"ERNIE-Image: text encoder {n_loaded}/{len(inner_sd)} weights loaded "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )

        text_model = text_model.to(dtype)
        text_model.eval()
        return text_model

    except Exception as exc:
        logger.warning(f"ERNIE-Image: text encoder build failed - {exc}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# VAE tiling helpers
# ---------------------------------------------------------------------------

def _enable_vae_tiling(pipeline) -> None:
    vae = getattr(pipeline, "vae", None)
    if vae is None:
        return
    for m in ("enable_tiling", "enable_vae_tiling"):
        fn = getattr(vae, m, None)
        if fn is not None:
            try:
                fn(); return
            except Exception:
                pass


def _disable_vae_tiling(pipeline) -> None:
    vae = getattr(pipeline, "vae", None)
    if vae is None:
        return
    for m in ("disable_tiling", "disable_vae_tiling"):
        fn = getattr(vae, m, None)
        if fn is not None:
            try:
                fn(); return
            except Exception:
                pass


def _patch_cosine_blend(pipeline) -> None:
    """Replace linear tile blending with cosine for smoother seams."""
    vae = getattr(pipeline, "vae", None)
    if vae is None or getattr(vae, "_cosine_blend_patched", False):
        return

    import types

    def _cosine_blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            alpha = (1.0 - math.cos(math.pi * y / blend_extent)) / 2.0
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - alpha) + b[:, :, y, :] * alpha
        return b

    def _cosine_blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            alpha = (1.0 - math.cos(math.pi * x / blend_extent)) / 2.0
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - alpha) + b[:, :, :, x] * alpha
        return b

    vae.blend_v = types.MethodType(_cosine_blend_v, vae)
    vae.blend_h = types.MethodType(_cosine_blend_h, vae)
    vae._cosine_blend_patched = True
    logger.debug("ERNIE-Image: VAE tile blending patched to cosine.")


# ---------------------------------------------------------------------------
# Lazy diffusers import
# ---------------------------------------------------------------------------

def _get_pipeline_class():
    try:
        from diffusers import ErnieImagePipeline
        return ErnieImagePipeline
    except ImportError as exc:
        raise ImportError(
            "diffusers with ErnieImagePipeline support is required.\n"
            "Install: pip install git+https://github.com/huggingface/diffusers"
        ) from exc


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

RESOLUTION_PRESETS: dict[str, tuple[int, int] | None] = {
    # Organized widest -> tallest. ★ = official ERNIE training presets.
    "3:1 panoramic":          (1776,  592),
    "21:9 cinemascope":       (1568,  672),
    "2:1 wide":               (1440,  720),
    "16:9 ★":                 (1376,  768),   # official ERNIE preset
    "8:5":                    (1280,  800),
    "3:2 ★":                  (1264,  848),   # official ERNIE preset
    "7:5":                    (1232,  880),
    "4:3 ★":                  (1200,  896),   # official ERNIE preset
    "5:4":                    (1120,  896),
    "1:1 square ★":           (1024, 1024),   # official ERNIE preset
    "4:5":                    ( 896, 1120),
    "3:4 ★":                  ( 896, 1200),   # official ERNIE preset
    "5:7":                    ( 880, 1232),
    "2:3 ★":                  ( 848, 1264),   # official ERNIE preset
    "5:8":                    ( 800, 1280),
    "9:16 ★":                 ( 768, 1376),   # official ERNIE preset
    "1:2":                    ( 720, 1440),
    "9:21 cinemascope":       ( 672, 1568),
    "1:3 panoramic":          ( 592, 1776),
    "custom":                 None,
}
LATENT_ALIGN = 16


def _resolve_dimensions(resolution, custom_width, custom_height, max_mp):
    if resolution == "custom":
        w, h = custom_width, custom_height
    else:
        w, h = RESOLUTION_PRESETS[resolution]  # type: ignore[misc]
    max_px = int(max_mp * 1_048_576)
    if w * h > max_px:
        scale = math.sqrt(max_px / (w * h))
        w = int(w * scale)
        h = int(h * scale)
    w = max(LATENT_ALIGN, (w // LATENT_ALIGN) * LATENT_ALIGN)
    h = max(LATENT_ALIGN, (h // LATENT_ALIGN) * LATENT_ALIGN)
    return w, h


def _pil_to_comfy(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# ErnieImageLoadModel
# ---------------------------------------------------------------------------

class ErnieImageLoadModel:
    """
    Load ERNIE-Image (SFT) or ERNIE-Image-Turbo with all quality patches.

    The sigma_schedule setting has been moved to ErnieImageGenerate so you can
    A/B test different schedules without reloading the model.
    """

    _CACHE: dict[tuple, object] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {
                        "default": "H:/Training/ERNIE-Image",
                        "multiline": False,
                        "tooltip": "Path to ERNIE-Image or ERNIE-Image-Turbo directory.",
                    },
                ),
                "precision": (
                    ["bf16", "fp16", "fp32"],
                    {
                        "default": "bf16",
                        "tooltip": "fp16 may give sharper VAE decode than bf16.",
                    },
                ),
                "device":    (["cuda", "cuda:0", "cuda:1", "cpu"], {"default": "cuda"}),
                "keep_in_vram": ("BOOLEAN", {"default": True}),
                "load_pe": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Requires transformers >= 5.x. Use Prompt Rewriter instead.",
                    },
                ),
            },
            "optional": {
                "attention_slicing": ("BOOLEAN", {"default": False}),
                "sequential_offload": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "CPU offload. Saves VRAM, slower."},
                ),
            },
        }

    RETURN_TYPES  = ("ERNIE_PIPELINE",)
    RETURN_NAMES  = ("pipeline",)
    FUNCTION      = "load_model"
    CATEGORY      = "Eric ERNIE Image"
    DESCRIPTION   = "Load ERNIE-Image or ERNIE-Image-Turbo with all quality patches."

    def load_model(
        self,
        model_path: str,
        precision: str,
        device: str,
        keep_in_vram: bool,
        load_pe: bool = False,
        attention_slicing: bool = False,
        sequential_offload: bool = False,
    ) -> tuple:

        cache_key = (model_path, precision, device, sequential_offload, load_pe)
        if keep_in_vram and cache_key in self._CACHE:
            logger.info("ERNIE-Image: returning cached pipeline.")
            return (self._CACHE[cache_key],)

        _patch_transformers_compat()
        _patch_mistral_for_explicit_head_dim()

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[precision]

        logger.info(
            f"ERNIE-Image: loading from {model_path!r}  precision={precision}  load_pe={load_pe}"
        )

        prebuilt_text_encoder = _build_text_encoder(model_path, dtype)

        load_kwargs: dict = {
            "torch_dtype": dtype,
            "local_files_only": True,
        }
        if prebuilt_text_encoder is not None:
            load_kwargs["text_encoder"] = prebuilt_text_encoder
        if not load_pe:
            load_kwargs["pe"] = None
            load_kwargs["pe_tokenizer"] = None

        ErnieImagePipeline = _get_pipeline_class()

        try:
            pipe = ErnieImagePipeline.from_pretrained(model_path, **load_kwargs)
        except Exception as exc:
            err_str = str(exc)
            if load_pe and ("'list' object has no attribute 'keys'" in err_str
                            or "TokenizersBackend" in err_str):
                raise RuntimeError(
                    "Loading the PE model requires transformers >= 5.x.\n"
                    "Set load_pe=False and use the ERNIE Prompt Rewriter node instead.\n"
                    f"Original error: {exc}"
                ) from exc
            if isinstance(exc, OSError):
                raise RuntimeError(
                    f"Could not load from {model_path!r}.\nOriginal error: {exc}"
                ) from exc
            raise

        if sequential_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to(device)

        if attention_slicing:
            pipe.enable_attention_slicing()

        if keep_in_vram:
            self._CACHE[cache_key] = pipe

        pipe._ernie_pe_loaded = load_pe
        logger.info("ERNIE-Image: pipeline ready.")
        return (pipe,)


# ---------------------------------------------------------------------------
# ErnieImageEncode  (standalone text encoder node)
# ---------------------------------------------------------------------------

class ErnieImageEncode:
    """
    Encode a prompt to embeddings using the loaded pipeline's text encoder.

    Output ERNIE_EMBEDS can be wired to ErnieImageGenerate's prompt_embeds input,
    bypassing the text encoder during generation.  This enables:
      - Reusing the same embeddings across multiple seeds/settings without
        re-running the text encoder each time
      - External embedding manipulation between this node and Generate
      - Caching embeddings for batch workflows

    The pipeline's encode_prompt() returns a List[Tensor] where each tensor is
    hidden_states[-2][0], shape [T, 3072].  This node wraps that list as a
    single Python object (ERNIE_EMBEDS type) that Generate knows how to consume.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("ERNIE_PIPELINE",),
                "prompt":   ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES  = ("ERNIE_EMBEDS",)
    RETURN_NAMES  = ("embeds",)
    FUNCTION      = "encode"
    CATEGORY      = "Eric ERNIE Image"
    DESCRIPTION   = "Pre-compute prompt embeddings for reuse across multiple generations."

    def encode(self, pipeline, prompt: str) -> tuple:
        device = next(pipeline.transformer.parameters()).device
        embeds = pipeline.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
        )
        logger.info(
            f"ERNIE-Image Encode: {len(embeds)} embedding(s), "
            f"shape={embeds[0].shape}, dtype={embeds[0].dtype}"
        )
        return (embeds,)


# ---------------------------------------------------------------------------
# ErnieImageGenerate
# ---------------------------------------------------------------------------

class ErnieImageGenerate:
    """
    Generate images with ERNIE-Image or ERNIE-Image-Turbo.

    sigma_schedule (on this node, not the loader)
    ----------------------------------------------
    Controls the denoising timestep distribution.  Switch without reloading:
      uniform                   -- pipeline default (shift=4.0 always applied)
      fixed shift               -- override shift to shift_value
      karras                    -- shift + karras transform (both extremes)
      beta                      -- shift + beta distribution (good detail)
      dynamic (resolution-based) -- resolution-dependent shift

    Quality settings
    ----------------
    vae_decode_fp32  -- upcast VAE to float32 during decode (big detail win)
    guidance_rescale -- reduce CFG-induced saturation (0.5-0.7 for high cfg)

    prompt_embeds / negative_prompt_embeds (optional)
    -------------------------------------------------
    Wire an ErnieImageEncode node output here to bypass the text encoder.

    Settings
    --------
    SFT   : steps=50, guidance_scale=4.0
    Turbo : steps=8,  guidance_scale=1.0
    """

    @classmethod
    def INPUT_TYPES(cls):
        presets = list(RESOLUTION_PRESETS.keys())
        return {
            "required": {
                "pipeline":       ("ERNIE_PIPELINE",),
                "prompt":         ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "Ignored when prompt_embeds is connected."}),
                "resolution":     (presets, {"default": "1024x1024 (1:1)"}),
                "steps":          ("INT", {"default": 50, "min": 1, "max": 100,
                                           "tooltip": "50 for SFT, 8 for Turbo."}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0,
                                             "step": 0.1,
                                             "tooltip": "4.0 for SFT, 1.0 for Turbo."}),
                "seed":           ("INT", {"default": 0, "min": 0,
                                           "max": 0xFFFFFFFFFFFFFFFF}),
                "use_pe":         ("BOOLEAN", {"default": False,
                                               "tooltip": "Requires load_pe=True in loader."}),
                "sigma_schedule": (
                    SIGMA_SCHEDULES,
                    {
                        "default": "uniform",
                        "tooltip": (
                            "'uniform' = pipeline default (shift=4.0 from scheduler config). "
                            "'fixed shift' = override shift to shift_value below. "
                            "'karras' = shift + karras transform (more steps at both extremes). "
                            "'beta' = shift + beta distribution (good balance for fine detail). "
                            "'dynamic' = resolution-based shift. "
                            "Change without reloading."
                        ),
                    },
                ),
                "shift_value": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.1,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": (
                            "Override scheduler shift for 'fixed shift', 'karras', 'beta' schedules. "
                            "1.0 = truly uniform (no shift bias). "
                            "4.0 = default (strong structural emphasis). "
                            "Ignored when schedule = 'uniform' or 'dynamic'."
                        ),
                    },
                ),
                "max_mp": (
                    "FLOAT",
                    {
                        "default": 1.5, "min": 0.25, "max": 8.0, "step": 0.25,
                        "tooltip": "Stay at 1.0-1.5 MP. ERNIE trained on 7 ~1MP presets.",
                    },
                ),
                "guidance_rescale": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                        "tooltip": (
                            "Rescale CFG output toward unit variance to reduce saturation. "
                            "0.0 = off. 0.5-0.7 = moderate rescaling. "
                            "Helps preserve fine detail at high guidance_scale."
                        ),
                    },
                ),
                "vae_decode_fp32": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Upcast VAE to float32 for decode. "
                            "Prevents bf16/fp16 quantization of fine textures. "
                            "Tiny speed cost, significant quality win."
                        ),
                    },
                ),
            },
            "optional": {
                "width":  ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 16,
                                   "tooltip": "Used only when resolution = 'custom'."}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 16,
                                   "tooltip": "Used only when resolution = 'custom'."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt_embeds": (
                    "ERNIE_EMBEDS",
                    {
                        "tooltip": (
                            "Pre-computed embeddings from ErnieImageEncode. "
                            "When connected, the prompt text input is ignored. "
                            "Enables reuse across seeds without re-running the text encoder."
                        ),
                    },
                ),
                "negative_prompt_embeds": (
                    "ERNIE_EMBEDS",
                    {
                        "tooltip": (
                            "Pre-computed negative embeddings from ErnieImageEncode. "
                            "When connected, negative_prompt text is ignored."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "generate"
    CATEGORY      = "Eric ERNIE Image"
    DESCRIPTION   = "Generate with ERNIE-Image or Turbo. Accepts pre-computed embeddings."

    def generate(
        self,
        pipeline,
        prompt: str,
        resolution: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        use_pe: bool,
        sigma_schedule: str,
        shift_value: float,
        max_mp: float,
        guidance_rescale: float,
        vae_decode_fp32: bool,
        width: int = 1024,
        height: int = 1024,
        negative_prompt: str = "",
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ) -> tuple:

        pe_loaded = getattr(pipeline, "_ernie_pe_loaded", False)
        if use_pe and not pe_loaded:
            logger.warning("ERNIE-Image: use_pe=True but PE not loaded. Using use_pe=False.")
            use_pe = False

        # Embeddings take priority over text - log which path we're taking
        using_embeds = prompt_embeds is not None
        if using_embeds:
            logger.info("ERNIE-Image: using pre-computed prompt_embeds (text encoder bypassed).")

        out_w, out_h = _resolve_dimensions(resolution, width, height, max_mp)
        logger.info(
            f"ERNIE-Image: {out_w}x{out_h} | steps={steps} | cfg={guidance_scale} | "
            f"schedule={sigma_schedule!r}(shift={shift_value}) | pe={use_pe} | seed={seed} | "
            f"embeds={'yes' if using_embeds else 'no'} | "
            f"cfg_rescale={guidance_rescale} | vae_fp32={vae_decode_fp32}"
        )

        # Apply sigma schedule on each call (cheap - just wraps __call__)
        _apply_sigma_schedule_to_call(pipeline, sigma_schedule, shift_value)

        # ---- VAE tiling (cosine blend for smoother seams) ------------- #
        if out_w * out_h > 2 * 1_048_576:
            _enable_vae_tiling(pipeline)
            _patch_cosine_blend(pipeline)
        else:
            _disable_vae_tiling(pipeline)

        # ---- VAE float32 upcast for decode quality -------------------- #
        vae = pipeline.vae
        vae_orig_dtype = vae.dtype
        if vae_decode_fp32 and vae_orig_dtype != torch.float32:
            vae.to(torch.float32)
            logger.debug("ERNIE-Image: VAE upcast to float32 for decode.")

        # ---- Guidance rescale (reduce CFG saturation) ----------------- #
        _orig_sched_step = None
        if guidance_rescale > 0 and guidance_scale > 1.0:
            _orig_sched_step = pipeline.scheduler.step
            _gr = guidance_rescale
            def _rescaled_step(model_output, timestep, sample, **kwargs):
                dims = list(range(1, model_output.ndim))
                std = model_output.std(dim=dims, keepdim=True).clamp(min=1e-6)
                rescaled = model_output / std
                model_output = (1.0 - _gr) * model_output + _gr * rescaled
                return _orig_sched_step(model_output, timestep, sample, **kwargs)
            pipeline.scheduler.step = _rescaled_step
            logger.debug(f"ERNIE-Image: guidance_rescale={guidance_rescale}")

        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(steps)
        except ImportError:
            pbar = None

        def _step_callback(pipe, step_index, timestep, callback_kwargs):
            if pbar is not None:
                pbar.update(1)
            return callback_kwargs

        generator = torch.Generator("cpu").manual_seed(seed)

        # Build call kwargs - use embeds when provided, text otherwise
        call_kwargs: dict = dict(
            height=out_h,
            width=out_w,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            use_pe=use_pe,
            callback_on_step_end=_step_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        if using_embeds:
            call_kwargs["prompt_embeds"] = prompt_embeds
            if negative_prompt_embeds is not None:
                call_kwargs["negative_prompt_embeds"] = negative_prompt_embeds
            elif guidance_scale > 1.0:
                # Need uncond embeds for CFG - encode empty string
                device = next(pipeline.transformer.parameters()).device
                call_kwargs["negative_prompt_embeds"] = pipeline.encode_prompt(
                    "", device=device, num_images_per_prompt=1
                )
        else:
            call_kwargs["prompt"] = prompt
            neg = negative_prompt.strip() or None
            if neg:
                call_kwargs["negative_prompt"] = neg

        try:
            with torch.inference_mode():
                result = pipeline(**call_kwargs)
        finally:
            # Restore VAE dtype
            if vae_decode_fp32 and vae_orig_dtype != torch.float32:
                vae.to(vae_orig_dtype)
            # Restore scheduler.step
            if _orig_sched_step is not None:
                pipeline.scheduler.step = _orig_sched_step

        return (_pil_to_comfy(result.images[0]),)


# ---------------------------------------------------------------------------
# ErnieImageUnload
# ---------------------------------------------------------------------------

class ErnieImageUnload:
    """
    Release ERNIE-Image from VRAM.

    Design: NO pipeline input.
    -------------------------
    The previous design required the pipeline to be wired in, which caused
    ComfyUI to re-execute the Load node first - OOMing before Unload could run.

    This node clears the class-level pipeline cache and calls torch.cuda.empty_cache()
    directly, without needing the pipeline object.  Connect nothing - just run it
    as a standalone node via the Queue button.

    The optional 'images' passthrough lets you trigger it at the end of a workflow
    without causing any upstream re-execution of the load path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "images": ("IMAGE", {"tooltip": "Optional passthrough - does not trigger model reload."}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("images", "status")
    FUNCTION      = "unload"
    CATEGORY      = "Eric ERNIE Image"
    OUTPUT_NODE   = True
    DESCRIPTION   = "Clear ERNIE-Image from VRAM. No pipeline input needed - avoids reload OOM."

    def unload(self, images=None) -> tuple:
        import gc, torch

        lines = []

        # Clear class-level pipeline cache
        n_cached = len(ErnieImageLoadModel._CACHE)
        if n_cached:
            cached_pipes = list(ErnieImageLoadModel._CACHE.values())
            ErnieImageLoadModel._CACHE.clear()

            for pipe in cached_pipes:
                # Remove accelerate hooks first (sequential_offload pins
                # modules to GPU via forward hooks — must remove before .to)
                if hasattr(pipe, "remove_all_hooks"):
                    try:
                        pipe.remove_all_hooks()
                    except Exception:
                        pass

                # Move every nn.Module component to CPU
                for attr in ("transformer", "vae", "text_encoder", "tokenizer",
                             "pe", "pe_tokenizer"):
                    submodel = getattr(pipe, attr, None)
                    if submodel is not None and hasattr(submodel, "to"):
                        try:
                            submodel.to("cpu")
                        except Exception:
                            pass

                # Move scheduler GPU tensors (sigmas/timesteps) to CPU
                sched = getattr(pipe, "scheduler", None)
                if sched is not None:
                    for t_attr in ("sigmas", "timesteps"):
                        val = getattr(sched, t_attr, None)
                        if isinstance(val, torch.Tensor) and val.is_cuda:
                            setattr(sched, t_attr, val.cpu())

            # Drop local refs so GC can actually collect the pipeline objects
            del cached_pipes, pipe
            lines.append(f"Cleared {n_cached} cached pipeline(s).")
        else:
            lines.append("No cached pipelines found.")

        # Two passes: first breaks circular refs, second collects freed objects
        gc.collect()
        gc.collect()
        lines.append("GC collected.")

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                before = torch.cuda.memory_allocated(i) / 1024**3
                torch.cuda.empty_cache()
                after  = torch.cuda.memory_allocated(i) / 1024**3
                lines.append(f"GPU {i}: {before:.2f}GB -> {after:.2f}GB allocated.")
        else:
            lines.append("No CUDA devices.")

        status = "  ".join(lines)
        return (images, status)

