# Eric ERNIE-Image - ComfyUI Custom Nodes

High-fidelity diffusers-based ComfyUI nodes for **ERNIE-Image** (SFT) and **ERNIE-Image-Turbo**, built by Eric Hiss (GitHub: EricRollei).

---

## Why This Is Better Than the Native ComfyUI ERNIE Nodes

The official ComfyUI core implementation loads ERNIE-Image through the standard UNet/KSampler pipeline - the same path used for SD1.5 and SDXL. This introduces several quality compromises:

| Issue | Native ComfyUI | This Package |
|-------|---------------|--------------|
| Text encoder | Loads wrong Mistral3Model at 5120-dim (broken) | Correct MistralModel at 3072-dim, all 236 weights loaded |
| Attention projections | All 104 attention matrices silently skipped (random weights) | head_dim=128 patched, correct q/k/v/o shapes [4096,3072] |
| Rotary embedding | inv_freq shape [48] (head_dim 96), causes runtime error | Fixed to [64] for head_dim=128 |
| Sigma schedule | Bypasses scheduler entirely | User-selectable: uniform / shift=4.0 / dynamic (resolution-based) |
| Pipeline class | UNet abstraction with shortcuts | Native `ErnieImagePipeline` from diffusers, full forward pass |
| VAE | Generic VAE handling | Direct `AutoencoderKLFlux2` with correct 32-channel latents |
| PE (Prompt Enhancer) | Not exposed | Bypassed cleanly; replaced by LM Studio rewriter node |
| Embedding reuse | Not possible | `ErnieImageEncode` node pre-computes embeddings for reuse |
| VRAM management | Manual | Dedicated Unload node, no pipeline connection required |

### The Text Encoder Bug (Critical)

Ministral 3B uses `head_dim=128` explicitly, but standard `MistralConfig` computes `head_dim = hidden_size / num_heads = 3072 / 32 = 96`. This mismatch means all 104 attention weight matrices (4 per layer × 26 layers) have wrong shapes and are silently skipped by `strict=False` during loading. The text encoder runs with **completely random attention weights**, producing near-zero semantic content.

This package patches `MistralAttention.__init__` to use the explicit `head_dim=128` from config, and replaces the shared `model.rotary_emb.inv_freq` buffer from shape [48] to [64] - the only reliable fix across transformers versions.

---

## Model Architecture

| Component | Details |
|-----------|---------|
| Transformer | 8B single-stream DiT, 36 layers, text_in_dim=3072 |
| Text encoder | Ministral 3B (hidden_size=3072, 26 layers, head_dim=128) |
| PE model | Separate 3B LLM in `pe/` subfolder - requires transformers ≥ 5.x |
| VAE | AutoencoderKLFlux2, 32-channel, patch_size [2,2] |
| Latent scale | 16× (8× VAE × 2× patch) - image dims must be multiples of 16 |
| Scheduler | FlowMatchEulerDiscreteScheduler, shift=4.0 |
| CFG | Zero-unconditional (empty string uncond, not true zero vector) |

**SFT vs Turbo:**
- **ERNIE-Image (SFT)**: steps=50, guidance_scale=4.0 - stronger prompt following, higher general quality
- **ERNIE-Image-Turbo**: steps=8, guidance_scale=1.0 - DMD+RL distilled, faster, aesthetically tuned

Same pipeline class, same loader node - just point to a different model path and adjust steps/CFG.

---

## Nodes

### Eric ERNIE-Image Load Model

Loads the pipeline from a local directory with all compatibility patches applied.

**Inputs:**
- `model_path` - Path to local ERNIE-Image or ERNIE-Image-Turbo directory
- `precision` - `bf16` (default) / `fp16` / `fp32`. Try `fp16` for sharper VAE decode than bf16
- `device` - `cuda` / `cuda:0` / `cuda:1` / `cpu`
- `keep_in_vram` - Cache the loaded pipeline; reuse on subsequent runs without reloading
- `load_pe` - Load the built-in Prompt Enhancer. **Requires transformers ≥ 5.x** - leave False on standard ComfyUI installs and use the Prompt Rewriter node instead

**Output:** `ERNIE_PIPELINE`

**Patches applied automatically:**
1. `TokenizersBackend` → `PreTrainedTokenizerFast` (three-route patch for transformers `_LazyModule`)
2. `ministral3` registration in `CONFIG_MAPPING`
3. `MistralAttention` projection shapes fixed for `head_dim=128`
4. `model.rotary_emb.inv_freq` buffer replaced `[48]→[64]`
5. Text encoder pre-built and passed to `from_pretrained` - the 7.7GB safetensors file is read **once** instead of twice

---

### Eric ERNIE-Image Encode

Pre-computes prompt embeddings using the pipeline's text encoder. The output can be wired directly into the Generate node, bypassing the text encoder on each generation run.

**Inputs:**
- `pipeline` - From Load Model node
- `prompt` - Text to encode

**Output:** `ERNIE_EMBEDS` - a `List[Tensor]` where each tensor is `hidden_states[-2][0]`, shape `[T, 3072]`

**Use cases:**
- Encode once, generate with multiple seeds - text encoder runs once instead of N times
- A/B test SFT vs Turbo with identical text conditioning
- Future: external embedding manipulation between Encode and Generate

---

### Eric ERNIE-Image Generate

Runs the diffusion pipeline to produce an image.

**Required inputs:**
- `pipeline` - From Load Model node
- `prompt` - Ignored when `prompt_embeds` is connected
- `resolution` - 7 official presets (~1MP each) + `custom`. **Stay at presets for best quality** - ERNIE was trained on these 7 specific aspect ratios
- `steps` - 50 for SFT, 8 for Turbo
- `guidance_scale` - 4.0 for SFT, 1.0 for Turbo (at 1.0, CFG is disabled - single-pass generation)
- `seed` - Deterministic generation
- `use_pe` - Use built-in Prompt Enhancer (requires `load_pe=True` in loader); falls back to False gracefully
- `sigma_schedule` - Controls denoising timestep distribution (see below)
- `max_mp` - Megapixel cap applied before 16px alignment. Stay at 1.0-1.5 MP

**Optional inputs:**
- `width` / `height` - Used only when resolution = `custom`
- `negative_prompt` - Works with SFT (guidance_scale > 1.0); ignored for Turbo
- `prompt_embeds` - Pre-computed embeddings from ErnieImageEncode; bypasses text encoder
- `negative_prompt_embeds` - Pre-computed negative embeddings. When `prompt_embeds` is connected but this is not, the node automatically encodes an empty string for the uncond path

**Output:** `IMAGE`

#### Sigma Schedule Options

The pipeline by default passes a uniform `linspace(1.0, 0.0, N+1)` directly to the scheduler, bypassing all shift computation. The scheduler config has three shift-related values:

| Setting | Value | Status |
|---------|-------|--------|
| `shift` | 4.0 | Fixed - used when `set_timesteps(num_inference_steps=N)` is called |
| `base_shift` | 0.5 | Dynamic floor - **inactive** (`use_dynamic_shifting=False`) |
| `max_shift` | 1.15 | Dynamic ceiling - **inactive** (`use_dynamic_shifting=False`) |

Node options:
- **`uniform (pipeline default)`** - Unchanged; what Baidu's pipeline code does. Start here.
- **`shift=4.0 (scheduler)`** - Applies the fixed shift from `scheduler_config.json`. Gives 64% of steps above sigma=0.7 (structural phase) vs 32% for uniform. Strong deviation - helps SFT, possibly hurts Turbo (which was distilled against uniform).
- **`dynamic (resolution-based)`** - Uses `base_shift=0.5` / `max_shift=1.15` per image size. At 1024×1024 this computes to ~1.15 (nearly uniform). Falls back to `shift=4.0` if the scheduler doesn't support the `mu` parameter.

Switching schedules takes effect immediately without reloading the model.

---

### Eric ERNIE-Image Unload

Releases all cached ERNIE-Image pipelines from VRAM.

**Design note:** This node has **no pipeline input** by design. The previous design (requiring a pipeline connection) caused ComfyUI to re-execute the Load node first, OOMing before Unload could run. This node clears the class-level cache and calls `torch.cuda.empty_cache()` directly.

**Optional input:**
- `images` - Passthrough only; wiring this does NOT trigger the load node

**Output:** `images` (passthrough), `status` (STRING describing what was cleared)

Run as a standalone node - just queue it with no connections. Or wire `images` at the end of a workflow to trigger cleanup after generation.

---

### Eric ERNIE Prompt Rewriter

Expands a short prompt into a rich 150-250 word visual description using any OpenAI-compatible LLM endpoint (LM Studio, Ollama, etc.).

**Why this instead of the built-in PE:**
- The built-in PE requires transformers ≥ 5.x (broken on standard ComfyUI)
- PE benchmarks show it actually **hurts** GENEval (instruction following) vs no PE
- This node: use any model you choose, see the expanded prompt before generation, zero extra VRAM, works with any language

**Inputs:**
- `prompt` - Short description to expand
- `api_url` - LM Studio: `http://localhost:1234/v1` / Ollama: `http://localhost:11434/v1`
- `model` - Model name as shown in LM Studio/Ollama (e.g. `qwen3-8b`)
- `language` - `English` or `Chinese` (Chinese matches training data origin)
- `width` / `height` - Passed to the LLM for aspect-ratio-aware expansion
- `temperature` - 0.6 matches PE defaults
- `passthrough` - Bypass rewriting, forward prompt unchanged (useful for A/B testing)
- `custom_instructions` - Extra instructions appended to the system prompt

**Output:** `STRING` - wire to Generate node's `prompt` input with `use_pe=False`

The system prompt mirrors the official PE's chat template exactly. The user message is `{"prompt": "...", "width": W, "height": H}` - same JSON format the PE uses.

---

## Recommended Workflows

### Basic SFT Generation
```
ErnieImageLoadModel (precision=bf16, load_pe=False)
    ↓
ErnieImageGenerate (steps=50, cfg=4.0, sigma_schedule=uniform)
    ↓
PreviewImage
```

### With Prompt Rewriter
```
[short prompt] → ErniePromptRewriter (LM Studio) → ErnieImageGenerate
                                                         ↑
                                                    ErnieImageLoadModel
```

### Embedding Reuse (multiple seeds)
```
ErnieImageLoadModel ──────────────────────────────────────────────┐
    ↓                                                               ↓
ErnieImageEncode (prompt) → [ERNIE_EMBEDS] → ErnieImageGenerate (seed=1)
                                           → ErnieImageGenerate (seed=2)
                                           → ErnieImageGenerate (seed=3)
```

Text encoder runs once. All three seeds share identical conditioning.

---

## Quality Notes

- **Stay at the 7 official presets** (~1MP) for best quality. The DiT's 2D RoPE positional encoding is out-of-distribution at non-preset resolutions, causing repetition artifacts above ~1.5MP.
- **Use long, detailed prompts** - all official gallery images used PE-expanded 200+ word descriptions. The model expects rich visual descriptions: lighting, materials, camera, atmosphere.
- **Negative prompts** work with SFT (guidance_scale=4.0, two-pass CFG). Ignored for Turbo (guidance_scale=1.0 disables CFG entirely).
- **fp16 precision** may give sharper VAE decode than bf16, at the cost of slightly higher OOM risk.

---

## Supported Resolutions

| Preset | Dimensions | Aspect Ratio |
|--------|-----------|--------------|
| 1024×1024 | 1024×1024 | 1:1 |
| 1264×848 | 1264×848 | 3:2 landscape |
| 848×1264 | 848×1264 | 2:3 portrait |
| 1376×768 | 1376×768 | 16:9 landscape |
| 768×1376 | 768×1376 | 9:16 portrait |
| 1200×896 | 1200×896 | 4:3 landscape |
| 896×1200 | 896×1200 | 3:4 portrait |
| custom | user-defined | any (capped by max_mp, aligned to 16px) |

---

## Installation

1. Clone or copy this folder to `ComfyUI/custom_nodes/Eric_Ernie_Image/`
2. Download ERNIE-Image and/or ERNIE-Image-Turbo from HuggingFace (`baidu/ERNIE-Image`, `baidu/ERNIE-Image-Turbo`)
3. No additional pip installs required - uses diffusers, transformers, and safetensors already present in ComfyUI's environment
4. Restart ComfyUI

**Requirements:**
- diffusers with `ErnieImagePipeline` support (`pip install git+https://github.com/huggingface/diffusers` if not present)
- transformers (any recent version - compatibility patches are applied automatically)
- safetensors
- PyTorch with CUDA

---

## Author

Eric Hiss - GitHub: [EricRollei](https://github.com/EricRollei)  
Photography: [rollei.us](https://rollei.us) · [rolleiflex.us](https://rolleiflex.us)
