"""
Eric ERNIE-Image Prompt Rewriter  -  nodes/node_ernie_prompter.py
=================================================================
LM Studio / Ollama-based prompt rewriter for ERNIE-Image.

Replicates the built-in PE (Prompt Enhancer) model's exact behavior using any
OpenAI-compatible LLM endpoint.  This is strictly better than the native PE:

  - Use any model in LM Studio (Qwen3-8B gives excellent results)
  - Output is inspectable and editable before generation
  - No extra 3B model to load into VRAM
  - No transformers 5.x requirement
  - Full temperature / language control

The PE's system prompt and user message format are preserved exactly so the
enhanced prompts are compatible with how ERNIE-Image was trained.

Wire: [prompt text] -> ErniePromptRewriter -> ErnieImageGenerate (use_pe=False)

Author : Eric Hiss  (GitHub: EricRollei)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ERNIE-Image PE system prompt (translated from the official chat_template.jinja)
# This is the exact system instruction the PE model receives.
# The English version produces equivalent quality for English prompts.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_EN = """You are a professional text-to-image prompt enhancement assistant. \
You will receive the user's brief image description and the target generation resolution. \
Expand it into a rich, detailed visual description (150-250 words) that will help the \
text-to-image model generate a high-quality image. Focus on: subject details, lighting, \
composition, color palette, atmosphere, style, and technical quality indicators. \
Output only the enhanced description - no explanations, no prefix, no quotes."""

_SYSTEM_PROMPT_ZH = (
    "你是一个专业的文生图 Prompt 增强助手。"
    "你将收到用户的简短图片描述及目标生成分辨率，"
    "请据此扩写为一段内容丰富、细节充分的视觉描述（150-250字），"
    "以帮助文生图模型生成高质量的图片。"
    "重点关注：主体细节、光线、构图、色彩、氛围、风格和质量描述。"
    "仅输出增强后的描述，不要包含任何解释或前缀。"
)


class ErniePromptRewriter:
    """
    Expand a short prompt into a rich ~200-word visual description suitable for
    ERNIE-Image using any OpenAI-compatible LLM endpoint (LM Studio, Ollama, etc.).

    The PE (Prompt Enhancer) built into ERNIE-Image requires transformers 5.x and
    loads a fixed 3B LLM.  This node is a drop-in replacement that:
      - Uses whichever model is running in LM Studio / Ollama
      - Outputs the enhanced prompt as an inspectable STRING
      - Connects directly to the ErnieImageGenerate 'prompt' input
      - Requires use_pe=False in the Generate node (PE is this node)

    Passthrough mode
    ----------------
    Set passthrough=True to bypass rewriting and forward the raw prompt unchanged.
    Useful for A/B testing with the same seed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Short prompt to expand into a detailed description.",
                    },
                ),
                "api_url": (
                    "STRING",
                    {
                        "default": "http://localhost:1234/v1",
                        "tooltip": (
                            "OpenAI-compatible API base URL. "
                            "LM Studio default: http://localhost:1234/v1  "
                            "Ollama default: http://localhost:11434/v1"
                        ),
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "qwen3-8b",
                        "tooltip": (
                            "Model name as shown in LM Studio / Ollama. "
                            "Qwen3-8B, Llama3-8B, or any instruct model works well."
                        ),
                    },
                ),
                "language": (
                    ["English", "Chinese"],
                    {
                        "default": "English",
                        "tooltip": (
                            "Language for the enhanced prompt. "
                            "Chinese matches ERNIE-Image's training data origin "
                            "but English works equally well."
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024, "min": 256, "max": 8192, "step": 16,
                        "tooltip": "Target width - passed to the LLM for aspect-ratio-aware expansion.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024, "min": 256, "max": 8192, "step": 16,
                        "tooltip": "Target height - passed to the LLM for aspect-ratio-aware expansion.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05,
                        "tooltip": "LLM temperature. 0.6 matches the PE defaults. Lower = more faithful.",
                    },
                ),
                "passthrough": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Skip rewriting and pass the prompt unchanged. Useful for A/B testing.",
                    },
                ),
            },
            "optional": {
                "custom_instructions": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Extra instructions appended to the system prompt.",
                    },
                ),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("enhanced_prompt",)
    FUNCTION      = "rewrite"
    CATEGORY      = "Eric ERNIE Image"
    DESCRIPTION   = "Expand a short prompt into a detailed ERNIE-Image description via LM Studio."

    # ------------------------------------------------------------------ #

    def rewrite(
        self,
        prompt: str,
        api_url: str,
        model: str,
        language: str,
        width: int,
        height: int,
        temperature: float,
        passthrough: bool,
        custom_instructions: str = "",
    ) -> tuple:

        if passthrough or not prompt.strip():
            return (prompt,)

        system_prompt = (
            _SYSTEM_PROMPT_ZH if language == "Chinese" else _SYSTEM_PROMPT_EN
        )
        if custom_instructions.strip():
            system_prompt += "\n\nAdditional instructions: " + custom_instructions.strip()

        # Match the PE's exact user message format so the LLM knows target resolution
        user_content = json.dumps(
            {"prompt": prompt.strip(), "width": width, "height": height},
            ensure_ascii=False,
        )

        try:
            enhanced = _call_llm(
                api_url=api_url.rstrip("/"),
                model=model,
                system=system_prompt,
                user=user_content,
                temperature=temperature,
                max_tokens=1024,
            )
        except Exception as exc:
            logger.warning(
                f"ERNIE Prompt Rewriter: LLM call failed ({exc}). "
                "Returning original prompt."
            )
            return (prompt,)

        # Strip any wrapping quotes or preamble the model may have added
        enhanced = _clean_llm_output(enhanced)

        if not enhanced:
            logger.warning("ERNIE Prompt Rewriter: empty response, returning original prompt.")
            return (prompt,)

        logger.info(
            f"ERNIE Prompt Rewriter: {len(prompt.split())} words -> "
            f"{len(enhanced.split())} words"
        )
        return (enhanced,)


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _call_llm(
    api_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    POST to an OpenAI-compatible /v1/chat/completions endpoint.
    Returns the assistant's text content.
    Uses urllib (stdlib) to avoid adding any extra dependencies.
    """
    import urllib.request
    import urllib.error

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    url = f"{api_url}/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Could not reach LLM at {url}. Is LM Studio / Ollama running?\n"
            f"Error: {exc}"
        ) from exc

    # Handle both standard OpenAI format and Ollama's slight variation
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"Unexpected LLM response format: {list(body.keys())}"
        ) from exc

    return text.strip()


def _clean_llm_output(text: str) -> str:
    """
    Remove common LLM preamble/postamble that some models add despite instructions.
    Examples: 'Here is the enhanced prompt:', leading quotes, markdown fences.
    """
    # Strip markdown code fences
    text = re.sub(r"^```[^\n]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())

    # Strip common preamble phrases (case-insensitive)
    preamble = re.compile(
        r"^(here(?:'s| is)(?: the)? (?:enhanced |expanded |improved )?prompt[:.]?\s*\n?|"
        r"enhanced prompt[:.]?\s*\n?|"
        r"improved prompt[:.]?\s*\n?)",
        re.IGNORECASE,
    )
    text = preamble.sub("", text.strip())

    # Strip surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]

    return text.strip()
