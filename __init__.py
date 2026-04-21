"""
Eric ERNIE-Image - ComfyUI custom node package
"""

from .nodes import (
    ErnieImageLoadModel,
    ErnieImageEncode,
    ErnieImageGenerate,
    ErnieImageUnload,
    ErniePromptRewriter,
    ErnieImageUltraGen,
)

NODE_CLASS_MAPPINGS = {
    "ErnieImageLoadModel":  ErnieImageLoadModel,
    "ErnieImageEncode":     ErnieImageEncode,
    "ErnieImageGenerate":   ErnieImageGenerate,
    "ErnieImageUnload":     ErnieImageUnload,
    "ErniePromptRewriter":  ErniePromptRewriter,
    "ErnieImageUltraGen":   ErnieImageUltraGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ErnieImageLoadModel":  "Eric ERNIE-Image Load Model",
    "ErnieImageEncode":     "Eric ERNIE-Image Encode",
    "ErnieImageGenerate":   "Eric ERNIE-Image Generate",
    "ErnieImageUnload":     "Eric ERNIE-Image Unload",
    "ErniePromptRewriter":  "Eric ERNIE Prompt Rewriter",
    "ErnieImageUltraGen":   "Eric ERNIE-Image UltraGen",
}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
