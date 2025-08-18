"""Model wrappers and utilities."""

from importlib import import_module

__all__ = [
    "MaskFace",
]

# Re-export for convenience
MaskFace = import_module("src.models.mask_face").MaskFace
