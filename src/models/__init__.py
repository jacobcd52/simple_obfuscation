"""Model wrappers and utilities."""

from importlib import import_module

__all__ = [
    "MindFace",
]

# Re-export for convenience
MindFace = import_module("src.models.mind_face").MindFace
