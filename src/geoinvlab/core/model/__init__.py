"""
Physical model components and ASCII input/output utilities for GeoInvLab3D.
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from .model import Model3D
from .model_io import (
    load_model,
    read_model,
    save_model,
    write_model,
)

__all__ = [
    "Model3D",
    "load_model",
    "read_model",
    "save_model",
    "write_model",
]
