"""
Mesh components for GeoInvLab3D.
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from .base import Mesh3D
from .structured import StructuredMesh3D
from .unstructured import UnstructuredMesh3D
from .mesh_io import (
    load_mesh,
    read_mesh,
    save_mesh,
    write_mesh,
)

__all__ = [
    "Mesh3D",
    "StructuredMesh3D",
    "UnstructuredMesh3D",
    "load_mesh",
    "read_mesh",
    "save_mesh",
    "write_mesh",
]
