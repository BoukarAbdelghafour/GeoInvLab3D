"""
Core components for GeoInvLab3D.

Core components are imported lazily to avoid circular imports
between the core packages.
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "DataSet": ("geoinvlab.core.data", "DataSet"),
    "StructuredMesh3D": ("geoinvlab.core.mesh", "StructuredMesh3D"),
    "Model3D": ("geoinvlab.core.model", "Model3D"),
    "ForwardOperator": (
        "geoinvlab.core.operators",
        "ForwardOperator",
    ),
    "LinearOperator": (
        "geoinvlab.core.operators",
        "LinearOperator",
    ),
    "InverseProblem": (
        "geoinvlab.core.inversion.problem",
        "InverseProblem",
    ),
    "LeastSquaresObjective": (
        "geoinvlab.core.inversion.objective",
        "LeastSquaresObjective",
    ),
    "LinearLeastSquaresSolver": (
        "geoinvlab.core.inversion.optimizer",
        "LinearLeastSquaresSolver",
    ),
    "Regularization": (
        "geoinvlab.core.inversion.regularization",
        "Regularization",
    ),
    "L2Regularization": (
        "geoinvlab.core.inversion.regularization",
        "L2Regularization",
    ),
}


__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load public core components on demand."""

    if name not in _EXPORTS:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, attribute_name = _EXPORTS[name]

    module = import_module(module_name)

    value = getattr(module, attribute_name)

    globals()[name] = value

    return value


def __dir__():
    """Return available public attributes."""

    return sorted(set(globals()) | set(__all__))
