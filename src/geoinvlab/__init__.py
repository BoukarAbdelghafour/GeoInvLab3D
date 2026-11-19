"""
GeoInvLab3D.  

A modular Python framework for 3D geophysical
forward modelling and inversion.

MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from importlib import import_module


__version__ = "0.1.0"


_EXPORTS = {
    "DataSet": ("geoinvlab.core.data", "DataSet"),
    "ForwardOperator": (
        "geoinvlab.core.operators",
        "ForwardOperator",
    ),
    "InverseProblem": (
        "geoinvlab.core.inversion.problem",
        "InverseProblem",
    ),
    "L2Regularization": (
        "geoinvlab.core.inversion.regularization",
        "L2Regularization",
    ),
    "LeastSquaresObjective": (
        "geoinvlab.core.inversion.objective",
        "LeastSquaresObjective",
    ),
    "LinearLeastSquaresSolver": (
        "geoinvlab.core.inversion.optimizer",
        "LinearLeastSquaresSolver",
    ),
    "LinearOperator": (
        "geoinvlab.core.operators",
        "LinearOperator",
    ),
    "Model3D": ("geoinvlab.core.model", "Model3D"),
    "Project": ("geoinvlab.project", "Project"),
    "Regularization": (
        "geoinvlab.core.inversion.regularization",
        "Regularization",
    ),
    "StructuredMesh3D": (
        "geoinvlab.core.mesh",
        "StructuredMesh3D",
    ),
}


__all__ = [
    "DataSet",
    "ForwardOperator",
    "InverseProblem",
    "L2Regularization",
    "LeastSquaresObjective",
    "LinearLeastSquaresSolver",
    "LinearOperator",
    "Model3D",
    "Project",
    "Regularization",
    "StructuredMesh3D",
    "__version__",
]


def __getattr__(name: str):
    """Load public GeoInvLab3D components on demand."""

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
