"""
Inversion tools for GeoInvLab3D.

This package provides objective functions, regularization
operators, inverse problem definitions, optimization algorithms,
and inversion engines.
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from geoinvlab.core.inversion.objective import (
    LeastSquaresObjective,
)

from geoinvlab.core.inversion.optimizer import (
    LinearLeastSquaresSolver,
    GaussNewtonSolver,
    Optimizer,
    PositiveBoundedLeastSquaresSolver,
)

from geoinvlab.core.inversion.problem import (
    InverseProblem,
)

from geoinvlab.core.inversion.regularization import (
    L2Regularization,
    Regularization,
)

from geoinvlab.core.inversion.engine import (
    InversionEngine,
    InversionResult,
)


__all__ = [
    "LeastSquaresObjective",
    "LinearLeastSquaresSolver",
    "GaussNewtonSolver",
    "Optimizer",
    "PositiveBoundedLeastSquaresSolver",
    "InverseProblem",
    "Regularization",
    "L2Regularization",
    "InversionEngine",
    "InversionResult",
]
