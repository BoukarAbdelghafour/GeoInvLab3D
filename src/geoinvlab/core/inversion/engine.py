"""
Inversion engine for GeoInvLab3D.

The inversion engine connects a Project/InverseProblem to an
inversion solver and produces an inversion result.
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from geoinvlab.core.inversion.problem import InverseProblem
from geoinvlab.core.inversion.optimizer import LinearLeastSquaresSolver
from geoinvlab.core.model import Model3D


@dataclass
class InversionResult:
    """Result produced by an inversion."""

    model: Model3D
    objective_value: float
    data_misfit: float
    regularization_value: float

    @property
    def values(self):
        """Return the inverted model values."""

        return self.model.values


class InversionEngine:
    """
    Main entry point for GeoInvLab3D inversion.

    The engine receives an InverseProblem, selects an appropriate
    solver, executes the inversion, and returns an InversionResult.
    """

    def __init__(
        self,
        solver: Any | None = None,
    ) -> None:
        """Create an inversion engine."""

        self.solver = (
            solver
            if solver is not None
            else LinearLeastSquaresSolver()
        )

    def solve(
        self,
        problem: InverseProblem,
    ) -> InversionResult:
        """
        Solve an inverse problem.

        Parameters
        ----------
        problem : InverseProblem
            Problem to solve.

        Returns
        -------
        InversionResult
            Result of the inversion.
        """

        if not isinstance(problem, InverseProblem):
            raise TypeError(
                "problem must be an InverseProblem instance."
            )

        # --------------------------------------------------------------
        # Solve the inverse problem.
        # --------------------------------------------------------------

        model = self.solver.solve(problem)

        # --------------------------------------------------------------
        # Evaluate the objective for the inverted model.
        #
        # Do not use problem.objective_value(), because that method
        # evaluates the original problem.model.
        # --------------------------------------------------------------

        objective_value = problem.objective.value(model)

        data_misfit = problem.objective.data_misfit(model)

        regularization_value = (
            problem.objective.regularization_value(model)
        )

        return InversionResult(
            model=model,
            objective_value=objective_value,
            data_misfit=data_misfit,
            regularization_value=regularization_value,
        )

    def run(
        self,
        project,
    ) -> InversionResult:
        """
        Run inversion directly from a Project.

        The project must contain an inverse problem.
        """

        if project.inverse_problem is None:
            raise ValueError(
                "Project does not contain an inverse problem."
            )

        result = self.solve(
            project.inverse_problem
        )

        project.set_result(result)

        return result

