"""
Optimization algorithms for geophysical inversion.

This module provides solvers for linear and nonlinear inverse
problems.

Available solvers
-----------------

LinearLeastSquaresSolver
    Weighted linear least-squares inversion with optional
    Tikhonov regularization.

GaussNewtonSolver
    Nonlinear Gauss-Newton inversion with optional Tikhonov
    regularization and safeguarded model updates.

Optimizer
    Generic public optimizer interface.

PositiveBoundedLeastSquaresSolver
    Weighted least-squares solver with non-negative model bounds.

MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator

from geoinvlab.core.inversion.problem import InverseProblem
from geoinvlab.core.model import Model3D


def _data_residual(
    problem: InverseProblem,
    predicted: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Return data residuals using the forward operator convention.

    Operators may provide ``data_residual(predicted, observed)`` to
    implement problem-specific residuals, such as circular MT phase
    differences. Other operators retain ordinary subtraction.
    """
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    observed = np.asarray(observed, dtype=float).reshape(-1)

    if predicted.shape != observed.shape:
        raise ValueError(
            "Predicted and observed data must have identical shapes."
        )

    residual_method = getattr(
        problem.operator,
        "data_residual",
        None,
    )

    if callable(residual_method):
        residual = np.asarray(
            residual_method(predicted, observed),
            dtype=float,
        ).reshape(-1)
    else:
        residual = predicted - observed

    if residual.shape != observed.shape:
        raise ValueError(
            "The operator data-residual method returned an invalid shape."
        )

    if not np.all(np.isfinite(residual)):
        raise ValueError(
            "Data residuals contain NaN or infinite values."
        )

    return residual


def _objective_value_from_prediction(
    problem: InverseProblem,
    model: Model3D,
    predicted: np.ndarray,
) -> float:
    """Evaluate the objective without repeating the forward calculation."""
    value_method = getattr(
        problem.objective,
        "value_from_prediction",
        None,
    )

    if callable(value_method):
        return float(value_method(model, predicted))

    return float(problem.objective.value(model))


class _TargetMisfitMonitor:
    """Shared, non-terminating target-misfit monitor for built-in solvers."""

    def _initialize_target_misfit(self, target_misfit: float) -> None:
        if not np.isfinite(target_misfit) or target_misfit <= 0.0:
            raise ValueError(
                "target_misfit must be finite and strictly positive."
            )

        self.target_misfit = float(target_misfit)
        self.target_misfit_reached = False

    def _check_target_misfit(
        self,
        problem: InverseProblem,
        model: Model3D,
        iteration: int | None = None,
        predicted: np.ndarray | None = None,
    ) -> float:
        """Report first attainment of normalized RMS without stopping."""
        if self.target_misfit_reached:
            return float("nan")

        observed = np.asarray(problem.data.values, dtype=float)
        if predicted is None:
            predicted = np.asarray(
                problem.operator.predict(model),
                dtype=float,
            )
        else:
            predicted = np.asarray(predicted, dtype=float)

        weights = np.asarray(problem.data.weights(), dtype=float)

        if predicted.shape != observed.shape:
            raise ValueError(
                "Forward prediction shape does not match observed data "
                "while evaluating target misfit."
            )

        if weights.shape != observed.shape:
            raise ValueError(
                "Data weights shape does not match observed data "
                "while evaluating target misfit."
            )

        if not (
            np.all(np.isfinite(observed))
            and np.all(np.isfinite(predicted))
            and np.all(np.isfinite(weights))
        ):
            raise ValueError(
                "Non-finite values encountered while evaluating target misfit."
            )

        residual = _data_residual(
            problem,
            predicted,
            observed,
        )
        normalized_rms = float(
            np.sqrt(np.mean((residual * weights) ** 2))
        )

        if normalized_rms <= self.target_misfit:
            self.target_misfit_reached = True

            iteration_text = (
                f" at iteration {iteration}"
                if iteration is not None
                else " after the direct solve"
            )

            print(
                f"Target misfit reached{iteration_text}: "
                f"NORM RMS = {normalized_rms:.6e} <= "
                f"{self.target_misfit:.6e}. "
                "Continuing optimization until numerical convergence "
                "(if applicable).",
                flush=True,
            )

        return normalized_rms


# ============================================================================
# OBJECTIVE FUNCTION COMPONENTS
# ============================================================================


def _calculate_objective_components(
    problem: InverseProblem,
    model: Model3D,
    observed: np.ndarray,
    weights: np.ndarray,
    regularization_parameter: float,
    predicted: np.ndarray | None = None,
) -> tuple[float, float, float, float, float]:
    """Calculate RMS metrics and the decomposed Tikhonov objective.

    phi_d = ||W(F(m)-d)||^2
    phi_m = lambda * ||L(m-m_ref)||^2
    phi   = phi_d + phi_m

    The returned phi_m already includes the regularization parameter.
    """
    if predicted is None:
        predicted = np.asarray(
            problem.operator.predict(model),
            dtype=float,
        )
    else:
        predicted = np.asarray(predicted, dtype=float)

    observed = np.asarray(observed, dtype=float)
    weights = np.asarray(weights, dtype=float)

    residual = _data_residual(
        problem,
        predicted,
        observed,
    )
    weighted_residual = residual * weights

    data_rms = float(np.sqrt(np.mean(residual**2)))
    normalized_rms = float(np.sqrt(np.mean(weighted_residual**2)))
    phi_d = float(np.dot(weighted_residual, weighted_residual))

    phi_m = 0.0
    regularization = problem.regularization
    lam = float(regularization_parameter)

    if regularization is not None and lam > 0.0:
        reference = LinearLeastSquaresSolver._get_reference_values(
            regularization,
            model,
        )

        difference = np.asarray(model.values, dtype=float) - reference

        from geoinvlab.core.inversion.regularization import L2Regularization

        if isinstance(regularization, L2Regularization):
            transformed = difference
        else:
            matrix = np.asarray(
                regularization.matrix(model),
                dtype=float,
            )
            transformed = matrix @ difference

        phi_m = lam * float(np.dot(transformed, transformed))

    return (
        data_rms,
        normalized_rms,
        phi_d,
        phi_m,
        float(phi_d + phi_m),
    )


# ============================================================================
# LINEAR LEAST-SQUARES SOLVER
# ============================================================================


class LinearLeastSquaresSolver(_TargetMisfitMonitor):
    """
    Weighted least-squares solver for linear inverse problems.

    This is a direct, non-iterative method; target misfit is therefore
    evaluated once after the direct solution has been computed.
    """

    def __init__(self, target_misfit: float = 1.0) -> None:
        self._initialize_target_misfit(target_misfit)

    def solve(self, problem: InverseProblem) -> Model3D:
        """Solve the linear inverse problem."""

        if not isinstance(problem, InverseProblem):
            raise TypeError(
                "problem must be an InverseProblem instance."
            )

        import time

        solve_start_time = time.perf_counter()
        initial_model = problem.model
        self.target_misfit_reached = False

        n_parameters = initial_model.n_parameters
        n_data = problem.data.n_data

        # Jacobian / forward matrix
        jacobian = np.asarray(
            problem.operator.jacobian(initial_model),
            dtype=float,
        )

        expected_shape = (n_data, n_parameters)

        if jacobian.ndim != 2:
            raise ValueError(
                "Jacobian must be a two-dimensional array."
            )

        if jacobian.shape != expected_shape:
            raise ValueError(
                "Jacobian shape does not match the data and model "
                f"dimensions. Expected {expected_shape}, got {jacobian.shape}."
            )

        if not np.all(np.isfinite(jacobian)):
            raise ValueError(
                "Jacobian contains NaN or infinite values."
            )

        # Observed data
        data = problem.data
        observed_values = np.asarray(data.values, dtype=float)

        if observed_values.shape != (n_data,):
            raise ValueError(
                f"Data values must have shape ({n_data},)."
            )

        if not np.all(np.isfinite(observed_values)):
            raise ValueError(
                "Observed data contain NaN or infinite values."
            )

        # Data weights
        weights = np.asarray(data.weights(), dtype=float)

        if weights.shape != (n_data,):
            raise ValueError(
                f"Data weights must have shape ({n_data},)."
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "Data weights contain NaN or infinite values."
            )

        if np.any(weights < 0.0):
            raise ValueError(
                "Data weights must be non-negative."
            )

        weighted_jacobian = jacobian * weights[:, np.newaxis]
        weighted_data = observed_values * weights

        # Regularization
        regularization = problem.regularization
        regularization_parameter = float(
            problem.regularization_parameter
        )

        if (
            not np.isfinite(regularization_parameter)
            or regularization_parameter < 0.0
        ):
            raise ValueError(
                "Regularization parameter must be finite and non-negative."
            )

        # Unregularized least squares
        if (
            regularization is None
            or regularization_parameter == 0.0
        ):
            model_values, _, _, _ = np.linalg.lstsq(
                weighted_jacobian,
                weighted_data,
                rcond=None,
            )

        # Regularized Tikhonov inversion
        else:
            regularization_matrix = np.asarray(
                regularization.matrix(initial_model),
                dtype=float,
            )

            if regularization_matrix.ndim != 2:
                raise ValueError(
                    "Regularization matrix must be two-dimensional."
                )

            if regularization_matrix.shape[1] != n_parameters:
                raise ValueError(
                    "Regularization matrix must have "
                    f"{n_parameters} columns. "
                    f"Got {regularization_matrix.shape[1]}."
                )

            if not np.all(np.isfinite(regularization_matrix)):
                raise ValueError(
                    "Regularization matrix contains "
                    "NaN or infinite values."
                )

            reference_values = self._get_reference_values(
                regularization,
                initial_model,
            )

            if reference_values.shape != (n_parameters,):
                raise ValueError(
                    "Reference model must have shape "
                    f"({n_parameters},). Got {reference_values.shape}."
                )

            if not np.all(np.isfinite(reference_values)):
                raise ValueError(
                    "Reference model contains NaN or infinite values."
                )

            sqrt_lambda = np.sqrt(regularization_parameter)

            augmented_jacobian = np.vstack(
                (
                    weighted_jacobian,
                    sqrt_lambda * regularization_matrix,
                )
            )

            augmented_data = np.concatenate(
                (
                    weighted_data,
                    sqrt_lambda * (
                        regularization_matrix @ reference_values
                    ),
                )
            )

            model_values, _, _, _ = np.linalg.lstsq(
                augmented_jacobian,
                augmented_data,
                rcond=None,
            )

        model_values = np.asarray(model_values, dtype=float)

        if model_values.shape != (n_parameters,):
            raise ValueError(
                "Solver returned an invalid model shape. "
                f"Expected ({n_parameters},), got {model_values.shape}."
            )

        if not np.all(np.isfinite(model_values)):
            raise ValueError(
                "Inversion produced NaN or infinite model parameters."
            )

        final_model = Model3D(
            mesh=initial_model.mesh,
            values=model_values,
            name=initial_model.name,
        )

        self.target_misfit_reached = False

        predicted = np.asarray(
            problem.operator.predict(final_model),
            dtype=float,
        )

        (
            data_rms,
            normalized_rms,
            phi_d,
            phi_m,
            phi,
        ) = _calculate_objective_components(
            problem,
            final_model,
            observed_values,
            weights,
            regularization_parameter,
            predicted=predicted,
        )

        print("===== LINEAR LEAST-SQUARES INVERSION SUMMARY =====")
        print(
            "ITER   DATA RMS       NORM RMS       PHI_D           PHI_M"
            "           PHI(M)          MODEL MIN      MODEL MAX      TIME(s)"
        )
        print("-" * 140)

        print(
            f"{'DIRECT':5s} "
            f"{data_rms:13.5e} "
            f"{normalized_rms:13.5e} "
            f"{phi_d:13.5e} "
            f"{phi_m:13.5e} "
            f"{phi:13.5e} "
            f"{float(model_values.min()):13.5e} "
            f"{float(model_values.max()):13.5e} "
            f"{time.perf_counter() - solve_start_time:9.2f}"
        )

        self._check_target_misfit(
            problem,
            final_model,
            iteration=1,
            predicted=predicted,
        )

        return final_model

    @staticmethod
    def _get_reference_values(
        regularization,
        model: Model3D,
    ) -> np.ndarray:
        """Retrieve the reference model values."""

        reference_values_method = getattr(
            regularization,
            "reference_values",
            None,
        )

        if callable(reference_values_method):
            values = reference_values_method(model)
            return np.asarray(values, dtype=float)

        reference_model = getattr(
            regularization,
            "reference_model",
            None,
        )

        if reference_model is None:
            return np.zeros(model.n_parameters, dtype=float)

        if not isinstance(reference_model, Model3D):
            raise TypeError(
                "regularization.reference_model must be a Model3D instance."
            )

        if reference_model.n_parameters != model.n_parameters:
            raise ValueError(
                "Reference model must have the same number of parameters "
                "as the inversion model."
            )

        if reference_model.mesh is not model.mesh:
            raise ValueError(
                "Reference model must use the same mesh as the inversion model."
            )

        return np.asarray(reference_model.values, dtype=float)


# ============================================================================
# GAUSS-NEWTON SOLVER
# ============================================================================


class GaussNewtonSolver(_TargetMisfitMonitor):
    """
    Nonlinear Gauss-Newton solver with safeguarded model updates.

    For positive MT parameterizations such as resistivity or
    conductivity, the solver guarantees that every accepted trial
    model remains strictly positive.

    A backtracking line search is used to avoid accepting an update
    that increases the objective function.

    If history_file is specified, the solver writes the initial model
    and every accepted iteration to an ASCII history file.
    """

    def __init__(
        self,
        max_iterations: int = 20,
        tolerance: float = 1e-6,
        damping: float = 0.0,
        verbose: bool = False,
        target_misfit: float = 1.0,
        history_file: str | Path | None = None,
    ) -> None:

        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1."
            )

        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "tolerance must be finite and strictly positive."
            )

        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError(
                "damping must be finite and non-negative."
            )

        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.damping = float(damping)
        self.verbose = bool(verbose)

        self.history_file = (
            Path(history_file) if history_file is not None else None
        )

        self.iteration_history: list[dict] = []
        self.converged = False
        self.status = "not_started"

        self._initialize_target_misfit(target_misfit)

    # ==================================================================
    # HISTORY FILE
    # ==================================================================

    def _initialize_history_file(self) -> None:
        """Create the history file and write its column description."""

        if self.history_file is None:
            return

        self.history_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.history_file.open(
            "w",
            encoding="ascii",
        ) as stream:
            stream.write(
                "# GeoInvLab3D Gauss-Newton inversion history\n"
            )
            stream.write(
                "# iteration data_rms normalized_rms phi_d phi_m phi "
                "model_min model_max elapsed_s alpha relative_update\n"
            )

    def _write_history_record(self, record: dict) -> None:
        """Append one iteration record to the ASCII history file."""

        if self.history_file is None:
            return

        with self.history_file.open(
            "a",
            encoding="ascii",
        ) as stream:
            stream.write(
                f"{int(record['iteration']):d} "
                f"{float(record['data_rms']):.12e} "
                f"{float(record['normalized_rms']):.12e} "
                f"{float(record['phi_d']):.12e} "
                f"{float(record['phi_m']):.12e} "
                f"{float(record['phi']):.12e} "
                f"{float(record['model_min']):.12e} "
                f"{float(record['model_max']):.12e} "
                f"{float(record['elapsed_s']):.12e} "
                f"{float(record['alpha']):.12e} "
                f"{float(record['relative_update']):.12e}\n"
            )

    # ==================================================================
    # POSITIVITY DETECTION
    # ==================================================================

    @staticmethod
    def _requires_positive_model(
        problem: InverseProblem,
        model: Model3D,
    ) -> bool:
        """Determine whether the current inversion uses a positive parameterization."""

        parameterization_method = getattr(
            problem.operator,
            "_model_parameterization",
            None,
        )

        if not callable(parameterization_method):
            return False

        try:
            parameterization = parameterization_method(model)
        except (TypeError, ValueError):
            return False

        parameterization = str(parameterization).lower()

        return parameterization in {
            "resistivity",
            "rho",
            "res",
            "conductivity",
            "sigma",
            "cond",
        }

    # ==================================================================
    # SAFE MODEL CONSTRUCTION
    # ==================================================================

    @classmethod
    def _build_trial_model(
        cls,
        problem: InverseProblem,
        current_model: Model3D,
        delta_model: np.ndarray,
        alpha: float,
    ) -> Model3D | None:
        """Build a trial model, enforcing positivity when required."""

        current_values = np.asarray(
            current_model.values,
            dtype=float,
        )

        trial_values = current_values + alpha * delta_model

        if not np.all(np.isfinite(trial_values)):
            return None

        if cls._requires_positive_model(problem, current_model):
            if np.any(trial_values <= 0.0):
                return None

        try:
            return Model3D(
                mesh=current_model.mesh,
                values=trial_values,
                name=current_model.name,
            )
        except (TypeError, ValueError):
            return None

    # ==================================================================
    # SOLVE
    # ==================================================================

    def solve(self, problem: InverseProblem) -> Model3D:
        """Solve a nonlinear inverse problem using Gauss-Newton."""

        if not isinstance(problem, InverseProblem):
            raise TypeError(
                "problem must be an InverseProblem instance."
            )

        import time

        current_model = problem.model
        self.target_misfit_reached = False
        self.converged = False
        self.status = "running"

        n_parameters = current_model.n_parameters
        n_data = problem.data.n_data

        observed_values = np.asarray(
            problem.data.values,
            dtype=float,
        )

        if observed_values.shape != (n_data,):
            raise ValueError(
                f"Observed data must have shape ({n_data},)."
            )

        if not np.all(np.isfinite(observed_values)):
            raise ValueError(
                "Observed data contain NaN or infinite values."
            )

        weights = np.asarray(
            problem.data.weights(),
            dtype=float,
        )

        if weights.shape != (n_data,):
            raise ValueError(
                f"Data weights must have shape ({n_data},)."
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "Data weights contain NaN or infinite values."
            )

        if np.any(weights < 0.0):
            raise ValueError(
                "Data weights must be non-negative."
            )

        regularization = problem.regularization

        regularization_parameter = float(
            problem.regularization_parameter
        )

        if (
            not np.isfinite(regularization_parameter)
            or regularization_parameter < 0.0
        ):
            raise ValueError(
                "Regularization parameter must be finite and non-negative."
            )

        max_line_search_steps = 20

        iteration_start_time = time.perf_counter()
        self.iteration_history = []

        # --------------------------------------------------------------
        # Initialize history file and record the initial model.
        # --------------------------------------------------------------

        self._initialize_history_file()

        initial_prediction = np.asarray(
            problem.operator.predict(current_model),
            dtype=float,
        )

        if initial_prediction.shape != (n_data,):
            raise ValueError(
                "Initial forward prediction has invalid shape. "
                f"Expected ({n_data},), got {initial_prediction.shape}."
            )

        if not np.all(np.isfinite(initial_prediction)):
            raise ValueError(
                "Initial forward prediction contains NaN or infinite values."
            )

        current_prediction = initial_prediction

        (
            initial_data_rms,
            initial_normalized_rms,
            initial_phi_d,
            initial_phi_m,
            initial_phi,
        ) = _calculate_objective_components(
            problem,
            current_model,
            observed_values,
            weights,
            regularization_parameter,
            predicted=initial_prediction,
        )

        initial_values = np.asarray(
            current_model.values,
            dtype=float,
        )

        initial_record = {
            "iteration": 0,
            "data_rms": initial_data_rms,
            "normalized_rms": initial_normalized_rms,
            "phi_d": initial_phi_d,
            "phi_m": initial_phi_m,
            "phi": initial_phi,
            "cost": initial_phi,
            "optimality": np.nan,
            "model_min": float(np.min(initial_values)),
            "model_max": float(np.max(initial_values)),
            "elapsed_s": 0.0,
            "regularization_lambda": regularization_parameter,
            "tolerance": self.tolerance,
            "target_misfit": self.target_misfit,
            "max_iterations": self.max_iterations,
            "damping": self.damping,
            "alpha": 0.0,
            "relative_update": 0.0,
        }

        self.iteration_history.append(initial_record)
        self._write_history_record(initial_record)

        if self.verbose:
            print("===== GAUSS-NEWTON ITERATION HISTORY =====")
            print(
                "ITER   DATA RMS       NORM RMS       PHI_D"
                "           PHI_M           PHI(M)"
                "          MODEL MIN      MODEL MAX      TIME(s)"
            )
            print("-" * 140)
            print(
                f"{0:4d} "
                f"{initial_data_rms:13.5e} "
                f"{initial_normalized_rms:13.5e} "
                f"{initial_phi_d:13.5e} "
                f"{initial_phi_m:13.5e} "
                f"{initial_phi:13.5e} "
                f"{float(np.min(initial_values)):13.5e} "
                f"{float(np.max(initial_values)):13.5e} "
                f"{0.0:9.2f}",
                flush=True,
            )

        # --------------------------------------------------------------
        # Main Gauss-Newton loop.
        # --------------------------------------------------------------

        for iteration in range(self.max_iterations):

            # Reuse the current prediction from the initial evaluation or
            # the previously accepted line-search trial.
            predicted = current_prediction

            # The objective residual is F(m) - d. The increment system
            # therefore uses its negative on the right-hand side.
            residual = _data_residual(
                problem,
                predicted,
                observed_values,
            )
            weighted_residual = residual * weights

            # Jacobian
            jacobian = np.asarray(
                problem.operator.jacobian(current_model),
                dtype=float,
            )

            expected_jacobian_shape = (n_data, n_parameters)

            if jacobian.ndim != 2:
                raise ValueError(
                    "Jacobian must be a two-dimensional array."
                )

            if jacobian.shape != expected_jacobian_shape:
                raise ValueError(
                    "Jacobian shape does not match the data and model "
                    f"dimensions. Expected {expected_jacobian_shape}, "
                    f"got {jacobian.shape}."
                )

            if not np.all(np.isfinite(jacobian)):
                raise ValueError(
                    "Jacobian contains NaN or infinite values."
                )

            weighted_jacobian = (
                jacobian * weights[:, np.newaxis]
            )

            # Build increment system
            system_matrix = weighted_jacobian
            system_rhs = -weighted_residual

            # Tikhonov regularization
            if (
                regularization is not None
                and regularization_parameter > 0.0
            ):
                regularization_matrix = np.asarray(
                    regularization.matrix(current_model),
                    dtype=float,
                )

                if regularization_matrix.ndim != 2:
                    raise ValueError(
                        "Regularization matrix must be two-dimensional."
                    )

                if regularization_matrix.shape[1] != n_parameters:
                    raise ValueError(
                        "Regularization matrix must have "
                        f"{n_parameters} columns. "
                        f"Got {regularization_matrix.shape[1]}."
                    )

                if not np.all(np.isfinite(regularization_matrix)):
                    raise ValueError(
                        "Regularization matrix contains "
                        "NaN or infinite values."
                    )

                reference_values = (
                    LinearLeastSquaresSolver._get_reference_values(
                        regularization,
                        current_model,
                    )
                )

                if reference_values.shape != (n_parameters,):
                    raise ValueError(
                        "Reference model must have shape "
                        f"({n_parameters},). Got {reference_values.shape}."
                    )

                if not np.all(np.isfinite(reference_values)):
                    raise ValueError(
                        "Reference model contains NaN or infinite values."
                    )

                sqrt_lambda = np.sqrt(regularization_parameter)

                regularization_matrix_increment = (
                    sqrt_lambda * regularization_matrix
                )

                regularization_rhs = sqrt_lambda * (
                    regularization_matrix
                    @ (reference_values - current_model.values)
                )

                system_matrix = np.vstack(
                    (
                        system_matrix,
                        regularization_matrix_increment,
                    )
                )

                system_rhs = np.concatenate(
                    (
                        system_rhs,
                        regularization_rhs,
                    )
                )

            # Optional damping
            if self.damping > 0.0:
                sqrt_damping = np.sqrt(self.damping)

                system_matrix = np.vstack(
                    (
                        system_matrix,
                        sqrt_damping * np.eye(
                            n_parameters,
                            dtype=float,
                        ),
                    )
                )

                system_rhs = np.concatenate(
                    (
                        system_rhs,
                        np.zeros(n_parameters, dtype=float),
                    )
                )

            # Solve increment
            delta_model, _, _, _ = np.linalg.lstsq(
                system_matrix,
                system_rhs,
                rcond=None,
            )

            delta_model = np.asarray(delta_model, dtype=float)

            if delta_model.shape != (n_parameters,):
                raise ValueError(
                    "Gauss-Newton increment has invalid shape. "
                    f"Expected ({n_parameters},), got {delta_model.shape}."
                )

            if not np.all(np.isfinite(delta_model)):
                raise ValueError(
                    "Gauss-Newton increment contains NaN or infinite values."
                )

            # Current objective
            objective_before = _objective_value_from_prediction(
                problem,
                current_model,
                predicted,
            )

            if not np.isfinite(objective_before):
                raise ValueError(
                    "Current objective is not finite."
                )

            # Backtracking line search
            alpha = 1.0
            accepted_model = None
            accepted_prediction = None
            objective_after = objective_before

            for _ in range(max_line_search_steps):
                trial_model = self._build_trial_model(
                    problem=problem,
                    current_model=current_model,
                    delta_model=delta_model,
                    alpha=alpha,
                )

                if trial_model is None:
                    alpha *= 0.5
                    continue

                trial_prediction = np.asarray(
                    problem.operator.predict(trial_model),
                    dtype=float,
                )

                if (
                    trial_prediction.shape != (n_data,)
                    or not np.all(np.isfinite(trial_prediction))
                ):
                    alpha *= 0.5
                    continue

                trial_objective = _objective_value_from_prediction(
                    problem,
                    trial_model,
                    trial_prediction,
                )

                if (
                    np.isfinite(trial_objective)
                    and trial_objective <= objective_before
                ):
                    accepted_model = trial_model
                    accepted_prediction = trial_prediction
                    objective_after = trial_objective
                    break

                alpha *= 0.5

            # No acceptable step
            if accepted_model is None:
                self.status = "line_search_failed"
                self.converged = False
                if self.verbose:
                    print(
                        f"Gauss-Newton iteration {iteration + 1:3d}: "
                        "no acceptable update; stopping."
                    )
                break

            # Accepted update
            current_values = np.asarray(
                current_model.values,
                dtype=float,
            )

            accepted_values = np.asarray(
                accepted_model.values,
                dtype=float,
            )

            actual_update = accepted_values - current_values

            model_norm = float(np.linalg.norm(current_values))
            update_norm = float(np.linalg.norm(actual_update))

            relative_update = (
                update_norm / max(model_norm, 1.0)
            )

            if self.verbose:
                print(
                    f"Gauss-Newton iteration {iteration + 1:3d}: "
                    f"objective={objective_after:.6e}, "
                    f"update={relative_update:.6e}, "
                    f"alpha={alpha:.3e}"
                )

            current_model = accepted_model
            predicted_after = accepted_prediction
            current_prediction = predicted_after

            (
                data_rms_after,
                normalized_rms_after,
                phi_d_after,
                phi_m_after,
                phi_after,
            ) = _calculate_objective_components(
                problem,
                current_model,
                observed_values,
                weights,
                regularization_parameter,
                predicted=predicted_after,
            )

            elapsed = time.perf_counter() - iteration_start_time
            model_min = float(np.min(accepted_values))
            model_max = float(np.max(accepted_values))

            iteration_record = {
                "iteration": iteration + 1,
                "data_rms": data_rms_after,
                "normalized_rms": normalized_rms_after,
                "phi_d": phi_d_after,
                "phi_m": phi_m_after,
                "phi": phi_after,
                "cost": phi_after,
                "optimality": np.nan,
                "model_min": model_min,
                "model_max": model_max,
                "elapsed_s": elapsed,
                "regularization_lambda": regularization_parameter,
                "tolerance": self.tolerance,
                "target_misfit": self.target_misfit,
                "max_iterations": self.max_iterations,
                "damping": self.damping,
                "alpha": alpha,
                "relative_update": relative_update,
            }

            self.iteration_history.append(iteration_record)

            # Write the accepted iteration immediately.
            self._write_history_record(iteration_record)

            if self.verbose:
                print(
                    f"{iteration + 1:4d} "
                    f"{data_rms_after:13.5e} "
                    f"{normalized_rms_after:13.5e} "
                    f"{phi_d_after:13.5e} "
                    f"{phi_m_after:13.5e} "
                    f"{phi_after:13.5e} "
                    f"{model_min:13.5e} "
                    f"{model_max:13.5e} "
                    f"{elapsed:9.2f}",
                    flush=True,
                )

            self._check_target_misfit(
                problem,
                current_model,
                iteration=iteration + 1,
                predicted=predicted_after,
            )

            # Convergence requires both a sufficiently small model update
            # and attainment of the requested normalized data misfit.
            if relative_update < self.tolerance:
                if normalized_rms_after <= self.target_misfit:
                    self.converged = True
                    self.status = "converged"
                    if self.verbose:
                        print(
                            "Gauss-Newton convergence reached: "
                            f"relative update = {relative_update:.6e}, "
                            f"normalized RMS = {normalized_rms_after:.6e}."
                        )
                else:
                    self.converged = False
                    self.status = "stagnated"
                    if self.verbose:
                        print(
                            "Gauss-Newton stagnation: the relative model "
                            f"update is small ({relative_update:.6e}), but "
                            "the target normalized RMS was not reached "
                            f"({normalized_rms_after:.6e} > "
                            f"{self.target_misfit:.6e})."
                        )
                break

        if self.status == "running":
            self.status = "maximum_iterations"
            self.converged = False
            if self.verbose:
                print(
                    "Gauss-Newton stopped after reaching the maximum "
                    f"number of iterations ({self.max_iterations})."
                )

        return current_model


# ============================================================================
# GENERIC OPTIMIZER
# ============================================================================


class Optimizer(_TargetMisfitMonitor):
    """Public optimizer interface for GeoInvLab3D."""

    def __init__(
        self,
        max_iterations: int = 20,
        tolerance: float = 1e-6,
        damping: float = 0.0,
        verbose: bool = False,
        solver=None,
        target_misfit: float = 1.0,
    ) -> None:

        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1."
            )

        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "tolerance must be finite and strictly positive."
            )

        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError(
                "damping must be finite and non-negative."
            )

        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.damping = float(damping)
        self.verbose = bool(verbose)
        self.solver = solver

        self._initialize_target_misfit(target_misfit)

    def solve(self, problem: InverseProblem) -> Model3D:
        """Solve an inverse problem."""

        if not isinstance(problem, InverseProblem):
            raise TypeError(
                "problem must be an InverseProblem instance."
            )

        if self.solver is not None:
            solve_method = getattr(
                self.solver,
                "solve",
                None,
            )

            if not callable(solve_method):
                raise TypeError(
                    "solver must provide a callable solve(problem) method."
                )

            if hasattr(self.solver, "target_misfit"):
                self.solver.target_misfit = self.target_misfit

            if hasattr(self.solver, "target_misfit_reached"):
                self.solver.target_misfit_reached = False

            result = solve_method(problem)

            if not hasattr(self.solver, "target_misfit_reached"):
                self._check_target_misfit(problem, result)

            return result

        solver = GaussNewtonSolver(
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            damping=self.damping,
            verbose=self.verbose,
            target_misfit=self.target_misfit,
        )
        self.solver = solver

        return solver.solve(problem)


# ============================================================================
# POSITIVE BOUNDED LEAST-SQUARES SOLVER
# ============================================================================


class PositiveBoundedLeastSquaresSolver:
    """
    Bounded weighted least-squares solver for positive physical models.

    Solves

        min_m ||W(G m - d)||^2
              + lambda ||L(m - m_ref)||^2

    subject to m >= 0.

    Diagonal column scaling is used to improve conditioning. The
    optimization is performed in scaled parameters, while the returned
    model is expressed in physical units.

    The solver uses scipy.optimize.least_squares with the trust-region
    reflective algorithm so that an iteration callback can report and
    save diagnostics while the inversion is running.
    """

    def __init__(
        self,
        tolerance: float = 1e-8,
        max_iterations: int = 200,
        lsmr_tolerance: str | float = "auto",
        verbose: bool = False,
        history_file: str | None = None,
        model_history_file: str | None = None,
        parameters_file: str | None = None,
        callback=None,
        target_misfit: float = 1.0,
    ) -> None:

        if tolerance <= 0.0:
            raise ValueError(
                "tolerance must be strictly positive."
            )

        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1."
            )

        if not np.isfinite(target_misfit) or target_misfit <= 0.0:
            raise ValueError(
                "target_misfit must be finite and strictly positive."
            )

        self.tolerance = float(tolerance)
        self.target_misfit = float(target_misfit)
        self.target_misfit_reached = False
        self.max_iterations = int(max_iterations)
        self.lsmr_tolerance = lsmr_tolerance
        self.verbose = bool(verbose)
        self.history_file = history_file
        self.model_history_file = model_history_file
        self.parameters_file = parameters_file
        self.callback = callback
        self.iteration_history = []
        self.result = None
        self.converged = False
        self.status = "not_started"

    @staticmethod
    def _write_parameters(path, values: dict) -> None:
        """Write solver parameters to an ASCII file."""

        if path is None:
            return

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="ascii") as stream:
            stream.write(
                "# GeoInvLab3D positive bounded inversion parameters\n"
            )
            stream.write("# KEY VALUE\n")

            for key, value in values.items():
                stream.write(f"{key} {value}\n")

    @staticmethod
    def _initialize_model_history(path, model: Model3D) -> None:
        """Create the ASCII model-history file and save iteration zero."""

        if path is None:
            return

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        centers = np.asarray(model.mesh.cell_centers(), dtype=float)
        values = np.asarray(model.values, dtype=float).reshape(-1)

        if centers.shape != (values.size, 3):
            raise ValueError(
                "Mesh cell centers do not match the model parameter count."
            )

        with output_path.open("w", encoding="ascii") as stream:
            stream.write(
                "# GeoInvLab3D positive bounded magnetic model history\n"
            )
            stream.write(
                "# One row per cell and iteration; cell_index is zero-based.\n"
            )
            stream.write(
                "# iteration cell_index x_center_m y_center_m z_center_m "
                "susceptibility_SI\n"
            )

        PositiveBoundedLeastSquaresSolver._append_model_history(
            output_path,
            iteration=0,
            model_values=values,
            centers=centers,
        )

    @staticmethod
    def _append_model_history(
        path,
        iteration: int,
        model_values: np.ndarray,
        centers: np.ndarray,
    ) -> None:
        """Append all cell susceptibilities for one iteration."""

        if path is None:
            return

        values = np.asarray(model_values, dtype=float).reshape(-1)
        centers = np.asarray(centers, dtype=float)

        if centers.shape != (values.size, 3):
            raise ValueError(
                "Model-history coordinates and values have incompatible shapes."
            )

        table = np.column_stack(
            (
                np.full(values.size, int(iteration), dtype=int),
                np.arange(values.size, dtype=int),
                centers,
                values,
            )
        )

        with Path(path).open("a", encoding="ascii") as stream:
            np.savetxt(
                stream,
                table,
                fmt=("%d", "%d", "%.10e", "%.10e", "%.10e", "%.12e"),
            )

    def solve(self, problem: InverseProblem) -> Model3D:
        """Solve the bounded linear inverse problem and retain the last iterate."""

        import time

        from scipy.optimize import least_squares

        if not isinstance(problem, InverseProblem):
            raise TypeError(
                "problem must be an InverseProblem instance."
            )

        initial_model = problem.model
        n_parameters = initial_model.n_parameters
        n_data = problem.data.n_data

        observed = np.asarray(problem.data.values, dtype=float)

        if observed.shape != (n_data,):
            raise ValueError(
                f"Observed data must have shape ({n_data},)."
            )

        if not np.all(np.isfinite(observed)):
            raise ValueError(
                "Observed data contain non-finite values."
            )

        weights = np.asarray(problem.data.weights(), dtype=float)

        if weights.shape != (n_data,):
            raise ValueError(
                f"Data weights must have shape ({n_data},)."
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "Data weights contain non-finite values."
            )

        if np.any(weights < 0.0):
            raise ValueError(
                "Data weights must be non-negative."
            )

        print("Building sensitivity matrix...")

        jacobian = np.asarray(
            problem.operator.jacobian(initial_model),
            dtype=float,
        )

        expected_shape = (n_data, n_parameters)

        if jacobian.shape != expected_shape:
            raise ValueError(
                f"Jacobian must have shape {expected_shape}; "
                f"got {jacobian.shape}."
            )

        if not np.all(np.isfinite(jacobian)):
            raise ValueError(
                "Jacobian contains non-finite values."
            )

        weighted_jacobian = jacobian * weights[:, np.newaxis]
        weighted_data = observed * weights

        column_norms = np.linalg.norm(weighted_jacobian, axis=0)

        if self.verbose:
            print("\n===== JACOBIAN DIAGNOSTICS =====")
            print(f"Minimum column norm: {column_norms.min():.6e}")
            print(f"Maximum column norm: {column_norms.max():.6e}")
            print(f"Mean column norm:    {column_norms.mean():.6e}")
            print(f"Median column norm:  {np.median(column_norms):.6e}")
            print(
                "Near-zero columns:  "
                f"{np.count_nonzero(column_norms < 1e-15)}"
            )

        max_column_norm = float(column_norms.max())

        if max_column_norm <= 0.0:
            raise ValueError(
                "The weighted Jacobian contains no sensitivity."
            )

        scaling_floor = np.finfo(float).eps * max_column_norm
        column_scales = 1.0 / np.maximum(
            column_norms,
            scaling_floor,
        )

        scaled_jacobian = (
            weighted_jacobian * column_scales[np.newaxis, :]
        )

        if self.verbose:
            scaled_norms = np.linalg.norm(scaled_jacobian, axis=0)

            print("\n===== PARAMETER SCALING =====")
            print(f"Minimum scale: {column_scales.min():.6e}")
            print(f"Maximum scale: {column_scales.max():.6e}")
            print(
                "Scaled Jacobian norm range: "
                f"{scaled_norms.min():.6e} to {scaled_norms.max():.6e}"
            )

        regularization = problem.regularization
        regularization_parameter = float(
            problem.regularization_parameter
        )

        if (
            not np.isfinite(regularization_parameter)
            or regularization_parameter < 0.0
        ):
            raise ValueError(
                "Regularization parameter must be finite and non-negative."
            )

        use_regularization = (
            regularization is not None
            and regularization_parameter > 0.0
        )

        sqrt_lambda = np.sqrt(regularization_parameter)
        regularization_matrix = None
        reference_values = np.zeros(n_parameters, dtype=float)

        if use_regularization:
            reference_values = (
                LinearLeastSquaresSolver._get_reference_values(
                    regularization,
                    initial_model,
                )
            )

            if reference_values.shape != (n_parameters,):
                raise ValueError(
                    "Reference model has an invalid shape."
                )

            if not np.all(np.isfinite(reference_values)):
                raise ValueError(
                    "Reference model contains non-finite values."
                )

            from geoinvlab.core.inversion.regularization import L2Regularization

            if not isinstance(regularization, L2Regularization):
                regularization_matrix = np.asarray(
                    regularization.matrix(initial_model),
                    dtype=float,
                )

                if regularization_matrix.ndim != 2:
                    raise ValueError(
                        "Regularization matrix must be two-dimensional."
                    )

                if regularization_matrix.shape[1] != n_parameters:
                    raise ValueError(
                        "Regularization matrix has an invalid number of columns."
                    )

                if not np.all(np.isfinite(regularization_matrix)):
                    raise ValueError(
                        "Regularization matrix contains non-finite values."
                    )

        # Construct the augmented least-squares system A x - b in scaled
        # coordinates, where physical model m = column_scales * x.
        if use_regularization:
            if regularization_matrix is None:
                n_regularization = n_parameters

                def matvec(values):
                    physical = column_scales * values

                    return np.concatenate(
                        (
                            weighted_jacobian @ physical,
                            sqrt_lambda * physical,
                        )
                    )

                def rmatvec(values):
                    data_part = values[:n_data]
                    reg_part = values[n_data:]

                    gradient = (
                        weighted_jacobian.T @ data_part
                        + sqrt_lambda * reg_part
                    )

                    return column_scales * gradient

                regularization_rhs = sqrt_lambda * reference_values

            else:
                n_regularization = regularization_matrix.shape[0]

                def matvec(values):
                    physical = column_scales * values

                    return np.concatenate(
                        (
                            weighted_jacobian @ physical,
                            sqrt_lambda * (
                                regularization_matrix @ physical
                            ),
                        )
                    )

                def rmatvec(values):
                    data_part = values[:n_data]
                    reg_part = values[n_data:]

                    gradient = (
                        weighted_jacobian.T @ data_part
                        + sqrt_lambda * (
                            regularization_matrix.T @ reg_part
                        )
                    )

                    return column_scales * gradient

                regularization_rhs = (
                    sqrt_lambda
                    * (regularization_matrix @ reference_values)
                )

            operator_shape = (
                n_data + n_regularization,
                n_parameters,
            )

            rhs = np.concatenate(
                (
                    weighted_data,
                    regularization_rhs,
                )
            )

        else:
            operator_shape = (n_data, n_parameters)
            rhs = weighted_data

            def matvec(values):
                return weighted_jacobian @ (
                    column_scales * values
                )

            def rmatvec(values):
                return column_scales * (
                    weighted_jacobian.T @ values
                )

        system_operator = LinearOperator(
            shape=operator_shape,
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float64,
        )

        lower_bounds = np.zeros(n_parameters, dtype=float)
        upper_bounds = np.full(n_parameters, np.inf, dtype=float)

        initial_physical_values = np.asarray(
            initial_model.values,
            dtype=float,
        )

        x0 = np.maximum(
            initial_physical_values / column_scales,
            1.0e-2,
        )

        # Initialize iteration and model history files.
        history_path = (
            Path(self.history_file)
            if self.history_file is not None
            else None
        )
        model_history_path = (
            Path(self.model_history_file)
            if self.model_history_file is not None
            else None
        )

        if history_path is not None:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("w", encoding="ascii") as stream:
                stream.write(
                    "# iteration data_rms normalized_rms phi_d phi_m phi "
                    "model_min model_max elapsed_s\n"
                )

        cell_centers = np.asarray(
            initial_model.mesh.cell_centers(),
            dtype=float,
        )

        if cell_centers.shape != (n_parameters, 3):
            raise ValueError(
                "Mesh cell centers do not match the number of model parameters."
            )

        self._initialize_model_history(
            model_history_path,
            initial_model,
        )

        parameter_values = {
            "solver": "scipy.optimize.least_squares(method=trf)",
            "tolerance": self.tolerance,
            "target_misfit": self.target_misfit,
            "max_iterations": self.max_iterations,
            "lsmr_tolerance": self.lsmr_tolerance,
            "regularization_lambda": regularization_parameter,
            "n_data": n_data,
            "n_parameters": n_parameters,
            "n_cells": n_parameters,
            "history_file": str(history_path) if history_path else "disabled",
            "model_history_file": (
                str(model_history_path) if model_history_path else "disabled"
            ),
            "lower_bound": 0.0,
            "upper_bound": "inf",
            "weighted_rms_definition": (
                "sqrt(mean((residual * weights)**2))"
            ),
            "target_misfit_behavior": (
                "report_only_continue_optimization"
            ),
        }

        self._write_parameters(
            self.parameters_file,
            parameter_values,
        )

        start_time = time.perf_counter()
        previous_model_values = np.asarray(
            initial_physical_values,
            dtype=float,
        ).copy()
        iteration_counter = 0

        self.iteration_history = []
        self.target_misfit_reached = False

        # Record the starting model as iteration zero, matching the
        # Gauss-Newton history convention used by the MT inversion.
        initial_prediction = np.asarray(
            problem.operator.predict(initial_model),
            dtype=float,
        ).reshape(-1)

        if initial_prediction.shape != (n_data,):
            raise ValueError(
                "Initial forward prediction has an invalid shape."
            )

        if not np.all(np.isfinite(initial_prediction)):
            raise ValueError(
                "Initial forward prediction contains non-finite values."
            )

        (
            initial_data_rms,
            initial_normalized_rms,
            initial_phi_d,
            initial_phi_m,
            initial_phi,
        ) = _calculate_objective_components(
            problem,
            initial_model,
            observed,
            weights,
            regularization_parameter,
            predicted=initial_prediction,
        )

        initial_record = {
            "iteration": 0,
            "elapsed_s": 0.0,
            "data_rms": initial_data_rms,
            "normalized_rms": initial_normalized_rms,
            "phi_d": initial_phi_d,
            "phi_m": initial_phi_m,
            "phi": initial_phi,
            "total_cost": initial_phi,
            "data_cost": 0.5 * initial_phi_d,
            "regularization_cost": 0.5 * initial_phi_m,
            "optimality": np.nan,
            "model_min": float(np.min(initial_physical_values)),
            "model_max": float(np.max(initial_physical_values)),
            "step_norm": 0.0,
            "alpha": np.nan,
            "regularization_lambda": regularization_parameter,
            "tolerance": self.tolerance,
            "target_misfit": self.target_misfit,
            "max_iterations": self.max_iterations,
            "lsmr_tolerance": self.lsmr_tolerance,
        }
        self.iteration_history.append(initial_record)

        if history_path is not None:
            with history_path.open("a", encoding="ascii") as stream:
                stream.write(
                    f"0 {initial_data_rms:.9e} "
                    f"{initial_normalized_rms:.9e} "
                    f"{initial_phi_d:.9e} "
                    f"{initial_phi_m:.9e} "
                    f"{initial_phi:.9e} "
                    f"{initial_record['model_min']:.9e} "
                    f"{initial_record['model_max']:.9e} 0.000000000e+00\n"
                )

        if self.verbose:
            print(
                "===== INVERSION ITERATION HISTORY =====",
                flush=True,
            )
            print(
                "ITER   DATA RMS       NORM RMS       PHI_D"
                "           PHI_M           PHI(M)"
                "          MODEL MIN      MODEL MAX      TIME(s)"
            )
            print("-" * 130)
            print(
                f"{0:4d} "
                f"{initial_data_rms:13.5e} "
                f"{initial_normalized_rms:13.5e} "
                f"{initial_phi_d:13.5e} "
                f"{initial_phi_m:13.5e} "
                f"{initial_phi:13.5e} "
                f"{initial_record['model_min']:13.5e} "
                f"{initial_record['model_max']:13.5e} "
                f"{0.0:9.2f}",
                flush=True,
            )

        if callable(self.callback):
            self.callback(dict(initial_record), initial_physical_values.copy())

        def residual_function(scaled_values):
            return np.asarray(
                system_operator @ scaled_values - rhs,
                dtype=float,
            )

        def jacobian_function(_scaled_values):
            return system_operator

        def iteration_callback(intermediate_result):
            nonlocal iteration_counter, previous_model_values

            iteration_counter += 1

            scaled_values = np.asarray(
                intermediate_result.x,
                dtype=float,
            )

            model_values = np.maximum(
                column_scales * scaled_values,
                0.0,
            )

            predicted = jacobian @ model_values

            (
                data_rms,
                normalized_rms,
                phi_d,
                phi_m,
                phi,
            ) = _calculate_objective_components(
                problem,
                Model3D(
                    mesh=initial_model.mesh,
                    values=model_values,
                    name=initial_model.name,
                ),
                observed,
                weights,
                regularization_parameter,
                predicted=predicted,
            )

            elapsed = time.perf_counter() - start_time

            optimality = float(
                getattr(intermediate_result, "optimality", np.nan)
            )

            step_norm = float(
                np.linalg.norm(model_values - previous_model_values)
            )

            model_min = float(model_values.min())
            model_max = float(model_values.max())

            row = {
                "iteration": iteration_counter,
                "elapsed_s": elapsed,
                "data_rms": data_rms,
                "normalized_rms": normalized_rms,
                "phi_d": phi_d,
                "phi_m": phi_m,
                "phi": phi,
                "total_cost": phi,
                "data_cost": 0.5 * phi_d,
                "regularization_cost": 0.5 * phi_m,
                "optimality": optimality,
                "model_min": model_min,
                "model_max": model_max,
                "step_norm": step_norm,
                "alpha": np.nan,
                "regularization_lambda": regularization_parameter,
                "tolerance": self.tolerance,
                "target_misfit": self.target_misfit,
                "max_iterations": self.max_iterations,
                "lsmr_tolerance": self.lsmr_tolerance,
            }

            self.iteration_history.append(row)

            self._append_model_history(
                model_history_path,
                iteration=iteration_counter,
                model_values=model_values,
                centers=cell_centers,
            )

            if history_path is not None:
                with history_path.open("a", encoding="ascii") as stream:
                    stream.write(
                        f"{iteration_counter:d} "
                        f"{data_rms:.9e} "
                        f"{normalized_rms:.9e} "
                        f"{phi_d:.9e} "
                        f"{phi_m:.9e} "
                        f"{phi:.9e} "
                        f"{model_min:.9e} "
                        f"{model_max:.9e} "
                        f"{elapsed:.9e}\n"
                    )

            if self.verbose:
                print(
                    f"{iteration_counter:4d} "
                    f"{data_rms:13.5e} "
                    f"{normalized_rms:13.5e} "
                    f"{phi_d:13.5e} "
                    f"{phi_m:13.5e} "
                    f"{phi:13.5e} "
                    f"{model_min:13.5e} "
                    f"{model_max:13.5e} "
                    f"{elapsed:9.2f}",
                    flush=True,
                )

            # The target misfit is diagnostic only.
            if (
                not self.target_misfit_reached
                and normalized_rms <= self.target_misfit
            ):
                self.target_misfit_reached = True

                print(
                    f"Target misfit reached at iteration "
                    f"{iteration_counter}: NORM RMS = "
                    f"{normalized_rms:.6e} <= "
                    f"{self.target_misfit:.6e}. "
                    "Continuing optimization until numerical convergence."
                )

            if callable(self.callback):
                self.callback(dict(row), model_values.copy())

            previous_model_values = model_values.copy()

            if iteration_counter >= self.max_iterations:
                raise StopIteration

        print("Solving positive bounded least-squares system...")
        print(f"Jacobian shape: {jacobian.shape}")
        print(f"Regularization lambda: {regularization_parameter:g}")

        result = least_squares(
            residual_function,
            x0,
            jac=jacobian_function,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            tr_solver="lsmr",
            tr_options={
                "atol": self.lsmr_tolerance,
                "btol": self.lsmr_tolerance,
            }
            if isinstance(self.lsmr_tolerance, (int, float))
            else {},
            ftol=self.tolerance,
            xtol=self.tolerance,
            gtol=self.tolerance,
            max_nfev=None,
            verbose=0,
            callback=iteration_callback,
        )

        self.result = result
        self.converged = bool(result.success)
        self.status = str(result.message)

        scaled_model_values = np.asarray(result.x, dtype=float)

        if scaled_model_values.shape != (n_parameters,):
            raise ValueError(
                "Solver returned an invalid model shape."
            )

        if not np.all(np.isfinite(scaled_model_values)):
            raise ValueError(
                "Inversion produced non-finite scaled parameters."
            )

        model_values = np.maximum(
            column_scales * scaled_model_values,
            0.0,
        )

        if not np.all(np.isfinite(model_values)):
            raise ValueError(
                "Inversion produced non-finite physical parameters."
            )

        if result.status == -2:
            self.converged = False
            self.status = (
                f"Maximum number of iterations reached "
                f"({self.max_iterations}); last iterate retained."
            )

        final_model = Model3D(
            mesh=initial_model.mesh,
            values=model_values,
            name=initial_model.name,
        )

        parameter_values.update(
            {
                "converged": self.converged,
                "status": self.status.replace("\n", " "),
                "iterations_completed": iteration_counter,
                "final_cost": float(result.cost),
                "first_order_optimality": float(result.optimality),
                "target_misfit_reached": self.target_misfit_reached,
            }
        )
        self._write_parameters(
            self.parameters_file,
            parameter_values,
        )

        print("\nPositive bounded inversion finished.")
        print(f"Solver status: {self.status}")
        print(f"Converged: {self.converged}")
        print(f"Solver iterations: {iteration_counter}")
        print(f"Final cost: {float(result.cost):.6e}")
        print(f"First-order optimality: {float(result.optimality):.6e}")
        print(f"Minimum model value: {model_values.min():.6e}")
        print(f"Maximum model value: {model_values.max():.6e}")

        return final_model
