"""
3D Maxwell linear solver for the MT discretization.

This module solves the linear system obtained after discretizing the
3D Maxwell equations and applying Dirichlet boundary conditions.

The solver supports:

    - direct sparse LU solution with iterative refinement
    - iterative GMRES solution
    - elimination of prescribed boundary edge degrees of freedom
    - residual and relative-residual diagnostics

The unknown vector is the electric field on mesh edges:

    E = [Ex, Ey, Ez]

The matrix is expected to have the form:

    A E = b

where A is assembled by MaxwellSystem.system_matrix().

Dirichlet treatment
-------------------
Let B denote prescribed boundary edge degrees of freedom and F
the remaining free degrees of freedom. The full system

    A E = b

with

    E_B = g

is reduced to

    A_FF E_F = b_F - A_FB g.

The complete electric field is then reconstructed as

    E = [E_F, g].

Important MT convention
-----------------------
This solver does not modify the phase or sign of the imposed electric
field. Boundary values are inserted exactly as supplied by the MT
source.

For GeoInvLab3D the physical convention is

    exp(+i omega t)

with

    curl(E) = -i omega mu H

and therefore

    H = +i curl(E) / (omega mu).

The Maxwell matrix is consequently expected to contain

    A = K + i omega M_sigma.

No conjugation, phase rotation, or artificial ``-1j`` factor is applied
in this module.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import gmres, splu

from geoinvlab.methods.mt.discretization.boundary import (
    BoundaryCondition,
)


@dataclass(frozen=True)
class LinearSolveResult:
    """
    Result of a 3D Maxwell linear solve.

    Parameters
    ----------
    field:
        Complete electric-field vector, including prescribed boundary
        degrees of freedom.

    residual:
        Absolute residual norm of the reduced system.

    relative_residual:
        Relative residual norm of the reduced system.

    iterations:
        Number of solver iterations.

    converged:
        True when the requested tolerance was reached.

    method:
        Solver method used.
    """

    field: np.ndarray
    residual: float
    relative_residual: float
    iterations: int
    converged: bool
    method: str

    def __post_init__(self) -> None:
        field = np.asarray(
            self.field,
            dtype=complex,
        )

        if field.ndim != 1:
            raise ValueError(
                "Solved field must be one-dimensional."
            )

        if not np.all(
            np.isfinite(field.real)
        ):
            raise ValueError(
                "Solved field contains non-finite real values."
            )

        if not np.all(
            np.isfinite(field.imag)
        ):
            raise ValueError(
                "Solved field contains non-finite imaginary values."
            )

        residual = float(self.residual)

        if not np.isfinite(residual):
            raise ValueError(
                "Residual must be finite."
            )

        if residual < 0.0:
            raise ValueError(
                "Residual must be non-negative."
            )

        relative_residual = float(
            self.relative_residual
        )

        if not np.isfinite(relative_residual):
            raise ValueError(
                "Relative residual must be finite."
            )

        if relative_residual < 0.0:
            raise ValueError(
                "Relative residual must be non-negative."
            )

        iterations = int(
            self.iterations
        )

        if iterations < 0:
            raise ValueError(
                "Number of iterations must be non-negative."
            )

        object.__setattr__(
            self,
            "field",
            field,
        )

        object.__setattr__(
            self,
            "residual",
            residual,
        )

        object.__setattr__(
            self,
            "relative_residual",
            relative_residual,
        )

        object.__setattr__(
            self,
            "iterations",
            iterations,
        )

        object.__setattr__(
            self,
            "converged",
            bool(self.converged),
        )

        object.__setattr__(
            self,
            "method",
            str(self.method),
        )


@dataclass(frozen=True)
class MaxwellSolver:
    """
    Sparse linear solver for a discretized 3D Maxwell system.

    Parameters
    ----------
    matrix:
        Complex sparse Maxwell system matrix.

    rhs:
        Complex right-hand-side vector.

    boundary:
        Dirichlet boundary condition describing prescribed electric
        field values on boundary edge degrees of freedom.
    """

    matrix: sparse.spmatrix
    rhs: np.ndarray
    boundary: BoundaryCondition

    def __post_init__(self) -> None:
        matrix = sparse.csr_matrix(
            self.matrix,
            dtype=complex,
        )

        rhs = np.asarray(
            self.rhs,
            dtype=complex,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Maxwell matrix must be two-dimensional."
            )

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "Maxwell matrix must be square."
            )

        if rhs.ndim != 1:
            raise ValueError(
                "Maxwell right-hand side must be one-dimensional."
            )

        if rhs.size != matrix.shape[0]:
            raise ValueError(
                "Maxwell matrix and right-hand side "
                "have incompatible sizes."
            )

        if not isinstance(
            self.boundary,
            BoundaryCondition,
        ):
            raise TypeError(
                "boundary must be a BoundaryCondition."
            )

        if self.boundary.n_edges != matrix.shape[0]:
            raise ValueError(
                "Boundary condition and Maxwell matrix "
                "have incompatible sizes."
            )

        indices = np.asarray(
            self.boundary.indices,
            dtype=int,
        )

        values = np.asarray(
            self.boundary.values,
            dtype=complex,
        )

        if indices.ndim != 1:
            raise ValueError(
                "Boundary indices must be one-dimensional."
            )

        if values.ndim != 1:
            raise ValueError(
                "Boundary values must be one-dimensional."
            )

        if indices.size != values.size:
            raise ValueError(
                "Boundary indices and values must have "
                "the same size."
            )

        if not np.all(
            np.isfinite(matrix.data.real)
        ):
            raise ValueError(
                "Maxwell matrix contains non-finite real values."
            )

        if not np.all(
            np.isfinite(matrix.data.imag)
        ):
            raise ValueError(
                "Maxwell matrix contains non-finite imaginary values."
            )

        if not np.all(
            np.isfinite(rhs.real)
        ):
            raise ValueError(
                "Maxwell right-hand side contains "
                "non-finite real values."
            )

        if not np.all(
            np.isfinite(rhs.imag)
        ):
            raise ValueError(
                "Maxwell right-hand side contains "
                "non-finite imaginary values."
            )

        if indices.size:
            if np.any(indices < 0):
                raise ValueError(
                    "Boundary indices must be non-negative."
                )

            if np.any(
                indices >= matrix.shape[0]
            ):
                raise ValueError(
                    "Boundary index exceeds the Maxwell "
                    "system size."
                )

            if np.unique(indices).size != indices.size:
                raise ValueError(
                    "Boundary indices must be unique."
                )

        if not np.all(
            np.isfinite(values.real)
        ):
            raise ValueError(
                "Boundary values contain non-finite "
                "real values."
            )

        if not np.all(
            np.isfinite(values.imag)
        ):
            raise ValueError(
                "Boundary values contain non-finite "
                "imaginary values."
            )

        object.__setattr__(
            self,
            "matrix",
            matrix,
        )

        object.__setattr__(
            self,
            "rhs",
            rhs,
        )

    @property
    def n_dof(self) -> int:
        """
        Total number of electric-field degrees of freedom.
        """
        return int(
            self.matrix.shape[0]
        )

    @property
    def boundary_indices(self) -> np.ndarray:
        """
        Indices of prescribed boundary degrees of freedom.
        """
        return np.asarray(
            self.boundary.indices,
            dtype=int,
        )

    @property
    def boundary_values(self) -> np.ndarray:
        """
        Prescribed electric-field values on boundary DOFs.
        """
        return np.asarray(
            self.boundary.values,
            dtype=complex,
        )

    @property
    def free_indices(self) -> np.ndarray:
        """
        Indices of unknown/free degrees of freedom.
        """
        mask = np.ones(
            self.n_dof,
            dtype=bool,
        )

        indices = self.boundary_indices

        if indices.size:
            mask[indices] = False

        return np.flatnonzero(mask)

    @property
    def n_free(self) -> int:
        """
        Number of free degrees of freedom.
        """
        return int(
            self.free_indices.size
        )

    def reduced_system(
        self,
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """
        Build the reduced Dirichlet system.

        Starting from

            A E = b

        with prescribed boundary values

            E_B = g,

        the free system is

            A_FF E_F
                =
            b_F - A_FB g.

        Returns
        -------
        matrix:
            Reduced sparse matrix A_FF.

        rhs:
            Reduced right-hand side
            b_F - A_FB g.
        """
        free = self.free_indices
        boundary = self.boundary_indices

        if free.size == 0:
            return (
                sparse.csr_matrix(
                    (0, 0),
                    dtype=complex,
                ),
                np.zeros(
                    0,
                    dtype=complex,
                ),
            )

        matrix_ff = (
            self.matrix[free][:, free]
            .tocsr()
        )

        rhs_reduced = np.asarray(
            self.rhs[free],
            dtype=complex,
        ).copy()

        if boundary.size:
            matrix_fb = (
                self.matrix[free][:, boundary]
            )

            rhs_reduced -= (
                matrix_fb
                @ self.boundary_values
            )

        if not np.all(
            np.isfinite(rhs_reduced.real)
        ):
            raise RuntimeError(
                "Reduced Maxwell right-hand side contains "
                "non-finite real values."
            )

        if not np.all(
            np.isfinite(rhs_reduced.imag)
        ):
            raise RuntimeError(
                "Reduced Maxwell right-hand side contains "
                "non-finite imaginary values."
            )

        return (
            matrix_ff,
            rhs_reduced,
        )

    def initial_field(self) -> np.ndarray:
        """
        Create the complete electric-field vector.

        Free degrees of freedom are initialized to zero.
        Prescribed boundary values are inserted exactly as supplied.
        """
        field = np.zeros(
            self.n_dof,
            dtype=complex,
        )

        boundary = self.boundary_indices

        if boundary.size:
            field[boundary] = (
                self.boundary_values
            )

        return field

    def solve(
        self,
        method: str = "direct",
        tolerance: float = 1.0e-8,
        maxiter: int | None = None,
    ) -> LinearSolveResult:
        """
        Solve the Maxwell system with Dirichlet boundary conditions.

        Parameters
        ----------
        method:
            ``"direct"`` for sparse LU or ``"gmres"`` for GMRES.

        tolerance:
            Requested relative residual tolerance.

        maxiter:
            Maximum GMRES iterations. If omitted, SciPy's default
            behaviour is used.

        Returns
        -------
        LinearSolveResult
            Complete electric field and solver diagnostics.
        """
        method = str(
            method
        ).strip().lower()

        if method not in {
            "direct",
            "gmres",
        }:
            raise ValueError(
                "Unknown solver method. "
                "Expected 'direct' or 'gmres'."
            )

        tolerance = float(
            tolerance
        )

        if (
            not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError(
                "Solver tolerance must be finite and positive."
            )

        if maxiter is not None:
            maxiter = int(
                maxiter
            )

            if maxiter <= 0:
                raise ValueError(
                    "maxiter must be positive when provided."
                )

        matrix, rhs = self.reduced_system()

        field = self.initial_field()

        # --------------------------------------------------------------
        # No free degrees of freedom.
        # --------------------------------------------------------------
        if self.n_free == 0:
            full_residual = (
                self.matrix @ field
                - self.rhs
            )

            residual = float(
                np.linalg.norm(
                    full_residual
                )
            )

            relative_residual = _relative_residual(
                full_residual,
                self.rhs,
            )

            converged = (
                relative_residual <= tolerance
            )

            if not converged:
                raise RuntimeError(
                    "Prescribed boundary field does not satisfy "
                    "the Maxwell system within the requested "
                    f"tolerance: "
                    f"{relative_residual:.3e} > "
                    f"{tolerance:.3e}."
                )

            return LinearSolveResult(
                field=field,
                residual=residual,
                relative_residual=relative_residual,
                iterations=0,
                converged=True,
                method=method,
            )

        # --------------------------------------------------------------
        # Solve the free system:
        #
        #     A_FF E_F =
        #         b_F - A_FB E_B
        # --------------------------------------------------------------
        if method == "direct":
            free_field = _solve_direct(
                matrix=matrix,
                rhs=rhs,
                tolerance=tolerance,
            )

            iterations = 1

        else:
            (
                free_field,
                iterations,
                gmres_converged,
            ) = _solve_gmres(
                matrix=matrix,
                rhs=rhs,
                tolerance=tolerance,
                maxiter=maxiter,
            )

            if not gmres_converged:
                raise RuntimeError(
                    "GMRES did not converge within the "
                    "requested tolerance."
                )

        free_field = np.asarray(
            free_field,
            dtype=complex,
        )

        if free_field.ndim != 1:
            raise RuntimeError(
                "Maxwell solver returned a non-vector "
                "free electric field."
            )

        if free_field.size != self.n_free:
            raise RuntimeError(
                "Maxwell solver returned an invalid number "
                "of free electric-field degrees of freedom."
            )

        _validate_solution(
            free_field,
            "Maxwell solver",
        )

        field[
            self.free_indices
        ] = free_field

        # --------------------------------------------------------------
        # Residual of the reduced system.
        #
        # Boundary equations have intentionally been eliminated,
        # therefore the solver convergence criterion is evaluated
        # on the free system.
        # --------------------------------------------------------------
        reduced_residual_vector = (
            matrix @ free_field
            - rhs
        )

        residual = float(
            np.linalg.norm(
                reduced_residual_vector
            )
        )

        relative_residual = _relative_residual(
            reduced_residual_vector,
            rhs,
        )

        converged = (
            relative_residual <= tolerance
        )

        if not converged:
            raise RuntimeError(
                "Maxwell solve produced an insufficiently "
                "small free-DOF residual: "
                f"{relative_residual:.3e} > "
                f"{tolerance:.3e}. "
                f"||rhs_reduced||="
                f"{np.linalg.norm(rhs):.3e}, "
                f"||residual||="
                f"{residual:.3e}."
            )

        return LinearSolveResult(
            field=field,
            residual=residual,
            relative_residual=relative_residual,
            iterations=iterations,
            converged=True,
            method=method,
        )


def _solve_direct(
    matrix: sparse.spmatrix,
    rhs: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """
    Solve a sparse complex system with LU factorization.

    Iterative refinement reuses the LU factorization to improve
    the final residual.
    """
    matrix = sparse.csc_matrix(
        matrix,
        dtype=complex,
    )

    rhs = np.asarray(
        rhs,
        dtype=complex,
    )

    if matrix.shape[0] == 0:
        return np.zeros(
            0,
            dtype=complex,
        )

    try:
        lu = splu(
            matrix
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Sparse LU factorization failed for the "
            "reduced Maxwell system."
        ) from exc

    try:
        solution = lu.solve(
            rhs
        )
    except Exception as exc:
        raise RuntimeError(
            "Sparse LU solve failed for the "
            "reduced Maxwell system."
        ) from exc

    solution = np.asarray(
        solution,
        dtype=complex,
    )

    _validate_solution(
        solution,
        "Direct Maxwell solver",
    )

    max_refinement_iterations = 8

    for _ in range(
        max_refinement_iterations
    ):
        residual_vector = (
            rhs
            - matrix @ solution
        )

        relative_residual = _relative_residual(
            residual_vector,
            rhs,
        )

        if relative_residual <= tolerance:
            break

        try:
            correction = lu.solve(
                residual_vector
            )
        except Exception as exc:
            raise RuntimeError(
                "Sparse LU iterative refinement failed."
            ) from exc

        correction = np.asarray(
            correction,
            dtype=complex,
        )

        _validate_solution(
            correction,
            "Direct Maxwell refinement",
        )

        solution += correction

        _validate_solution(
            solution,
            "Direct Maxwell refinement",
        )

    return solution


def _solve_gmres(
    matrix: sparse.spmatrix,
    rhs: np.ndarray,
    tolerance: float,
    maxiter: int | None,
) -> tuple[np.ndarray, int, bool]:
    """
    Solve a sparse complex system using GMRES.

    The implementation supports both SciPy APIs:

    Newer SciPy
        rtol=...

    Older SciPy
        tol=...
    """
    iterations = 0

    def callback(_: float) -> None:
        nonlocal iterations
        iterations += 1

    try:
        solution, info = gmres(
            matrix,
            rhs,
            rtol=tolerance,
            atol=0.0,
            maxiter=maxiter,
            callback=callback,
            callback_type="pr_norm",
        )

    except TypeError as error:
        if (
            "unexpected keyword argument 'rtol'"
            not in str(error)
        ):
            raise

        solution, info = gmres(
            matrix,
            rhs,
            tol=tolerance,
            atol=0.0,
            maxiter=maxiter,
            callback=callback,
            callback_type="pr_norm",
        )

    solution = np.asarray(
        solution,
        dtype=complex,
    )

    _validate_solution(
        solution,
        "GMRES",
    )

    # SciPy convention:
    #
    #   info == 0  -> converged
    #   info > 0   -> maximum iterations/restarts reached
    #   info < 0   -> numerical breakdown/error
    if info < 0:
        raise RuntimeError(
            "GMRES failed with a numerical breakdown "
            f"(info={info})."
        )

    return (
        solution,
        iterations,
        info == 0,
    )


def _validate_solution(
    solution: np.ndarray,
    name: str,
) -> None:
    """
    Validate a numerical solution vector.
    """
    solution = np.asarray(
        solution,
        dtype=complex,
    )

    if solution.ndim != 1:
        raise RuntimeError(
            f"{name} returned an invalid field: "
            "the result must be one-dimensional."
        )

    if not np.all(
        np.isfinite(solution.real)
    ):
        raise RuntimeError(
            f"{name} returned non-finite real values."
        )

    if not np.all(
        np.isfinite(solution.imag)
    ):
        raise RuntimeError(
            f"{name} returned non-finite imaginary values."
        )


def _relative_residual(
    residual_vector: np.ndarray,
    rhs: np.ndarray,
) -> float:
    """
    Compute the relative residual norm.

    For a non-zero right-hand side:

        ||r|| / ||b||

    For a zero right-hand side:

        ||r||

    is returned.

    This convention is useful for the MT problem because the
    external plane-wave excitation is imposed through Dirichlet
    boundary values while the assembled volumetric RHS is normally
    zero.
    """
    residual_vector = np.asarray(
        residual_vector,
        dtype=complex,
    )

    rhs = np.asarray(
        rhs,
        dtype=complex,
    )

    residual_norm = float(
        np.linalg.norm(
            residual_vector
        )
    )

    rhs_norm = float(
        np.linalg.norm(
            rhs
        )
    )

    if rhs_norm == 0.0:
        return residual_norm

    return (
        residual_norm
        / rhs_norm
    )


def solve_maxwell(
    matrix: sparse.spmatrix,
    rhs: np.ndarray,
    boundary: BoundaryCondition,
    *,
    method: str = "direct",
    tolerance: float = 1.0e-8,
    maxiter: int | None = None,
) -> LinearSolveResult:
    """
    Convenience function for solving a 3D Maxwell system.

    Parameters
    ----------
    matrix:
        Maxwell system matrix.

    rhs:
        Maxwell right-hand side.

    boundary:
        Dirichlet boundary conditions.

    method:
        ``"direct"`` or ``"gmres"``.

    tolerance:
        Solver tolerance.

    maxiter:
        Maximum GMRES iterations.

    Returns
    -------
    LinearSolveResult
        Complete electric field and convergence diagnostics.
    """
    solver = MaxwellSolver(
        matrix=matrix,
        rhs=rhs,
        boundary=boundary,
    )

    return solver.solve(
        method=method,
        tolerance=tolerance,
        maxiter=maxiter,
    )
 

