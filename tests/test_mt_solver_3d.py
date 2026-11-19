import numpy as np
import pytest
from scipy import sparse

from geoinvlab.methods.mt.discretization.boundary import (
    BoundaryCondition,
)
from geoinvlab.methods.mt.discretization.solver_3d import (
    LinearSolveResult,
    MaxwellSolver,
    solve_maxwell,
)


def make_boundary(
    indices,
    values,
    n_edges,
):
    return BoundaryCondition(
        indices=np.asarray(indices, dtype=int),
        values=np.asarray(values, dtype=complex),
        n_edges=n_edges,
    )


def test_solver_accepts_valid_system():
    matrix = sparse.eye(
        3,
        dtype=complex,
        format="csr",
    )

    rhs = np.array(
        [1.0, 2.0, 3.0],
        dtype=complex,
    )

    boundary = make_boundary(
        [],
        [],
        3,
    )

    solver = MaxwellSolver(
        matrix,
        rhs,
        boundary,
    )

    assert solver.n_dof == 3
    assert solver.n_free == 3


def test_solver_rejects_non_square_matrix():
    matrix = sparse.csr_matrix(
        np.ones((2, 3)),
        dtype=complex,
    )

    rhs = np.zeros(2, dtype=complex)

    boundary = make_boundary(
        [],
        [],
        2,
    )

    with pytest.raises(ValueError, match="square"):
        MaxwellSolver(
            matrix,
            rhs,
            boundary,
        )


def test_solver_rejects_incompatible_rhs():
    matrix = sparse.eye(
        3,
        dtype=complex,
        format="csr",
    )

    rhs = np.zeros(2, dtype=complex)

    boundary = make_boundary(
        [],
        [],
        3,
    )

    with pytest.raises(
        ValueError,
        match="incompatible sizes",
    ):
        MaxwellSolver(
            matrix,
            rhs,
            boundary,
        )


def test_solver_rejects_incompatible_boundary():
    matrix = sparse.eye(
        3,
        dtype=complex,
        format="csr",
    )

    rhs = np.zeros(3, dtype=complex)

    boundary = make_boundary(
        [],
        [],
        4,
    )

    with pytest.raises(
        ValueError,
        match="incompatible sizes",
    ):
        MaxwellSolver(
            matrix,
            rhs,
            boundary,
        )


def test_direct_solver_without_boundary_condition():
    matrix = sparse.csr_matrix(
        [
            [4.0, 1.0, 0.0],
            [1.0, 5.0, 1.0],
            [0.0, 1.0, 3.0],
        ],
        dtype=complex,
    )

    rhs = np.array(
        [1.0, 2.0, 3.0],
        dtype=complex,
    )

    boundary = make_boundary(
        [],
        [],
        3,
    )

    result = solve_maxwell(
        matrix,
        rhs,
        boundary,
        method="direct",
    )

    assert isinstance(
        result,
        LinearSolveResult,
    )

    assert result.converged
    assert result.method == "direct"

    np.testing.assert_allclose(
        matrix @ result.field,
        rhs,
        atol=1.0e-12,
    )

    assert result.residual < 1.0e-12
    assert result.relative_residual < 1.0e-12


def test_direct_solver_with_dirichlet_condition():
    matrix = sparse.csr_matrix(
        [
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=complex,
    )

    rhs = np.zeros(
        3,
        dtype=complex,
    )

    boundary = make_boundary(
        [0],
        [2.0],
        3,
    )

    result = solve_maxwell(
        matrix,
        rhs,
        boundary,
    )

    assert isinstance(
        result,
        LinearSolveResult,
    )

    assert result.converged

    # ---------------------------------------------------------------
    # Dirichlet condition must be preserved.
    # ---------------------------------------------------------------
    np.testing.assert_allclose(
        result.field[0],
        2.0,
        atol=1.0e-12,
    )

    # ---------------------------------------------------------------
    # Only the free equations must be satisfied.
    #
    # The boundary row is intentionally excluded because its
    # corresponding degree of freedom was prescribed explicitly.
    # ---------------------------------------------------------------
    free_indices = np.array(
        [1, 2],
        dtype=int,
    )

    free_residual = (
        matrix @ result.field - rhs
    )[free_indices]

    np.testing.assert_allclose(
        free_residual,
        0.0,
        atol=1.0e-12,
    )

    assert result.residual < 1.0e-12
    assert result.relative_residual < 1.0e-12


def test_boundary_values_are_preserved():
    matrix = sparse.eye(
        4,
        dtype=complex,
        format="csr",
    )

    rhs = np.zeros(
        4,
        dtype=complex,
    )

    boundary = make_boundary(
        [1, 3],
        [2.0 + 1.0j, -3.0 + 2.0j],
        4,
    )

    result = solve_maxwell(
        matrix,
        rhs,
        boundary,
    )

    np.testing.assert_allclose(
        result.field[1],
        2.0 + 1.0j,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        result.field[3],
        -3.0 + 2.0j,
        atol=1.0e-12,
    )

    # Free DOFs 0 and 2 solve the identity equations.
    np.testing.assert_allclose(
        result.field[[0, 2]],
        0.0,
        atol=1.0e-12,
    )

    free_indices = np.array(
        [0, 2],
        dtype=int,
    )

    free_residual = (
        matrix @ result.field - rhs
    )[free_indices]

    np.testing.assert_allclose(
        free_residual,
        0.0,
        atol=1.0e-12,
    )

    assert result.converged


def test_reduced_system_without_boundary():
    matrix = sparse.csr_matrix(
        [
            [2.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=complex,
    )

    rhs = np.array(
        [1.0, 2.0],
        dtype=complex,
    )

    boundary = make_boundary(
        [],
        [],
        2,
    )

    solver = MaxwellSolver(
        matrix,
        rhs,
        boundary,
    )

    reduced_matrix, reduced_rhs = (
        solver.reduced_system()
    )

    np.testing.assert_allclose(
        reduced_matrix.toarray(),
        matrix.toarray(),
    )

    np.testing.assert_allclose(
        reduced_rhs,
        rhs,
    )


def test_reduced_system_with_boundary():
    matrix = sparse.csr_matrix(
        [
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=complex,
    )

    rhs = np.zeros(
        3,
        dtype=complex,
    )

    boundary = make_boundary(
        [0],
        [2.0],
        3,
    )

    solver = MaxwellSolver(
        matrix,
        rhs,
        boundary,
    )

    reduced_matrix, reduced_rhs = (
        solver.reduced_system()
    )

    expected_matrix = sparse.csr_matrix(
        [
            [3.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=complex,
    )

    expected_rhs = np.array(
        [-2.0, 0.0],
        dtype=complex,
    )

    np.testing.assert_allclose(
        reduced_matrix.toarray(),
        expected_matrix.toarray(),
    )

    np.testing.assert_allclose(
        reduced_rhs,
        expected_rhs,
    )


def test_initial_field_contains_boundary_values():
    matrix = sparse.eye(
        4,
        dtype=complex,
        format="csr",
    )

    rhs = np.zeros(
        4,
        dtype=complex,
    )

    boundary = make_boundary(
        [0, 2],
        [1.0 + 2.0j, -3.0 + 4.0j],
        4,
    )

    solver = MaxwellSolver(
        matrix,
        rhs,
        boundary,
    )

    field = solver.initial_field()

    expected = np.array(
        [
            1.0 + 2.0j,
            0.0 + 0.0j,
            -3.0 + 4.0j,
            0.0 + 0.0j,
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(
        field,
        expected,
    )


def test_direct_and_gmres_have_same_solution():
    matrix = sparse.csr_matrix(
        [
            [4.0, 1.0, 0.0],
            [1.0, 5.0, 1.0],
            [0.0, 1.0, 3.0],
        ],
        dtype=complex,
    )

    rhs = np.array(
        [
            1.0 + 1.0j,
            2.0 - 1.0j,
            3.0 + 0.5j,
        ],
        dtype=complex,
    )

    boundary = make_boundary(
        [],
        [],
        3,
    )

    direct = solve_maxwell(
        matrix,
        rhs,
        boundary,
        method="direct",
    )

    iterative = solve_maxwell(
        matrix,
        rhs,
        boundary,
        method="gmres",
        tolerance=1.0e-10,
    )

    assert direct.converged
    assert iterative.converged

    np.testing.assert_allclose(
        iterative.field,
        direct.field,
        atol=1.0e-9,
    )

    assert iterative.relative_residual < 1.0e-10


def test_gmres_solution_matches_direct_solution():
    matrix = sparse.csr_matrix(
        [
            [4.0, 1.0, 0.0],
            [1.0, 5.0, 1.0],
            [0.0, 1.0, 3.0],
        ],
        dtype=complex,
    )

    rhs = np.array(
        [
            1.0 + 1.0j,
            2.0 - 1.0j,
            3.0 + 0.5j,
        ],
        dtype=complex,
    )

    boundary = make_boundary(
        [],
        [],
        3,
    )

    direct = solve_maxwell(
        matrix,
        rhs,
        boundary,
        method="direct",
    )

    iterative = solve_maxwell(
        matrix,
        rhs,
        boundary,
        method="gmres",
        tolerance=1.0e-10,
    )

    assert iterative.converged

    np.testing.assert_allclose(
        iterative.field,
        direct.field,
        atol=1.0e-9,
    )

    assert iterative.relative_residual < 1.0e-10


def test_invalid_solver_method():
    matrix = sparse.eye(
        2,
        dtype=complex,
        format="csr",
    )

    rhs = np.ones(
        2,
        dtype=complex,
    )

    boundary = make_boundary(
        [],
        [],
        2,
    )

    with pytest.raises(
        ValueError,
        match="Unknown solver method",
    ):
        solve_maxwell(
            matrix,
            rhs,
            boundary,
            method="invalid",
        )


def test_invalid_tolerance():
    matrix = sparse.eye(
        2,
        dtype=complex,
        format="csr",
    )

    rhs = np.ones(
        2,
        dtype=complex,
    )

    boundary = make_boundary(
        [],
        [],
        2,
    )

    with pytest.raises(
        ValueError,
        match="tolerance",
    ):
        solve_maxwell(
            matrix,
            rhs,
            boundary,
            tolerance=0.0,
        )
