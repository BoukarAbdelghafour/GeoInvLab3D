import numpy as np
import pytest

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.methods.mt.discretization.boundary import (
    BoundaryCondition,
    build_boundary_edges,
    make_dirichlet_boundary,
)


def make_mesh():
    return StructuredMesh3D(
        nx=2,
        ny=3,
        nz=2,
        dx=10.0,
        dy=20.0,
        dz=30.0,
    )


def test_boundary_edge_count_is_positive():
    mesh = make_mesh()

    boundary = build_boundary_edges(mesh)

    assert boundary.n_boundary_edges > 0
    assert boundary.n_boundary_edges < boundary.n_edges


def test_boundary_indices_are_unique():
    mesh = make_mesh()

    boundary = build_boundary_edges(mesh)

    assert np.unique(boundary.all).size == boundary.all.size


def test_boundary_indices_are_sorted():
    mesh = make_mesh()

    boundary = build_boundary_edges(mesh)

    assert np.all(np.diff(boundary.all) > 0)


def test_boundary_indices_are_valid():
    mesh = make_mesh()

    boundary = build_boundary_edges(mesh)

    assert np.all(boundary.all >= 0)
    assert np.all(boundary.all < boundary.n_edges)


def test_boundary_mask_has_correct_size():
    mesh = make_mesh()

    boundary = build_boundary_edges(mesh)
    mask = boundary.mask()

    assert mask.dtype == bool
    assert mask.shape == (boundary.n_edges,)
    assert mask.sum() == boundary.n_boundary_edges


def test_each_boundary_face_contains_edges():
    mesh = make_mesh()

    boundary = build_boundary_edges(mesh)

    assert boundary.xmin.size > 0
    assert boundary.xmax.size > 0
    assert boundary.ymin.size > 0
    assert boundary.ymax.size > 0
    assert boundary.zmin.size > 0
    assert boundary.zmax.size > 0


def test_boundary_faces_are_subsets_of_all_boundary_edges():
    mesh = make_mesh()

    boundary = build_boundary_edges(mesh)
    all_edges = set(boundary.all.tolist())

    for indices in (
        boundary.xmin,
        boundary.xmax,
        boundary.ymin,
        boundary.ymax,
        boundary.zmin,
        boundary.zmax,
    ):
        assert set(indices.tolist()).issubset(all_edges)


def test_boundary_condition_validation():
    condition = BoundaryCondition(
        indices=np.array([1, 3, 5]),
        values=np.array(
            [1.0 + 0.0j, 2.0 + 1.0j, 3.0 - 1.0j]
        ),
        n_edges=10,
    )

    assert condition.size == 3


def test_boundary_condition_vector():
    condition = BoundaryCondition(
        indices=np.array([1, 4]),
        values=np.array(
            [2.0 + 3.0j, -1.0 + 0.5j]
        ),
        n_edges=6,
    )

    vector = condition.vector()

    expected = np.zeros(6, dtype=complex)
    expected[1] = 2.0 + 3.0j
    expected[4] = -1.0 + 0.5j

    np.testing.assert_array_equal(vector, expected)


def test_make_dirichlet_boundary():
    condition = make_dirichlet_boundary(
        indices=np.array([0, 2]),
        values=np.array([1.0, 2.0]),
        n_edges=5,
    )

    assert isinstance(condition, BoundaryCondition)
    assert condition.size == 2
    assert condition.values.dtype == complex


def test_duplicate_boundary_indices_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        BoundaryCondition(
            indices=np.array([1, 1]),
            values=np.array([1.0, 2.0]),
            n_edges=5,
        )


def test_invalid_boundary_indices_are_rejected():
    with pytest.raises(ValueError, match="invalid"):
        BoundaryCondition(
            indices=np.array([0, 5]),
            values=np.array([1.0, 2.0]),
            n_edges=5,
        )


def test_mismatched_boundary_sizes_are_rejected():
    with pytest.raises(ValueError, match="same size"):
        BoundaryCondition(
            indices=np.array([0, 1]),
            values=np.array([1.0]),
            n_edges=5,
        )


def test_non_finite_boundary_values_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        BoundaryCondition(
            indices=np.array([0]),
            values=np.array([np.nan + 0.0j]),
            n_edges=5,
        )
