"""
Tests for the MT structured-grid metric layer.
"""

import numpy as np
import pytest

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.methods.mt.discretization.metric import (
    build_metric,
)
from geoinvlab.methods.mt.discretization.topology import (
    build_topology,
)


def make_mesh():
    return StructuredMesh3D(
        nx=2,
        ny=3,
        nz=4,
        dx=10.0,
        dy=20.0,
        dz=30.0,
    )


def test_metric_dimensions_match_topology():
    mesh = make_mesh()

    topology = build_topology(mesh)
    metric = build_metric(mesh)

    assert metric.mesh_shape == mesh.shape

    assert metric.n_edges == topology.n_edges
    assert metric.n_faces == topology.n_faces
    assert metric.n_cells == mesh.n_cells


def test_edge_lengths_for_uniform_mesh():
    mesh = make_mesh()
    metric = build_metric(mesh)

    nx, ny, nz = mesh.shape

    n_ex = nx * (ny + 1) * (nz + 1)
    n_ey = (nx + 1) * ny * (nz + 1)
    n_ez = (nx + 1) * (ny + 1) * nz

    ex = metric.edges.lengths[:n_ex]
    ey = metric.edges.lengths[n_ex:n_ex + n_ey]
    ez = metric.edges.lengths[n_ex + n_ey:]

    assert np.all(ex == 10.0)
    assert np.all(ey == 20.0)
    assert np.all(ez == 30.0)

    assert ex.size == n_ex
    assert ey.size == n_ey
    assert ez.size == n_ez


def test_face_areas_for_uniform_mesh():
    mesh = make_mesh()
    metric = build_metric(mesh)

    nx, ny, nz = mesh.shape

    n_fx = (nx + 1) * ny * nz
    n_fy = nx * (ny + 1) * nz

    fx = metric.faces.areas[:n_fx]
    fy = metric.faces.areas[n_fx:n_fx + n_fy]
    fz = metric.faces.areas[n_fx + n_fy:]

    # Fx normal to x -> dy * dz
    assert np.all(fx == 20.0 * 30.0)

    # Fy normal to y -> dx * dz
    assert np.all(fy == 10.0 * 30.0)

    # Fz normal to z -> dx * dy
    assert np.all(fz == 10.0 * 20.0)


def test_cell_volumes_for_uniform_mesh():
    mesh = make_mesh()
    metric = build_metric(mesh)

    expected_volume = 10.0 * 20.0 * 30.0

    assert metric.n_cells == mesh.n_cells
    assert np.all(metric.cells.volumes == expected_volume)


def test_metric_values_are_positive_and_finite():
    mesh = make_mesh()
    metric = build_metric(mesh)

    assert np.all(np.isfinite(metric.edges.lengths))
    assert np.all(metric.edges.lengths > 0.0)

    assert np.all(np.isfinite(metric.faces.areas))
    assert np.all(metric.faces.areas > 0.0)

    assert np.all(np.isfinite(metric.cells.volumes))
    assert np.all(metric.cells.volumes > 0.0)


def test_metric_with_padding():
    mesh = StructuredMesh3D(
        nx=2,
        ny=2,
        nz=3,
        dx=10.0,
        dy=20.0,
        dz=30.0,
        padding_x=(2, 2),
        padding_y=(1, 1),
        padding_z=(2, 2),
        padding_factor=1.5,
    )

    metric = build_metric(mesh)
    topology = build_topology(mesh)

    assert metric.mesh_shape == mesh.shape
    assert metric.n_edges == topology.n_edges
    assert metric.n_faces == topology.n_faces
    assert metric.n_cells == mesh.n_cells

    assert np.all(metric.edges.lengths > 0.0)
    assert np.all(metric.faces.areas > 0.0)
    assert np.all(metric.cells.volumes > 0.0)


def test_metric_rejects_non_structured_mesh():
    class DummyMesh:
        pass

    with pytest.raises(TypeError):
        build_metric(DummyMesh())


def test_metric_is_consistent_with_mesh_volumes():
    mesh = make_mesh()
    metric = build_metric(mesh)

    expected = np.asarray(mesh.cell_volumes(), dtype=float)

    np.testing.assert_allclose(
        metric.cells.volumes,
        expected,
    )


def test_edge_count():
    mesh = make_mesh()
    metric = build_metric(mesh)

    nx, ny, nz = mesh.shape

    expected = (
        nx * (ny + 1) * (nz + 1)
        + (nx + 1) * ny * (nz + 1)
        + (nx + 1) * (ny + 1) * nz
    )

    assert metric.n_edges == expected


def test_face_count():
    mesh = make_mesh()
    metric = build_metric(mesh)

    nx, ny, nz = mesh.shape

    expected = (
        (nx + 1) * ny * nz
        + nx * (ny + 1) * nz
        + nx * ny * (nz + 1)
    )

    assert metric.n_faces == expected
