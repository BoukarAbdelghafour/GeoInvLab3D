import numpy as np
import pytest

from geoinvlab.methods.mt.discretization.source import (
    MTPlaneWaveSource,
    MTSourcePolarization,
    build_mt_source,
)
from geoinvlab.methods.mt.discretization.topology import (
    build_topology,
)
from geoinvlab.core.mesh.structured import StructuredMesh3D


def make_mesh():
    return StructuredMesh3D(
        nx=2,
        ny=2,
        nz=2,
        dx=100.0,
        dy=100.0,
        dz=100.0,
    )


# ---------------------------------------------------------------------------
# MTPlaneWaveSource
# ---------------------------------------------------------------------------


def test_source_x_polarization():
    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
    )

    assert source.polarization == MTSourcePolarization.X
    assert source.amplitude == 1.0
    assert source.phase == 0.0
    assert source.frequency == 1.0


def test_source_y_polarization():
    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.Y,
    )

    assert source.polarization == MTSourcePolarization.Y


def test_source_custom_parameters():
    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
        amplitude=3.0,
        phase=np.pi / 2,
        frequency=10.0,
    )

    assert source.amplitude == 3.0
    assert source.phase == np.pi / 2
    assert source.frequency == 10.0


def test_complex_amplitude_without_phase():
    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
        amplitude=3.0,
        phase=0.0,
    )

    np.testing.assert_allclose(
        source.complex_amplitude,
        3.0 + 0.0j,
    )


def test_complex_amplitude_with_phase():
    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
        amplitude=3.0,
        phase=np.pi / 2,
    )

    np.testing.assert_allclose(
        source.complex_amplitude,
        3.0j,
        atol=1e-12,
    )


def test_negative_amplitude_is_allowed():
    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
        amplitude=-2.0,
    )

    np.testing.assert_allclose(
        source.complex_amplitude,
        -2.0 + 0.0j,
    )


def test_invalid_amplitude_nan():
    with pytest.raises(ValueError):
        MTPlaneWaveSource(
            polarization=MTSourcePolarization.X,
            amplitude=np.nan,
        )


def test_invalid_amplitude_inf():
    with pytest.raises(ValueError):
        MTPlaneWaveSource(
            polarization=MTSourcePolarization.X,
            amplitude=np.inf,
        )


def test_invalid_phase_nan():
    with pytest.raises(ValueError):
        MTPlaneWaveSource(
            polarization=MTSourcePolarization.X,
            phase=np.nan,
        )


def test_invalid_phase_inf():
    with pytest.raises(ValueError):
        MTPlaneWaveSource(
            polarization=MTSourcePolarization.X,
            phase=np.inf,
        )


def test_invalid_frequency_zero():
    with pytest.raises(ValueError):
        MTPlaneWaveSource(
            polarization=MTSourcePolarization.X,
            frequency=0.0,
        )


def test_invalid_frequency_negative():
    with pytest.raises(ValueError):
        MTPlaneWaveSource(
            polarization=MTSourcePolarization.X,
            frequency=-1.0,
        )


def test_invalid_frequency_nan():
    with pytest.raises(ValueError):
        MTPlaneWaveSource(
            polarization=MTSourcePolarization.X,
            frequency=np.nan,
        )


def test_invalid_frequency_inf():
    with pytest.raises(ValueError):
        MTPlaneWaveSource(
            polarization=MTSourcePolarization.X,
            frequency=np.inf,
        )


# ---------------------------------------------------------------------------
# build_mt_source
# ---------------------------------------------------------------------------


def test_x_source_returns_boundary_condition():
    mesh = make_mesh()

    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
        amplitude=3.0,
    )

    boundary = build_mt_source(mesh, source)

    topology = build_topology(mesh)

    assert boundary.n_edges == (
        topology.edges.ex.size
        + topology.edges.ey.size
        + topology.edges.ez.size
    )

    assert boundary.size > 0


def test_y_source_returns_boundary_condition():
    mesh = make_mesh()

    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.Y,
        amplitude=3.0,
    )

    boundary = build_mt_source(mesh, source)

    topology = build_topology(mesh)

    assert boundary.n_edges == (
        topology.edges.ex.size
        + topology.edges.ey.size
        + topology.edges.ez.size
    )

    assert boundary.size > 0


def test_x_source_has_nonzero_ex():
    mesh = make_mesh()

    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
        amplitude=3.0,
    )

    boundary = build_mt_source(mesh, source)
    vector = boundary.vector()

    topology = build_topology(mesh)

    n_ex = topology.edges.ex.size
    n_ey = topology.edges.ey.size

    ex_values = vector[:n_ex]

    # Il y a des arêtes Ex sur la frontière supérieure.
    assert np.any(np.abs(ex_values) > 0.0)


def test_x_source_has_zero_ey():
    mesh = make_mesh()

    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
        amplitude=3.0,
    )

    boundary = build_mt_source(mesh, source)
    vector = boundary.vector()

    topology = build_topology(mesh)

    n_ex = topology.edges.ex.size
    n_ey = topology.edges.ey.size

    ex_values = vector[:n_ex]
    ey_values = vector[n_ex:n_ex + n_ey]

    # Polarisation X : Ey = 0.
    np.testing.assert_allclose(
        ey_values,
        0.0,
    )

    # Ex doit contenir la valeur imposée.
    assert np.any(np.abs(ex_values) > 0.0)


def test_y_source_has_nonzero_ey():
    mesh = make_mesh()

    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.Y,
        amplitude=3.0,
    )

    boundary = build_mt_source(mesh, source)
    vector = boundary.vector()

    topology = build_topology(mesh)

    n_ex = topology.edges.ex.size
    n_ey = topology.edges.ey.size

    ey_values = vector[n_ex:n_ex + n_ey]

    # Il y a des arêtes Ey sur la frontière supérieure.
    assert np.any(np.abs(ey_values) > 0.0)


def test_y_source_has_zero_ex():
    mesh = make_mesh()

    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.Y,
        amplitude=3.0,
    )

    boundary = build_mt_source(mesh, source)
    vector = boundary.vector()

    topology = build_topology(mesh)

    n_ex = topology.edges.ex.size
    n_ey = topology.edges.ey.size

    ex_values = vector[:n_ex]
    ey_values = vector[n_ex:n_ex + n_ey]

    # Polarisation Y : Ex = 0.
    np.testing.assert_allclose(
        ex_values,
        0.0,
    )

    # Ey doit contenir la valeur imposée.
    assert np.any(np.abs(ey_values) > 0.0)


def test_x_source_amplitude_is_preserved():
    mesh = make_mesh()

    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
        amplitude=3.0,
    )

    boundary = build_mt_source(mesh, source)
    vector = boundary.vector()

    topology = build_topology(mesh)

    n_ex = topology.edges.ex.size

    ex_values = vector[:n_ex]

    nonzero = np.abs(ex_values) > 0.0

    assert np.any(nonzero)

    np.testing.assert_allclose(
        ex_values[nonzero],
        3.0 + 0.0j,
    )


def test_y_source_amplitude_is_preserved():
    mesh = make_mesh()

    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.Y,
        amplitude=3.0,
    )

    boundary = build_mt_source(mesh, source)
    vector = boundary.vector()

    topology = build_topology(mesh)

    n_ex = topology.edges.ex.size
    n_ey = topology.edges.ey.size

    ey_values = vector[n_ex:n_ex + n_ey]

    nonzero = np.abs(ey_values) > 0.0

    assert np.any(nonzero)

    np.testing.assert_allclose(
        ey_values[nonzero],
        3.0 + 0.0j,
    )


def test_source_phase_is_preserved():
    mesh = make_mesh()

    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
        amplitude=2.0,
        phase=np.pi / 2,
    )

    boundary = build_mt_source(mesh, source)
    vector = boundary.vector()

    topology = build_topology(mesh)

    n_ex = topology.edges.ex.size

    ex_values = vector[:n_ex]

    nonzero = np.abs(ex_values) > 0.0

    assert np.any(nonzero)

    np.testing.assert_allclose(
        ex_values[nonzero],
        2.0j,
        atol=1e-12,
    )


def test_source_boundary_contains_only_tangential_edges():
    mesh = make_mesh()

    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
    )

    boundary = build_mt_source(mesh, source)

    topology = build_topology(mesh)

    n_ex = topology.edges.ex.size
    n_ey = topology.edges.ey.size
    n_ez = topology.edges.ez.size

    ez_start = n_ex + n_ey
    ez_end = ez_start + n_ez

    vector = boundary.vector()

    # Aucun Ez ne doit être imposé.
    np.testing.assert_allclose(
        vector[ez_start:ez_end],
        0.0,
    )


def test_source_does_not_prescribe_interior_edges():
    mesh = make_mesh()

    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
        amplitude=2.0,
    )

    boundary = build_mt_source(mesh, source)

    topology = build_topology(mesh)

    n_ex = topology.edges.ex.size
    n_ey = topology.edges.ey.size

    vector = boundary.vector()

    ex_values = vector[:n_ex]
    ey_values = vector[n_ex:n_ex + n_ey]

    # Pour une source X, les valeurs non nulles doivent correspondre
    # uniquement aux Ex de la frontière supérieure.
    assert np.all(
        np.logical_or(
            np.isclose(ex_values, 0.0),
            np.isclose(ex_values, 2.0),
        )
    )

    # Ey doit rester nul pour la polarisation X.
    np.testing.assert_allclose(
        ey_values,
        0.0,
    )


def test_source_vector_has_correct_size():
    mesh = make_mesh()

    source = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
    )

    boundary = build_mt_source(mesh, source)

    topology = build_topology(mesh)

    n_edges = (
        topology.edges.ex.size
        + topology.edges.ey.size
        + topology.edges.ez.size
    )

    vector = boundary.vector()

    assert vector.shape == (n_edges,)


def test_x_and_y_sources_have_same_boundary_structure():
    mesh = make_mesh()

    source_x = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
        amplitude=1.0,
    )

    source_y = MTPlaneWaveSource(
        polarization=MTSourcePolarization.Y,
        amplitude=1.0,
    )

    boundary_x = build_mt_source(mesh, source_x)
    boundary_y = build_mt_source(mesh, source_y)

    # Les deux polarisations utilisent le même ensemble d'arêtes
    # tangentielles sur la frontière supérieure.
    np.testing.assert_array_equal(
        boundary_x.indices,
        boundary_y.indices,
    )


def test_x_and_y_sources_are_orthogonal():
    mesh = make_mesh()

    source_x = MTPlaneWaveSource(
        polarization=MTSourcePolarization.X,
        amplitude=1.0,
    )

    source_y = MTPlaneWaveSource(
        polarization=MTSourcePolarization.Y,
        amplitude=1.0,
    )

    boundary_x = build_mt_source(mesh, source_x)
    boundary_y = build_mt_source(mesh, source_y)

    vector_x = boundary_x.vector()
    vector_y = boundary_y.vector()

    topology = build_topology(mesh)

    n_ex = topology.edges.ex.size
    n_ey = topology.edges.ey.size

    ex_x = vector_x[:n_ex]
    ey_x = vector_x[n_ex:n_ex + n_ey]

    ex_y = vector_y[:n_ex]
    ey_y = vector_y[n_ex:n_ex + n_ey]

    # X : Ex non nul, Ey nul.
    assert np.any(np.abs(ex_x) > 0.0)
    np.testing.assert_allclose(ey_x, 0.0)

    # Y : Ex nul, Ey non nul.
    np.testing.assert_allclose(ex_y, 0.0)
    assert np.any(np.abs(ey_y) > 0.0)
