"""
Tests for the 3D MT forward solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from geoinvlab.core.model import Model3D
from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.methods.mt.discretization.forward import (
    MTForwardResult,
    MTForwardSolver,
    forward_mt,
)
from geoinvlab.methods.mt.discretization.station import (
    MTStation,
    MTStationSet,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def mesh() -> StructuredMesh3D:
    """Return a small structured mesh."""

    return StructuredMesh3D(
        nx=2,
        ny=2,
        nz=2,
        dx=100.0,
        dy=100.0,
        dz=100.0,
    )


@pytest.fixture
def model(mesh: StructuredMesh3D) -> Model3D:
    """Return a homogeneous resistivity model."""

    return Model3D(
        mesh=mesh,
        values=np.full(
            mesh.n_cells,
            100.0,
            dtype=float,
        ),
        name="resistivity",
    )


@pytest.fixture
def stations() -> MTStationSet:
    """Return two observation stations."""

    return MTStationSet(
        (
            MTStation(
                x=50.0,
                y=50.0,
                z=100.0,
                name="STA01",
            ),
            MTStation(
                x=150.0,
                y=150.0,
                z=100.0,
                name="STA02",
            ),
        )
    )


@pytest.fixture
def solver(mesh: StructuredMesh3D) -> MTForwardSolver:
    """Return a small direct forward solver."""

    return MTForwardSolver(
        mesh=mesh,
        frequencies=np.array(
            [1.0, 10.0],
            dtype=float,
        ),
        solver_method="direct",
    )


# ----------------------------------------------------------------------
# Basic validation
# ----------------------------------------------------------------------


def test_solver_accepts_structured_mesh(
    mesh: StructuredMesh3D,
) -> None:
    """The solver must accept StructuredMesh3D."""

    solver = MTForwardSolver(
        mesh=mesh,
        frequencies=[1.0],
    )

    assert solver.mesh is mesh
    assert solver.n_frequencies == 1


def test_solver_rejects_invalid_frequency(
    mesh: StructuredMesh3D,
) -> None:
    """Frequencies must be strictly positive."""

    with pytest.raises(ValueError):
        MTForwardSolver(
            mesh=mesh,
            frequencies=[0.0],
        )


def test_solver_rejects_negative_frequency(
    mesh: StructuredMesh3D,
) -> None:
    """Negative frequencies are invalid."""

    with pytest.raises(ValueError):
        MTForwardSolver(
            mesh=mesh,
            frequencies=[-1.0],
        )


def test_solver_rejects_invalid_method(
    mesh: StructuredMesh3D,
) -> None:
    """Only direct and gmres are supported."""

    with pytest.raises(ValueError):
        MTForwardSolver(
            mesh=mesh,
            frequencies=[1.0],
            solver_method="invalid",
        )


# ----------------------------------------------------------------------
# Model conversion
# ----------------------------------------------------------------------


def test_resistivity_to_conductivity(
    solver: MTForwardSolver,
    model: Model3D,
) -> None:
    """Resistivity must be converted to conductivity."""

    conductivity = solver.conductivity_from_model(
        model,
        parameterization="resistivity",
    )

    assert conductivity.shape == (
        model.mesh.n_cells,
    )

    assert np.allclose(
        conductivity,
        1.0 / 100.0,
    )


def test_conductivity_parameterization(
    solver: MTForwardSolver,
    mesh: StructuredMesh3D,
) -> None:
    """Conductivity parameterization must be accepted."""

    model = Model3D(
        mesh=mesh,
        values=np.full(
            mesh.n_cells,
            0.01,
        ),
        name="conductivity",
    )

    conductivity = solver.conductivity_from_model(
        model,
        parameterization="conductivity",
    )

    assert np.allclose(
        conductivity,
        0.01,
    )


def test_invalid_resistivity_is_rejected(
    solver: MTForwardSolver,
    mesh: StructuredMesh3D,
) -> None:
    """Non-positive resistivity must be rejected."""

    model = Model3D(
        mesh=mesh,
        values=np.ones(mesh.n_cells),
    )

    model.values[0] = 0.0

    with pytest.raises(ValueError):
        solver.conductivity_from_model(
            model,
            parameterization="resistivity",
        )


def test_invalid_parameterization_is_rejected(
    solver: MTForwardSolver,
    model: Model3D,
) -> None:
    """Unknown parameterizations must be rejected."""

    with pytest.raises(ValueError):
        solver.conductivity_from_model(
            model,
            parameterization="unknown",
        )


# ----------------------------------------------------------------------
# Maxwell system
# ----------------------------------------------------------------------


def test_build_system(
    solver: MTForwardSolver,
    model: Model3D,
) -> None:
    """The forward solver must build a Maxwell system."""

    conductivity = solver.conductivity_from_model(
        model
    )

    system = solver.build_system(
        conductivity
    )

    assert system.n_edges > 0
    assert system.n_faces > 0
    assert system.n_cells == model.mesh.n_cells


# ----------------------------------------------------------------------
# Single source
# ----------------------------------------------------------------------


def test_solve_x_source(
    solver: MTForwardSolver,
    model: Model3D,
) -> None:
    """The X-polarized source must produce an edge field."""

    from geoinvlab.methods.mt.discretization.source import (
        MTSourcePolarization,
    )

    conductivity = solver.conductivity_from_model(
        model
    )

    system = solver.build_system(
        conductivity
    )

    electric, magnetic, result = solver.solve_source(
        system=system,
        frequency=1.0,
        polarization=MTSourcePolarization.X,
    )

    assert electric.values.shape == (
        system.n_edges,
    )

    assert magnetic.shape == (
        system.n_faces,
    )

    assert np.all(
        np.isfinite(electric.values.real)
    )

    assert np.all(
        np.isfinite(electric.values.imag)
    )

    assert np.all(
        np.isfinite(magnetic.real)
    )

    assert np.all(
        np.isfinite(magnetic.imag)
    )

    assert result.field.shape == (
        system.n_edges,
    )


def test_solve_y_source(
    solver: MTForwardSolver,
    model: Model3D,
) -> None:
    """The Y-polarized source must produce an edge field."""

    from geoinvlab.methods.mt.discretization.source import (
        MTSourcePolarization,
    )

    conductivity = solver.conductivity_from_model(
        model
    )

    system = solver.build_system(
        conductivity
    )

    electric, magnetic, result = solver.solve_source(
        system=system,
        frequency=1.0,
        polarization=MTSourcePolarization.Y,
    )

    assert electric.values.shape == (
        system.n_edges,
    )

    assert magnetic.shape == (
        system.n_faces,
    )

    assert result.field.shape == (
        system.n_edges,
    )


# ----------------------------------------------------------------------
# Impedance construction
# ----------------------------------------------------------------------


def test_impedance_from_two_polarizations() -> None:
    """
    Two independent E/H responses must reconstruct Z.

    Synthetic example:

        Z = [[0, 2],
             [-3, 0]]

    H = I

    therefore E = Z.
    """

    from geoinvlab.methods.mt.discretization.observation import (
        MTObservedFields,
    )

    observations_x = [
        MTObservedFields(
            ex=0.0 + 0.0j,
            ey=-3.0 + 0.0j,
            hx=1.0 + 0.0j,
            hy=0.0 + 0.0j,
        )
    ]

    observations_y = [
        MTObservedFields(
            ex=2.0 + 0.0j,
            ey=0.0 + 0.0j,
            hx=0.0 + 0.0j,
            hy=1.0 + 0.0j,
        )
    ]

    result = MTForwardSolver.impedance_from_observations(
        observations_x,
        observations_y,
    )

    assert result.shape == (
        1,
        2,
        2,
    )

    expected = np.array(
        [
            [
                0.0 + 0.0j,
                2.0 + 0.0j,
            ],
            [
                -3.0 + 0.0j,
                0.0 + 0.0j,
            ],
        ]
    )

    assert np.allclose(
        result[0],
        expected,
    )


def test_singular_magnetic_matrix_is_rejected() -> None:
    """A singular H matrix cannot define an impedance tensor."""

    from geoinvlab.methods.mt.discretization.observation import (
        MTObservedFields,
    )

    observations_x = [
        MTObservedFields(
            ex=1.0 + 0.0j,
            ey=2.0 + 0.0j,
            hx=1.0 + 0.0j,
            hy=0.0 + 0.0j,
        )
    ]

    observations_y = [
        MTObservedFields(
            ex=3.0 + 0.0j,
            ey=4.0 + 0.0j,
            hx=2.0 + 0.0j,
            hy=0.0 + 0.0j,
        )
    ]

    with pytest.raises(np.linalg.LinAlgError):
        MTForwardSolver.impedance_from_observations(
            observations_x,
            observations_y,
        )


# ----------------------------------------------------------------------
# Full forward
# ----------------------------------------------------------------------


def test_forward_result_shape(
    solver: MTForwardSolver,
    model: Model3D,
    stations: MTStationSet,
) -> None:
    """The complete forward result must have the expected shape."""

    result = solver.forward(
        model=model,
        stations=stations,
    )

    assert isinstance(
        result,
        MTForwardResult,
    )

    assert result.impedance.shape == (
        len(stations),
        solver.n_frequencies,
        2,
        2,
    )


def test_forward_result_frequencies(
    solver: MTForwardSolver,
    model: Model3D,
    stations: MTStationSet,
) -> None:
    """The result must preserve the requested frequencies."""

    result = solver.forward(
        model=model,
        stations=stations,
    )

    assert np.allclose(
        result.frequencies,
        [1.0, 10.0],
    )


def test_forward_result_station_count(
    solver: MTForwardSolver,
    model: Model3D,
    stations: MTStationSet,
) -> None:
    """The result must preserve the station count."""

    result = solver.forward(
        model=model,
        stations=stations,
    )

    assert result.n_stations == 2


def test_forward_result_accessors(
    solver: MTForwardSolver,
    model: Model3D,
    stations: MTStationSet,
) -> None:
    """Tensor component accessors must have consistent shapes."""

    result = solver.forward(
        model=model,
        stations=stations,
    )

    expected_shape = (
        len(stations),
        solver.n_frequencies,
    )

    assert result.zxx.shape == expected_shape
    assert result.zxy.shape == expected_shape
    assert result.zyx.shape == expected_shape
    assert result.zyy.shape == expected_shape


def test_forward_tensor_accessor(
    solver: MTForwardSolver,
    model: Model3D,
    stations: MTStationSet,
) -> None:
    """A single station/frequency tensor must be accessible."""

    result = solver.forward(
        model=model,
        stations=stations,
    )

    tensor = result.tensor(
        station_index=0,
        frequency_index=0,
    )

    assert tensor.tensor.shape == (
        2,
        2,
    )

    assert np.allclose(
        tensor.tensor,
        result.impedance[0, 0],
    )


# ----------------------------------------------------------------------
# Field storage
# ----------------------------------------------------------------------


def test_forward_stores_fields(
    solver: MTForwardSolver,
    model: Model3D,
    stations: MTStationSet,
) -> None:
    """Complete E and H fields must optionally be stored."""

    result = solver.forward(
        model=model,
        stations=stations,
        store_fields=True,
    )

    assert result.electric_fields is not None
    assert result.magnetic_fields is not None

    assert result.electric_fields.shape[0] == (
        solver.n_frequencies
    )

    assert result.electric_fields.shape[1] == 2

    assert result.magnetic_fields.shape[0] == (
        solver.n_frequencies
    )

    assert result.magnetic_fields.shape[1] == 2


def test_forward_can_skip_fields(
    solver: MTForwardSolver,
    model: Model3D,
    stations: MTStationSet,
) -> None:
    """Field storage can be disabled."""

    result = solver.forward(
        model=model,
        stations=stations,
        store_fields=False,
    )

    assert result.electric_fields is None
    assert result.magnetic_fields is None


# ----------------------------------------------------------------------
# Convenience interface
# ----------------------------------------------------------------------


def test_forward_mt_convenience_function(
    mesh: StructuredMesh3D,
    model: Model3D,
    stations: MTStationSet,
) -> None:
    """The forward_mt convenience function must work."""

    result = forward_mt(
        mesh=mesh,
        model=model,
        stations=stations,
        frequencies=[1.0],
    )

    assert isinstance(
        result,
        MTForwardResult,
    )

    assert result.impedance.shape == (
        len(stations),
        1,
        2,
        2,
    )


# ----------------------------------------------------------------------
# Two-source architecture
# ----------------------------------------------------------------------


def test_two_source_fields_are_independent(
    solver: MTForwardSolver,
    model: Model3D,
) -> None:
    """
    X and Y source solves must be represented independently.

    This verifies the architecture rather than imposing a physical
    symmetry on the approximate current source/boundary formulation.
    """

    from geoinvlab.methods.mt.discretization.source import (
        MTSourcePolarization,
    )

    conductivity = solver.conductivity_from_model(
        model
    )

    system = solver.build_system(
        conductivity
    )

    electric_x, magnetic_x, _ = solver.solve_source(
        system=system,
        frequency=1.0,
        polarization=MTSourcePolarization.X,
    )

    electric_y, magnetic_y, _ = solver.solve_source(
        system=system,
        frequency=1.0,
        polarization=MTSourcePolarization.Y,
    )

    assert electric_x.values.shape == electric_y.values.shape
    assert magnetic_x.shape == magnetic_y.shape

    assert np.all(
        np.isfinite(electric_x.values)
    )

    assert np.all(
        np.isfinite(electric_y.values)
    )


# ----------------------------------------------------------------------
# Validation of input model
# ----------------------------------------------------------------------


def test_forward_rejects_non_model(
    solver: MTForwardSolver,
    stations: MTStationSet,
) -> None:
    """forward() must require Model3D."""

    with pytest.raises(TypeError):
        solver.forward(
            model=np.ones(8),
            stations=stations,
        )


def test_forward_rejects_invalid_station_input(
    solver: MTForwardSolver,
    model: Model3D,
) -> None:
    """An empty station set must be rejected."""

    with pytest.raises(ValueError):
        solver.forward(
            model=model,
            stations=(),
        )


# ----------------------------------------------------------------------
# Numerical sanity
# ----------------------------------------------------------------------


def test_forward_impedance_is_finite(
    solver: MTForwardSolver,
    model: Model3D,
    stations: MTStationSet,
) -> None:
    """
    The computed impedance must remain finite for this small test
    problem.
    """

    result = solver.forward(
        model=model,
        stations=stations,
    )

    assert np.all(
        np.isfinite(
            result.impedance.real
        )
    )

    assert np.all(
        np.isfinite(
            result.impedance.imag
        )
    )
