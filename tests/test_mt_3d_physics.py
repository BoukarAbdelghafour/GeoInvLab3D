"""
Physical validation tests for the 3D magnetotelluric forward solver.

These tests are deliberately stronger than algebraic/unit tests.

They verify:

    - structured topology dimensions
    - electric/magnetic field dimensions
    - finite homogeneous 3D MT response
    - 1D/3D homogeneous consistency
    - 1D tensor symmetry
    - approximate 1D tensor symmetry of the 3D solver
    - physically meaningful apparent resistivity
    - frequency dependence
    - station-to-discrete-observation mapping
    - both source polarizations
    - stored electric and magnetic fields
    - raw field amplitudes for physical debugging

The tests do not assume exact equality between the simplified 3D
discretization and the analytical 1D solution. The comparison is
therefore deliberately tolerant.

Important:
    These are physics-validation tests, not merely software API tests.
    If they fail while the unit-test suite passes, the failure should
    be investigated in the Maxwell/source/field/observation chain
    rather than hidden by weakening the thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.core.model.model import Model3D

from geoinvlab.methods.mt.discretization.field import (
    EdgeElectricField,
    build_edge_electric_field,
)
from geoinvlab.methods.mt.discretization.forward import (
    MTForwardSolver,
)
from geoinvlab.methods.mt.discretization.station import (
    MTStation,
    MTStationSet,
    stations_to_observations,
)
from geoinvlab.methods.mt.discretization.topology import (
    build_topology,
)
from geoinvlab.methods.mt.kernel import MTKernel


# ----------------------------------------------------------------------
# Test configuration
# ----------------------------------------------------------------------

RESISTIVITY = 100.0

FREQUENCIES = np.array(
    [0.1, 1.0, 10.0],
    dtype=float,
)

MU0 = 4.0 * np.pi * 1.0e-7


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def _make_mesh() -> StructuredMesh3D:
    """
    Build a padded structured mesh.

    Padding is intentional here because station mapping and physical
    coordinates must remain correct on non-uniform meshes.
    """
    return StructuredMesh3D(
        nx=4,
        ny=4,
        nz=4,
        dx=100.0,
        dy=100.0,
        dz=100.0,
        origin=(-250.0, -250.0, -400.0),
        padding_x=(1, 1),
        padding_y=(1, 1),
        padding_z=(1, 1),
        padding_factor=1.5,
    )


def _make_uniform_mesh() -> StructuredMesh3D:
    """
    Build an unpadded uniform mesh.

    This is useful for tests where padding would add an unnecessary
    complication to the expected geometry.
    """
    return StructuredMesh3D(
        nx=4,
        ny=4,
        nz=4,
        dx=100.0,
        dy=100.0,
        dz=100.0,
        origin=(-200.0, -200.0, -400.0),
    )


def _make_stations(
    mesh: StructuredMesh3D,
) -> MTStationSet:
    """
    Build stations inside the physical computational domain.

    The coordinates are taken from actual mesh coordinates rather than
    reconstructed from origin + constant spacing, so this remains valid
    for padded meshes.
    """
    x0 = float(mesh.edges_x[1])
    x1 = float(mesh.edges_x[-2])

    y0 = float(mesh.edges_y[1])
    y1 = float(mesh.edges_y[-2])

    z = float(mesh.edges_z[-1])

    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)

    return MTStationSet(
        stations=(
            MTStation(
                x=x0,
                y=y0,
                z=z,
                name="STA_00",
            ),
            MTStation(
                x=xm,
                y=ym,
                z=z,
                name="STA_01",
            ),
            MTStation(
                x=x1,
                y=y1,
                z=z,
                name="STA_02",
            ),
        )
    )


def _make_homogeneous_model(
    mesh: StructuredMesh3D,
    resistivity: float = RESISTIVITY,
) -> Model3D:
    """Create a homogeneous resistivity model."""
    values = np.full(
        mesh.n_cells,
        resistivity,
        dtype=float,
    )

    return Model3D(
        mesh=mesh,
        values=values,
        name="resistivity",
    )


def _make_forward_solver(
    mesh: StructuredMesh3D,
    frequencies: np.ndarray = FREQUENCIES,
) -> MTForwardSolver:
    """Create the 3D MT forward solver."""
    return MTForwardSolver(
        mesh=mesh,
        frequencies=frequencies,
        parameterization="resistivity",
        mu_r=1.0,
        solver_method="direct",
    )


def _run_homogeneous_forward(
    *,
    mesh: StructuredMesh3D,
    frequencies: np.ndarray = FREQUENCIES,
    store_fields: bool = False,
):
    """
    Run the 3D forward problem for a homogeneous half-space.
    """
    model = _make_homogeneous_model(mesh)

    stations = _make_stations(mesh)

    solver = _make_forward_solver(
        mesh=mesh,
        frequencies=frequencies,
    )

    result = solver.forward(
        model=model,
        stations=stations,
        store_fields=store_fields,
    )

    return result, model, stations, solver


def _scalar_impedance_from_tensor(
    impedance: np.ndarray,
) -> np.ndarray:
    """
    Extract the off-diagonal scalar MT impedance.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_stations, n_frequencies)``.
    """
    impedance = np.asarray(
        impedance,
        dtype=complex,
    )

    if impedance.ndim != 4:
        raise ValueError(
            "impedance must have shape "
            "(n_stations, n_frequencies, 2, 2)"
        )

    return impedance[:, :, 0, 1]


def _analytical_homogeneous_impedance(
    frequencies: np.ndarray,
    resistivity: float = RESISTIVITY,
) -> np.ndarray:
    """
    Analytical homogeneous half-space impedance.

    With the project's convention exp(+i omega t):

        gamma = sqrt(i omega mu sigma)
        Z = i omega mu / gamma
    """
    frequencies = np.asarray(
        frequencies,
        dtype=float,
    )

    omega = 2.0 * np.pi * frequencies
    sigma = 1.0 / resistivity

    gamma = np.sqrt(
        1j * omega * MU0 * sigma
    )

    return (
        1j
        * omega
        * MU0
        / gamma
    )


def _relative_error(
    numerical: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Element-wise relative error."""
    numerical = np.asarray(
        numerical,
        dtype=complex,
    )
    reference = np.asarray(
        reference,
        dtype=complex,
    )

    denominator = np.maximum(
        np.abs(reference),
        1.0e-30,
    )

    return np.abs(
        numerical - reference
    ) / denominator


# ----------------------------------------------------------------------
# Topology and field dimensions
# ----------------------------------------------------------------------


def test_structured_topology_dimensions():
    """The topology must expose consistent edge and face dimensions."""
    mesh = _make_mesh()

    topology = build_topology(mesh)

    nx, ny, nz = mesh.shape

    expected_ex = nx * (ny + 1) * (nz + 1)
    expected_ey = (nx + 1) * ny * (nz + 1)
    expected_ez = (nx + 1) * (ny + 1) * nz

    expected_fx = (nx + 1) * ny * nz
    expected_fy = nx * (ny + 1) * nz
    expected_fz = nx * ny * (nz + 1)

    expected_edges = (
        expected_ex
        + expected_ey
        + expected_ez
    )

    expected_faces = (
        expected_fx
        + expected_fy
        + expected_fz
    )

    assert topology.n_edges == expected_edges
    assert topology.n_faces == expected_faces


def test_edge_electric_field_dimensions():
    """
    The electric field builder must produce a valid edge field.

    This test uses an unpadded mesh because the current
    build_edge_electric_field implementation is based on mesh.nx/ny/nz.
    """
    mesh = _make_uniform_mesh()

    topology = build_topology(mesh)

    values = np.zeros(
        topology.n_edges,
        dtype=complex,
    )

    field = build_edge_electric_field(
        mesh,
        values,
    )

    assert isinstance(
        field,
        EdgeElectricField,
    )

    assert field.values.shape == (
        topology.n_edges,
    )

    assert field.n_edges == topology.n_edges

    assert field.ex.size > 0
    assert field.ey.size > 0
    assert field.ez.size > 0

    assert (
        field.ex.size
        + field.ey.size
        + field.ez.size
        == topology.n_edges
    )


# ----------------------------------------------------------------------
# Homogeneous 3D response
# ----------------------------------------------------------------------


def test_homogeneous_forward_returns_finite_impedance():
    """The homogeneous 3D forward response must be finite."""
    mesh = _make_mesh()

    result, _, stations, _ = _run_homogeneous_forward(
        mesh=mesh,
    )

    assert result.impedance.shape == (
        len(stations),
        FREQUENCIES.size,
        2,
        2,
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


def test_homogeneous_3d_impedance_has_physical_magnitude():
    """
    The 3D homogeneous response must have a non-negligible impedance.

    For rho = 100 ohm.m, the analytical 1D response is approximately
    0.0089, 0.028, and 0.089 ohm at 0.1, 1, and 10 Hz respectively.

    This test deliberately rejects numerical values close to machine
    zero, which would indicate a broken source/observation chain.
    """
    mesh = _make_mesh()

    result, _, _, _ = _run_homogeneous_forward(
        mesh=mesh,
    )

    zxy = _scalar_impedance_from_tensor(
        result.impedance
    )

    analytical = _analytical_homogeneous_impedance(
        FREQUENCIES,
    )

    # The response should be within one to two orders of magnitude of
    # the homogeneous half-space characteristic impedance.
    ratio = (
        np.abs(zxy)
        / np.abs(analytical)[None, :]
    )

    assert np.all(
        ratio > 0.05
    ), (
        "3D impedance is orders of magnitude too small. "
        f"Observed |Zxy|={np.abs(zxy)!r}, "
        f"reference |Z|={np.abs(analytical)!r}."
    )

    assert np.all(
        ratio < 20.0
    ), (
        "3D impedance is orders of magnitude too large. "
        f"Observed |Zxy|={np.abs(zxy)!r}, "
        f"reference |Z|={np.abs(analytical)!r}."
    )


def test_homogeneous_3d_vs_1d_reference():
    """
    Compare the homogeneous 3D response with the 1D reference kernel.

    Exact equality is not expected because the 3D discretization is
    still an initial simplified discretization.
    """
    mesh = _make_mesh()

    result, model, stations, _ = _run_homogeneous_forward(
        mesh=mesh,
    )

    kernel = MTKernel(
        mesh=mesh,
        frequencies=FREQUENCIES,
        mu_r=1.0,
    )

    reference = kernel.solve(
        model=model,
        locations=np.vstack(
            [
                station.coordinates
                for station in stations.stations
            ]
        ),
    )

    zxy_3d = result.impedance[
        :,
        :,
        0,
        1,
    ]

    zxy_1d = reference[
        :,
        :,
        0,
        1,
    ]

    ratio = (
        np.abs(zxy_3d)
        / np.maximum(
            np.abs(zxy_1d),
            1.0e-30,
        )
    )

    assert np.all(
        ratio > 0.05
    ), (
        "3D impedance is much too small compared with the 1D "
        f"reference. Ratio={ratio!r}."
    )

    assert np.all(
        ratio < 20.0
    ), (
        "3D impedance is much too large compared with the 1D "
        f"reference. Ratio={ratio!r}."
    )


# ----------------------------------------------------------------------
# Tensor structure
# ----------------------------------------------------------------------


def test_homogeneous_1d_reference_tensor_symmetry():
    """
    The 1D reference kernel must satisfy the exact tensor symmetry.
    """
    mesh = _make_mesh()

    model = _make_homogeneous_model(mesh)

    stations = _make_stations(mesh)

    kernel = MTKernel(
        mesh=mesh,
        frequencies=FREQUENCIES,
    )

    reference = kernel.solve(
        model=model,
        locations=np.vstack(
            [
                station.coordinates
                for station in stations.stations
            ]
        ),
    )

    assert np.allclose(
        reference[:, :, 0, 0],
        0.0,
        atol=1.0e-14,
    )

    assert np.allclose(
        reference[:, :, 1, 1],
        0.0,
        atol=1.0e-14,
    )

    assert np.allclose(
        reference[:, :, 1, 0],
        -reference[:, :, 0, 1],
        rtol=1.0e-12,
        atol=1.0e-14,
    )


def test_homogeneous_3d_tensor_is_approximately_antisymmetric():
    """
    In a homogeneous isotropic half-space, the 3D tensor should be
    approximately antisymmetric.

    The tolerance is intentionally loose because the current 3D
    station mapping uses nearest-neighbour sampling.
    """
    mesh = _make_mesh()

    result, _, _, _ = _run_homogeneous_forward(
        mesh=mesh,
    )

    zxx = result.impedance[
        :,
        :,
        0,
        0,
    ]

    zyy = result.impedance[
        :,
        :,
        1,
        1,
    ]

    zxy = result.impedance[
        :,
        :,
        0,
        1,
    ]

    zyx = result.impedance[
        :,
        :,
        1,
        0,
    ]

    scale = np.maximum(
        np.abs(zxy),
        1.0e-30,
    )

    diagonal_x_error = (
        np.abs(zxx)
        / scale
    )

    diagonal_y_error = (
        np.abs(zyy)
        / scale
    )

    antisymmetry_error = (
        np.abs(zxy + zyx)
        / scale
    )

    assert np.all(
        diagonal_x_error < 0.5
    ), (
        "Zxx is unexpectedly large relative to Zxy. "
        f"Relative error={diagonal_x_error!r}."
    )

    assert np.all(
        diagonal_y_error < 0.5
    ), (
        "Zyy is unexpectedly large relative to Zxy. "
        f"Relative error={diagonal_y_error!r}."
    )

    assert np.all(
        antisymmetry_error < 0.5
    ), (
        "Zyx is not approximately -Zxy. "
        f"Relative error={antisymmetry_error!r}."
    )


# ----------------------------------------------------------------------
# Phase and apparent resistivity
# ----------------------------------------------------------------------


def test_homogeneous_3d_phase_is_reasonable():
    """
    Compare the 3D impedance phase with the analytical homogeneous
    half-space phase.

    For the exp(+i omega t) convention used here, the analytical
    homogeneous response has a phase close to +45 degrees.
    """
    mesh = _make_mesh()

    result, _, _, _ = _run_homogeneous_forward(
        mesh=mesh,
    )

    zxy = result.impedance[
        :,
        :,
        0,
        1,
    ]

    kernel = MTKernel(
        mesh=mesh,
        frequencies=FREQUENCIES,
    )

    reference = kernel.homogeneous_half_space(
        RESISTIVITY,
    )

    phase_3d = np.angle(
        zxy,
        deg=True,
    )

    phase_1d = np.angle(
        reference,
        deg=True,
    )[None, :]

    phase_difference = np.abs(
        phase_3d - phase_1d
    )

    # Allow for the simplified 3D discretization and nearest-neighbour
    # station sampling.
    assert np.all(
        phase_difference < 45.0
    ), (
        "3D phase differs too strongly from the 1D reference. "
        f"3D phase={phase_3d!r}, "
        f"1D phase={phase_1d!r}."
    )


def test_homogeneous_3d_apparent_resistivity_is_physical():
    """
    Check apparent resistivity derived from the 3D impedance.

    rho_a = |Z|^2 / (mu0 * omega)

    For a homogeneous 100 ohm.m half-space, the expected result is
    approximately 100 ohm.m.
    """
    mesh = _make_mesh()

    result, _, _, _ = _run_homogeneous_forward(
        mesh=mesh,
    )

    zxy = result.impedance[
        :,
        :,
        0,
        1,
    ]

    omega = (
        2.0
        * np.pi
        * FREQUENCIES
    )

    rho_a = (
        np.abs(zxy) ** 2
        / (
            MU0
            * omega[None, :]
        )
    )

    assert np.all(
        np.isfinite(rho_a)
    )

    assert np.all(
        rho_a > 1.0
    ), (
        "Apparent resistivity is unrealistically small. "
        f"rho_a={rho_a!r}."
    )

    assert np.all(
        rho_a < 1.0e4
    ), (
        "Apparent resistivity is unrealistically large. "
        f"rho_a={rho_a!r}."
    )

    # The homogeneous model is 100 ohm.m. The simplified 3D
    # discretization does not need to reproduce it exactly, but should
    # remain within two orders of magnitude.
    ratio = (
        rho_a
        / RESISTIVITY
    )

    assert np.all(
        ratio > 0.01
    ), (
        "Apparent resistivity is more than two orders of magnitude "
        f"below the model resistivity. rho_a={rho_a!r}."
    )

    assert np.all(
        ratio < 100.0
    ), (
        "Apparent resistivity is more than two orders of magnitude "
        f"above the model resistivity. rho_a={rho_a!r}."
    )


def test_homogeneous_3d_response_depends_on_frequency():
    """
    A homogeneous MT response must vary with frequency.

    For a homogeneous half-space:

        |Z| proportional to sqrt(omega)
    """
    mesh = _make_mesh()

    result, _, _, _ = _run_homogeneous_forward(
        mesh=mesh,
        frequencies=FREQUENCIES,
    )

    zxy = result.impedance[
        0,
        :,
        0,
        1,
    ]

    magnitude = np.abs(zxy)

    assert np.all(
        np.isfinite(magnitude)
    )

    assert np.all(
        magnitude > 0.0
    )

    assert not np.allclose(
        magnitude[0],
        magnitude[-1],
        rtol=1.0e-3,
        atol=1.0e-30,
    )


# ----------------------------------------------------------------------
# Station mapping
# ----------------------------------------------------------------------


def test_station_mapping_returns_valid_indices():
    """
    Physical stations must map to valid Ex/Ey/Fx/Fy indices.
    """
    mesh = _make_mesh()

    stations = _make_stations(mesh)

    observations = stations_to_observations(
        mesh=mesh,
        stations=stations,
    )

    topology = build_topology(mesh)

    n_ex = (
        mesh.shape[0]
        * (mesh.shape[1] + 1)
        * (mesh.shape[2] + 1)
    )

    n_ey = (
        (mesh.shape[0] + 1)
        * mesh.shape[1]
        * (mesh.shape[2] + 1)
    )

    n_fx = (
        (mesh.shape[0] + 1)
        * mesh.shape[1]
        * mesh.shape[2]
    )

    n_fy = (
        mesh.shape[0]
        * (mesh.shape[1] + 1)
        * mesh.shape[2]
    )

    assert len(observations) == len(stations)

    for station_observation in observations:
        observation = station_observation.observation

        assert 0 <= observation.ex_index < n_ex
        assert 0 <= observation.ey_index < n_ey
        assert 0 <= observation.hx_index < n_fx
        assert 0 <= observation.hy_index < n_fy

    assert topology.n_edges > 0
    assert topology.n_faces > 0


def test_station_mapping_uses_physical_padding_coordinates():
    """
    Stations near the physical main-domain boundaries must still map
    to valid discrete locations when padding is present.
    """
    mesh = _make_mesh()

    stations = _make_stations(mesh)

    observations = stations_to_observations(
        mesh=mesh,
        stations=stations,
    )

    assert len(observations) == len(stations)

    for mapped in observations:
        assert np.all(
            np.isfinite(
                mapped.station.coordinates
            )
        )

        assert mapped.observation.ex_index >= 0
        assert mapped.observation.ey_index >= 0
        assert mapped.observation.hx_index >= 0
        assert mapped.observation.hy_index >= 0


# ----------------------------------------------------------------------
# Source polarizations
# ----------------------------------------------------------------------


def test_both_source_polarizations_produce_nonzero_impedance():
    """
    Both MT source polarizations must produce a measurable response.

    A near-zero response for both polarizations indicates that the
    source excitation or station extraction is disconnected from the
    solved field.
    """
    mesh = _make_mesh()

    result, _, _, _ = _run_homogeneous_forward(
        mesh=mesh,
    )

    zxy = result.impedance[
        :,
        :,
        0,
        1,
    ]

    zyx = result.impedance[
        :,
        :,
        1,
        0,
    ]

    assert np.any(
        np.abs(zxy) > 1.0e-8
    ), (
        "Zxy is effectively zero for every station/frequency. "
        f"|Zxy|={np.abs(zxy)!r}."
    )

    assert np.any(
        np.abs(zyx) > 1.0e-8
    ), (
        "Zyx is effectively zero for every station/frequency. "
        f"|Zyx|={np.abs(zyx)!r}."
    )


# ----------------------------------------------------------------------
# Stored fields
# ----------------------------------------------------------------------


def test_forward_stores_fields():
    """
    Stored electric fields must contain one EdgeElectricField per
    source polarization and finite numerical values.
    """
    mesh = _make_mesh()

    result, _, _, _ = _run_homogeneous_forward(
        mesh=mesh,
        store_fields=True,
    )

    assert result.electric_fields is not None
    assert result.magnetic_fields is not None

    assert len(
        result.electric_fields
    ) == FREQUENCIES.size

    assert len(
        result.magnetic_fields
    ) == FREQUENCIES.size

    for frequency_fields in result.electric_fields:
        # One field for each of the two MT source polarizations.
        assert np.shape(frequency_fields) == (2,)

        for electric_field in frequency_fields:
            assert isinstance(
                electric_field,
                EdgeElectricField,
            )

            values = np.asarray(
                electric_field.values,
                dtype=complex,
            )

            assert values.ndim == 1

            assert np.all(
                np.isfinite(values.real)
            )

            assert np.all(
                np.isfinite(values.imag)
            )

            assert values.size > 0

    for frequency_fields in result.magnetic_fields:
        assert np.shape(frequency_fields) == (2,)

        for magnetic_field in frequency_fields:
            values = np.asarray(
                getattr(
                    magnetic_field,
                    "values",
                    magnetic_field,
                ),
                dtype=complex,
            )

            assert values.ndim == 1
            assert values.size > 0

            assert np.all(
                np.isfinite(values.real)
            )

            assert np.all(
                np.isfinite(values.imag)
            )


# ----------------------------------------------------------------------
# Raw field diagnostics
# ----------------------------------------------------------------------


def test_debug_homogeneous_field_amplitudes():
    """
    Diagnostic test exposing the raw field amplitudes.

    This test is intentionally useful during development. If the
    impedance is near zero, this tells us whether the problem starts
    with E, H, or the station extraction.
    """
    mesh = _make_mesh()

    result, _, stations, _ = _run_homogeneous_forward(
        mesh=mesh,
        store_fields=True,
    )

    print()
    print("=== DEBUG MT 3D ===")

    for frequency_index, frequency in enumerate(
        FREQUENCIES
    ):
        print(
            f"frequency = {frequency:.6g} Hz"
        )

        electric_fields = (
            result.electric_fields[
                frequency_index
            ]
        )

        magnetic_fields = (
            result.magnetic_fields[
                frequency_index
            ]
        )

        for polarization in range(2):
            electric_field = (
                electric_fields[
                    polarization
                ]
            )

            magnetic_field = (
                magnetic_fields[
                    polarization
                ]
            )

            e_values = np.asarray(
                electric_field.values,
                dtype=complex,
            )

            h_values = np.asarray(
                getattr(
                    magnetic_field,
                    "values",
                    magnetic_field,
                ),
                dtype=complex,
            )

            print(
                f"  polarization = {polarization}"
            )

            print(
                "    E type   :",
                type(electric_field),
            )

            print(
                "    E shape  :",
                e_values.shape,
            )

            print(
                "    |E|max   :",
                float(np.max(np.abs(e_values))),
            )

            print(
                "    H type   :",
                type(magnetic_field),
            )

            print(
                "    H shape  :",
                h_values.shape,
            )

            print(
                "    |H|max   :",
                float(np.max(np.abs(h_values))),
            )

            print(
                "    |E| mean :",
                float(np.mean(np.abs(e_values))),
            )

            print(
                "    |H| mean :",
                float(np.mean(np.abs(h_values))),
            )

    print(
        "station count:",
        len(stations),
    )

    print(
        "|Zxy|:",
        np.abs(
            result.impedance[
                :,
                :,
                0,
                1,
            ]
        ),
    )

    print(
        "|Zyx|:",
        np.abs(
            result.impedance[
                :,
                :,
                1,
                0,
            ]
        ),
    )

    # The diagnostic itself should not fail merely because the physical
    # chain is currently broken. It only verifies that stored fields
    # exist and are finite.
    assert result.electric_fields is not None
    assert result.magnetic_fields is not None


# ----------------------------------------------------------------------
# Reference kernel checks
# ----------------------------------------------------------------------


def test_kernel_homogeneous_half_space_matches_analytical_formula():
    """
    The 1D reference kernel must reproduce its analytical homogeneous
    half-space formula.
    """
    mesh = _make_mesh()

    kernel = MTKernel(
        mesh=mesh,
        frequencies=FREQUENCIES,
        mu_r=1.0,
    )

    kernel_result = kernel.homogeneous_half_space(
        RESISTIVITY,
    )

    analytical = _analytical_homogeneous_impedance(
        FREQUENCIES,
        RESISTIVITY,
    )

    assert np.allclose(
        kernel_result,
        analytical,
        rtol=1.0e-12,
        atol=1.0e-14,
    )


def test_kernel_homogeneous_half_space_has_expected_apparent_resistivity():
    """
    The analytical homogeneous half-space must recover the model
    resistivity exactly through the MT apparent-resistivity formula.
    """
    mesh = _make_mesh()

    kernel = MTKernel(
        mesh=mesh,
        frequencies=FREQUENCIES,
        mu_r=1.0,
    )

    z = kernel.homogeneous_half_space(
        RESISTIVITY,
    )

    omega = (
        2.0
        * np.pi
        * FREQUENCIES
    )

    rho_a = (
        np.abs(z) ** 2
        / (
            MU0
            * omega
        )
    )

    assert np.allclose(
        rho_a,
        RESISTIVITY,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
