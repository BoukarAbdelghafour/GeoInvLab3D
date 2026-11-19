"""
Tests for the 1D magnetotelluric reference kernel.

GeoInvLab3D MT convention
-------------------------
Time dependence:

    exp(+i * omega * t)

Faraday's law:

    curl(E) = -i * omega * mu * H

Complex wavenumber:

    k^2 = -i * omega * mu * sigma

Physical branch:

    Re(k) > 0
    Im(k) < 0

Downward electric field:

    E(z) = E0 * exp(-i * k * z)

For X polarization:

    Zxy = Ex / Hy = omega * mu / k

For Y polarization:

    Zyx = Ey / Hx = -omega * mu / k

Therefore, for a homogeneous half-space:

    phase(Zxy) = +45 degrees
    phase(Zyx) = -135 degrees

and

    rho_a = |Z|^2 / (mu0 * omega).
"""

from __future__ import annotations

import numpy as np
import pytest

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.core.model.model import Model3D
from geoinvlab.methods.mt.kernel import (
    MU0,
    MTKernel,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RHO = 100.0
SIGMA = 1.0 / RHO

FREQUENCIES = np.array(
    [
        0.01,
        0.1,
        1.0,
        10.0,
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Mesh and model helpers
# ---------------------------------------------------------------------------


def _make_mesh() -> StructuredMesh3D:
    """
    Build a small structured 3D mesh.
    """
    return StructuredMesh3D(
        nx=2,
        ny=2,
        nz=4,
        dx=100.0,
        dy=100.0,
        dz=100.0,
    )


def _make_resistivity_model(
    mesh: StructuredMesh3D,
    resistivity: float = RHO,
) -> Model3D:
    """
    Build a homogeneous resistivity model.
    """
    return Model3D(
        mesh=mesh,
        values=np.full(
            mesh.n_cells,
            float(resistivity),
            dtype=float,
        ),
        name="resistivity",
    )


def _make_conductivity_model(
    mesh: StructuredMesh3D,
    conductivity: float = SIGMA,
) -> Model3D:
    """
    Build a homogeneous conductivity model.
    """
    return Model3D(
        mesh=mesh,
        values=np.full(
            mesh.n_cells,
            float(conductivity),
            dtype=float,
        ),
        name="conductivity",
    )


def _make_locations() -> np.ndarray:
    """
    Return representative surface stations.
    """
    return np.array(
        [
            [50.0, 50.0, 0.0],
            [150.0, 150.0, 0.0],
        ],
        dtype=float,
    )


def _make_kernel(
    *,
    mu_r: float = 1.0,
) -> MTKernel:
    """
    Construct an MTKernel using its current public API.
    """
    return MTKernel(
        mesh=_make_mesh(),
        frequencies=FREQUENCIES,
        mu_r=mu_r,
    )


# ---------------------------------------------------------------------------
# Analytical reference
# ---------------------------------------------------------------------------


def _physical_wavenumber(
    frequency: float,
    conductivity: float,
    mu_r: float = 1.0,
) -> complex:
    """
    Return the physical MT wavenumber.
    """
    frequency = float(frequency)
    conductivity = float(conductivity)
    mu_r = float(mu_r)

    omega = (
        2.0
        * np.pi
        * frequency
    )

    mu = MU0 * mu_r

    k = np.sqrt(
        -1j
        * omega
        * mu
        * conductivity
    )

    if (
        k.real < 0.0
        or (
            np.isclose(k.real, 0.0)
            and k.imag > 0.0
        )
    ):
        k = -k

    assert k.real > 0.0
    assert k.imag < 0.0

    return complex(k)


def _analytical_zxy(
    frequency: float,
    resistivity: float,
    mu_r: float = 1.0,
) -> complex:
    """
    Analytical homogeneous half-space Zxy.
    """
    conductivity = 1.0 / float(
        resistivity
    )

    omega = (
        2.0
        * np.pi
        * float(frequency)
    )

    mu = MU0 * float(mu_r)

    k = _physical_wavenumber(
        frequency=frequency,
        conductivity=conductivity,
        mu_r=mu_r,
    )

    return (
        omega
        * mu
        / k
    )


def _analytical_zyx(
    frequency: float,
    resistivity: float,
    mu_r: float = 1.0,
) -> complex:
    """
    Analytical homogeneous half-space Zyx.
    """
    return -_analytical_zxy(
        frequency=frequency,
        resistivity=resistivity,
        mu_r=mu_r,
    )


# ---------------------------------------------------------------------------
# Wavenumber tests
# ---------------------------------------------------------------------------


def test_physical_wavenumber_has_correct_branch():
    """
    The physical branch must satisfy

        Re(k) > 0
        Im(k) < 0.
    """
    for frequency in FREQUENCIES:
        k = _physical_wavenumber(
            frequency=frequency,
            conductivity=SIGMA,
        )

        assert np.isfinite(k.real)
        assert np.isfinite(k.imag)

        assert k.real > 0.0
        assert k.imag < 0.0


def test_physical_wavenumber_has_equal_magnitude_parts():
    """
    For a homogeneous isotropic half-space,

        k = a - i*a.

    Therefore the real and imaginary parts have equal magnitude.
    """
    for frequency in FREQUENCIES:
        k = _physical_wavenumber(
            frequency=frequency,
            conductivity=SIGMA,
        )

        assert np.isclose(
            abs(k.real),
            abs(k.imag),
            rtol=1.0e-12,
            atol=1.0e-15,
        )


# ---------------------------------------------------------------------------
# Analytical impedance tests
# ---------------------------------------------------------------------------


def test_analytical_zxy_has_plus_45_degree_phase():
    """
    Zxy must have +45 degree phase.
    """
    for frequency in FREQUENCIES:
        zxy = _analytical_zxy(
            frequency=frequency,
            resistivity=RHO,
        )

        phase = np.angle(
            zxy,
            deg=True,
        )

        assert np.isclose(
            phase,
            45.0,
            atol=1.0e-12,
        )


def test_analytical_zyx_is_negative_zxy():
    """
    For an isotropic half-space,

        Zyx = -Zxy.
    """
    for frequency in FREQUENCIES:
        zxy = _analytical_zxy(
            frequency=frequency,
            resistivity=RHO,
        )

        zyx = _analytical_zyx(
            frequency=frequency,
            resistivity=RHO,
        )

        np.testing.assert_allclose(
            zyx,
            -zxy,
            rtol=1.0e-12,
            atol=1.0e-15,
        )


def test_analytical_zyx_has_wrapped_minus_135_degree_phase():
    """
    Zyx has wrapped phase -135 degrees.
    """
    for frequency in FREQUENCIES:
        zyx = _analytical_zyx(
            frequency=frequency,
            resistivity=RHO,
        )

        phase = np.angle(
            zyx,
            deg=True,
        )

        assert np.isclose(
            phase,
            -135.0,
            atol=1.0e-12,
        )


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------


def test_kernel_construction():
    """
    MTKernel must use the current public constructor.
    """
    kernel = _make_kernel()

    assert isinstance(
        kernel,
        MTKernel,
    )

    assert kernel.n_frequencies == (
        FREQUENCIES.size
    )

    np.testing.assert_allclose(
        kernel.frequencies,
        FREQUENCIES,
    )

    assert np.isclose(
        kernel.permeability,
        MU0,
    )


def test_kernel_angular_frequencies():
    """
    Angular frequencies must be 2*pi*f.
    """
    kernel = _make_kernel()

    expected = (
        2.0
        * np.pi
        * FREQUENCIES
    )

    np.testing.assert_allclose(
        kernel.angular_frequencies,
        expected,
        rtol=1.0e-14,
        atol=1.0e-15,
    )


def test_kernel_rejects_empty_frequencies():
    """
    An empty frequency vector is invalid.
    """
    mesh = _make_mesh()

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        MTKernel(
            mesh=mesh,
            frequencies=np.array(
                [],
                dtype=float,
            ),
        )


def test_kernel_rejects_non_positive_frequencies():
    """
    Zero and negative frequencies are invalid.
    """
    mesh = _make_mesh()

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        MTKernel(
            mesh=mesh,
            frequencies=np.array(
                [0.0, 1.0],
                dtype=float,
            ),
        )


def test_kernel_rejects_invalid_mu_r():
    """
    Relative permeability must be strictly positive.
    """
    mesh = _make_mesh()

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        MTKernel(
            mesh=mesh,
            frequencies=FREQUENCIES,
            mu_r=0.0,
        )


# ---------------------------------------------------------------------------
# Physical wavenumber API
# ---------------------------------------------------------------------------


def test_kernel_physical_wavenumber_matches_reference():
    """
    The kernel's public wavenumber calculation must match the
    analytical reference.
    """
    kernel = _make_kernel()

    for frequency in FREQUENCIES:
        actual = kernel.physical_wavenumber(
            frequency=frequency,
            conductivity=SIGMA,
        )

        expected = _physical_wavenumber(
            frequency=frequency,
            conductivity=SIGMA,
        )

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1.0e-14,
            atol=1.0e-15,
        )


def test_kernel_physical_wavenumber_has_plus_real_minus_imaginary():
    """
    The selected physical branch must have

        Re(k) > 0
        Im(k) < 0.
    """
    kernel = _make_kernel()

    for frequency in FREQUENCIES:
        k = kernel.physical_wavenumber(
            frequency=frequency,
            conductivity=SIGMA,
        )

        assert k.real > 0.0
        assert k.imag < 0.0


# ---------------------------------------------------------------------------
# Forward solve
# ---------------------------------------------------------------------------


def test_kernel_returns_expected_impedance_shape():
    """
    solve() must return

        (n_stations, n_frequencies, 2, 2).
    """
    mesh = _make_mesh()

    model = _make_resistivity_model(
        mesh
    )

    locations = _make_locations()

    kernel = MTKernel(
        mesh=mesh,
        frequencies=FREQUENCIES,
    )

    impedance = kernel.solve(
        model=model,
        locations=locations,
    )

    assert impedance.shape == (
        locations.shape[0],
        FREQUENCIES.size,
        2,
        2,
    )

    assert np.iscomplexobj(
        impedance
    )


def test_kernel_homogeneous_zxy_matches_analytical_solution():
    """
    Zxy from the kernel must match the analytical homogeneous
    half-space impedance.
    """
    mesh = _make_mesh()

    model = _make_resistivity_model(
        mesh
    )

    locations = _make_locations()

    kernel = _make_kernel()

    impedance = kernel.solve(
        model=model,
        locations=locations,
    )

    actual = impedance[
        :,
        :,
        0,
        1,
    ]

    expected = np.array(
        [
            _analytical_zxy(
                frequency=frequency,
                resistivity=RHO,
            )
            for frequency in FREQUENCIES
        ],
        dtype=complex,
    )

    for station_index in range(
        locations.shape[0]
    ):
        np.testing.assert_allclose(
            actual[station_index],
            expected,
            rtol=1.0e-10,
            atol=1.0e-14,
        )


def test_kernel_homogeneous_zyx_matches_analytical_solution():
    """
    Zyx from the kernel must match -Zxy.
    """
    mesh = _make_mesh()

    model = _make_resistivity_model(
        mesh
    )

    locations = _make_locations()

    kernel = _make_kernel()

    impedance = kernel.solve(
        model=model,
        locations=locations,
    )

    actual = impedance[
        :,
        :,
        1,
        0,
    ]

    expected = np.array(
        [
            _analytical_zyx(
                frequency=frequency,
                resistivity=RHO,
            )
            for frequency in FREQUENCIES
        ],
        dtype=complex,
    )

    for station_index in range(
        locations.shape[0]
    ):
        np.testing.assert_allclose(
            actual[station_index],
            expected,
            rtol=1.0e-10,
            atol=1.0e-14,
        )


def test_kernel_homogeneous_diagonal_terms_are_zero():
    """
    Zxx and Zyy must vanish for a homogeneous isotropic 1D Earth.
    """
    mesh = _make_mesh()

    model = _make_resistivity_model(
        mesh
    )

    impedance = _make_kernel().solve(
        model=model,
        locations=_make_locations(),
    )

    np.testing.assert_allclose(
        impedance[:, :, 0, 0],
        0.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        impedance[:, :, 1, 1],
        0.0,
        atol=1.0e-14,
    )


def test_kernel_homogeneous_impedance_is_antisymmetric():
    """
    A homogeneous isotropic 1D Earth satisfies

        Zyx = -Zxy.
    """
    mesh = _make_mesh()

    model = _make_resistivity_model(
        mesh
    )

    impedance = _make_kernel().solve(
        model=model,
        locations=_make_locations(),
    )

    np.testing.assert_allclose(
        impedance[:, :, 1, 0],
        -impedance[:, :, 0, 1],
        rtol=1.0e-10,
        atol=1.0e-14,
    )


# ---------------------------------------------------------------------------
# Phase tests
# ---------------------------------------------------------------------------


def test_kernel_zxy_has_plus_45_degree_phase():
    """
    The kernel must produce the definitive +45 degree Zxy phase.
    """
    mesh = _make_mesh()

    model = _make_resistivity_model(
        mesh
    )

    impedance = _make_kernel().solve(
        model=model,
        locations=_make_locations(),
    )

    phases = np.angle(
        impedance[:, :, 0, 1],
        deg=True,
    )

    np.testing.assert_allclose(
        phases,
        45.0,
        atol=1.0e-10,
    )


def test_kernel_zyx_has_wrapped_minus_135_degree_phase():
    """
    The kernel must produce the wrapped -135 degree Zyx phase.
    """
    mesh = _make_mesh()

    model = _make_resistivity_model(
        mesh
    )

    impedance = _make_kernel().solve(
        model=model,
        locations=_make_locations(),
    )

    phases = np.angle(
        impedance[:, :, 1, 0],
        deg=True,
    )

    np.testing.assert_allclose(
        phases,
        -135.0,
        atol=1.0e-10,
    )


# ---------------------------------------------------------------------------
# Apparent resistivity
# ---------------------------------------------------------------------------


def test_kernel_homogeneous_apparent_resistivity_is_100_ohm_m():
    """
    For the homogeneous 100 ohm.m half-space,

        rho_a = |Zxy|^2 / (mu0 * omega)

    must recover 100 ohm.m.
    """
    mesh = _make_mesh()

    model = _make_resistivity_model(
        mesh
    )

    impedance = _make_kernel().solve(
        model=model,
        locations=_make_locations(),
    )

    zxy = impedance[
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

    np.testing.assert_allclose(
        rho_a,
        RHO,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


# ---------------------------------------------------------------------------
# Resistivity / conductivity parameterizations
# ---------------------------------------------------------------------------


def test_kernel_resistivity_and_conductivity_models_are_equivalent():
    """
    Equivalent resistivity and conductivity models must give the
    same impedance.
    """
    mesh = _make_mesh()
    locations = _make_locations()

    resistivity_model = _make_resistivity_model(
        mesh,
        resistivity=RHO,
    )

    conductivity_model = _make_conductivity_model(
        mesh,
        conductivity=SIGMA,
    )

    kernel = _make_kernel()

    z_rho = kernel.solve(
        model=resistivity_model,
        locations=locations,
    )

    z_sigma = kernel.solve(
        model=conductivity_model,
        locations=locations,
    )

    np.testing.assert_allclose(
        z_sigma,
        z_rho,
        rtol=1.0e-10,
        atol=1.0e-14,
    )


# ---------------------------------------------------------------------------
# Station independence
# ---------------------------------------------------------------------------


def test_kernel_homogeneous_response_is_station_independent():
    """
    A homogeneous 1D Earth must have the same impedance at every
    station.
    """
    mesh = _make_mesh()

    model = _make_resistivity_model(
        mesh
    )

    locations = np.array(
        [
            [10.0, 20.0, 0.0],
            [50.0, 50.0, 0.0],
            [150.0, 250.0, 0.0],
            [350.0, 450.0, 0.0],
        ],
        dtype=float,
    )

    impedance = _make_kernel().solve(
        model=model,
        locations=locations,
    )

    reference = impedance[0]

    for station_index in range(
        1,
        locations.shape[0],
    ):
        np.testing.assert_allclose(
            impedance[station_index],
            reference,
            rtol=1.0e-10,
            atol=1.0e-14,
        )


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------


def test_forward_alias_matches_solve():
    """
    forward() must be equivalent to solve().
    """
    mesh = _make_mesh()

    model = _make_resistivity_model(
        mesh
    )

    locations = _make_locations()
    kernel = _make_kernel()

    expected = kernel.solve(
        model=model,
        locations=locations,
    )

    actual = kernel.forward(
        model=model,
        locations=locations,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-14,
        atol=1.0e-15,
    )


def test_compute_impedance_alias_matches_solve():
    """
    compute_impedance() must be equivalent to solve().
    """
    mesh = _make_mesh()

    model = _make_resistivity_model(
        mesh
    )

    locations = _make_locations()
    kernel = _make_kernel()

    expected = kernel.solve(
        model=model,
        locations=locations,
    )

    actual = kernel.compute_impedance(
        model=model,
        locations=locations,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-14,
        atol=1.0e-15,
    )
 

