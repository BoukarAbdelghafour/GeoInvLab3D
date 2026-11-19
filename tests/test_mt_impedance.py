import numpy as np
import pytest

from geoinvlab.methods.mt.discretization.impedance import (
    MTImpedance,
    MTResponse,
    apparent_resistivity,
    impedance_from_fields,
    impedance_phase,
    off_diagonal_impedance,
    response_from_impedance,
)
from geoinvlab.methods.mt.discretization.maxwell import MU0


# ============================================================================
# MTImpedance
# ============================================================================


def test_mt_impedance_stores_components():
    impedance = MTImpedance(
        zxx=1.0 + 2.0j,
        zxy=3.0 + 4.0j,
        zyx=5.0 + 6.0j,
        zyy=7.0 + 8.0j,
    )

    assert impedance.zxx == 1.0 + 2.0j
    assert impedance.zxy == 3.0 + 4.0j
    assert impedance.zyx == 5.0 + 6.0j
    assert impedance.zyy == 7.0 + 8.0j


def test_mt_impedance_tensor_shape():
    impedance = MTImpedance(
        zxx=1.0 + 0.0j,
        zxy=2.0 + 0.0j,
        zyx=3.0 + 0.0j,
        zyy=4.0 + 0.0j,
    )

    tensor = impedance.tensor

    assert tensor.shape == (2, 2)
    assert tensor.dtype == complex


def test_mt_impedance_tensor_values():
    impedance = MTImpedance(
        zxx=1.0 + 2.0j,
        zxy=3.0 + 4.0j,
        zyx=5.0 + 6.0j,
        zyy=7.0 + 8.0j,
    )

    expected = np.array(
        [
            [1.0 + 2.0j, 3.0 + 4.0j],
            [5.0 + 6.0j, 7.0 + 8.0j],
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(
        impedance.tensor,
        expected,
    )


def test_mt_impedance_rejects_nan():
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        MTImpedance(
            zxx=np.nan + 0.0j,
            zxy=0.0 + 0.0j,
            zyx=0.0 + 0.0j,
            zyy=0.0 + 0.0j,
        )


def test_mt_impedance_rejects_infinite_value():
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        MTImpedance(
            zxx=1.0 + np.inf * 1j,
            zxy=0.0 + 0.0j,
            zyx=0.0 + 0.0j,
            zyy=0.0 + 0.0j,
        )


# ============================================================================
# impedance_from_fields
# ============================================================================


def test_impedance_from_fields_computes_all_ratios():
    ex = 2.0 + 4.0j
    ey = 6.0 + 8.0j
    hx = 1.0 + 0.0j
    hy = 2.0 + 0.0j

    impedance = impedance_from_fields(
        ex,
        ey,
        hx,
        hy,
    )

    np.testing.assert_allclose(
        impedance.zxx,
        ex / hx,
    )

    np.testing.assert_allclose(
        impedance.zxy,
        ex / hy,
    )

    np.testing.assert_allclose(
        impedance.zyx,
        ey / hx,
    )

    np.testing.assert_allclose(
        impedance.zyy,
        ey / hy,
    )


def test_impedance_from_fields_handles_zero_hx():
    impedance = impedance_from_fields(
        ex=1.0 + 1.0j,
        ey=2.0 + 2.0j,
        hx=0.0 + 0.0j,
        hy=1.0 + 0.0j,
    )

    assert impedance.zxx == 0.0 + 0.0j
    assert impedance.zyx == 0.0 + 0.0j

    np.testing.assert_allclose(
        impedance.zxy,
        1.0 + 1.0j,
    )

    np.testing.assert_allclose(
        impedance.zyy,
        2.0 + 2.0j,
    )


def test_impedance_from_fields_handles_zero_hy():
    impedance = impedance_from_fields(
        ex=1.0 + 1.0j,
        ey=2.0 + 2.0j,
        hx=1.0 + 0.0j,
        hy=0.0 + 0.0j,
    )

    np.testing.assert_allclose(
        impedance.zxx,
        1.0 + 1.0j,
    )

    np.testing.assert_allclose(
        impedance.zyx,
        2.0 + 2.0j,
    )

    assert impedance.zxy == 0.0 + 0.0j
    assert impedance.zyy == 0.0 + 0.0j


def test_impedance_from_fields_rejects_invalid_tolerance():
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        impedance_from_fields(
            1.0,
            2.0,
            1.0,
            1.0,
            denominator_tolerance=0.0,
        )


def test_impedance_from_fields_rejects_nan_field():
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        impedance_from_fields(
            np.nan + 0.0j,
            0.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
        )


# ============================================================================
# off_diagonal_impedance
# ============================================================================


def test_off_diagonal_impedance():
    ex = 4.0 + 2.0j
    ey = -3.0 + 5.0j
    hx = 2.0 + 0.0j
    hy = 4.0 + 0.0j

    zxy, zyx = off_diagonal_impedance(
        ex,
        ey,
        hx,
        hy,
    )

    np.testing.assert_allclose(
        zxy,
        ex / hy,
    )

    np.testing.assert_allclose(
        zyx,
        ey / hx,
    )


def test_off_diagonal_impedance_rejects_zero_hx():
    with pytest.raises(
        ValueError,
        match="Hx is too small",
    ):
        off_diagonal_impedance(
            ex=1.0,
            ey=2.0,
            hx=0.0,
            hy=1.0,
        )


def test_off_diagonal_impedance_rejects_zero_hy():
    with pytest.raises(
        ValueError,
        match="Hy is too small",
    ):
        off_diagonal_impedance(
            ex=1.0,
            ey=2.0,
            hx=1.0,
            hy=0.0,
        )


def test_off_diagonal_impedance_rejects_small_hx():
    with pytest.raises(
        ValueError,
        match="Hx is too small",
    ):
        off_diagonal_impedance(
            ex=1.0,
            ey=2.0,
            hx=1e-16,
            hy=1.0,
            denominator_tolerance=1e-14,
        )


def test_off_diagonal_impedance_rejects_small_hy():
    with pytest.raises(
        ValueError,
        match="Hy is too small",
    ):
        off_diagonal_impedance(
            ex=1.0,
            ey=2.0,
            hx=1.0,
            hy=1e-16,
            denominator_tolerance=1e-14,
        )


# ============================================================================
# Apparent resistivity
# ============================================================================


def test_apparent_resistivity_matches_formula():
    frequency = 1.0
    omega = 2.0 * np.pi * frequency

    z = 1.0 + 2.0j

    expected = abs(z) ** 2 / (MU0 * omega)

    result = apparent_resistivity(
        z,
        omega,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_apparent_resistivity_real_impedance():
    omega = 2.0 * np.pi

    z = 2.0 + 0.0j

    expected = 4.0 / (MU0 * omega)

    result = apparent_resistivity(
        z,
        omega,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_apparent_resistivity_accepts_array():
    omega = 2.0 * np.pi

    z = np.array(
        [
            1.0 + 0.0j,
            2.0 + 0.0j,
            3.0 + 4.0j,
        ],
        dtype=complex,
    )

    expected = np.abs(z) ** 2 / (MU0 * omega)

    result = apparent_resistivity(
        z,
        omega,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_apparent_resistivity_zero_impedance():
    omega = 2.0 * np.pi

    result = apparent_resistivity(
        0.0 + 0.0j,
        omega,
    )

    assert result == 0.0


def test_apparent_resistivity_rejects_zero_omega():
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        apparent_resistivity(
            1.0 + 0.0j,
            0.0,
        )


def test_apparent_resistivity_rejects_negative_omega():
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        apparent_resistivity(
            1.0 + 0.0j,
            -1.0,
        )


# ============================================================================
# Phase
# ============================================================================


def test_impedance_phase_zero():
    result = impedance_phase(
        1.0 + 0.0j,
    )

    np.testing.assert_allclose(
        result,
        0.0,
    )


def test_impedance_phase_positive_imaginary():
    result = impedance_phase(
        1.0 + 1.0j,
    )

    np.testing.assert_allclose(
        result,
        45.0,
    )


def test_impedance_phase_negative_imaginary():
    result = impedance_phase(
        1.0 - 1.0j,
    )

    np.testing.assert_allclose(
        result,
        -45.0,
    )


def test_impedance_phase_pure_positive_imaginary():
    result = impedance_phase(
        0.0 + 1.0j,
    )

    np.testing.assert_allclose(
        result,
        90.0,
    )


def test_impedance_phase_array():
    z = np.array(
        [
            1.0 + 0.0j,
            1.0 + 1.0j,
            1.0 - 1.0j,
        ],
        dtype=complex,
    )

    result = impedance_phase(z)

    expected = np.array(
        [
            0.0,
            45.0,
            -45.0,
        ]
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_impedance_phase_rejects_nan():
    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        impedance_phase(
            np.nan + 0.0j,
        )


# ============================================================================
# MTResponse
# ============================================================================


def make_impedance():
    return MTImpedance(
        zxx=0.1 + 0.2j,
        zxy=1.0 + 2.0j,
        zyx=-1.0 - 2.0j,
        zyy=0.3 + 0.4j,
    )


def test_mt_response_stores_frequency():
    impedance = make_impedance()

    response = MTResponse(
        impedance=impedance,
        frequency=10.0,
    )

    assert response.frequency == 10.0


def test_mt_response_computes_omega():
    impedance = make_impedance()

    response = MTResponse(
        impedance=impedance,
        frequency=10.0,
    )

    np.testing.assert_allclose(
        response.omega,
        20.0 * np.pi,
    )


def test_mt_response_apparent_resistivity():
    impedance = make_impedance()

    frequency = 10.0

    response = MTResponse(
        impedance=impedance,
        frequency=frequency,
    )

    expected = apparent_resistivity(
        impedance.tensor,
        2.0 * np.pi * frequency,
    )

    np.testing.assert_allclose(
        response.apparent_resistivity,
        expected,
    )


def test_mt_response_phase():
    impedance = make_impedance()

    response = MTResponse(
        impedance=impedance,
        frequency=10.0,
    )

    expected = impedance_phase(
        impedance.tensor,
    )

    np.testing.assert_allclose(
        response.phase,
        expected,
    )


def test_mt_response_rejects_zero_frequency():
    impedance = make_impedance()

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        MTResponse(
            impedance=impedance,
            frequency=0.0,
        )


def test_mt_response_rejects_negative_frequency():
    impedance = make_impedance()

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        MTResponse(
            impedance=impedance,
            frequency=-1.0,
        )


def test_mt_response_rejects_wrong_impedance_type():
    with pytest.raises(
        TypeError,
        match="MTImpedance",
    ):
        response_from_impedance(
            impedance=object(),
            frequency=1.0,
        )


# ============================================================================
# response_from_impedance
# ============================================================================


def test_response_from_impedance():
    impedance = make_impedance()

    response = response_from_impedance(
        impedance,
        frequency=5.0,
    )

    assert isinstance(
        response,
        MTResponse,
    )

    assert response.impedance is impedance
    assert response.frequency == 5.0


# ============================================================================
# Homogeneous half-space analytical reference
# ============================================================================


def test_homogeneous_impedance_gives_expected_resistivity():
    """
    For a homogeneous half-space, the impedance magnitude satisfies:

        |Z|^2 = rho * mu0 * omega

    therefore:

        rho_a = rho.
    """
    rho = 100.0
    frequency = 1.0
    omega = 2.0 * np.pi * frequency

    # The exact complex phase/sign depends on the time convention.
    # Only the magnitude is relevant for apparent resistivity.
    z = np.sqrt(
        rho * MU0 * omega
    ) * (1.0 + 1.0j) / np.sqrt(2.0)

    result = apparent_resistivity(
        z,
        omega,
    )

    np.testing.assert_allclose(
        result,
        rho,
        rtol=1e-12,
        atol=1e-12,
    )


def test_homogeneous_tensor_has_zero_diagonal():
    impedance = MTImpedance(
        zxx=0.0 + 0.0j,
        zxy=1.0 + 2.0j,
        zyx=-1.0 - 2.0j,
        zyy=0.0 + 0.0j,
    )

    np.testing.assert_allclose(
        impedance.tensor[0, 0],
        0.0,
    )

    np.testing.assert_allclose(
        impedance.tensor[1, 1],
        0.0,
    )


def test_homogeneous_tensor_off_diagonal_antisymmetry():
    z = 1.0 + 2.0j

    impedance = MTImpedance(
        zxx=0.0,
        zxy=z,
        zyx=-z,
        zyy=0.0,
    )

    np.testing.assert_allclose(
        impedance.zyx,
        -impedance.zxy,
    )
