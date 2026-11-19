"""
Tests for the magnetic rectangular-prism kernel.

Coordinate convention
---------------------
x : East
y : North
z : Up

Magnetic field convention
-------------------------
D : declination, positive clockwise from North toward East.

I : inclination, positive downward.

F : magnetic field intensity in nT.

"""

import numpy as np
import pytest

from geoinvlab.methods.magnetic.kernel import (
    _field_direction,
    _magnetic_tensor,
    magnetic_kernel,
)


# ----------------------------------------------------------------------
# Reference geometry
# ----------------------------------------------------------------------

OBSERVATION = np.array(
    [
        0.0,
        0.0,
        10.0,
    ],
    dtype=float,
)

CELL_CENTER = np.array(
    [
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)

DX = 10.0
DY = 10.0
DZ = 10.0

INCLINATION = 60.0
DECLINATION = 20.0
FIELD_INTENSITY = 50000.0


# ----------------------------------------------------------------------
# Field direction
# ----------------------------------------------------------------------


def test_field_direction_has_unit_norm():
    """
    The magnetic field direction must have unit norm.
    """

    direction = _field_direction(
        inclination=INCLINATION,
        declination=DECLINATION,
    )

    assert np.isclose(
        np.linalg.norm(direction),
        1.0,
        atol=1.0e-14,
    )


def test_field_direction_horizontal_north():
    """
    D=0 and I=0 must point toward North.

    [East, North, Up] = [0, 1, 0]
    """

    direction = _field_direction(
        inclination=0.0,
        declination=0.0,
    )

    expected = np.array(
        [
            0.0,
            1.0,
            0.0,
        ],
        dtype=float,
    )

    assert np.allclose(
        direction,
        expected,
        atol=1.0e-14,
    )


def test_field_direction_horizontal_east():
    """
    D=90 and I=0 must point toward East.

    [East, North, Up] = [1, 0, 0]
    """

    direction = _field_direction(
        inclination=0.0,
        declination=90.0,
    )

    expected = np.array(
        [
            1.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    assert np.allclose(
        direction,
        expected,
        atol=1.0e-14,
    )


def test_field_direction_vertical_down():
    """
    I=+90 must point downward.

    Since z=Up:

        Down = [0, 0, -1]
    """

    direction = _field_direction(
        inclination=90.0,
        declination=0.0,
    )

    expected = np.array(
        [
            0.0,
            0.0,
            -1.0,
        ],
        dtype=float,
    )

    assert np.allclose(
        direction,
        expected,
        atol=1.0e-14,
    )


def test_field_direction_vertical_up():
    """
    I=-90 must point upward.

    Since z=Up:

        Up = [0, 0, 1]
    """

    direction = _field_direction(
        inclination=-90.0,
        declination=0.0,
    )

    expected = np.array(
        [
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    assert np.allclose(
        direction,
        expected,
        atol=1.0e-14,
    )


# ----------------------------------------------------------------------
# Magnetic tensor
# ----------------------------------------------------------------------


def test_magnetic_tensor_shape():
    """
    The magnetic tensor must have shape (3, 3).
    """

    tensor = _magnetic_tensor(
        observation=OBSERVATION,
        cell_center=CELL_CENTER,
        dx=DX,
        dy=DY,
        dz=DZ,
    )

    assert tensor.shape == (3, 3)


def test_magnetic_tensor_is_symmetric():
    """
    The magnetic tensor must be symmetric.
    """

    tensor = _magnetic_tensor(
        observation=OBSERVATION,
        cell_center=CELL_CENTER,
        dx=DX,
        dy=DY,
        dz=DZ,
    )

    assert np.allclose(
        tensor,
        tensor.T,
        atol=1.0e-12,
    )


def test_magnetic_tensor_changes_with_cell_geometry():
    """
    Changing the cell geometry must change the tensor.
    """

    tensor_1 = _magnetic_tensor(
        observation=OBSERVATION,
        cell_center=CELL_CENTER,
        dx=DX,
        dy=DY,
        dz=DZ,
    )

    tensor_2 = _magnetic_tensor(
        observation=OBSERVATION,
        cell_center=CELL_CENTER,
        dx=20.0,
        dy=DY,
        dz=DZ,
    )

    assert not np.allclose(
        tensor_1,
        tensor_2,
    )


# ----------------------------------------------------------------------
# Magnetic kernel
# ----------------------------------------------------------------------


def test_magnetic_kernel_returns_float():
    """
    The magnetic kernel must return a Python float.
    """

    response = magnetic_kernel(
        observation=OBSERVATION,
        cell_center=CELL_CENTER,
        dx=DX,
        dy=DY,
        dz=DZ,
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    assert isinstance(
        response,
        float,
    )


def test_magnetic_kernel_scales_linearly_with_field_intensity():
    """
    Magnetic response must scale linearly with field intensity.
    """

    response_1 = magnetic_kernel(
        observation=OBSERVATION,
        cell_center=CELL_CENTER,
        dx=DX,
        dy=DY,
        dz=DZ,
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    response_2 = magnetic_kernel(
        observation=OBSERVATION,
        cell_center=CELL_CENTER,
        dx=DX,
        dy=DY,
        dz=DZ,
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=2.0 * FIELD_INTENSITY,
    )

    assert np.isclose(
        response_2,
        2.0 * response_1,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_magnetic_kernel_changes_with_field_direction():
    """
    Changing D or I must change the total-field response.
    """

    response_1 = magnetic_kernel(
        observation=OBSERVATION,
        cell_center=CELL_CENTER,
        dx=DX,
        dy=DY,
        dz=DZ,
        inclination=30.0,
        declination=0.0,
        field_intensity=FIELD_INTENSITY,
    )

    response_2 = magnetic_kernel(
        observation=OBSERVATION,
        cell_center=CELL_CENTER,
        dx=DX,
        dy=DY,
        dz=DZ,
        inclination=60.0,
        declination=45.0,
        field_intensity=FIELD_INTENSITY,
    )

    assert not np.isclose(
        response_1,
        response_2,
    )


def test_magnetic_kernel_scales_linearly_with_susceptibility():
    """
    The kernel is the response per unit susceptibility.

    Therefore:

        anomaly = kernel * susceptibility
    """

    response = magnetic_kernel(
        observation=OBSERVATION,
        cell_center=CELL_CENTER,
        dx=DX,
        dy=DY,
        dz=DZ,
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    susceptibility_1 = 1.0
    susceptibility_2 = 3.0

    anomaly_1 = (
        susceptibility_1
        * response
    )

    anomaly_2 = (
        susceptibility_2
        * response
    )

    assert np.isclose(
        anomaly_2,
        3.0 * anomaly_1,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_magnetic_kernel_rejects_invalid_dimensions():
    """
    Invalid cell dimensions must raise ValueError.
    """

    with pytest.raises(ValueError):

        magnetic_kernel(
            observation=OBSERVATION,
            cell_center=CELL_CENTER,
            dx=0.0,
            dy=DY,
            dz=DZ,
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=FIELD_INTENSITY,
        )

    with pytest.raises(ValueError):

        magnetic_kernel(
            observation=OBSERVATION,
            cell_center=CELL_CENTER,
            dx=DX,
            dy=-1.0,
            dz=DZ,
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=FIELD_INTENSITY,
        )

    with pytest.raises(ValueError):

        magnetic_kernel(
            observation=OBSERVATION,
            cell_center=CELL_CENTER,
            dx=DX,
            dy=DY,
            dz=0.0,
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=FIELD_INTENSITY,
        )


def test_magnetic_kernel_rejects_negative_field_intensity():
    """
    Negative field intensity must be rejected.
    """

    with pytest.raises(ValueError):

        magnetic_kernel(
            observation=OBSERVATION,
            cell_center=CELL_CENTER,
            dx=DX,
            dy=DY,
            dz=DZ,
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=-1.0,
        )


def test_magnetic_kernel_rejects_non_finite_field_intensity():
    """
    Non-finite field intensity must be rejected.
    """

    with pytest.raises(ValueError):

        magnetic_kernel(
            observation=OBSERVATION,
            cell_center=CELL_CENTER,
            dx=DX,
            dy=DY,
            dz=DZ,
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=np.nan,
        )

    with pytest.raises(ValueError):

        magnetic_kernel(
            observation=OBSERVATION,
            cell_center=CELL_CENTER,
            dx=DX,
            dy=DY,
            dz=DZ,
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=np.inf,
        )


def test_magnetic_kernel_rejects_observation_on_cell_corner():
    """
    Observation exactly on a cell corner must be rejected.
    """

    observation = np.array(
        [
            DX / 2.0,
            DY / 2.0,
            DZ / 2.0,
        ],
        dtype=float,
    )

    with pytest.raises(ValueError):

        magnetic_kernel(
            observation=observation,
            cell_center=CELL_CENTER,
            dx=DX,
            dy=DY,
            dz=DZ,
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=FIELD_INTENSITY,
        )


# ----------------------------------------------------------------------
# Distance behaviour
# ----------------------------------------------------------------------


def test_magnetic_kernel_decreases_with_distance():
    """
    The magnitude of the magnetic response should decrease
    as the observation point moves farther from the source.
    """

    observation_near = np.array(
        [
            0.0,
            0.0,
            10.0,
        ],
        dtype=float,
    )

    observation_far = np.array(
        [
            0.0,
            0.0,
            50.0,
        ],
        dtype=float,
    )

    response_near = magnetic_kernel(
        observation=observation_near,
        cell_center=CELL_CENTER,
        dx=DX,
        dy=DY,
        dz=DZ,
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    response_far = magnetic_kernel(
        observation=observation_far,
        cell_center=CELL_CENTER,
        dx=DX,
        dy=DY,
        dz=DZ,
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    assert abs(response_far) < abs(
        response_near
    )
"""
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""
