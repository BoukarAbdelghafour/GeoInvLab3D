import numpy as np
import pytest

from geoinvlab.methods.gravity.kernel import gravity_kernel


def test_gravity_kernel_returns_finite_value():
    observation = np.array([0.0, 0.0, 10.0])
    cell_center = np.array([0.0, 0.0, 0.0])

    value = gravity_kernel(
        observation=observation,
        cell_center=cell_center,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )

    assert np.isfinite(value)


def test_gravity_kernel_is_nonzero():
    observation = np.array([0.0, 0.0, 10.0])
    cell_center = np.array([0.0, 0.0, 0.0])

    value = gravity_kernel(
        observation,
        cell_center,
        1.0,
        1.0,
        1.0,
    )

    assert value != 0.0


def test_gravity_kernel_decreases_with_distance():
    cell_center = np.array([0.0, 0.0, 0.0])

    near = gravity_kernel(
        np.array([0.0, 0.0, 10.0]),
        cell_center,
        1.0,
        1.0,
        1.0,
    )

    far = gravity_kernel(
        np.array([0.0, 0.0, 20.0]),
        cell_center,
        1.0,
        1.0,
        1.0,
    )

    assert abs(far) < abs(near)


def test_gravity_kernel_invalid_dimensions():
    with pytest.raises(ValueError):
        gravity_kernel(
            np.array([0.0, 0.0, 10.0]),
            np.array([0.0, 0.0, 0.0]),
            -1.0,
            1.0,
            1.0,
        )


def test_gravity_kernel_invalid_observation_shape():
    with pytest.raises(ValueError):
        gravity_kernel(
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            1.0,
            1.0,
            1.0,
        )
"""
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""
