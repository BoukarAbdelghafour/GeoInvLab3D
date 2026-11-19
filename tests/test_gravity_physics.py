import numpy as np

from geoinvlab.methods.gravity.kernel import gravity_kernel


def test_gravity_symmetry_x():
    observation = np.array([0.0, 0.0, 10.0])

    left = gravity_kernel(
        observation,
        np.array([-2.0, 0.0, 0.0]),
        1.0,
        1.0,
        1.0,
    )

    right = gravity_kernel(
        observation,
        np.array([2.0, 0.0, 0.0]),
        1.0,
        1.0,
        1.0,
    )

    np.testing.assert_allclose(
        abs(left),
        abs(right),
        rtol=1e-12,
    )


def test_gravity_symmetry_y():
    observation = np.array([0.0, 0.0, 10.0])

    front = gravity_kernel(
        observation,
        np.array([0.0, -2.0, 0.0]),
        1.0,
        1.0,
        1.0,
    )

    back = gravity_kernel(
        observation,
        np.array([0.0, 2.0, 0.0]),
        1.0,
        1.0,
        1.0,
    )

    np.testing.assert_allclose(
        abs(front),
        abs(back),
        rtol=1e-12,
    )


def test_gravity_decreases_with_horizontal_distance():
    observation = np.array([0.0, 0.0, 10.0])

    near = gravity_kernel(
        observation,
        np.array([0.0, 0.0, 0.0]),
        1.0,
        1.0,
        1.0,
    )

    far = gravity_kernel(
        observation,
        np.array([10.0, 0.0, 0.0]),
        1.0,
        1.0,
        1.0,
    )

    assert abs(far) < abs(near)


def test_gravity_decreases_with_vertical_distance():
    cell = np.array([0.0, 0.0, 0.0])

    near = gravity_kernel(
        np.array([0.0, 0.0, 5.0]),
        cell,
        1.0,
        1.0,
        1.0,
    )

    far = gravity_kernel(
        np.array([0.0, 0.0, 20.0]),
        cell,
        1.0,
        1.0,
        1.0,
    )

    assert abs(far) < abs(near)


def test_gravity_kernel_is_finite():
    value = gravity_kernel(
        np.array([0.0, 0.0, 10.0]),
        np.array([0.0, 0.0, 0.0]),
        1.0,
        1.0,
        1.0,
    )

    assert np.isfinite(value)
def test_gravity_far_field_matches_point_mass():
    import numpy as np

    from geoinvlab.methods.gravity.kernel import (
        G,
        gravity_kernel,
    )

    observation = np.array(
        [0.0, 0.0, 100.0]
    )

    cell_center = np.array(
        [0.0, 0.0, 0.0]
    )

    dx = 1.0
    dy = 1.0
    dz = 1.0

    numerical = gravity_kernel(
        observation,
        cell_center,
        dx,
        dy,
        dz,
    )

    # Unit density: rho = 1 kg/m^3.
    mass = dx * dy * dz

    distance = 100.0

    analytical = (
        G
        * mass
        / distance**2
    )

    np.testing.assert_allclose(
        abs(numerical),
        analytical,
        rtol=1e-2,
    )
"""
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""
