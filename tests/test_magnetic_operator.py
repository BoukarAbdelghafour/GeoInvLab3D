"""
Tests for the magnetic forward operator.
"""

import numpy as np
import pytest

from geoinvlab.core.mesh import StructuredMesh3D
from geoinvlab.core.model import Model3D

from geoinvlab.methods.magnetic.operator import MagneticOperator


# ----------------------------------------------------------------------
# Test configuration
# ----------------------------------------------------------------------

OBSERVATION_LOCATIONS = np.array(
    [
        [0.0, 0.0, 10.0],
        [20.0, 0.0, 10.0],
        [0.0, 20.0, 10.0],
    ],
    dtype=float,
)

INCLINATION = 60.0
DECLINATION = 20.0
FIELD_INTENSITY = 50000.0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _create_mesh():
    """
    Create a small structured 3D mesh for testing.
    """

    return StructuredMesh3D(
        nx=2,
        ny=2,
        nz=1,
        dx=10.0,
        dy=10.0,
        dz=10.0,
        origin=np.array(
            [
                -5.0,
                -5.0,
                0.0,
            ],
            dtype=float,
        ),
    )


def _create_model(values=None):
    """
    Create a small magnetic model.
    """

    mesh = _create_mesh()

    n_parameters = (
        mesh.nx
        * mesh.ny
        * mesh.nz
    )

    if values is None:
        values = np.ones(
            n_parameters,
            dtype=float,
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    return Model3D(
        mesh=mesh,
        values=values,
    )


def _create_operator(
    observation_locations=None,
    inclination=INCLINATION,
    declination=DECLINATION,
    field_intensity=FIELD_INTENSITY,
):
    """
    Create a magnetic operator for testing.
    """

    if observation_locations is None:
        observation_locations = (
            OBSERVATION_LOCATIONS
        )

    return MagneticOperator(
        observation_locations=observation_locations,
        inclination=inclination,
        declination=declination,
        field_intensity=field_intensity,
    )


# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------


def test_magnetic_operator_initialization():
    """
    The magnetic operator must initialize correctly.
    """

    operator = _create_operator()

    assert operator.n_data == 3

    assert operator.n_parameters is None


def test_magnetic_operator_observation_locations_shape():
    """
    Observation locations must have shape (n_data, 3).
    """

    with pytest.raises(ValueError):

        MagneticOperator(
            observation_locations=np.array(
                [
                    [0.0, 0.0],
                    [10.0, 10.0],
                ]
            ),
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=FIELD_INTENSITY,
        )


def test_magnetic_operator_rejects_invalid_observation_locations():
    """
    Non-finite observation coordinates must be rejected.
    """

    observations = np.array(
        [
            [0.0, 0.0, np.nan],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError):

        MagneticOperator(
            observation_locations=observations,
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=FIELD_INTENSITY,
        )


# ----------------------------------------------------------------------
# Magnetic field parameters
# ----------------------------------------------------------------------


def test_magnetic_operator_rejects_non_finite_inclination():
    """
    Non-finite inclination must be rejected.
    """

    with pytest.raises(ValueError):

        MagneticOperator(
            observation_locations=OBSERVATION_LOCATIONS,
            inclination=np.nan,
            declination=DECLINATION,
            field_intensity=FIELD_INTENSITY,
        )


def test_magnetic_operator_rejects_non_finite_declination():
    """
    Non-finite declination must be rejected.
    """

    with pytest.raises(ValueError):

        MagneticOperator(
            observation_locations=OBSERVATION_LOCATIONS,
            inclination=INCLINATION,
            declination=np.inf,
            field_intensity=FIELD_INTENSITY,
        )


def test_magnetic_operator_rejects_non_finite_field_intensity():
    """
    Non-finite field intensity must be rejected.
    """

    with pytest.raises(ValueError):

        MagneticOperator(
            observation_locations=OBSERVATION_LOCATIONS,
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=np.nan,
        )


def test_magnetic_operator_rejects_negative_field_intensity():
    """
    Negative field intensity must be rejected.
    """

    with pytest.raises(ValueError):

        MagneticOperator(
            observation_locations=OBSERVATION_LOCATIONS,
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=-100.0,
        )


# ----------------------------------------------------------------------
# Jacobian
# ----------------------------------------------------------------------


def test_magnetic_operator_jacobian_shape():
    """
    The Jacobian must have shape:

        (n_data, n_parameters)
    """

    model = _create_model()

    operator = _create_operator()

    jacobian = operator.jacobian(
        model
    )

    assert jacobian.shape == (
        operator.n_data,
        model.n_parameters,
    )


def test_magnetic_operator_n_parameters_after_jacobian():
    """
    n_parameters must be available after Jacobian construction.
    """

    model = _create_model()

    operator = _create_operator()

    assert operator.n_parameters is None

    operator.jacobian(
        model
    )

    assert (
        operator.n_parameters
        == model.n_parameters
    )


def test_magnetic_operator_jacobian_is_finite():
    """
    The Jacobian must contain only finite values.
    """

    model = _create_model()

    operator = _create_operator()

    jacobian = operator.jacobian(
        model
    )

    assert np.all(
        np.isfinite(jacobian)
    )


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------


def test_magnetic_operator_prediction_shape():
    """
    Predicted magnetic anomalies must have shape (n_data,).
    """

    model = _create_model()

    operator = _create_operator()

    predicted = operator.predict(
        model
    )

    assert predicted.shape == (
        operator.n_data,
    )


def test_magnetic_operator_prediction_is_finite():
    """
    Predicted magnetic anomalies must be finite.
    """

    model = _create_model()

    operator = _create_operator()

    predicted = operator.predict(
        model
    )

    assert np.all(
        np.isfinite(predicted)
    )


def test_magnetic_operator_prediction_equals_matrix_product():
    """
    Prediction must satisfy:

        d_pred = G @ chi
    """

    model = _create_model(
        values=np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            dtype=float,
        )
    )

    operator = _create_operator()

    jacobian = operator.jacobian(
        model
    )

    predicted = operator.predict(
        model
    )

    expected = (
        jacobian
        @ model.values
    )

    assert np.allclose(
        predicted,
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


# ----------------------------------------------------------------------
# Linearity with susceptibility
# ----------------------------------------------------------------------


def test_magnetic_operator_is_linear_in_susceptibility():
    """
    The magnetic forward problem must be linear in susceptibility.

        G @ (a * chi)
        =
        a * (G @ chi)
    """

    model_1 = _create_model(
        values=np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            dtype=float,
        )
    )

    model_2 = _create_model(
        values=3.0 * model_1.values
    )

    operator = _create_operator()

    prediction_1 = operator.predict(
        model_1
    )

    prediction_2 = operator.predict(
        model_2
    )

    assert np.allclose(
        prediction_2,
        3.0 * prediction_1,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_magnetic_operator_is_linear_under_superposition():
    """
    The forward operator must satisfy:

        G @ (m1 + m2)
        =
        G @ m1 + G @ m2
    """

    model_1 = _create_model(
        values=np.array(
            [
                1.0,
                0.5,
                2.0,
                1.5,
            ],
            dtype=float,
        )
    )

    model_2 = _create_model(
        values=np.array(
            [
                0.2,
                1.0,
                0.5,
                2.0,
            ],
            dtype=float,
        )
    )

    model_sum = _create_model(
        values=(
            model_1.values
            + model_2.values
        )
    )

    operator = _create_operator()

    prediction_1 = operator.predict(
        model_1
    )

    prediction_2 = operator.predict(
        model_2
    )

    prediction_sum = operator.predict(
        model_sum
    )

    assert np.allclose(
        prediction_sum,
        prediction_1 + prediction_2,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


# ----------------------------------------------------------------------
# Field intensity
# ----------------------------------------------------------------------


def test_magnetic_operator_scales_with_field_intensity():
    """
    The Jacobian must scale linearly with F.
    """

    model = _create_model()

    operator_1 = _create_operator(
        field_intensity=50000.0
    )

    operator_2 = _create_operator(
        field_intensity=100000.0
    )

    jacobian_1 = operator_1.jacobian(
        model
    )

    jacobian_2 = operator_2.jacobian(
        model
    )

    assert np.allclose(
        jacobian_2,
        2.0 * jacobian_1,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


# ----------------------------------------------------------------------
# Field direction
# ----------------------------------------------------------------------


def test_magnetic_operator_changes_with_inclination():
    """
    Changing inclination must change the Jacobian.
    """

    model = _create_model()

    operator_1 = _create_operator(
        inclination=20.0
    )

    operator_2 = _create_operator(
        inclination=70.0
    )

    jacobian_1 = operator_1.jacobian(
        model
    )

    jacobian_2 = operator_2.jacobian(
        model
    )

    assert not np.allclose(
        jacobian_1,
        jacobian_2,
    )


def test_magnetic_operator_changes_with_declination():
    """
    Changing declination must change the Jacobian.
    """

    model = _create_model()

    operator_1 = _create_operator(
        declination=0.0
    )

    operator_2 = _create_operator(
        declination=90.0
    )

    jacobian_1 = operator_1.jacobian(
        model
    )

    jacobian_2 = operator_2.jacobian(
        model
    )

    assert not np.allclose(
        jacobian_1,
        jacobian_2,
    )


# ----------------------------------------------------------------------
# Jacobian caching
# ----------------------------------------------------------------------


def test_magnetic_operator_caches_jacobian():
    """
    Repeated calls to jacobian() must return the cached matrix.
    """

    model = _create_model()

    operator = _create_operator()

    jacobian_1 = operator.jacobian(
        model
    )

    jacobian_2 = operator.jacobian(
        model
    )

    assert jacobian_1 is jacobian_2


def test_magnetic_operator_clear_cache():
    """
    clear_cache() must invalidate the cached Jacobian.
    """

    model = _create_model()

    operator = _create_operator()

    jacobian_1 = operator.jacobian(
        model
    )

    operator.clear_cache()

    assert operator.n_parameters is None

    jacobian_2 = operator.jacobian(
        model
    )

    assert jacobian_2 is not jacobian_1

    assert np.allclose(
        jacobian_1,
        jacobian_2,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


# ----------------------------------------------------------------------
# Mesh dependency
# ----------------------------------------------------------------------


def test_magnetic_operator_recomputes_for_different_mesh():
    """
    A different mesh must invalidate the cached Jacobian.
    """

    model_1 = _create_model()

    operator = _create_operator()

    jacobian_1 = operator.jacobian(
        model_1
    )

    # Create a different mesh.
    mesh_2 = StructuredMesh3D(
        nx=3,
        ny=2,
        nz=1,
        dx=10.0,
        dy=10.0,
        dz=10.0,
        origin=np.array(
            [
                -15.0,
                -5.0,
                0.0,
            ],
            dtype=float,
        ),
    )

    model_2 = Model3D(
        mesh=mesh_2,
        values=np.ones(
            mesh_2.nx
            * mesh_2.ny
            * mesh_2.nz,
            dtype=float,
        ),
    )

    jacobian_2 = operator.jacobian(
        model_2
    )

    assert jacobian_1.shape != jacobian_2.shape


# ----------------------------------------------------------------------
# Empty observations
# ----------------------------------------------------------------------


def test_magnetic_operator_accepts_empty_observations():
    """
    An empty observation array with shape (0, 3) is structurally valid.
    """

    observations = np.empty(
        (0, 3),
        dtype=float,
    )

    operator = _create_operator(
        observation_locations=observations
    )

    assert operator.n_data == 0
"""
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""
