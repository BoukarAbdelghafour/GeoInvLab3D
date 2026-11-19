"""
Tests for the magnetic inversion method.
"""

import numpy as np
import pytest

from geoinvlab.core.operators import ForwardOperator
from geoinvlab.methods.magnetic.method import MagneticMethod
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
# Initialization
# ----------------------------------------------------------------------


def test_magnetic_method_initialization():
    """
    The magnetic method must initialize correctly.
    """

    method = MagneticMethod(
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    assert method.inclination == INCLINATION
    assert method.declination == DECLINATION
    assert method.field_intensity == FIELD_INTENSITY


# ----------------------------------------------------------------------
# Method identity
# ----------------------------------------------------------------------


def test_magnetic_method_name():
    """
    The human-readable method name must be 'Magnetic'.
    """

    method = MagneticMethod(
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    assert method.name == "Magnetic"


def test_magnetic_method_identifier():
    """
    The method identifier must be 'magnetic'.
    """

    method = MagneticMethod(
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    assert method.identifier == "magnetic"


# ----------------------------------------------------------------------
# Operator creation
# ----------------------------------------------------------------------


def test_magnetic_method_creates_operator():
    """
    MagneticMethod must create a MagneticOperator.
    """

    method = MagneticMethod(
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    operator = method.create_operator(
        OBSERVATION_LOCATIONS
    )

    assert isinstance(
        operator,
        MagneticOperator,
    )


def test_magnetic_method_operator_is_forward_operator():
    """
    The created operator must implement ForwardOperator.
    """

    method = MagneticMethod(
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    operator = method.create_operator(
        OBSERVATION_LOCATIONS
    )

    assert isinstance(
        operator,
        ForwardOperator,
    )


def test_magnetic_method_passes_field_parameters():
    """
    MagneticMethod must pass D, I and F to MagneticOperator.
    """

    method = MagneticMethod(
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    operator = method.create_operator(
        OBSERVATION_LOCATIONS
    )

    assert operator.inclination == INCLINATION

    assert operator.declination == DECLINATION

    assert (
        operator.field_intensity
        == FIELD_INTENSITY
    )


def test_magnetic_method_passes_observation_locations():
    """
    MagneticMethod must pass observation locations
    unchanged to MagneticOperator.
    """

    method = MagneticMethod(
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    operator = method.create_operator(
        OBSERVATION_LOCATIONS
    )

    assert np.array_equal(
        operator.observation_locations,
        OBSERVATION_LOCATIONS,
    )


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_magnetic_method_rejects_nan_inclination():
    """
    NaN inclination must be rejected.
    """

    with pytest.raises(ValueError):

        MagneticMethod(
            inclination=np.nan,
            declination=DECLINATION,
            field_intensity=FIELD_INTENSITY,
        )


def test_magnetic_method_rejects_infinite_inclination():
    """
    Infinite inclination must be rejected.
    """

    with pytest.raises(ValueError):

        MagneticMethod(
            inclination=np.inf,
            declination=DECLINATION,
            field_intensity=FIELD_INTENSITY,
        )


def test_magnetic_method_rejects_nan_declination():
    """
    NaN declination must be rejected.
    """

    with pytest.raises(ValueError):

        MagneticMethod(
            inclination=INCLINATION,
            declination=np.nan,
            field_intensity=FIELD_INTENSITY,
        )


def test_magnetic_method_rejects_infinite_declination():
    """
    Infinite declination must be rejected.
    """

    with pytest.raises(ValueError):

        MagneticMethod(
            inclination=INCLINATION,
            declination=np.inf,
            field_intensity=FIELD_INTENSITY,
        )


def test_magnetic_method_rejects_nan_field_intensity():
    """
    NaN field intensity must be rejected.
    """

    with pytest.raises(ValueError):

        MagneticMethod(
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=np.nan,
        )


def test_magnetic_method_rejects_infinite_field_intensity():
    """
    Infinite field intensity must be rejected.
    """

    with pytest.raises(ValueError):

        MagneticMethod(
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=np.inf,
        )


def test_magnetic_method_rejects_negative_field_intensity():
    """
    Negative field intensity must be rejected.
    """

    with pytest.raises(ValueError):

        MagneticMethod(
            inclination=INCLINATION,
            declination=DECLINATION,
            field_intensity=-1.0,
        )


def test_magnetic_method_accepts_zero_field_intensity():
    """
    Zero field intensity is mathematically valid.
    """

    method = MagneticMethod(
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=0.0,
    )

    assert method.field_intensity == 0.0


# ----------------------------------------------------------------------
# Operator behavior
# ----------------------------------------------------------------------


def test_magnetic_method_operator_has_correct_number_of_data():
    """
    The created operator must contain the correct number
    of observations.
    """

    method = MagneticMethod(
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    operator = method.create_operator(
        OBSERVATION_LOCATIONS
    )

    assert operator.n_data == (
        OBSERVATION_LOCATIONS.shape[0]
    )


def test_magnetic_method_creates_independent_operators():
    """
    Calling create_operator twice must produce two
    independent operator instances.
    """

    method = MagneticMethod(
        inclination=INCLINATION,
        declination=DECLINATION,
        field_intensity=FIELD_INTENSITY,
    )

    operator_1 = method.create_operator(
        OBSERVATION_LOCATIONS
    )

    operator_2 = method.create_operator(
        OBSERVATION_LOCATIONS
    )

    assert operator_1 is not operator_2


# ----------------------------------------------------------------------
# Parameter types
# ----------------------------------------------------------------------


def test_magnetic_method_converts_parameters_to_float():
    """
    Numeric parameters must be stored as floats.
    """

    method = MagneticMethod(
        inclination=60,
        declination=20,
        field_intensity=50000,
    )

    assert isinstance(
        method.inclination,
        float,
    )

    assert isinstance(
        method.declination,
        float,
    )

    assert isinstance(
        method.field_intensity,
        float,
    )
"""
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""
