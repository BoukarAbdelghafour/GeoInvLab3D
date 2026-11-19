"""
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""
import numpy as np
import pytest

from geoinvlab.core.data import DataSet


def create_test_data():

    values = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    locations = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [30.0, 0.0, 0.0],
    ])

    uncertainties = np.array([
        0.1,
        0.2,
        0.1,
        0.2,
    ])

    return DataSet(
        values=values,
        locations=locations,
        uncertainties=uncertainties,
        name="gravity",
        units="mGal",
    )


def test_data_creation():

    data = create_test_data()

    assert data.n_data == 4
    assert data.dimension == 3
    assert data.name == "gravity"
    assert data.units == "mGal"


def test_data_locations():

    data = create_test_data()

    assert data.locations.shape == (4, 3)


def test_data_weights():

    data = create_test_data()

    weights = data.weights()

    assert np.allclose(
        weights,
        [10.0, 5.0, 10.0, 5.0],
    )


def test_data_copy():

    data = create_test_data()

    copied = data.copy()

    assert copied is not data
    assert np.array_equal(
        copied.values,
        data.values,
    )


def test_invalid_locations():

    values = np.array([1.0, 2.0])

    locations = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
    ])

    with pytest.raises(ValueError):

        DataSet(
            values=values,
            locations=locations,
        )

