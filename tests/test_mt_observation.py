import numpy as np
import pytest

from geoinvlab.methods.mt.discretization.field import EdgeElectricField
from geoinvlab.methods.mt.discretization.observation import (
    MTObservation,
    MTObservationSet,
    make_observation,
    make_observation_set,
    observe_fields,
    observe_multiple,
    validate_observation,
)


def test_observation_stores_four_distinct_indices():
    observation = MTObservation(
        ex_index=1,
        ey_index=5,
        hx_index=12,
        hy_index=18,
    )

    assert observation.ex_index == 1
    assert observation.ey_index == 5
    assert observation.hx_index == 12
    assert observation.hy_index == 18


@pytest.mark.parametrize(
    "field",
    ["ex_index", "ey_index", "hx_index", "hy_index"],
)
def test_observation_rejects_negative_index(field):
    values = {
        "ex_index": 1,
        "ey_index": 2,
        "hx_index": 3,
        "hy_index": 4,
    }
    values[field] = -1

    with pytest.raises(ValueError, match="non-negative"):
        MTObservation(**values)


@pytest.mark.parametrize(
    "field",
    ["ex_index", "ey_index", "hx_index", "hy_index"],
)
def test_observation_rejects_non_integer_index(field):
    values = {
        "ex_index": 1,
        "ey_index": 2,
        "hx_index": 3,
        "hy_index": 4,
    }
    values[field] = 1.5

    with pytest.raises(ValueError, match="integer"):
        MTObservation(**values)


def test_observation_set_requires_at_least_one_observation():
    with pytest.raises(ValueError, match="must not be empty"):
        MTObservationSet(())


def test_observation_set():
    first = MTObservation(0, 1, 2, 3)
    second = MTObservation(4, 5, 6, 7)

    observations = MTObservationSet((first, second))

    assert len(observations) == 2
    assert observations[0] == first
    assert observations[1] == second


def test_validate_observation():
    observation = MTObservation(
        ex_index=2,
        ey_index=4,
        hx_index=7,
        hy_index=9,
    )

    result = validate_observation(
        observation,
        n_edges=10,
        n_faces=12,
    )

    assert result == observation


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"ex_index": 10, "ey_index": 1, "hx_index": 2, "hy_index": 3},
            "ex_index",
        ),
        (
            {"ex_index": 1, "ey_index": 10, "hx_index": 2, "hy_index": 3},
            "ey_index",
        ),
        (
            {"ex_index": 1, "ey_index": 2, "hx_index": 12, "hy_index": 3},
            "hx_index",
        ),
        (
            {"ex_index": 1, "ey_index": 2, "hx_index": 3, "hy_index": 12},
            "hy_index",
        ),
    ],
)
def test_validate_observation_rejects_out_of_range(kwargs, expected):
    observation = MTObservation(**kwargs)

    with pytest.raises(ValueError, match=expected):
        validate_observation(
            observation,
            n_edges=10,
            n_faces=12,
        )


def test_observe_fields_extracts_distinct_edge_and_face_values():
    electric = EdgeElectricField(
        values=np.array(
            [
                10 + 1j,
                20 + 2j,
                30 + 3j,
                40 + 4j,
                50 + 5j,
            ],
            dtype=complex,
        ),
        n_ex=2,
        n_ey=2,
        n_ez=1,
    )

    magnetic = np.array(
        [
            100 + 1j,
            200 + 2j,
            300 + 3j,
            400 + 4j,
            500 + 5j,
        ],
        dtype=complex,
    )

    observation = MTObservation(
        ex_index=0,
        ey_index=3,
        hx_index=1,
        hy_index=4,
    )

    result = observe_fields(
        electric_field=electric,
        magnetic_field_values=magnetic,
        observation=observation,
    )

    assert result.ex == 10 + 1j
    assert result.ey == 40 + 4j
    assert result.hx == 200 + 2j
    assert result.hy == 500 + 5j


def test_observed_fields_vectors():
    electric = EdgeElectricField(
        values=np.array(
            [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j],
            dtype=complex,
        ),
        n_ex=1,
        n_ey=1,
        n_ez=2,
    )

    magnetic = np.array(
        [5 + 5j, 6 + 6j, 7 + 7j, 8 + 8j],
        dtype=complex,
    )

    observation = MTObservation(
        ex_index=0,
        ey_index=1,
        hx_index=2,
        hy_index=3,
    )

    result = observe_fields(electric, magnetic, observation)

    np.testing.assert_array_equal(
        result.electric,
        np.array([1 + 1j, 2 + 2j]),
    )

    np.testing.assert_array_equal(
        result.magnetic,
        np.array([7 + 7j, 8 + 8j]),
    )


def test_observe_multiple():
    electric = EdgeElectricField(
        values=np.arange(8, dtype=float).astype(complex),
        n_ex=2,
        n_ey=2,
        n_ez=4,
    )

    magnetic = np.arange(10, dtype=float).astype(complex)

    observations = MTObservationSet(
        (
            MTObservation(0, 2, 1, 3),
            MTObservation(1, 3, 4, 5),
        )
    )

    results = observe_multiple(
        electric_field=electric,
        magnetic_field_values=magnetic,
        observations=observations,
    )

    assert len(results) == 2

    assert results[0].ex == 0
    assert results[0].ey == 2
    assert results[0].hx == 1
    assert results[0].hy == 3

    assert results[1].ex == 1
    assert results[1].ey == 3
    assert results[1].hx == 4
    assert results[1].hy == 5


def test_observe_rejects_wrong_magnetic_dimension():
    electric = EdgeElectricField(
        values=np.ones(4, dtype=complex),
        n_ex=1,
        n_ey=1,
        n_ez=2,
    )

    magnetic = np.ones((2, 2), dtype=complex)

    observation = MTObservation(0, 1, 0, 1)

    with pytest.raises(ValueError, match="one-dimensional"):
        observe_fields(electric, magnetic, observation)


def test_make_observation():
    observation = make_observation(
        ex_index=1,
        ey_index=2,
        hx_index=3,
        hy_index=4,
    )

    assert observation == MTObservation(1, 2, 3, 4)


def test_make_observation_set():
    first = MTObservation(0, 1, 2, 3)
    second = MTObservation(4, 5, 6, 7)

    result = make_observation_set([first, second])

    assert isinstance(result, MTObservationSet)
    assert len(result) == 2
