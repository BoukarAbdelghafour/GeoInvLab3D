import numpy as np
import pytest

from geoinvlab.core.mesh.structured import StructuredMesh3D

from geoinvlab.methods.mt.discretization.observation import MTObservation
from geoinvlab.methods.mt.discretization.station import (
    MTStation,
    MTStationSet,
    StationObservation,
    make_station,
    make_station_set,
    station_to_observation,
    stations_to_observations,
)


def make_mesh():
    return StructuredMesh3D(
        nx=2,
        ny=2,
        nz=2,
        dx=10.0,
        dy=20.0,
        dz=30.0,
        origin=(0.0, 0.0, 0.0),
    )


def test_station_stores_coordinates():
    station = MTStation(
        x=10.0,
        y=20.0,
        z=-30.0,
        name="S01",
    )

    assert station.x == 10.0
    assert station.y == 20.0
    assert station.z == -30.0
    assert station.name == "S01"


def test_station_coordinates():
    station = MTStation(1.0, 2.0, 3.0)

    np.testing.assert_array_equal(
        station.coordinates,
        np.array([1.0, 2.0, 3.0]),
    )


@pytest.mark.parametrize("coordinate", ["x", "y", "z"])
def test_station_rejects_non_finite_coordinate(coordinate):
    values = {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
    }

    values[coordinate] = np.nan

    with pytest.raises(ValueError, match="finite"):
        MTStation(**values)


def test_station_rejects_invalid_name():
    with pytest.raises(ValueError, match="string"):
        MTStation(1.0, 2.0, 3.0, name=123)


def test_station_set():
    first = MTStation(1.0, 2.0, 3.0, "S01")
    second = MTStation(4.0, 5.0, 6.0, "S02")

    stations = MTStationSet((first, second))

    assert len(stations) == 2
    assert stations[0] == first
    assert stations[1] == second


def test_station_set_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        MTStationSet(())


def test_station_set_rejects_wrong_type():
    with pytest.raises(ValueError, match="MTStation"):
        MTStationSet((MTStation(0.0, 0.0, 0.0), "bad"))


def test_station_maps_to_discrete_observation():
    mesh = make_mesh()

    station = MTStation(
        x=5.0,
        y=10.0,
        z=15.0,
    )

    result = station_to_observation(mesh, station)

    assert isinstance(result, StationObservation)
    assert result.station == station
    assert isinstance(result.observation, MTObservation)

    assert 0 <= result.observation.ex_index
    assert 0 <= result.observation.ey_index
    assert 0 <= result.observation.hx_index
    assert 0 <= result.observation.hy_index


def test_station_at_origin_maps_to_valid_indices():
    mesh = make_mesh()

    station = MTStation(0.0, 0.0, 0.0)

    result = station_to_observation(mesh, station)

    observation = result.observation

    assert observation.ex_index >= 0
    assert observation.ey_index >= 0
    assert observation.hx_index >= 0
    assert observation.hy_index >= 0


def test_station_at_mesh_center_maps_to_valid_indices():
    mesh = make_mesh()

    station = MTStation(
        x=10.0,
        y=20.0,
        z=30.0,
    )

    result = station_to_observation(mesh, station)

    observation = result.observation

    assert observation.ex_index >= 0
    assert observation.ey_index >= 0
    assert observation.hx_index >= 0
    assert observation.hy_index >= 0


def test_station_mapping_is_deterministic():
    mesh = make_mesh()

    station = MTStation(
        x=7.0,
        y=13.0,
        z=21.0,
    )

    first = station_to_observation(mesh, station)
    second = station_to_observation(mesh, station)

    assert first == second


def test_multiple_station_mapping():
    mesh = make_mesh()

    stations = MTStationSet(
        (
            MTStation(1.0, 2.0, 3.0, "S01"),
            MTStation(10.0, 20.0, 30.0, "S02"),
            MTStation(15.0, 30.0, 45.0, "S03"),
        )
    )

    results = stations_to_observations(mesh, stations)

    assert len(results) == 3

    for result, station in zip(results, stations.stations):
        assert result.station == station
        assert isinstance(result.observation, MTObservation)


def test_make_station():
    station = make_station(
        x=1.0,
        y=2.0,
        z=3.0,
        name="TEST",
    )

    assert station == MTStation(
        1.0,
        2.0,
        3.0,
        "TEST",
    )


def test_make_station_set():
    first = MTStation(0.0, 0.0, 0.0)
    second = MTStation(1.0, 1.0, 1.0)

    result = make_station_set([first, second])

    assert isinstance(result, MTStationSet)
    assert len(result) == 2


def test_station_mapping_does_not_require_matching_edge_and_face_indices():
    mesh = make_mesh()

    station = MTStation(
        x=5.0,
        y=10.0,
        z=15.0,
    )

    result = station_to_observation(mesh, station)

    observation = result.observation

    # The four discrete entities have independent index spaces.
    # They are not required to share the same integer.
    assert isinstance(observation.ex_index, int)
    assert isinstance(observation.ey_index, int)
    assert isinstance(observation.hx_index, int)
    assert isinstance(observation.hy_index, int)
