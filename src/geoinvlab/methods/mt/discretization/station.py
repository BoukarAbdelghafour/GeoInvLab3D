"""
MT station geometry and discrete observation mapping.

A station is defined in physical coordinates (x, y, z).

The electric field lives on edges while the magnetic field lives on faces.
Therefore the station mapper independently locates:

    Ex -> nearest Ex edge
    Ey -> nearest Ey edge
    Hx -> nearest Fx face
    Hy -> nearest Fy face

This module intentionally uses nearest-neighbour localization only.
Weighted interpolation will be introduced separately.

IMPORTANT
---------
GeoInvLab3D uses z-fastest ordering for structured discrete quantities.

Therefore all coordinate arrays generated here must use the same
ordering as the corresponding discrete field vectors:

    ix -> iy -> iz

with iz varying fastest.

The implementation uses the actual physical mesh edge arrays, so it
remains compatible with non-uniform structured meshes and padding cells.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geoinvlab.core.mesh.structured import StructuredMesh3D

from .observation import MTObservation


# ============================================================================
# MT station
# ============================================================================


@dataclass(frozen=True)
class MTStation:
    """
    Physical MT observation station.

    Parameters
    ----------
    x, y, z:
        Physical coordinates of the station.

    name:
        Optional station name.
    """

    x: float
    y: float
    z: float
    name: str | None = None

    def __post_init__(self) -> None:
        coordinates = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }

        for name, value in coordinates.items():
            value = float(value)

            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        if (
            self.name is not None
            and not isinstance(self.name, str)
        ):
            raise ValueError(
                "name must be a string or None"
            )

    @property
    def coordinates(self) -> np.ndarray:
        """Return station coordinates as [x, y, z]."""
        return np.array(
            [
                self.x,
                self.y,
                self.z,
            ],
            dtype=float,
        )


# ============================================================================
# Station set
# ============================================================================


@dataclass(frozen=True)
class MTStationSet:
    """Collection of MT stations."""

    stations: tuple[MTStation, ...]

    def __post_init__(self) -> None:
        stations = tuple(self.stations)

        if len(stations) == 0:
            raise ValueError(
                "stations must not be empty"
            )

        for station in stations:
            if not isinstance(
                station,
                MTStation,
            ):
                raise ValueError(
                    "stations must contain "
                    "MTStation instances"
                )

        object.__setattr__(
            self,
            "stations",
            stations,
        )

    def __len__(self) -> int:
        """Return the number of stations."""
        return len(self.stations)

    @property
    def n_stations(self) -> int:
        """Return the number of stations."""
        return len(self.stations)

    def __getitem__(
        self,
        index: int,
    ) -> MTStation:
        """Return one station by index."""
        return self.stations[index]

    def __iter__(self):
        """Iterate over stations."""
        return iter(self.stations)


# ============================================================================
# Station observation
# ============================================================================


@dataclass(frozen=True)
class StationObservation:
    """
    Mapping between one physical station and discrete MT fields.

    Parameters
    ----------
    station:
        Physical station.

    observation:
        Discrete Ex/Ey/Hx/Hy indices.
    """

    station: MTStation
    observation: MTObservation


# ============================================================================
# Mesh coordinates
# ============================================================================


def _axis_centers(
    edges: np.ndarray,
) -> np.ndarray:
    """
    Return physical cell-center coordinates from mesh edges.
    """
    edges = np.asarray(
        edges,
        dtype=float,
    )

    if edges.ndim != 1:
        raise ValueError(
            "edges must be one-dimensional"
        )

    if edges.size < 2:
        raise ValueError(
            "edges must contain at least two coordinates"
        )

    return 0.5 * (
        edges[:-1]
        + edges[1:]
    )


def _mesh_coordinates(
    mesh: StructuredMesh3D,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Return physical node and cell-center coordinates.

    Actual mesh edge arrays are used so that non-uniform meshes and
    padding cells are handled correctly.
    """
    x_edges = np.asarray(
        mesh.edges_x,
        dtype=float,
    )

    y_edges = np.asarray(
        mesh.edges_y,
        dtype=float,
    )

    z_edges = np.asarray(
        mesh.edges_z,
        dtype=float,
    )

    x_centers = _axis_centers(
        x_edges
    )

    y_centers = _axis_centers(
        y_edges
    )

    z_centers = _axis_centers(
        z_edges
    )

    return (
        (
            x_edges,
            y_edges,
            z_edges,
        ),
        (
            x_centers,
            y_centers,
            z_centers,
        ),
    )


# ============================================================================
# Edge coordinates
# ============================================================================


def _edge_coordinates(
    mesh: StructuredMesh3D,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return physical coordinates for Ex, Ey and Ez edges.

    GeoInvLab3D ordering is z-fastest.

    Ex:
        shape = (nx, ny+1, nz+1)

        physical location:
            (x_center, y_node, z_node)

    Ey:
        shape = (nx+1, ny, nz+1)

        physical location:
            (x_node, y_center, z_node)

    Ez:
        shape = (nx+1, ny+1, nz)

        physical location:
            (x_node, y_node, z_center)

    The flattened coordinate arrays follow exactly the same ordering
    as the corresponding discrete field vectors:

        for ix
            for iy
                for iz

    so that iz is the fastest-varying index.
    """
    (
        x_edges,
        y_edges,
        z_edges,
    ), (
        x_centers,
        y_centers,
        z_centers,
    ) = _mesh_coordinates(mesh)

    nx, ny, nz = mesh.shape

    # ------------------------------------------------------------------
    # Ex edges
    #
    # Ex has:
    #
    #     nx * (ny + 1) * (nz + 1)
    #
    # degrees of freedom.
    # ------------------------------------------------------------------
    ex = np.array(
        [
            [
                x_centers[ix],
                y_edges[iy],
                z_edges[iz],
            ]
            for ix in range(nx)
            for iy in range(ny + 1)
            for iz in range(nz + 1)
        ],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Ey edges
    #
    # Ey has:
    #
    #     (nx + 1) * ny * (nz + 1)
    #
    # degrees of freedom.
    # ------------------------------------------------------------------
    ey = np.array(
        [
            [
                x_edges[ix],
                y_centers[iy],
                z_edges[iz],
            ]
            for ix in range(nx + 1)
            for iy in range(ny)
            for iz in range(nz + 1)
        ],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Ez edges
    #
    # Ez has:
    #
    #     (nx + 1) * (ny + 1) * nz
    #
    # degrees of freedom.
    # ------------------------------------------------------------------
    ez = np.array(
        [
            [
                x_edges[ix],
                y_edges[iy],
                z_centers[iz],
            ]
            for ix in range(nx + 1)
            for iy in range(ny + 1)
            for iz in range(nz)
        ],
        dtype=float,
    )

    return (
        ex,
        ey,
        ez,
    )


# ============================================================================
# Face coordinates
# ============================================================================


def _face_coordinates(
    mesh: StructuredMesh3D,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return physical coordinates for Fx, Fy and Fz faces.

    GeoInvLab3D ordering is z-fastest.

    Fx:
        shape = (nx+1, ny, nz)

        physical location:
            (x_node, y_center, z_center)

    Fy:
        shape = (nx, ny+1, nz)

        physical location:
            (x_center, y_node, z_center)

    Fz:
        shape = (nx, ny, nz+1)

        physical location:
            (x_center, y_center, z_node)

    The flattened arrays use:

        for ix
            for iy
                for iz

    so that iz is the fastest-varying index.
    """
    (
        x_edges,
        y_edges,
        z_edges,
    ), (
        x_centers,
        y_centers,
        z_centers,
    ) = _mesh_coordinates(mesh)

    nx, ny, nz = mesh.shape

    # ------------------------------------------------------------------
    # Fx faces
    # ------------------------------------------------------------------
    fx = np.array(
        [
            [
                x_edges[ix],
                y_centers[iy],
                z_centers[iz],
            ]
            for ix in range(nx + 1)
            for iy in range(ny)
            for iz in range(nz)
        ],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Fy faces
    # ------------------------------------------------------------------
    fy = np.array(
        [
            [
                x_centers[ix],
                y_edges[iy],
                z_centers[iz],
            ]
            for ix in range(nx)
            for iy in range(ny + 1)
            for iz in range(nz)
        ],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Fz faces
    # ------------------------------------------------------------------
    fz = np.array(
        [
            [
                x_centers[ix],
                y_centers[iy],
                z_edges[iz],
            ]
            for ix in range(nx)
            for iy in range(ny)
            for iz in range(nz + 1)
        ],
        dtype=float,
    )

    return (
        fx,
        fy,
        fz,
    )


# ============================================================================
# Nearest neighbour
# ============================================================================


def _nearest_index(
    coordinates: np.ndarray,
    point: np.ndarray,
) -> int:
    """
    Return the index of the closest discrete location.
    """
    coordinates = np.asarray(
        coordinates,
        dtype=float,
    )

    point = np.asarray(
        point,
        dtype=float,
    )

    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
    ):
        raise ValueError(
            "coordinates must have shape (n, 3)"
        )

    if point.shape != (3,):
        raise ValueError(
            "point must have shape (3,)"
        )

    if coordinates.shape[0] == 0:
        raise ValueError(
            "coordinates must not be empty"
        )

    if not np.all(
        np.isfinite(coordinates)
    ):
        raise ValueError(
            "coordinates must contain finite values"
        )

    if not np.all(
        np.isfinite(point)
    ):
        raise ValueError(
            "point must contain finite values"
        )

    distances = np.sum(
        (
            coordinates
            - point
        ) ** 2,
        axis=1,
    )

    return int(
        np.argmin(distances)
    )


# ============================================================================
# Station mapping
# ============================================================================


def station_to_observation(
    mesh: StructuredMesh3D,
    station: MTStation,
) -> StationObservation:
    """
    Map a physical station to the nearest discrete MT locations.

    The mapping is:

        Ex -> Ex edge
        Ey -> Ey edge
        Hx -> Fx face
        Hy -> Fy face

    All returned indices are COMPONENT-LOCAL.

    The coordinate arrays use exactly the same z-fastest ordering as
    the discrete field vectors.
    """
    if not isinstance(
        mesh,
        StructuredMesh3D,
    ):
        raise TypeError(
            "mesh must be a StructuredMesh3D"
        )

    if not isinstance(
        station,
        MTStation,
    ):
        raise TypeError(
            "station must be an MTStation"
        )

    point = station.coordinates

    (
        ex_coordinates,
        ey_coordinates,
        _,
    ) = _edge_coordinates(mesh)

    (
        fx_coordinates,
        fy_coordinates,
        _,
    ) = _face_coordinates(mesh)

    ex_index = _nearest_index(
        ex_coordinates,
        point,
    )

    ey_index = _nearest_index(
        ey_coordinates,
        point,
    )

    hx_index = _nearest_index(
        fx_coordinates,
        point,
    )

    hy_index = _nearest_index(
        fy_coordinates,
        point,
    )

    observation = MTObservation(
        ex_index=ex_index,
        ey_index=ey_index,
        hx_index=hx_index,
        hy_index=hy_index,
    )

    return StationObservation(
        station=station,
        observation=observation,
    )


def stations_to_observations(
    mesh: StructuredMesh3D,
    stations: MTStationSet,
) -> tuple[StationObservation, ...]:
    """
    Map several physical stations to discrete observations.
    """
    if not isinstance(
        stations,
        MTStationSet,
    ):
        raise TypeError(
            "stations must be an MTStationSet"
        )

    return tuple(
        station_to_observation(
            mesh=mesh,
            station=station,
        )
        for station in stations
    )


# ============================================================================
# Convenience constructors
# ============================================================================


def make_station(
    x: float,
    y: float,
    z: float,
    name: str | None = None,
) -> MTStation:
    """Convenience constructor for one MT station."""
    return MTStation(
        x=x,
        y=y,
        z=z,
        name=name,
    )


def make_station_set(
    stations: list[MTStation]
    | tuple[MTStation, ...],
) -> MTStationSet:
    """Convenience constructor for several MT stations."""
    return MTStationSet(
        tuple(stations)
    )


__all__ = [
    "MTStation",
    "MTStationSet",
    "StationObservation",
    "station_to_observation",
    "stations_to_observations",
    "make_station",
    "make_station_set",
]
 

