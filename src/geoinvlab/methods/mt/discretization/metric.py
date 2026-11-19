"""
Metric quantities for the 3D MT discretization.

This module provides metric information associated with the structured
Cartesian topology:

    - edge lengths
    - face areas
    - cell volumes

The topological operators (curl/divergence) contain only incidence
information. This module supplies the geometric factors required to turn
those topological quantities into physical differential operators.

The ordering is identical to StructuredTopology:

    Edges:
        [Ex, Ey, Ez]

    Faces:
        [Fx, Fy, Fz]

    Cells:
        x -> y -> z, with z varying fastest.
        
         MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.methods.mt.discretization.topology import (
    StructuredTopology,
    build_topology,
)


@dataclass(frozen=True)
class EdgeMetric:
    """
    Metric quantities associated with oriented edges.

    Parameters
    ----------
    lengths:
        Length of every edge, ordered as [Ex, Ey, Ez].
    """

    lengths: np.ndarray

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return self.lengths.size


@dataclass(frozen=True)
class FaceMetric:
    """
    Metric quantities associated with oriented faces.

    Parameters
    ----------
    areas:
        Area of every face, ordered as [Fx, Fy, Fz].
    """

    areas: np.ndarray

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return self.areas.size


@dataclass(frozen=True)
class CellMetric:
    """
    Metric quantities associated with cells.

    Parameters
    ----------
    volumes:
        Volume of every cell.
    """

    volumes: np.ndarray

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.volumes.size


@dataclass(frozen=True)
class StructuredMetric:
    """
    Complete metric description of a structured Cartesian mesh.

    The metric contains only geometric quantities. Physical parameters
    such as conductivity and magnetic permeability are handled elsewhere.
    """

    mesh_shape: tuple[int, int, int]
    edges: EdgeMetric
    faces: FaceMetric
    cells: CellMetric

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return self.edges.n_edges

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return self.faces.n_faces

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.cells.n_cells


def _build_edge_lengths(
    mesh: StructuredMesh3D,
) -> EdgeMetric:
    """
    Build the physical length of every oriented edge.

    Ex edges have length dx.
    Ey edges have length dy.
    Ez edges have length dz.
    """

    nx, ny, nz = mesh.shape

    dx = np.asarray(mesh.cell_widths_x, dtype=float)
    dy = np.asarray(mesh.cell_widths_y, dtype=float)
    dz = np.asarray(mesh.cell_widths_z, dtype=float)

    # --------------------------------------------------------------
    # x-directed edges
    #
    # There are nx * (ny + 1) * (nz + 1) of them.
    # --------------------------------------------------------------

    ex = np.empty(
        nx * (ny + 1) * (nz + 1),
        dtype=float,
    )

    offset = 0

    for ix in range(nx):
        count = (ny + 1) * (nz + 1)

        ex[offset:offset + count] = dx[ix]

        offset += count

    # --------------------------------------------------------------
    # y-directed edges
    # --------------------------------------------------------------

    ey = np.empty(
        (nx + 1) * ny * (nz + 1),
        dtype=float,
    )

    offset = 0

    for ix in range(nx + 1):
        count = ny * (nz + 1)

        ey[offset:offset + count] = np.repeat(
            dy,
            nz + 1,
        )

        offset += count

    # --------------------------------------------------------------
    # z-directed edges
    # --------------------------------------------------------------

    ez = np.empty(
        (nx + 1) * (ny + 1) * nz,
        dtype=float,
    )

    offset = 0

    for ix in range(nx + 1):
        for iy in range(ny + 1):
            ez[
                offset:offset + nz
            ] = dz

            offset += nz

    lengths = np.concatenate(
        [ex, ey, ez]
    )

    return EdgeMetric(lengths=lengths)


def _build_face_areas(
    mesh: StructuredMesh3D,
) -> FaceMetric:
    """
    Build the physical area of every oriented face.

    Fx:
        area = dy * dz

    Fy:
        area = dx * dz

    Fz:
        area = dx * dy
    """

    nx, ny, nz = mesh.shape

    dx = np.asarray(mesh.cell_widths_x, dtype=float)
    dy = np.asarray(mesh.cell_widths_y, dtype=float)
    dz = np.asarray(mesh.cell_widths_z, dtype=float)

    # --------------------------------------------------------------
    # Fx faces
    #
    # x-normal faces:
    #
    #     (nx + 1) * ny * nz
    #
    # Each face spans one dy and one dz.
    # --------------------------------------------------------------

    fx = np.empty(
        (nx + 1) * ny * nz,
        dtype=float,
    )

    offset = 0

    for ix in range(nx + 1):
        values = (
            dy[:, None]
            * dz[None, :]
        ).ravel()

        count = ny * nz

        fx[offset:offset + count] = values

        offset += count

    # --------------------------------------------------------------
    # Fy faces
    #
    #     nx * (ny + 1) * nz
    #
    # Each face spans one dx and one dz.
    # --------------------------------------------------------------

    fy = np.empty(
        nx * (ny + 1) * nz,
        dtype=float,
    )

    offset = 0

    for ix in range(nx):
        values = np.repeat(
            dx[ix] * dz,
            ny + 1,
        )

        # The previous expression is not ordered correctly for
        # y -> z indexing. Build explicitly.
        values = np.tile(
            dx[ix] * dz,
            ny + 1,
        )

        count = (ny + 1) * nz

        fy[offset:offset + count] = values

        offset += count

    # --------------------------------------------------------------
    # Fz faces
    #
    #     nx * ny * (nz + 1)
    #
    # Each face spans one dx and one dy.
    # --------------------------------------------------------------

    fz = np.empty(
        nx * ny * (nz + 1),
        dtype=float,
    )

    offset = 0

    for ix in range(nx):
        values = np.repeat(
            dx[ix] * dy,
            nz + 1,
        )

        count = ny * (nz + 1)

        fz[offset:offset + count] = values

        offset += count

    areas = np.concatenate(
        [fx, fy, fz]
    )

    return FaceMetric(areas=areas)


def _build_cell_volumes(
    mesh: StructuredMesh3D,
) -> CellMetric:
    """
    Build the physical volume of every cell.

    Ordering follows StructuredMesh3D:

        index = ix * ny * nz + iy * nz + iz
    """

    volumes = np.asarray(
        mesh.cell_volumes(),
        dtype=float,
    ).copy()

    return CellMetric(volumes=volumes)


def build_metric(
    mesh: StructuredMesh3D,
) -> StructuredMetric:
    """
    Build the complete metric description of a structured mesh.

    Parameters
    ----------
    mesh:
        Structured Cartesian mesh.

    Returns
    -------
    StructuredMetric
        Edge lengths, face areas and cell volumes.
    """

    if not isinstance(mesh, StructuredMesh3D):
        raise TypeError(
            "build_metric requires a StructuredMesh3D."
        )

    topology: StructuredTopology = build_topology(mesh)

    edges = _build_edge_lengths(mesh)
    faces = _build_face_areas(mesh)
    cells = _build_cell_volumes(mesh)

    if edges.n_edges != topology.n_edges:
        raise RuntimeError(
            "Edge metric size is inconsistent with topology."
        )

    if faces.n_faces != topology.n_faces:
        raise RuntimeError(
            "Face metric size is inconsistent with topology."
        )

    if cells.n_cells != mesh.n_cells:
        raise RuntimeError(
            "Cell metric size is inconsistent with mesh."
        )

    if not np.all(np.isfinite(edges.lengths)):
        raise ValueError(
            "Edge lengths contain non-finite values."
        )

    if not np.all(edges.lengths > 0.0):
        raise ValueError(
            "Edge lengths must be strictly positive."
        )

    if not np.all(np.isfinite(faces.areas)):
        raise ValueError(
            "Face areas contain non-finite values."
        )

    if not np.all(faces.areas > 0.0):
        raise ValueError(
            "Face areas must be strictly positive."
        )

    if not np.all(np.isfinite(cells.volumes)):
        raise ValueError(
            "Cell volumes contain non-finite values."
        )

    if not np.all(cells.volumes > 0.0):
        raise ValueError(
            "Cell volumes must be strictly positive."
        )

    return StructuredMetric(
        mesh_shape=mesh.shape,
        edges=edges,
        faces=faces,
        cells=cells,
    )
