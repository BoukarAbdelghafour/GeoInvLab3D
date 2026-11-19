"""
Unstructured three-dimensional mesh.

This module provides the mesh representation used by
the magnetotelluric forward problem.
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Mesh3D


@dataclass
class UnstructuredMesh3D(Mesh3D):
    """
    Generic unstructured 3D mesh.

    Parameters
    ----------
    vertices : array-like, shape (n_vertices, 3)
        Coordinates of mesh vertices.

    cells : sequence of array-like
        Cell connectivity. Each entry contains the indices
        of vertices belonging to one cell.

    volumes : array-like, optional
        Volume of each cell.

        If omitted, volumes are automatically computed for
        tetrahedral cells.

    Notes
    -----
    The mesh does not impose a particular cell topology.

    Supported cell types can therefore be extended later to:

        - tetrahedra
        - hexahedra
        - prisms
        - pyramids
        - other polyhedral cells

    For MT, the mesh geometry can later be extended with
    edges, faces, boundary information, and electrode/station
    associations.
    """

    vertices: np.ndarray
    cells: list[np.ndarray]
    volumes: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate the unstructured mesh."""

        self.vertices = np.asarray(
            self.vertices,
            dtype=float,
        )

        if self.vertices.ndim != 2:
            raise ValueError(
                "vertices must be a 2D array."
            )

        if self.vertices.shape[1] != 3:
            raise ValueError(
                "vertices must have shape (n_vertices, 3)."
            )

        if self.vertices.shape[0] == 0:
            raise ValueError(
                "The mesh must contain at least one vertex."
            )

        if not np.all(
            np.isfinite(self.vertices)
        ):
            raise ValueError(
                "vertices contains NaN or infinite values."
            )

        normalized_cells: list[np.ndarray] = []

        for index, cell in enumerate(self.cells):

            cell_array = np.asarray(
                cell,
                dtype=int,
            ).reshape(-1)

            if cell_array.size < 4:
                raise ValueError(
                    f"Cell {index} must contain at least "
                    "four vertices."
                )

            if np.any(cell_array < 0):
                raise ValueError(
                    f"Cell {index} contains negative "
                    "vertex indices."
                )

            if np.any(
                cell_array >= self.n_vertices
            ):
                raise ValueError(
                    f"Cell {index} contains a vertex index "
                    "outside the mesh."
                )

            if np.unique(
                cell_array
            ).size != cell_array.size:

                raise ValueError(
                    f"Cell {index} contains duplicate "
                    "vertex indices."
                )

            normalized_cells.append(
                cell_array
            )

        if not normalized_cells:
            raise ValueError(
                "The mesh must contain at least one cell."
            )

        self.cells = normalized_cells

        # ------------------------------------------------------
        # Cell volumes
        # ------------------------------------------------------

        if self.volumes is not None:

            self.volumes = np.asarray(
                self.volumes,
                dtype=float,
            ).reshape(-1)

            if self.volumes.size != self.n_cells:
                raise ValueError(
                    "volumes must contain one value "
                    "per cell."
                )

            if not np.all(
                np.isfinite(self.volumes)
            ):
                raise ValueError(
                    "volumes contains NaN or infinite values."
                )

            if np.any(
                self.volumes <= 0.0
            ):
                raise ValueError(
                    "volumes must be strictly positive."
                )

        else:

            self.volumes = self._compute_volumes()

        # Validate the common Mesh3D interface.
        self.validate()

    # ==========================================================
    # BASIC PROPERTIES
    # ==========================================================

    @property
    def n_vertices(self) -> int:
        """Return the number of vertices."""

        return int(
            self.vertices.shape[0]
        )

    @property
    def n_cells(self) -> int:
        """Return the number of cells."""

        return len(self.cells)

    @property
    def shape(self) -> tuple[int]:
        """
        Return the logical shape of the mesh.

        Unstructured meshes do not have a Cartesian
        (nx, ny, nz) shape, therefore the natural shape
        is simply the number of cells.
        """

        return (self.n_cells,)

    # ==========================================================
    # CELL GEOMETRY
    # ==========================================================

    def cell_centers(self) -> np.ndarray:
        """
        Return the geometric center of each cell.

        Returns
        -------
        numpy.ndarray
            Array with shape (n_cells, 3).
        """

        centers = np.empty(
            (self.n_cells, 3),
            dtype=float,
        )

        for index, cell in enumerate(
            self.cells
        ):

            centers[index] = np.mean(
                self.vertices[cell],
                axis=0,
            )

        return centers

    def cell_volumes(self) -> np.ndarray:
        """
        Return the volume of every cell.

        Returns
        -------
        numpy.ndarray
            Array with shape (n_cells,).
        """

        return self.volumes.copy()

    def cell_vertices(
        self,
        index: int,
    ) -> np.ndarray:
        """
        Return the coordinates of one cell's vertices.

        Parameters
        ----------
        index : int
            Cell index.

        Returns
        -------
        numpy.ndarray
            Vertex coordinates with shape (n_vertices_cell, 3).
        """

        if index < 0 or index >= self.n_cells:
            raise IndexError(
                "Cell index out of range."
            )

        return self.vertices[
            self.cells[index]
        ]

    # ==========================================================
    # VOLUME CALCULATION
    # ==========================================================

    def _compute_volumes(self) -> np.ndarray:
        """
        Compute cell volumes when possible.

        Currently automatic computation is supported for
        tetrahedral cells only.

        Returns
        -------
        numpy.ndarray
            Cell volumes.

        Raises
        ------
        ValueError
            If a cell topology is not currently supported.
        """

        volumes = np.empty(
            self.n_cells,
            dtype=float,
        )

        for index, cell in enumerate(
            self.cells
        ):

            if cell.size != 4:

                raise ValueError(
                    "Automatic volume calculation is currently "
                    "implemented only for tetrahedral cells. "
                    f"Cell {index} contains {cell.size} vertices. "
                    "Provide volumes explicitly for this mesh."
                )

            points = self.vertices[cell]

            p0 = points[0]
            p1 = points[1]
            p2 = points[2]
            p3 = points[3]

            vectors = np.column_stack(
                (
                    p1 - p0,
                    p2 - p0,
                    p3 - p0,
                )
            )

            volume = abs(
                np.linalg.det(vectors)
            ) / 6.0

            if volume <= 0.0:

                raise ValueError(
                    f"Cell {index} has zero volume."
                )

            volumes[index] = volume

        return volumes

    # ==========================================================
    # COPY
    # ==========================================================

    def copy(self) -> "UnstructuredMesh3D":
        """Return a deep copy of the mesh."""

        return UnstructuredMesh3D(
            vertices=self.vertices.copy(),
            cells=[
                cell.copy()
                for cell in self.cells
            ],
            volumes=self.volumes.copy(),
        )

    # ==========================================================
    # REPRESENTATION
    # ==========================================================

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            "UnstructuredMesh3D("
            f"n_vertices={self.n_vertices}, "
            f"n_cells={self.n_cells})"
        )

