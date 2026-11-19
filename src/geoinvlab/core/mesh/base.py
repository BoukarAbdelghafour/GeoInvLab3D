"""
Base classes for GeoInvLab3D meshes.

This module defines the common interface shared by all
three-dimensional meshes used by GeoInvLab3D.

MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Mesh3D(ABC):
    """
    Abstract base class for three-dimensional meshes.

    A Mesh3D provides the common interface required by the
    forward operators, physical models, and inversion engine.

    Concrete implementations may include:

        - StructuredMesh3D
        - UnstructuredMesh3D

    The mesh is responsible for describing the computational
    geometry. Physical parameters are stored separately in
    Model3D.
    """

    # ==========================================================
    # BASIC PROPERTIES
    # ==========================================================

    @property
    @abstractmethod
    def n_cells(self) -> int:
        """
        Return the number of cells in the mesh.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        Return the logical shape of the mesh.

        Structured meshes normally return:

            (nx, ny, nz)

        Unstructured meshes may return a one-dimensional
        representation such as:

            (n_cells,)
        """
        raise NotImplementedError

    # ==========================================================
    # GEOMETRY
    # ==========================================================

    @abstractmethod
    def cell_centers(self) -> np.ndarray:
        """
        Return the coordinates of all cell centers.

        Returns
        -------
        numpy.ndarray
            Array with shape:

                (n_cells, 3)

            containing x, y, z coordinates.
        """
        raise NotImplementedError

    # ==========================================================
    # CELL VOLUMES
    # ==========================================================

    @abstractmethod
    def cell_volumes(self) -> np.ndarray:
        """
        Return the volume of every mesh cell.

        Returns
        -------
        numpy.ndarray
            One-dimensional array with shape:

                (n_cells,)
        """
        raise NotImplementedError

    # ==========================================================
    # VALIDATION
    # ==========================================================

    def validate(self) -> None:
        """
        Validate the basic geometry of the mesh.

        Concrete meshes can override this method if they need
        additional validation.
        """

        if self.n_cells <= 0:
            raise ValueError(
                "A mesh must contain at least one cell."
            )

        centers = np.asarray(
            self.cell_centers(),
            dtype=float,
        )

        if centers.ndim != 2:
            raise ValueError(
                "cell_centers() must return a 2D array."
            )

        if centers.shape != (self.n_cells, 3):
            raise ValueError(
                "cell_centers() must return an array with "
                f"shape ({self.n_cells}, 3), "
                f"got {centers.shape}."
            )

        if not np.all(np.isfinite(centers)):
            raise ValueError(
                "Mesh cell centers contain NaN or infinite values."
            )

        volumes = np.asarray(
            self.cell_volumes(),
            dtype=float,
        )

        if volumes.ndim != 1:
            raise ValueError(
                "cell_volumes() must return a 1D array."
            )

        if volumes.size != self.n_cells:
            raise ValueError(
                "cell_volumes() must return exactly "
                f"{self.n_cells} values."
            )

        if not np.all(np.isfinite(volumes)):
            raise ValueError(
                "Mesh cell volumes contain NaN or infinite values."
            )

        if np.any(volumes <= 0.0):
            raise ValueError(
                "All mesh cell volumes must be strictly positive."
            )

    # ==========================================================
    # CONVENIENCE
    # ==========================================================

    def __len__(self) -> int:
        """
        Return the number of cells.
        """
        return self.n_cells

    def __repr__(self) -> str:
        """
        Return a concise mesh representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"n_cells={self.n_cells})"
        )
