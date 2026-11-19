"""
Generic 3D physical model representation.

This module provides the Model3D class used to associate
physical parameters with a GeoInvLab3D mesh.
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geoinvlab.core.mesh.base import Mesh3D


@dataclass
class Model3D:
    """
    Generic physical model defined on a 3D mesh.

    Parameters
    ----------
    mesh : Mesh3D
        Computational mesh associated with the model.

        The mesh can be structured or unstructured.

    values : numpy.ndarray
        Physical parameter value for each mesh cell.

    name : str, optional
        Name of the physical parameter.

    Notes
    -----
    The model stores one physical parameter per mesh cell.

    Examples
    --------
    Structured model::

        mesh = StructuredMesh3D(...)
        model = Model3D(
            mesh=mesh,
            values=np.ones(mesh.n_cells),
            name="resistivity",
        )

    Unstructured model::

        mesh = UnstructuredMesh3D(...)
        model = Model3D(
            mesh=mesh,
            values=np.ones(mesh.n_cells),
            name="resistivity",
        )
    """

    mesh: Mesh3D
    values: np.ndarray
    name: str = "model"

    def __post_init__(self) -> None:
        """Validate the physical model."""

        if not isinstance(self.mesh, Mesh3D):
            raise TypeError(
                "mesh must be an instance of Mesh3D."
            )

        values = np.asarray(
            self.values,
            dtype=float,
        )

        if values.ndim != 1:
            raise ValueError(
                "Model values must be a one-dimensional array."
            )

        if values.size != self.mesh.n_cells:
            raise ValueError(
                f"Expected {self.mesh.n_cells} model values, "
                f"but received {values.size}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Model values contain NaN or infinite values."
            )

        self.values = values
        self.name = str(self.name)

    # ==============================================================
    # BASIC PROPERTIES
    # ==============================================================

    @property
    def n_parameters(self) -> int:
        """Return the number of model parameters."""

        return int(self.values.size)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return the spatial shape of the model.

        For structured meshes this is the mesh shape.

        For unstructured meshes the model is cell-based and
        therefore has shape ``(n_cells,)``.
        """

        if hasattr(self.mesh, "shape"):
            return tuple(self.mesh.shape)

        return (self.mesh.n_cells,)

    # ==============================================================
    # GEOMETRY
    # ==============================================================

    @property
    def cell_centers(self) -> np.ndarray:
        """
        Return the coordinates of all model cell centers.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_cells, 3)``.
        """

        return self.mesh.cell_centers()

    # ==============================================================
    # COPY
    # ==============================================================

    def copy(self) -> "Model3D":
        """Return an independent copy of the model."""

        return Model3D(
            mesh=self.mesh,
            values=self.values.copy(),
            name=self.name,
        )

    # ==============================================================
    # ARRAY REPRESENTATION
    # ==============================================================

    def reshape(self) -> np.ndarray:
        """
        Return model values using the mesh dimensions.

        For a structured mesh, the returned array has the
        three-dimensional mesh shape.

        For an unstructured mesh, the values are returned as
        a one-dimensional cell array.

        Returns
        -------
        numpy.ndarray
            Model values arranged according to the mesh type.
        """

        if hasattr(self.mesh, "shape"):
            return self.values.reshape(self.mesh.shape)

        return self.values.copy()

    # ==============================================================
    # STATISTICS
    # ==============================================================

    def min(self) -> float:
        """Return the minimum model value."""

        return float(np.min(self.values))

    def max(self) -> float:
        """Return the maximum model value."""

        return float(np.max(self.values))

    def mean(self) -> float:
        """Return the mean model value."""

        return float(np.mean(self.values))

    def norm(self) -> float:
        """Return the Euclidean norm of the model."""

        return float(np.linalg.norm(self.values))

    # ==============================================================
    # REPRESENTATION
    # ==============================================================

    def __len__(self) -> int:
        """Return the number of model parameters."""

        return self.n_parameters

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"Model3D("
            f"name={self.name!r}, "
            f"n_parameters={self.n_parameters}, "
            f"mesh={type(self.mesh).__name__}"
            f")"
        )

