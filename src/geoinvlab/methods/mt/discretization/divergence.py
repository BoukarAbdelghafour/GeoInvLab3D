"""
Discrete divergence operator for structured Cartesian meshes.

The operator maps face-based vector fields to cell-centered divergence:

    D : flux_faces -> div(flux)_cells

The field components are ordered as:

    [Fx, Fy, Fz]

where:

    Fx : x-normal faces
    Fy : y-normal faces
    Fz : z-normal faces

This module is purely topological. Metric factors such as cell widths
and cell volumes are intentionally not included here.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.methods.mt.discretization.topology import (
    StructuredTopology,
    build_topology,
)


@dataclass(frozen=True)
class DivergenceOperator:
    """
    Discrete face-to-cell divergence operator.

    Parameters
    ----------
    matrix:
        Sparse cell-face incidence matrix.
    topology:
        Mesh topology used to construct the operator.
    """

    matrix: sparse.csr_matrix
    topology: StructuredTopology

    @property
    def shape(self) -> tuple[int, int]:
        """Return the matrix shape."""
        return self.matrix.shape

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        nx, ny, nz = self.topology.mesh_shape
        return nx * ny * nz

    @property
    def n_faces(self) -> int:
        """Number of face unknowns."""
        return self.topology.n_faces

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """
        Apply the discrete divergence.

        Parameters
        ----------
        field:
            Face-based field with shape ``(n_faces,)``.

        Returns
        -------
        numpy.ndarray
            Cell-based divergence with shape ``(n_cells,)``.
        """

        field = np.asarray(field)

        if field.ndim != 1:
            raise ValueError(
                "Face field must be a one-dimensional array."
            )

        if field.size != self.n_faces:
            raise ValueError(
                f"Expected {self.n_faces} face values, "
                f"got {field.size}."
            )

        return self.matrix @ field


def _face_x(
    topology: StructuredTopology,
    ix: int,
    iy: int,
    iz: int,
) -> int:
    """Return the global index of an x-normal face."""

    nx, ny, nz = topology.mesh_shape

    return (
        ix * ny * nz
        + iy * nz
        + iz
    )


def _face_y(
    topology: StructuredTopology,
    ix: int,
    iy: int,
    iz: int,
) -> int:
    """Return the global index of a y-normal face."""

    nx, ny, nz = topology.mesh_shape

    n_fx = (nx + 1) * ny * nz

    return (
        n_fx
        + ix * (ny + 1) * nz
        + iy * nz
        + iz
    )


def _face_z(
    topology: StructuredTopology,
    ix: int,
    iy: int,
    iz: int,
) -> int:
    """Return the global index of a z-normal face."""

    nx, ny, nz = topology.mesh_shape

    n_fx = (nx + 1) * ny * nz
    n_fy = nx * (ny + 1) * nz

    return (
        n_fx
        + n_fy
        + ix * ny * (nz + 1)
        + iy * (nz + 1)
        + iz
    )


def _cell_index(
    topology: StructuredTopology,
    ix: int,
    iy: int,
    iz: int,
) -> int:
    """
    Return the cell index.

    The convention is the same as StructuredMesh3D:

        index = ix * ny * nz + iy * nz + iz

    with z varying fastest.
    """

    nx, ny, nz = topology.mesh_shape

    return (
        ix * ny * nz
        + iy * nz
        + iz
    )


def _build_divergence_incidence(
    topology: StructuredTopology,
) -> sparse.csr_matrix:
    """
    Build the topological face-to-cell incidence matrix.

    Each cell has six oriented boundary faces:

        -Fx at x-min
        +Fx at x-max

        -Fy at y-min
        +Fy at y-max

        -Fz at z-min
        +Fz at z-max

    Therefore every row contains at most six non-zero entries.
    """

    nx, ny, nz = topology.mesh_shape

    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):

                cell = _cell_index(
                    topology,
                    ix,
                    iy,
                    iz,
                )

                # ------------------------------------------------------
                # x faces
                # ------------------------------------------------------

                fx_left = _face_x(
                    topology,
                    ix,
                    iy,
                    iz,
                )

                fx_right = _face_x(
                    topology,
                    ix + 1,
                    iy,
                    iz,
                )

                rows.extend([cell, cell])
                cols.extend([fx_left, fx_right])
                data.extend([-1, +1])

                # ------------------------------------------------------
                # y faces
                # ------------------------------------------------------

                fy_front = _face_y(
                    topology,
                    ix,
                    iy,
                    iz,
                )

                fy_back = _face_y(
                    topology,
                    ix,
                    iy + 1,
                    iz,
                )

                rows.extend([cell, cell])
                cols.extend([fy_front, fy_back])
                data.extend([-1, +1])

                # ------------------------------------------------------
                # z faces
                # ------------------------------------------------------

                fz_bottom = _face_z(
                    topology,
                    ix,
                    iy,
                    iz,
                )

                fz_top = _face_z(
                    topology,
                    ix,
                    iy,
                    iz + 1,
                )

                rows.extend([cell, cell])
                cols.extend([fz_bottom, fz_top])
                data.extend([-1, +1])

    n_cells = nx * ny * nz
    n_faces = topology.n_faces

    matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(n_cells, n_faces),
        dtype=float,
    )

    return matrix.tocsr()


def build_divergence(
    mesh: StructuredMesh3D,
) -> DivergenceOperator:
    """
    Build the discrete divergence operator.

    Parameters
    ----------
    mesh:
        Structured Cartesian mesh.

    Returns
    -------
    DivergenceOperator
        Sparse face-to-cell divergence operator.
    """

    if not isinstance(mesh, StructuredMesh3D):
        raise TypeError(
            "build_divergence requires a StructuredMesh3D."
        )

    topology = build_topology(mesh)

    matrix = _build_divergence_incidence(topology)

    return DivergenceOperator(
        matrix=matrix,
        topology=topology,
    )
