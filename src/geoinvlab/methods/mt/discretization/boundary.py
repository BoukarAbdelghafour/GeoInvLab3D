"""
Boundary degrees of freedom for the structured 3D MT discretization.

The electric field is represented on mesh edges. This module identifies
the edge degrees of freedom located on the six outer boundary faces of
a structured Cartesian mesh.

No physical MT source is imposed here.

The purpose of this module is to provide the geometric/topological
information required later by the Maxwell solver to impose boundary
conditions.

Edge ordering
-------------
The global edge vector is ordered as:

    [Ex, Ey, Ez]

with:

    Ex : x-directed edges
    Ey : y-directed edges
    Ez : z-directed edges

For a mesh with shape (nx, ny, nz):

    Ex : nx * (ny + 1) * (nz + 1)
    Ey : (nx + 1) * ny * (nz + 1)
    Ez : (nx + 1) * (ny + 1) * nz

Boundary faces
--------------
The six boundary faces are identified as:

    xmin : x = xmin
    xmax : x = xmax
    ymin : y = ymin
    ymax : y = ymax
    zmin : z = zmin
    zmax : z = zmax

An edge belongs to a boundary face when the edge lies entirely in
that face.

Therefore:

    Ex is on xmin/xmax when ix is on the x boundary,
    Ex is on ymin/ymax when iy is on the y boundary,
    Ex is on zmin/zmax when iz is on the z boundary.

and similarly for Ey and Ez.

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
class BoundaryEdges:
    """
    Edge degrees of freedom associated with the six outer boundaries.

    All indices are global edge indices in the ordering

        [Ex, Ey, Ez].

    The arrays contain unique indices and are sorted in ascending order.
    """

    xmin: np.ndarray
    xmax: np.ndarray
    ymin: np.ndarray
    ymax: np.ndarray
    zmin: np.ndarray
    zmax: np.ndarray

    n_edges: int

    @property
    def all(self) -> np.ndarray:
        """Return all unique boundary edge indices."""
        return np.unique(
            np.concatenate(
                [
                    self.xmin,
                    self.xmax,
                    self.ymin,
                    self.ymax,
                    self.zmin,
                    self.zmax,
                ]
            )
        )

    @property
    def n_boundary_edges(self) -> int:
        """Number of unique boundary edge degrees of freedom."""
        return self.all.size

    def mask(self) -> np.ndarray:
        """
        Return a boolean mask selecting all boundary edge DOFs.
        """
        result = np.zeros(self.n_edges, dtype=bool)
        result[self.all] = True
        return result


@dataclass(frozen=True)
class BoundaryCondition:
    """
    Dirichlet boundary condition on edge-based electric-field DOFs.

    Parameters
    ----------
    indices:
        Global edge indices where the field value is prescribed.

    values:
        Complex prescribed electric-field values.

    n_edges:
        Total number of electric-field edge DOFs.
    """

    indices: np.ndarray
    values: np.ndarray
    n_edges: int

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=int)
        values = np.asarray(self.values, dtype=complex)

        if indices.ndim != 1:
            raise ValueError("Boundary indices must be one-dimensional.")

        if values.ndim != 1:
            raise ValueError("Boundary values must be one-dimensional.")

        if indices.size != values.size:
            raise ValueError(
                "Boundary indices and values must have the same size."
            )

        if np.any(indices < 0) or np.any(indices >= self.n_edges):
            raise ValueError(
                "Boundary indices contain invalid edge indices."
            )

        if np.unique(indices).size != indices.size:
            raise ValueError(
                "Boundary indices must be unique."
            )

        if not np.all(np.isfinite(values.real)):
            raise ValueError(
                "Boundary values contain non-finite real parts."
            )

        if not np.all(np.isfinite(values.imag)):
            raise ValueError(
                "Boundary values contain non-finite imaginary parts."
            )

        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "values", values)

    @property
    def size(self) -> int:
        """Number of constrained edge DOFs."""
        return self.indices.size

    def vector(self) -> np.ndarray:
        """
        Return the complete prescribed-field vector.

        Unconstrained DOFs are filled with zero.
        """
        result = np.zeros(self.n_edges, dtype=complex)
        result[self.indices] = self.values
        return result


def _edge_x(
    topology: StructuredTopology,
    ix: int,
    iy: int,
    iz: int,
) -> int:
    nx, ny, nz = topology.mesh_shape

    if not (0 <= ix < nx):
        raise IndexError("Invalid Ex x-index.")
    if not (0 <= iy <= ny):
        raise IndexError("Invalid Ex y-index.")
    if not (0 <= iz <= nz):
        raise IndexError("Invalid Ex z-index.")

    return ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz


def _edge_y(
    topology: StructuredTopology,
    ix: int,
    iy: int,
    iz: int,
) -> int:
    nx, ny, nz = topology.mesh_shape

    if not (0 <= ix <= nx):
        raise IndexError("Invalid Ey x-index.")
    if not (0 <= iy < ny):
        raise IndexError("Invalid Ey y-index.")
    if not (0 <= iz <= nz):
        raise IndexError("Invalid Ey z-index.")

    n_ex = nx * (ny + 1) * (nz + 1)

    return (
        n_ex
        + ix * ny * (nz + 1)
        + iy * (nz + 1)
        + iz
    )


def _edge_z(
    topology: StructuredTopology,
    ix: int,
    iy: int,
    iz: int,
) -> int:
    nx, ny, nz = topology.mesh_shape

    if not (0 <= ix <= nx):
        raise IndexError("Invalid Ez x-index.")
    if not (0 <= iy <= ny):
        raise IndexError("Invalid Ez y-index.")
    if not (0 <= iz < nz):
        raise IndexError("Invalid Ez z-index.")

    n_ex = nx * (ny + 1) * (nz + 1)
    n_ey = (nx + 1) * ny * (nz + 1)

    return (
        n_ex
        + n_ey
        + ix * (ny + 1) * nz
        + iy * nz
        + iz
    )


def _build_boundary_edges(
    topology: StructuredTopology,
) -> BoundaryEdges:
    """
    Identify all edge DOFs lying on the six outer boundary faces.
    """
    nx, ny, nz = topology.mesh_shape

    xmin: list[int] = []
    xmax: list[int] = []
    ymin: list[int] = []
    ymax: list[int] = []
    zmin: list[int] = []
    zmax: list[int] = []

    # ---------------------------------------------------------------
    # Ex edges
    #
    # Ex[ix, iy, iz] runs in the x direction.
    #
    # It belongs to:
    #   xmin/xmax if ix is at an x boundary,
    #   ymin/ymax if iy is at a y boundary,
    #   zmin/zmax if iz is at a z boundary.
    # ---------------------------------------------------------------
    for ix in range(nx):
        for iy in range(ny + 1):
            for iz in range(nz + 1):
                edge = _edge_x(topology, ix, iy, iz)

                if iy == 0:
                    ymin.append(edge)
                if iy == ny:
                    ymax.append(edge)

                if iz == 0:
                    zmin.append(edge)
                if iz == nz:
                    zmax.append(edge)

    # ---------------------------------------------------------------
    # Ey edges
    #
    # Ey[ix, iy, iz] runs in the y direction.
    # ---------------------------------------------------------------
    for ix in range(nx + 1):
        for iy in range(ny):
            for iz in range(nz + 1):
                edge = _edge_y(topology, ix, iy, iz)

                if ix == 0:
                    xmin.append(edge)
                if ix == nx:
                    xmax.append(edge)

                if iz == 0:
                    zmin.append(edge)
                if iz == nz:
                    zmax.append(edge)

    # ---------------------------------------------------------------
    # Ez edges
    #
    # Ez[ix, iy, iz] runs in the z direction.
    # ---------------------------------------------------------------
    for ix in range(nx + 1):
        for iy in range(ny + 1):
            for iz in range(nz):
                edge = _edge_z(topology, ix, iy, iz)

                if ix == 0:
                    xmin.append(edge)
                if ix == nx:
                    xmax.append(edge)

                if iy == 0:
                    ymin.append(edge)
                if iy == ny:
                    ymax.append(edge)

    def unique_sorted(values: list[int]) -> np.ndarray:
        return np.unique(np.asarray(values, dtype=int))

    return BoundaryEdges(
        xmin=unique_sorted(xmin),
        xmax=unique_sorted(xmax),
        ymin=unique_sorted(ymin),
        ymax=unique_sorted(ymax),
        zmin=unique_sorted(zmin),
        zmax=unique_sorted(zmax),
        n_edges=topology.n_edges,
    )


def build_boundary_edges(
    mesh: StructuredMesh3D,
) -> BoundaryEdges:
    """
    Build the boundary-edge description for a structured mesh.
    """
    if not isinstance(mesh, StructuredMesh3D):
        raise TypeError(
            "build_boundary_edges requires a StructuredMesh3D."
        )

    topology = build_topology(mesh)
    return _build_boundary_edges(topology)


def make_dirichlet_boundary(
    indices: np.ndarray,
    values: np.ndarray,
    *,
    n_edges: int,
) -> BoundaryCondition:
    """
    Construct a Dirichlet boundary condition for edge-based E.

    This helper deliberately does not prescribe any physical MT source.
    The caller supplies the actual boundary values.
    """
    return BoundaryCondition(
        indices=np.asarray(indices, dtype=int),
        values=np.asarray(values, dtype=complex),
        n_edges=n_edges,
    )
