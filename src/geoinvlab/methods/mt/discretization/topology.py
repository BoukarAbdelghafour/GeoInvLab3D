"""
Topological operators for structured Cartesian meshes used by the 3D MT solver.

The topology is purely geometric/topological: it does not contain any
electromagnetic physics.

For a structured Cartesian grid:

    - electric fields are associated with edges;
    - magnetic fluxes are associated with faces;
    - the discrete curl maps edge quantities to face quantities.

The indexing convention follows StructuredMesh3D:

    cell index = ix * ny * nz + iy * nz + iz

with z varying fastest.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geoinvlab.core.mesh.structured import StructuredMesh3D


@dataclass(frozen=True)
class EdgeTopology:
    """
    Indexing information for the three families of oriented edges.

    Ex edges are parallel to x.
    Ey edges are parallel to y.
    Ez edges are parallel to z.
    """

    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray

    @property
    def n_ex(self) -> int:
        return self.ex.size

    @property
    def n_ey(self) -> int:
        return self.ey.size

    @property
    def n_ez(self) -> int:
        return self.ez.size

    @property
    def n_edges(self) -> int:
        return self.n_ex + self.n_ey + self.n_ez


@dataclass(frozen=True)
class FaceTopology:
    """
    Indexing information for the three families of oriented faces.

    Fx faces have normals along x.
    Fy faces have normals along y.
    Fz faces have normals along z.
    """

    fx: np.ndarray
    fy: np.ndarray
    fz: np.ndarray

    @property
    def n_fx(self) -> int:
        return self.fx.size

    @property
    def n_fy(self) -> int:
        return self.fy.size

    @property
    def n_fz(self) -> int:
        return self.fz.size

    @property
    def n_faces(self) -> int:
        return self.n_fx + self.n_fy + self.n_fz


@dataclass(frozen=True)
class StructuredTopology:
    """
    Complete topological description of a structured Cartesian mesh.

    The topology contains only integer connectivity/indexing information.
    Metric quantities such as dx, dy, dz remain in StructuredMesh3D.
    """

    mesh_shape: tuple[int, int, int]
    edges: EdgeTopology
    faces: FaceTopology

    @property
    def nx(self) -> int:
        return self.mesh_shape[0]

    @property
    def ny(self) -> int:
        return self.mesh_shape[1]

    @property
    def nz(self) -> int:
        return self.mesh_shape[2]

    @property
    def n_edges(self) -> int:
        return self.edges.n_edges

    @property
    def n_faces(self) -> int:
        return self.faces.n_faces


def _edge_index_x(
    ix: int,
    iy: int,
    iz: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Return the local index of an x-directed edge.

    An x-edge starts at a mesh vertex (ix, iy, iz), therefore:

        ix = 0 ... nx-1
        iy = 0 ... ny
        iz = 0 ... nz
    """

    n_x = nx * (ny + 1) * (nz + 1)

    return (
        ix * (ny + 1) * (nz + 1)
        + iy * (nz + 1)
        + iz
    )


def _edge_index_y(
    ix: int,
    iy: int,
    iz: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Return the local index of a y-directed edge.
    """

    n_x = nx * (ny + 1) * (nz + 1)

    return (
        n_x
        + ix * ny * (nz + 1)
        + iy * (nz + 1)
        + iz
    )


def _edge_index_z(
    ix: int,
    iy: int,
    iz: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Return the local index of a z-directed edge.
    """

    n_x = nx * (ny + 1) * (nz + 1)
    n_y = (nx + 1) * ny * (nz + 1)

    return (
        n_x
        + n_y
        + ix * (ny + 1) * nz
        + iy * nz
        + iz
    )


def _face_index_x(
    ix: int,
    iy: int,
    iz: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Return the local index of an x-normal face.

    x-normal faces are located at x-vertices:

        ix = 0 ... nx
        iy = 0 ... ny-1
        iz = 0 ... nz-1
    """

    return ix * ny * nz + iy * nz + iz


def _face_index_y(
    ix: int,
    iy: int,
    iz: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Return the local index of a y-normal face.
    """

    n_x = (nx + 1) * ny * nz

    return (
        n_x
        + ix * (ny + 1) * nz
        + iy * nz
        + iz
    )


def _face_index_z(
    ix: int,
    iy: int,
    iz: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Return the local index of a z-normal face.
    """

    n_x = (nx + 1) * ny * nz
    n_y = nx * (ny + 1) * nz

    return (
        n_x
        + n_y
        + ix * ny * (nz + 1)
        + iy * (nz + 1)
        + iz
    )


def build_edge_topology(
    mesh: StructuredMesh3D,
) -> EdgeTopology:
    """
    Build the indexing arrays for all oriented mesh edges.
    """

    nx, ny, nz = mesh.shape

    ex = np.arange(
        nx * (ny + 1) * (nz + 1),
        dtype=np.int64,
    )

    ey = np.arange(
        (nx + 1) * ny * (nz + 1),
        dtype=np.int64,
    )

    ez = np.arange(
        (nx + 1) * (ny + 1) * nz,
        dtype=np.int64,
    )

    offset_y = ex.size
    offset_z = offset_y + ey.size

    ey = ey + offset_y
    ez = ez + offset_z

    return EdgeTopology(
        ex=ex,
        ey=ey,
        ez=ez,
    )


def build_face_topology(
    mesh: StructuredMesh3D,
) -> FaceTopology:
    """
    Build the indexing arrays for all oriented mesh faces.
    """

    nx, ny, nz = mesh.shape

    fx = np.arange(
        (nx + 1) * ny * nz,
        dtype=np.int64,
    )

    fy = np.arange(
        nx * (ny + 1) * nz,
        dtype=np.int64,
    )

    fz = np.arange(
        nx * ny * (nz + 1),
        dtype=np.int64,
    )

    offset_y = fx.size
    offset_z = offset_y + fy.size

    fy = fy + offset_y
    fz = fz + offset_z

    return FaceTopology(
        fx=fx,
        fy=fy,
        fz=fz,
    )


def build_topology(
    mesh: StructuredMesh3D,
) -> StructuredTopology:
    """
    Build the complete topology of a structured Cartesian mesh.

    Parameters
    ----------
    mesh:
        Structured Cartesian mesh.

    Returns
    -------
    StructuredTopology
        Edge and face indexing information.
    """

    if not isinstance(mesh, StructuredMesh3D):
        raise TypeError(
            "build_topology requires a StructuredMesh3D."
        )

    return StructuredTopology(
        mesh_shape=mesh.shape,
        edges=build_edge_topology(mesh),
        faces=build_face_topology(mesh),
    )
