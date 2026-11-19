"""
Discrete 3D curl operator for GeoInvLab3D.

Topology
--------
The electric field is stored on primal edges:

    Ex : nx     * (ny + 1) * (nz + 1)
    Ey : (nx+1) * ny       * (nz + 1)
    Ez : (nx+1) * (ny + 1) * nz

The magnetic field / curl result is stored on primal faces:

    Fx : (nx + 1) * ny       * nz
    Fy : nx       * (ny + 1) * nz
    Fz : nx       * ny       * (nz + 1)

The z index is the fastest index inside every component block.

The global electric-edge ordering is

    [Ex, Ey, Ez]

and the global face ordering is

    [Fx, Fy, Fz].

The topological curl matrix is the edge-to-face incidence matrix

    C.

It contains only orientation information. Geometry is introduced
separately through

    C_h = D_face^{-1} C D_edge,

where

    D_edge = diag(edge lengths)
    D_face = diag(face areas).

Time convention
---------------
GeoInvLab3D uses

    exp(+i * omega * t)

so that

    curl(E) = -i * omega * mu * H

and therefore

    H = +i * curl(E) / (omega * mu).

Curl orientation
----------------
The oriented circulation of E around a face is

    curl(E)_x = dEz/dy - dEy/dz
    curl(E)_y = dEx/dz - dEz/dx
    curl(E)_z = dEy/dx - dEx/dy.

For a structured orthogonal grid, the topological incidence rows
are therefore

    Fx:
        +Ez(i,j+1,k)
        -Ez(i,j,k)
        -Ey(i,j,k+1)
        +Ey(i,j,k)

    Fy:
        +Ex(i,j,k+1)
        -Ex(i,j,k)
        -Ez(i+1,j,k)
        +Ez(i,j,k)

    Fz:
        +Ey(i+1,j,k)
        -Ey(i,j,k)
        -Ex(i,j+1,k)
        +Ex(i,j,k).

Every row contains exactly four distinct edge incidences:

    +1, -1, -1, +1.

No artificial forward/backward boundary stencil is used. Boundary
faces use the actual four boundary edges belonging to the face.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, diags

from geoinvlab.core.mesh import StructuredMesh3D


# ============================================================================
# Mesh helpers
# ============================================================================


def _mesh_dimensions(
    mesh: StructuredMesh3D,
) -> tuple[int, int, int]:
    """
    Return the canonical topological mesh dimensions.

    ``StructuredMesh3D.shape`` is deliberately used here instead of
    ``mesh.nx``, ``mesh.ny`` and ``mesh.nz``.

    This is important because a StructuredMesh3D may contain padding
    or other geometry information for which those convenience
    attributes do not necessarily represent the same topological
    dimensions used by ``shape``.

    The following modules must all use the same dimensions:

        topology.py
        metric.py
        curl.py
        MaxwellSystem
    """
    try:
        shape = tuple(int(value) for value in mesh.shape)
    except Exception as exc:
        raise TypeError(
            "StructuredMesh3D must expose a valid shape."
        ) from exc

    if len(shape) != 3:
        raise ValueError(
            "StructuredMesh3D.shape must contain exactly "
            "three dimensions."
        )

    nx, ny, nz = shape

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(
            "Mesh dimensions nx, ny and nz must be "
            "strictly positive."
        )

    return nx, ny, nz


def _cell_spacing(
    mesh: StructuredMesh3D,
) -> tuple[float, float, float]:
    """
    Return scalar cell spacings.

    This helper is retained for compatibility with the existing
    uniform-grid physical-curl construction.
    """
    dx = float(mesh.dx)
    dy = float(mesh.dy)
    dz = float(mesh.dz)

    if (
        not np.isfinite(dx)
        or not np.isfinite(dy)
        or not np.isfinite(dz)
        or dx <= 0.0
        or dy <= 0.0
        or dz <= 0.0
    ):
        raise ValueError(
            "Mesh spacings must be finite and strictly positive."
        )

    return dx, dy, dz


# ============================================================================
# Edge counts
# ============================================================================


def _edge_component_counts(
    mesh: StructuredMesh3D,
) -> tuple[int, int, int]:
    """
    Return the number of Ex, Ey and Ez edge unknowns.

    For a structured hexahedral mesh:

        Ex : nx     * (ny + 1) * (nz + 1)
        Ey : (nx+1) * ny       * (nz + 1)
        Ez : (nx+1) * (ny + 1) * nz
    """
    nx, ny, nz = _mesh_dimensions(mesh)

    n_ex = (
        nx
        * (ny + 1)
        * (nz + 1)
    )

    n_ey = (
        (nx + 1)
        * ny
        * (nz + 1)
    )

    n_ez = (
        (nx + 1)
        * (ny + 1)
        * nz
    )

    return n_ex, n_ey, n_ez


# ============================================================================
# Face counts
# ============================================================================


def _face_component_counts(
    mesh: StructuredMesh3D,
) -> tuple[int, int, int]:
    """
    Return the number of Fx, Fy and Fz face unknowns.

        Fx : (nx+1) * ny       * nz
        Fy : nx     * (ny+1) * nz
        Fz : nx     * ny       * (nz+1)
    """
    nx, ny, nz = _mesh_dimensions(mesh)

    n_fx = (
        (nx + 1)
        * ny
        * nz
    )

    n_fy = (
        nx
        * (ny + 1)
        * nz
    )

    n_fz = (
        nx
        * ny
        * (nz + 1)
    )

    return n_fx, n_fy, n_fz


# ============================================================================
# Edge indexing
# ============================================================================


def _edge_x(
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Return the global Ex-edge index.

    Ex edges run in the x direction.

    Valid ranges:

        i = 0 ... nx-1
        j = 0 ... ny
        k = 0 ... nz
    """
    if not (
        0 <= i < nx
        and 0 <= j <= ny
        and 0 <= k <= nz
    ):
        raise IndexError(
            "Invalid Ex edge index."
        )

    return (
        (i * (ny + 1) + j)
        * (nz + 1)
        + k
    )


def _edge_y(
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Return the global Ey-edge index.

    Ey edges run in the y direction.

    Valid ranges:

        i = 0 ... nx
        j = 0 ... ny-1
        k = 0 ... nz
    """
    if not (
        0 <= i <= nx
        and 0 <= j < ny
        and 0 <= k <= nz
    ):
        raise IndexError(
            "Invalid Ey edge index."
        )

    n_ex = (
        nx
        * (ny + 1)
        * (nz + 1)
    )

    return (
        n_ex
        + (
            i * ny
            + j
        )
        * (nz + 1)
        + k
    )


def _edge_z(
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Return the global Ez-edge index.

    Ez edges run in the z direction.

    Valid ranges:

        i = 0 ... nx
        j = 0 ... ny
        k = 0 ... nz-1
    """
    if not (
        0 <= i <= nx
        and 0 <= j <= ny
        and 0 <= k < nz
    ):
        raise IndexError(
            "Invalid Ez edge index."
        )

    n_ex = (
        nx
        * (ny + 1)
        * (nz + 1)
    )

    n_ey = (
        (nx + 1)
        * ny
        * (nz + 1)
    )

    return (
        n_ex
        + n_ey
        + (
            i * (ny + 1)
            + j
        )
        * nz
        + k
    )


# ============================================================================
# Face indexing
# ============================================================================


def _face_x(
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Return the global Fx-face index.

    Fx faces are normal to x.

    Valid ranges:

        i = 0 ... nx
        j = 0 ... ny-1
        k = 0 ... nz-1
    """
    if not (
        0 <= i <= nx
        and 0 <= j < ny
        and 0 <= k < nz
    ):
        raise IndexError(
            "Invalid Fx face index."
        )

    return (
        (i * ny + j)
        * nz
        + k
    )


def _face_y(
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Return the global Fy-face index.

    Fy faces are normal to y.

    Valid ranges:

        i = 0 ... nx-1
        j = 0 ... ny
        k = 0 ... nz-1
    """
    if not (
        0 <= i < nx
        and 0 <= j <= ny
        and 0 <= k < nz
    ):
        raise IndexError(
            "Invalid Fy face index."
        )

    n_fx = (
        (nx + 1)
        * ny
        * nz
    )

    return (
        n_fx
        + (
            i * (ny + 1)
            + j
        )
        * nz
        + k
    )


def _face_z(
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Return the global Fz-face index.

    Fz faces are normal to z.

    Valid ranges:

        i = 0 ... nx-1
        j = 0 ... ny-1
        k = 0 ... nz
    """
    if not (
        0 <= i < nx
        and 0 <= j < ny
        and 0 <= k <= nz
    ):
        raise IndexError(
            "Invalid Fz face index."
        )

    n_fx = (
        (nx + 1)
        * ny
        * nz
    )

    n_fy = (
        nx
        * (ny + 1)
        * nz
    )

    return (
        n_fx
        + n_fy
        + (
            i * ny
            + j
        )
        * (nz + 1)
        + k
    )


# ============================================================================
# Curl operator
# ============================================================================


@dataclass
class CurlOperator:
    """
    Discrete topological curl operator.

    ``matrix`` is the edge-to-face incidence matrix.

    ``topology`` is retained as a compatibility alias exposing the
    component counts required by MaxwellSystem.

    ``physical_matrix`` is

        D_face^{-1} C D_edge.
    """

    matrix: Any

    edge_count: int
    face_count: int

    n_ex: int
    n_ey: int
    n_ez: int

    n_fx: int
    n_fy: int
    n_fz: int

    physical_matrix: Any | None = None

    def __post_init__(self) -> None:
        self.matrix = self.matrix.tocsr()

        if self.matrix.shape != (
            self.face_count,
            self.edge_count,
        ):
            raise ValueError(
                "Invalid curl matrix shape: "
                f"expected "
                f"({self.face_count}, {self.edge_count}), "
                f"got {self.matrix.shape}."
            )

        if (
            self.n_ex
            + self.n_ey
            + self.n_ez
            != self.edge_count
        ):
            raise ValueError(
                "Ex/Ey/Ez counts do not match "
                "edge_count."
            )

        if (
            self.n_fx
            + self.n_fy
            + self.n_fz
            != self.face_count
        ):
            raise ValueError(
                "Fx/Fy/Fz counts do not match "
                "face_count."
            )

        # MaxwellSystem expects ``curl.topology`` to expose the
        # component counts. ``self`` provides this interface directly.
        self.topology = self

    @property
    def n_edges(self) -> int:
        """Compatibility alias for the total number of edge unknowns."""
        return int(self.edge_count)

    @property
    def n_faces(self) -> int:
        """Compatibility alias for the total number of face unknowns."""
        return int(self.face_count)

    @property
    def C_h(self):
        """
        Metric-aware curl matrix.

        Compatibility alias used by the Maxwell discretization.
        """
        if self.physical_matrix is None:
            raise AttributeError(
                "The physical curl matrix has not "
                "been constructed."
            )

        return self.physical_matrix


# ============================================================================
# Physical metrics
# ============================================================================


def _edge_lengths(
    mesh: StructuredMesh3D,
) -> np.ndarray:
    """
    Return one characteristic length for every edge unknown.

    The ordering is exactly

        [Ex, Ey, Ez]

    and the counts are derived from ``mesh.shape``.
    """
    nx, ny, nz = _mesh_dimensions(mesh)

    dx, dy, dz = _cell_spacing(mesh)

    n_ex, n_ey, n_ez = _edge_component_counts(
        mesh
    )

    values = np.empty(
        n_ex + n_ey + n_ez,
        dtype=float,
    )

    offset = 0

    values[
        offset : offset + n_ex
    ] = dx

    offset += n_ex

    values[
        offset : offset + n_ey
    ] = dy

    offset += n_ey

    values[
        offset : offset + n_ez
    ] = dz

    return values


def _face_areas(
    mesh: StructuredMesh3D,
) -> np.ndarray:
    """
    Return one area for every face unknown.

    The ordering is exactly

        [Fx, Fy, Fz].
    """
    nx, ny, nz = _mesh_dimensions(mesh)

    dx, dy, dz = _cell_spacing(mesh)

    n_fx, n_fy, n_fz = _face_component_counts(
        mesh
    )

    values = np.empty(
        n_fx + n_fy + n_fz,
        dtype=float,
    )

    offset = 0

    # Fx: face normal to x -> dy * dz.
    values[
        offset : offset + n_fx
    ] = dy * dz

    offset += n_fx

    # Fy: face normal to y -> dx * dz.
    values[
        offset : offset + n_fy
    ] = dx * dz

    offset += n_fy

    # Fz: face normal to z -> dx * dy.
    values[
        offset : offset + n_fz
    ] = dx * dy

    return values


# ============================================================================
# Internal row helper
# ============================================================================


def _append_row(
    rows: list[int],
    cols: list[int],
    values: list[float],
    row: int,
    columns: tuple[int, int, int, int],
    coefficients: tuple[float, float, float, float],
) -> None:
    """
    Append one four-edge incidence row.

    Every face of the structured hexahedral grid has four distinct
    boundary edges.
    """
    if len(set(columns)) != 4:
        raise RuntimeError(
            "Internal curl construction error: "
            "a curl row contains duplicate edge indices."
        )

    if len(coefficients) != 4:
        raise RuntimeError(
            "Internal curl construction error: "
            "exactly four incidence coefficients are required."
        )

    rows.extend(
        (
            row,
            row,
            row,
            row,
        )
    )

    cols.extend(columns)
    values.extend(coefficients)


# ============================================================================
# Topological construction
# ============================================================================


def build_curl(
    mesh: StructuredMesh3D,
) -> CurlOperator:
    """
    Build the structured-grid edge-to-face curl incidence matrix.

    The oriented circulation is

        Fx = dEz/dy - dEy/dz
        Fy = dEx/dz - dEz/dx
        Fz = dEy/dx - dEx/dy.

    No finite-difference boundary approximation is introduced.

    Each face directly references the four primal edges forming its
    boundary.
    """
    nx, ny, nz = _mesh_dimensions(mesh)

    n_ex, n_ey, n_ez = _edge_component_counts(
        mesh
    )

    n_fx, n_fy, n_fz = _face_component_counts(
        mesh
    )

    n_edges = (
        n_ex
        + n_ey
        + n_ez
    )

    n_faces = (
        n_fx
        + n_fy
        + n_fz
    )

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    # ==================================================================
    # Fx faces
    #
    # curl(E)_x = dEz/dy - dEy/dz
    #
    #   +Ez(i,j+1,k)
    #   -Ez(i,j,k)
    #   -Ey(i,j,k+1)
    #   +Ey(i,j,k)
    # ==================================================================

    for i in range(nx + 1):
        for j in range(ny):
            for k in range(nz):

                row = _face_x(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                )

                ez_y_plus = _edge_z(
                    i,
                    j + 1,
                    k,
                    nx,
                    ny,
                    nz,
                )

                ez_y_minus = _edge_z(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                )

                ey_z_plus = _edge_y(
                    i,
                    j,
                    k + 1,
                    nx,
                    ny,
                    nz,
                )

                ey_z_minus = _edge_y(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                )

                _append_row(
                    rows,
                    cols,
                    values,
                    row,
                    (
                        ez_y_plus,
                        ez_y_minus,
                        ey_z_plus,
                        ey_z_minus,
                    ),
                    (
                        +1.0,
                        -1.0,
                        -1.0,
                        +1.0,
                    ),
                )

    # ==================================================================
    # Fy faces
    #
    # curl(E)_y = dEx/dz - dEz/dx
    #
    #   +Ex(i,j,k+1)
    #   -Ex(i,j,k)
    #   -Ez(i+1,j,k)
    #   +Ez(i,j,k)
    # ==================================================================

    for i in range(nx):
        for j in range(ny + 1):
            for k in range(nz):

                row = _face_y(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                )

                ex_z_plus = _edge_x(
                    i,
                    j,
                    k + 1,
                    nx,
                    ny,
                    nz,
                )

                ex_z_minus = _edge_x(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                )

                ez_x_plus = _edge_z(
                    i + 1,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                )

                ez_x_minus = _edge_z(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                )

                _append_row(
                    rows,
                    cols,
                    values,
                    row,
                    (
                        ex_z_plus,
                        ex_z_minus,
                        ez_x_plus,
                        ez_x_minus,
                    ),
                    (
                        +1.0,
                        -1.0,
                        -1.0,
                        +1.0,
                    ),
                )

    # ==================================================================
    # Fz faces
    #
    # curl(E)_z = dEy/dx - dEx/dy
    #
    #   +Ey(i+1,j,k)
    #   -Ey(i,j,k)
    #   -Ex(i,j+1,k)
    #   +Ex(i,j,k)
    # ==================================================================

    for i in range(nx):
        for j in range(ny):
            for k in range(nz + 1):

                row = _face_z(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                )

                ey_x_plus = _edge_y(
                    i + 1,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                )

                ey_x_minus = _edge_y(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                )

                ex_y_plus = _edge_x(
                    i,
                    j + 1,
                    k,
                    nx,
                    ny,
                    nz,
                )

                ex_y_minus = _edge_x(
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                )

                _append_row(
                    rows,
                    cols,
                    values,
                    row,
                    (
                        ey_x_plus,
                        ey_x_minus,
                        ex_y_plus,
                        ex_y_minus,
                    ),
                    (
                        +1.0,
                        -1.0,
                        -1.0,
                        +1.0,
                    ),
                )

    # ==================================================================
    # Sparse topological matrix
    # ==================================================================

    matrix = csr_matrix(
        (
            np.asarray(
                values,
                dtype=float,
            ),
            (
                np.asarray(
                    rows,
                    dtype=int,
                ),
                np.asarray(
                    cols,
                    dtype=int,
                ),
            ),
        ),
        shape=(
            n_faces,
            n_edges,
        ),
    )

    # ==================================================================
    # Curl operator
    # ==================================================================

    operator = CurlOperator(
        matrix=matrix,
        edge_count=n_edges,
        face_count=n_faces,
        n_ex=n_ex,
        n_ey=n_ey,
        n_ez=n_ez,
        n_fx=n_fx,
        n_fy=n_fy,
        n_fz=n_fz,
    )

    # ==================================================================
    # Metric-aware curl
    #
    # C_h = D_face^{-1} C D_edge
    # ==================================================================

    edge_lengths = _edge_lengths(mesh)

    face_areas = _face_areas(mesh)

    D_edge = diags(
        edge_lengths,
        offsets=0,
        format="csr",
    )

    D_face_inv = diags(
        1.0 / face_areas,
        offsets=0,
        format="csr",
    )

    operator.physical_matrix = (
        D_face_inv
        @ operator.matrix
        @ D_edge
    ).tocsr()

    # ==================================================================
    # Structural validation
    # ==================================================================

    _validate_curl_structure(
        operator.matrix
    )

    return operator


# ============================================================================
# Validation
# ============================================================================


def _validate_curl_structure(
    matrix,
) -> None:
    """
    Validate the topological curl matrix.

    Every row must contain exactly four distinct entries:

        +1, +1, -1, -1.
    """
    matrix = matrix.tocsr()

    if matrix.ndim != 2:
        raise ValueError(
            "Curl matrix must be two-dimensional."
        )

    for row_index in range(
        matrix.shape[0]
    ):
        row = matrix.getrow(
            row_index
        )

        if row.nnz != 4:
            raise ValueError(
                "Each curl row must contain exactly "
                "four edge contributions; "
                f"row {row_index} contains "
                f"{row.nnz}."
            )

        values = np.asarray(
            row.data,
            dtype=float,
        )

        if not np.all(
            np.isin(
                values,
                (-1.0, +1.0),
            )
        ):
            raise ValueError(
                "Curl incidence entries must be ±1."
            )

        n_positive = np.count_nonzero(
            values > 0.0
        )

        n_negative = np.count_nonzero(
            values < 0.0
        )

        if (
            n_positive != 2
            or n_negative != 2
        ):
            raise ValueError(
                "Each curl row must contain two +1 "
                "and two -1 entries."
            )


# ============================================================================
# Physical curl helper
# ============================================================================


def physical_curl(
    mesh: StructuredMesh3D,
    edge_field: np.ndarray,
) -> np.ndarray:
    """
    Compute the metric-aware curl of an edge field.
    """
    operator = build_curl(mesh)

    values = np.asarray(
        edge_field
    ).reshape(-1)

    if values.size != operator.edge_count:
        raise ValueError(
            "edge_field has the wrong size: "
            f"expected {operator.edge_count}, "
            f"got {values.size}."
        )

    return np.asarray(
        operator.physical_matrix @ values
    )


__all__ = [
    "CurlOperator",
    "build_curl",
    "physical_curl",
]
 

