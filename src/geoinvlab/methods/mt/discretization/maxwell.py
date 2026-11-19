"""
Geometry-aware frequency-domain Maxwell system for 3D MT.

Time convention
---------------
GeoInvLab3D uses

    exp(+i * omega * t)

and therefore

    curl(E) = -i * omega * mu * H

and

    H = i * curl(E) / (omega * mu).

Coordinates
-----------
The project uses

    x = East
    y = North
    z = positive downward.

Unknowns
--------
The electric field is represented on primal edges:

    E = [Ex, Ey, Ez]

The magnetic field is represented on dual faces:

    H = [Hx, Hy, Hz].

The topological curl is the metric-free edge-to-face incidence matrix

    C.

The physical curl is

    C_h = D_face^{-1} C D_edge,

where

    D_edge = diag(edge lengths)
    D_face = diag(face areas).

For a conducting medium, the electric-field system is

    A E = 0

with

    A = K + i * omega * M_sigma,

and

    K = C_h.T M_mu_inv C_h.

The conductivity and permeability mass matrices use the corresponding
dual volumes.

MT propagation convention
-------------------------
For a homogeneous half-space,

    k**2 = -i * omega * mu * sigma,

with the physical branch

    Re(k) > 0,
    Im(k) < 0.

The downward-decaying plane wave is

    E(z) = E0 * exp(-i * k * z).

This gives

    Zxy = omega * mu / k

and

    Zyx = -omega * mu / k.
    
     MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from .curl import build_curl

try:
    from .metric import build_metric
except ImportError:  # pragma: no cover
    build_metric = None


MU0 = 4.0e-7 * np.pi


# ---------------------------------------------------------------------------
# Mesh and metric helpers
# ---------------------------------------------------------------------------


def _require_mesh(mesh: Any) -> None:
    """Validate the minimal structured-mesh interface."""
    for name in ("nx", "ny", "nz"):
        if not hasattr(mesh, name):
            raise TypeError(
                f"mesh must expose '{name}'."
            )


def _effective_mesh_shape(
    mesh: Any,
) -> tuple[int, int, int]:
    """Return the effective structured mesh dimensions."""
    shape = getattr(mesh, "shape", None)

    if callable(shape):
        shape = shape()

    if shape is not None:
        shape = tuple(
            int(value)
            for value in shape
        )

        if len(shape) != 3:
            raise ValueError(
                "mesh.shape must contain exactly three dimensions."
            )

        if any(value <= 0 for value in shape):
            raise ValueError(
                "All effective mesh dimensions must be positive."
            )

        return shape

    return (
        int(mesh.nx),
        int(mesh.ny),
        int(mesh.nz),
    )


def _effective_cell_count(mesh: Any) -> int:
    """Return the number of cells."""
    nx, ny, nz = _effective_mesh_shape(mesh)
    return nx * ny * nz


def _get_metric(mesh: Any) -> Any:
    """Construct or retrieve the geometry metric."""
    if build_metric is not None:
        try:
            return build_metric(mesh)
        except Exception:
            if hasattr(mesh, "metric"):
                return mesh.metric
            raise

    if hasattr(mesh, "metric"):
        return mesh.metric

    raise AttributeError(
        "Unable to construct the mesh metric."
    )


def _metric_cell_volumes(mesh: Any) -> np.ndarray:
    """Return primal cell volumes."""
    metric = _get_metric(mesh)

    if (
        not hasattr(metric, "cells")
        or not hasattr(metric.cells, "volumes")
    ):
        raise AttributeError(
            "metric.cells.volumes is required."
        )

    values = np.asarray(
        metric.cells.volumes,
        dtype=float,
    ).reshape(-1)

    expected = _effective_cell_count(mesh)

    if values.size != expected:
        raise ValueError(
            "The number of metric cell volumes does not match "
            f"the effective mesh: {values.size} != {expected}."
        )

    if (
        not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError(
            "Cell volumes must be finite and strictly positive."
        )

    return values


def _metric_edge_lengths(
    mesh: Any,
    n_edges: int,
) -> np.ndarray:
    """Return physical primal-edge lengths."""
    metric = _get_metric(mesh)

    if (
        not hasattr(metric, "edges")
        or not hasattr(metric.edges, "lengths")
    ):
        raise AttributeError(
            "metric.edges.lengths is required."
        )

    values = np.asarray(
        metric.edges.lengths,
        dtype=float,
    ).reshape(-1)

    if values.size != n_edges:
        raise ValueError(
            "The number of metric edge lengths does not match "
            f"the curl topology: {values.size} != {n_edges}."
        )

    if (
        not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError(
            "Edge lengths must be finite and strictly positive."
        )

    return values


def _metric_face_areas(
    mesh: Any,
    n_faces: int,
) -> np.ndarray:
    """Return physical dual-face areas."""
    metric = _get_metric(mesh)

    if (
        not hasattr(metric, "faces")
        or not hasattr(metric.faces, "areas")
    ):
        raise AttributeError(
            "metric.faces.areas is required."
        )

    values = np.asarray(
        metric.faces.areas,
        dtype=float,
    ).reshape(-1)

    if values.size != n_faces:
        raise ValueError(
            "The number of metric face areas does not match "
            f"the curl topology: {values.size} != {n_faces}."
        )

    if (
        not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError(
            "Face areas must be finite and strictly positive."
        )

    return values


# ---------------------------------------------------------------------------
# Structured topology counts
# ---------------------------------------------------------------------------


def _edge_counts_from_shape(
    mesh: Any,
) -> tuple[int, int, int]:
    """Return Ex, Ey and Ez edge counts."""
    nx, ny, nz = _effective_mesh_shape(mesh)

    return (
        nx * (ny + 1) * (nz + 1),
        (nx + 1) * ny * (nz + 1),
        (nx + 1) * (ny + 1) * nz,
    )


def _face_counts_from_shape(
    mesh: Any,
) -> tuple[int, int, int]:
    """Return Fx, Fy and Fz face counts."""
    nx, ny, nz = _effective_mesh_shape(mesh)

    return (
        (nx + 1) * ny * nz,
        nx * (ny + 1) * nz,
        nx * ny * (nz + 1),
    )


def _edge_counts(
    curl: Any,
    mesh: Any,
) -> tuple[int, int, int]:
    """Recover component-wise edge counts from the topology."""
    topology = getattr(
        curl,
        "topology",
        None,
    )

    for obj in (
        topology,
        getattr(topology, "edges", None),
    ):
        if obj is None:
            continue

        for names in (
            ("n_ex", "n_ey", "n_ez"),
            ("num_ex", "num_ey", "num_ez"),
        ):
            if all(
                hasattr(obj, name)
                for name in names
            ):
                return tuple(
                    int(getattr(obj, name))
                    for name in names
                )

        edges = getattr(
            obj,
            "edges",
            None,
        )

        if (
            edges is not None
            and all(
                hasattr(edges, name)
                for name in ("ex", "ey", "ez")
            )
        ):
            return (
                int(edges.ex.size),
                int(edges.ey.size),
                int(edges.ez.size),
            )

    return _edge_counts_from_shape(mesh)


def _face_counts(
    curl: Any,
    mesh: Any,
) -> tuple[int, int, int]:
    """Recover component-wise face counts from the topology."""
    topology = getattr(
        curl,
        "topology",
        None,
    )

    for obj in (
        topology,
        getattr(topology, "faces", None),
    ):
        if obj is None:
            continue

        for names in (
            ("n_fx", "n_fy", "n_fz"),
            ("num_fx", "num_fy", "num_fz"),
        ):
            if all(
                hasattr(obj, name)
                for name in names
            ):
                return tuple(
                    int(getattr(obj, name))
                    for name in names
                )

        faces = getattr(
            obj,
            "faces",
            None,
        )

        if (
            faces is not None
            and all(
                hasattr(faces, name)
                for name in ("fx", "fy", "fz")
            )
        ):
            return (
                int(faces.fx.size),
                int(faces.fy.size),
                int(faces.fz.size),
            )

    return _face_counts_from_shape(mesh)


def _cell_index(
    ix: int,
    iy: int,
    iz: int,
    ny: int,
    nz: int,
) -> int:
    """Return the z-fastest cell index."""
    return (
        (ix * ny + iy) * nz
        + iz
    )


# ---------------------------------------------------------------------------
# Property normalization and interpolation
# ---------------------------------------------------------------------------


def _as_cell_array(
    values: Any,
    n_cells: int,
    *,
    name: str,
) -> np.ndarray:
    """Normalize a scalar or cell vector."""
    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim == 0:
        array = np.full(
            n_cells,
            float(array),
            dtype=float,
        )

    elif array.ndim == 1:
        array = array.astype(
            float,
            copy=False,
        )

    else:
        raise ValueError(
            f"{name} must be a scalar or a one-dimensional "
            f"array of exactly {n_cells} values; "
            f"got shape {array.shape}."
        )

    if array.size != n_cells:
        raise ValueError(
            f"{name} must be a scalar or contain exactly "
            f"{n_cells} values; got {array.size}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return array


def _normalize_conductivity(
    conductivity: Any,
    n_cells: int,
) -> np.ndarray:
    """Normalize conductivity to a positive cell vector."""
    if (
        hasattr(conductivity, "values")
        and hasattr(conductivity, "mesh")
    ):
        conductivity = conductivity.values

    sigma = _as_cell_array(
        conductivity,
        n_cells,
        name="conductivity",
    )

    if np.any(sigma <= 0.0):
        raise ValueError(
            "conductivity must be strictly positive."
        )

    return sigma


def _normalize_mu_r(
    mu_r: Any,
    n_cells: int,
) -> np.ndarray:
    """Normalize relative permeability to a positive cell vector."""
    values = _as_cell_array(
        mu_r,
        n_cells,
        name="mu_r",
    )

    if np.any(values <= 0.0):
        raise ValueError(
            "mu_r must be strictly positive."
        )

    return values


def _cell_to_edge_conductivity(
    mesh: Any,
    conductivity: Any,
) -> np.ndarray:
    """Interpolate cell conductivity to electric-field edges.

    Ordering:

        [Ex, Ey, Ez]

    Component shapes:

        Ex: (nx,   ny+1, nz+1)
        Ey: (nx+1, ny,   nz+1)
        Ez: (nx+1, ny+1, nz)

    Internal edges receive the arithmetic mean of adjacent cells.
    Boundary edges use the mean of all available adjacent cells.
    """
    nx, ny, nz = _effective_mesh_shape(mesh)
    n_cells = nx * ny * nz

    sigma = np.asarray(
        conductivity,
        dtype=float,
    ).reshape(-1)

    if sigma.size == 1:
        sigma = np.full(
            n_cells,
            float(sigma[0]),
            dtype=float,
        )

    if sigma.size != n_cells:
        raise ValueError(
            f"conductivity has {sigma.size} values, "
            f"expected {n_cells}."
        )

    if (
        not np.all(np.isfinite(sigma))
        or np.any(sigma <= 0.0)
    ):
        raise ValueError(
            "conductivity must contain finite, "
            "strictly positive values."
        )

    cells = sigma.reshape(
        (nx, ny, nz),
        order="C",
    )

    # Ex
    ex = np.empty(
        (nx, ny + 1, nz + 1),
        dtype=float,
    )

    for ix in range(nx):
        for iy in range(ny + 1):
            for iz in range(nz + 1):
                values = []

                for dy in (-1, 0):
                    cy = iy + dy

                    if not 0 <= cy < ny:
                        continue

                    for dz in (-1, 0):
                        cz = iz + dz

                        if 0 <= cz < nz:
                            values.append(
                                cells[ix, cy, cz]
                            )

                ex[ix, iy, iz] = np.mean(values)

    # Ey
    #
    # Important: Ey has nz+1 z-nodes, exactly like Ex.
    ey = np.empty(
        (nx + 1, ny, nz + 1),
        dtype=float,
    )

    for ix in range(nx + 1):
        for iy in range(ny):
            for iz in range(nz + 1):
                values = []

                for dx in (-1, 0):
                    cx = ix + dx

                    if not 0 <= cx < nx:
                        continue

                    for dz in (-1, 0):
                        cz = iz + dz

                        if 0 <= cz < nz:
                            values.append(
                                cells[cx, iy, cz]
                            )

                ey[ix, iy, iz] = np.mean(values)

    # Ez
    ez = np.empty(
        (nx + 1, ny + 1, nz),
        dtype=float,
    )

    for ix in range(nx + 1):
        for iy in range(ny + 1):
            for iz in range(nz):
                values = []

                for dx in (-1, 0):
                    cx = ix + dx

                    if not 0 <= cx < nx:
                        continue

                    for dy in (-1, 0):
                        cy = iy + dy

                        if 0 <= cy < ny:
                            values.append(
                                cells[cx, cy, iz]
                            )

                ez[ix, iy, iz] = np.mean(values)

    return np.concatenate(
        (
            ex.ravel(),
            ey.ravel(),
            ez.ravel(),
        )
    )


def _cell_to_face_property(
    mesh: Any,
    values: Any,
) -> np.ndarray:
    """Interpolate a cell-centred scalar to [Fx, Fy, Fz] faces."""
    nx, ny, nz = _effective_mesh_shape(mesh)
    n_cells = nx * ny * nz

    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim == 0 or array.size == 1:
        cells = np.full(
            (nx, ny, nz),
            float(array.reshape(-1)[0]),
            dtype=float,
        )
    else:
        flat = array.reshape(-1)

        if flat.size != n_cells:
            raise ValueError(
                f"values has {flat.size} values, "
                f"expected {n_cells}."
            )

        cells = flat.reshape(
            (nx, ny, nz),
            order="C",
        )

    if not np.all(np.isfinite(cells)):
        raise ValueError(
            "Cell property must contain finite values."
        )

    # Fx
    fx = np.empty(
        (nx + 1, ny, nz),
        dtype=float,
    )

    fx[0] = cells[0]
    fx[-1] = cells[-1]

    if nx > 1:
        fx[1:-1] = 0.5 * (
            cells[:-1]
            + cells[1:]
        )

    # Fy
    fy = np.empty(
        (nx, ny + 1, nz),
        dtype=float,
    )

    fy[:, 0, :] = cells[:, 0, :]
    fy[:, -1, :] = cells[:, -1, :]

    if ny > 1:
        fy[:, 1:-1, :] = 0.5 * (
            cells[:, :-1, :]
            + cells[:, 1:, :]
        )

    # Fz
    fz = np.empty(
        (nx, ny, nz + 1),
        dtype=float,
    )

    fz[:, :, 0] = cells[:, :, 0]
    fz[:, :, -1] = cells[:, :, -1]

    if nz > 1:
        fz[:, :, 1:-1] = 0.5 * (
            cells[:, :, :-1]
            + cells[:, :, 1:]
        )

    return np.concatenate(
        (
            fx.ravel(),
            fy.ravel(),
            fz.ravel(),
        )
    )


# ---------------------------------------------------------------------------
# Backward-compatible interpolation helpers
# ---------------------------------------------------------------------------


def _cell_values_to_edge_values(
    mesh: Any,
    values: Any,
    n_edges: int,
) -> np.ndarray:
    """Backward-compatible cell-to-edge interpolation."""
    result = _cell_to_edge_conductivity(
        mesh,
        values,
    )

    if result.size != n_edges:
        raise ValueError(
            "Interpolated edge property has "
            f"{result.size} values, expected {n_edges}."
        )

    return result


def _cell_values_to_face_values(
    mesh: Any,
    values: Any,
    n_faces: int,
) -> np.ndarray:
    """Backward-compatible cell-to-face interpolation."""
    result = _cell_to_face_property(
        mesh,
        values,
    )

    if result.size != n_faces:
        raise ValueError(
            "Interpolated face property has "
            f"{result.size} values, expected {n_faces}."
        )

    return result


# ---------------------------------------------------------------------------
# Dual volumes
# ---------------------------------------------------------------------------


def _edge_dual_volumes(
    mesh: Any,
    n_edges: int | None = None,
) -> np.ndarray:
    """Construct lumped dual volumes associated with primal edges."""
    nx, ny, nz = _effective_mesh_shape(mesh)

    n_ex, n_ey, n_ez = _edge_counts_from_shape(
        mesh
    )

    expected_edges = (
        n_ex
        + n_ey
        + n_ez
    )

    if n_edges is None:
        n_edges = expected_edges

    if expected_edges != n_edges:
        raise ValueError(
            "The effective edge topology does not "
            "match the curl matrix."
        )

    cell_volumes = _metric_cell_volumes(
        mesh
    )

    dual = np.zeros(
        n_edges,
        dtype=float,
    )

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                value = (
                    0.25
                    * cell_volumes[
                        _cell_index(
                            ix,
                            iy,
                            iz,
                            ny,
                            nz,
                        )
                    ]
                )

                # Ex edges
                for dy in (0, 1):
                    for dz in (0, 1):
                        edge = (
                            ix
                            * (ny + 1)
                            * (nz + 1)
                            + (iy + dy)
                            * (nz + 1)
                            + (iz + dz)
                        )

                        dual[edge] += value

                # Ey edges
                for dx in (0, 1):
                    for dz in (0, 1):
                        edge = (
                            n_ex
                            + (ix + dx)
                            * ny
                            * (nz + 1)
                            + iy
                            * (nz + 1)
                            + (iz + dz)
                        )

                        dual[edge] += value

                # Ez edges
                for dx in (0, 1):
                    for dy in (0, 1):
                        edge = (
                            n_ex
                            + n_ey
                            + (ix + dx)
                            * (ny + 1)
                            * nz
                            + (iy + dy)
                            * nz
                            + iz
                        )

                        dual[edge] += value

    if (
        not np.all(np.isfinite(dual))
        or np.any(dual <= 0.0)
    ):
        raise ValueError(
            "Edge dual volumes must be finite and "
            "strictly positive."
        )

    return dual


def _face_dual_volumes(
    mesh: Any,
    n_faces: int | None = None,
) -> np.ndarray:
    """Construct lumped dual volumes associated with faces."""
    nx, ny, nz = _effective_mesh_shape(mesh)

    n_fx, n_fy, n_fz = _face_counts_from_shape(
        mesh
    )

    expected_faces = (
        n_fx
        + n_fy
        + n_fz
    )

    if n_faces is None:
        n_faces = expected_faces

    if expected_faces != n_faces:
        raise ValueError(
            "The effective face topology does not "
            "match the curl matrix."
        )

    cell_volumes = _metric_cell_volumes(
        mesh
    )

    dual = np.zeros(
        n_faces,
        dtype=float,
    )

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                value = (
                    0.5
                    * cell_volumes[
                        _cell_index(
                            ix,
                            iy,
                            iz,
                            ny,
                            nz,
                        )
                    ]
                )

                # Fx
                for dx in (0, 1):
                    face = (
                        (ix + dx)
                        * ny
                        * nz
                        + iy * nz
                        + iz
                    )

                    dual[face] += value

                # Fy
                for dy in (0, 1):
                    face = (
                        n_fx
                        + ix
                        * (ny + 1)
                        * nz
                        + (iy + dy)
                        * nz
                        + iz
                    )

                    dual[face] += value

                # Fz
                for dz in (0, 1):
                    face = (
                        n_fx
                        + n_fy
                        + ix
                        * ny
                        * (nz + 1)
                        + iy
                        * (nz + 1)
                        + (iz + dz)
                    )

                    dual[face] += value

    if (
        not np.all(np.isfinite(dual))
        or np.any(dual <= 0.0)
    ):
        raise ValueError(
            "Face dual volumes must be finite and "
            "strictly positive."
        )

    return dual


# ---------------------------------------------------------------------------
# Maxwell system
# ---------------------------------------------------------------------------


@dataclass
class MaxwellSystem:
    """Geometry-aware 3D frequency-domain Maxwell system."""

    mesh: Any
    conductivity: Any
    mu_r: Any = 1.0

    def __post_init__(self) -> None:
        _require_mesh(
            self.mesh
        )

        self.metric = _get_metric(
            self.mesh
        )

        self.cell_volumes = _metric_cell_volumes(
            self.mesh
        )

        self.n_cells = int(
            self.cell_volumes.size
        )

        self.curl = build_curl(
            self.mesh
        )

        if (
            not hasattr(self.curl, "matrix")
            or not hasattr(self.curl, "topology")
        ):
            raise TypeError(
                "build_curl(mesh) must return an object "
                "with matrix and topology."
            )

        curl_matrix = self.curl.matrix

        if not hasattr(
            curl_matrix,
            "shape",
        ):
            raise TypeError(
                "The curl matrix must expose a shape."
            )

        self.n_faces, self.n_edges = (
            int(value)
            for value in curl_matrix.shape
        )

        self.n_ex, self.n_ey, self.n_ez = _edge_counts(
            self.curl,
            self.mesh,
        )

        self.n_fx, self.n_fy, self.n_fz = _face_counts(
            self.curl,
            self.mesh,
        )

        if (
            self.n_ex
            + self.n_ey
            + self.n_ez
            != self.n_edges
        ):
            raise ValueError(
                "The Ex/Ey/Ez topology counts do not "
                "match the curl matrix."
            )

        if (
            self.n_fx
            + self.n_fy
            + self.n_fz
            != self.n_faces
        ):
            raise ValueError(
                "The Fx/Fy/Fz topology counts do not "
                "match the curl matrix."
            )

        self.conductivity = _normalize_conductivity(
            self.conductivity,
            self.n_cells,
        )

        self.mu_r = _normalize_mu_r(
            self.mu_r,
            self.n_cells,
        )

        self._build_operators()

    def _build_operators(self) -> None:
        """Build all geometry-aware Maxwell operators."""
        C_raw = self.curl.matrix

        if hasattr(
            C_raw,
            "toarray",
        ):
            C = np.asarray(
                C_raw.toarray(),
                dtype=float,
            )
        else:
            C = np.asarray(
                C_raw,
                dtype=float,
            )

        if C.shape != (
            self.n_faces,
            self.n_edges,
        ):
            raise ValueError(
                "The curl matrix has shape "
                f"{C.shape}, expected "
                f"{(self.n_faces, self.n_edges)}."
            )

        if not np.all(
            np.isfinite(C)
        ):
            raise ValueError(
                "The curl matrix contains non-finite values."
            )

        self.C = C

        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------

        self.edge_lengths = _metric_edge_lengths(
            self.mesh,
            self.n_edges,
        )

        self.face_areas = _metric_face_areas(
            self.mesh,
            self.n_faces,
        )

        self.D_edge = sparse.diags(
            self.edge_lengths,
            format="csr",
        )

        self.D_face_inv = sparse.diags(
            1.0 / self.face_areas,
            format="csr",
        )

        # ------------------------------------------------------------------
        # Physical curl
        # ------------------------------------------------------------------

        C_sparse = sparse.csr_matrix(
            C
        )

        self._physical_curl_matrix = (
            self.D_face_inv
            @ C_sparse
            @ self.D_edge
        ).tocsr()

        self.C_h = self._physical_curl_matrix

        # ------------------------------------------------------------------
        # Conductivity on electric edges
        # ------------------------------------------------------------------

        sigma_edges = _cell_to_edge_conductivity(
            self.mesh,
            self.conductivity,
        )

        if sigma_edges.size != self.n_edges:
            raise ValueError(
                "Interpolated edge conductivity "
                "has an invalid size."
            )

        self.sigma_edges = sigma_edges

        self.edge_dual_volumes = _edge_dual_volumes(
            self.mesh,
            self.n_edges,
        )

        self._conductivity_mass = sparse.diags(
            sigma_edges
            * self.edge_dual_volumes,
            format="csr",
        )

        self.M_sigma = (
            self._conductivity_mass
        )

        # ------------------------------------------------------------------
        # Permeability on magnetic faces
        # ------------------------------------------------------------------

        mu_r_faces = _cell_to_face_property(
            self.mesh,
            self.mu_r,
        )

        if mu_r_faces.size != self.n_faces:
            raise ValueError(
                "Interpolated face relative permeability "
                "has an invalid size."
            )

        if (
            np.any(mu_r_faces <= 0.0)
            or not np.all(
                np.isfinite(mu_r_faces)
            )
        ):
            raise ValueError(
                "Interpolated face relative permeability "
                "must be finite and strictly positive."
            )

        self.mu_r_faces = mu_r_faces

        self.mu_faces = (
            MU0
            * mu_r_faces
        )

        self.face_dual_volumes = _face_dual_volumes(
            self.mesh,
            self.n_faces,
        )

        self._permeability_mass_inverse = sparse.diags(
            self.face_dual_volumes
            / self.mu_faces,
            format="csr",
        )

        self.M_mu_inv = (
            self._permeability_mass_inverse
        )

        # ------------------------------------------------------------------
        # Curl-curl stiffness
        # ------------------------------------------------------------------

        self._stiffness_matrix = (
            self._physical_curl_matrix.T
            @ self._permeability_mass_inverse
            @ self._physical_curl_matrix
        ).tocsr()

        # Numerical roundoff should not destroy the exact structural
        # symmetry expected from C_h.T M_mu_inv C_h.
        self._stiffness_matrix = (
            0.5
            * (
                self._stiffness_matrix
                + self._stiffness_matrix.T
            )
        ).tocsr()

        self.K = (
            self._stiffness_matrix
        )

    # ------------------------------------------------------------------
    # Matrix accessors
    # ------------------------------------------------------------------

    def conductivity_mass(
        self,
    ) -> sparse.csr_matrix:
        """Return the diagonal conductivity mass matrix."""
        return self._conductivity_mass

    def permeability_mass_inverse(
        self,
    ) -> sparse.csr_matrix:
        """Return the inverse-permeability mass matrix."""
        return self._permeability_mass_inverse

    def stiffness_matrix(
        self,
    ) -> sparse.csr_matrix:
        """Return the real symmetric curl-curl stiffness matrix."""
        return self._stiffness_matrix

    def physical_curl_matrix(
        self,
    ) -> sparse.csr_matrix:
        """Return the geometry-aware physical curl matrix."""
        return self._physical_curl_matrix

    # ------------------------------------------------------------------
    # Frequency-domain system
    # ------------------------------------------------------------------

    def system_matrix(
        self,
        omega: float,
    ) -> sparse.csr_matrix:
        """Return

            A = K + i * omega * M_sigma.

        This sign is required by the ``exp(+i*omega*t)`` convention.
        """
        omega = float(
            omega
        )

        if not np.isfinite(omega):
            raise ValueError(
                "omega must be finite."
            )

        if omega <= 0.0:
            raise ValueError(
                "omega must be strictly positive."
            )

        matrix = (
            self._stiffness_matrix.astype(
                complex
            )
            + 1j
            * omega
            * self._conductivity_mass.astype(
                complex
            )
        ).tocsr()

        if not np.all(
            np.isfinite(
                matrix.data
            )
        ):
            raise ValueError(
                "The Maxwell system matrix contains "
                "non-finite values."
            )

        return matrix

    def matrix(
        self,
        omega: float,
    ) -> sparse.csr_matrix:
        """Compatibility alias for :meth:`system_matrix`."""
        return self.system_matrix(
            omega
        )

    def assemble(
        self,
        omega: float,
    ) -> sparse.csr_matrix:
        """Compatibility alias for :meth:`system_matrix`."""
        return self.system_matrix(
            omega
        )

    # ------------------------------------------------------------------
    # Right-hand side
    # ------------------------------------------------------------------

    def zero_rhs(self) -> np.ndarray:
        """Return a zero electric-field right-hand side."""
        return np.zeros(
            self.n_edges,
            dtype=complex,
        )

    def source_rhs(
        self,
        source: Any = None,
    ) -> np.ndarray:
        """Return the default zero volume source RHS.

        MT excitation is imposed through electric boundary conditions.
        """
        del source
        return self.zero_rhs()

    # ------------------------------------------------------------------
    # Physical curl and magnetic field
    # ------------------------------------------------------------------

    def physical_curl(
        self,
        electric_field: Any,
    ) -> np.ndarray:
        """Compute the geometry-aware physical curl of ``E``.

        The result is ordered

            [curl_x, curl_y, curl_z]

        on the corresponding dual faces.
        """
        E = np.asarray(
            electric_field,
            dtype=complex,
        ).reshape(-1)

        if E.size != self.n_edges:
            raise ValueError(
                "electric_field must contain exactly "
                f"{self.n_edges} edge values; "
                f"got {E.size}."
            )

        if not np.all(
            np.isfinite(E)
        ):
            raise ValueError(
                "electric_field contains non-finite values."
            )

        result = np.asarray(
            self._physical_curl_matrix @ E,
            dtype=complex,
        ).reshape(-1)

        if not np.all(
            np.isfinite(result)
        ):
            raise ValueError(
                "The physical electric curl contains "
                "non-finite values."
            )

        return result

    def magnetic_field(
        self,
        electric_field: Any,
        omega: float,
    ) -> np.ndarray:
        """Compute the physical magnetic field from the electric field.

        With

            exp(+i * omega * t)

        Faraday's law is

            curl(E) = -i * omega * mu * H.

        Therefore

            H = i * curl(E) / (omega * mu).

        The physical geometry-aware curl ``C_h`` is used here.
        """
        omega = float(
            omega
        )

        if (
            not np.isfinite(omega)
            or omega <= 0.0
        ):
            raise ValueError(
                "omega must be finite and strictly positive."
            )

        curl_E = self.physical_curl(
            electric_field
        )

        magnetic = (
            1j
            * curl_E
            / (
                omega
                * self.mu_faces
            )
        )

        if not np.all(
            np.isfinite(magnetic)
        ):
            raise ValueError(
                "The computed magnetic field contains "
                "non-finite values."
            )

        return magnetic

    def physical_magnetic_field(
        self,
        electric_field: Any,
        omega: float,
    ) -> np.ndarray:
        """Explicit alias for the physical Faraday magnetic field."""
        return self.magnetic_field(
            electric_field,
            omega,
        )

    # ------------------------------------------------------------------
    # Material properties
    # ------------------------------------------------------------------

    @property
    def sigma(
        self,
    ) -> np.ndarray:
        """Cell conductivity."""
        return self.conductivity

    @property
    def conductivity_cells(
        self,
    ) -> np.ndarray:
        """Cell conductivity."""
        return self.conductivity

    @property
    def permeability_relative(
        self,
    ) -> np.ndarray:
        """Cell relative permeability."""
        return self.mu_r

    def permeability(
        self,
    ) -> np.ndarray:
        """Return absolute cell permeability in H/m."""
        return (
            MU0
            * self.mu_r
        ).copy()

    def inverse_permeability(
        self,
    ) -> np.ndarray:
        """Return inverse absolute cell permeability."""
        return 1.0 / (
            MU0
            * self.mu_r
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_maxwell_system(
    mesh: Any,
    conductivity: Any = None,
    mu_r: Any = 1.0,
    *,
    sigma: Any = None,
    permeability: Any = None,
    **kwargs: Any,
) -> MaxwellSystem:
    """Build a :class:`MaxwellSystem`.

    Compatibility aliases
    ---------------------
    ``sigma`` is accepted as an alias for ``conductivity``.

    ``permeability`` is accepted as an alias for ``mu_r``.

    Historical ``frequency`` and ``omega`` keyword arguments are accepted
    and ignored because frequency belongs to ``system_matrix(omega)``.
    """
    if conductivity is None:
        conductivity = sigma

    if conductivity is None:
        raise ValueError(
            "conductivity (or sigma) must be provided."
        )

    if permeability is not None:
        mu_r = permeability

    unsupported = {
        key: value
        for key, value in kwargs.items()
        if key not in {
            "frequency",
            "omega",
        }
    }

    if unsupported:
        names = ", ".join(
            sorted(unsupported)
        )

        raise TypeError(
            f"Unexpected keyword argument(s): {names}"
        )

    return MaxwellSystem(
        mesh=mesh,
        conductivity=conductivity,
        mu_r=mu_r,
    )


__all__ = [
    "MU0",
    "MaxwellSystem",
    "build_maxwell_system",
    "_cell_to_edge_conductivity",
    "_cell_to_face_property",
]


