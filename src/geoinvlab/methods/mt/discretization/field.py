"""
Physical electric and magnetic fields for the 3D MT discretization.

This module provides lightweight containers and physical field
reconstruction utilities for the edge/face finite-volume formulation
used by GeoInvLab3D.

Coordinate convention
---------------------
GeoInvLab3D uses the right-handed coordinate system

    x = East
    y = North
    z = positive depth (downward)

Time convention
---------------
The whole MT implementation uses

    exp(+i * omega * t)

and therefore Faraday's law is

    curl(E) = -i * omega * mu * H.

Consequently,

    H = i * curl(E) / (omega * mu).

Plane-wave convention
---------------------
For a homogeneous half-space, the electric field is written as

    E(z) = E0 * exp(-i * k * z),

with the physical complex wavenumber

    k^2 = -i * omega * mu * sigma,

and the physical branch

    Re(k) > 0,
    Im(k) < 0.

Thus, for an X-polarized wave,

    E = Ex(z) e_x,

    (curl E)_y = dEx/dz
                = -i * k * Ex,

and Faraday's law gives

    Hy = k * Ex / (omega * mu).

Therefore,

    Zxy = Ex / Hy
        = omega * mu / k.

For a Y-polarized wave,

    Zyx = -omega * mu / k.

Hence, for a homogeneous isotropic half-space,

    Zyx = -Zxy.

Discrete field convention
-------------------------
The electric field is stored on mesh edges in the ordering

    [Ex, Ey, Ez]

and the magnetic field is stored on mesh faces in the ordering

    [Hx, Hy, Hz].

The physical curl is the geometry-aware operator

    C_h = D_face^{-1} C D_edge,

where

    C       : topological edge-to-face curl,
    D_edge  : edge lengths,
    D_face  : face areas.

Thus, if ``E`` contains edge-integrated electric quantities,

    curl(E) = C_h E.

The physical magnetic field is then

    H = i * C_h E / (omega * mu).

This definition must remain identical to the one used by
``MaxwellSystem``.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curl import build_curl
from .maxwell import MU0
from .metric import build_metric


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _as_complex_1d(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """
    Convert values to a one-dimensional complex NumPy array.

    A copy is intentionally avoided when possible.
    """
    array = np.asarray(
        values,
        dtype=np.complex128,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array; "
            f"got shape {array.shape!r}."
        )

    return array


def _validate_component_sizes(
    values: np.ndarray,
    *,
    n_x: int,
    n_y: int,
    n_z: int,
    field_name: str,
) -> None:
    """
    Validate component counts against a flattened field vector.
    """
    n_x = int(n_x)
    n_y = int(n_y)
    n_z = int(n_z)

    if n_x < 0 or n_y < 0 or n_z < 0:
        raise ValueError(
            f"{field_name} component counts must be non-negative."
        )

    expected = n_x + n_y + n_z

    if values.size != expected:
        raise ValueError(
            f"{field_name} has {values.size} values, but the supplied "
            f"component counts require {expected} values "
            f"(n_x={n_x}, n_y={n_y}, n_z={n_z})."
        )


# ---------------------------------------------------------------------------
# Electric field
# ---------------------------------------------------------------------------


@dataclass
class EdgeElectricField:
    """
    Electric field stored on mesh edges.

    Global ordering:

        [Ex, Ey, Ez]

    with

        x = East,
        y = North,
        z = positive depth.
    """

    values: np.ndarray
    n_ex: int
    n_ey: int
    n_ez: int

    def __post_init__(self) -> None:
        self.values = _as_complex_1d(
            self.values,
            name="Electric field values",
        )

        self.n_ex = int(self.n_ex)
        self.n_ey = int(self.n_ey)
        self.n_ez = int(self.n_ez)

        if not np.all(np.isfinite(self.values)):
            raise ValueError(
                "Electric field values must be finite "
                "(non-finite values detected)."
            )

        _validate_component_sizes(
            self.values,
            n_x=self.n_ex,
            n_y=self.n_ey,
            n_z=self.n_ez,
            field_name="Electric field",
        )

    def as_vector(self) -> np.ndarray:
        """Return an independent copy of the global electric-field vector."""
        return self.values.copy()

    @property
    def shape(self) -> tuple[int]:
        """Shape of the flattened field vector."""
        return self.values.shape

    @property
    def n_edges(self) -> int:
        """Total number of electric-field edge values."""
        return self.values.size

    @property
    def ex(self) -> np.ndarray:
        """Electric Ex component."""
        return self.values[: self.n_ex]

    @property
    def ey(self) -> np.ndarray:
        """Electric Ey component."""
        start = self.n_ex
        stop = start + self.n_ey
        return self.values[start:stop]

    @property
    def ez(self) -> np.ndarray:
        """Electric Ez component."""
        start = self.n_ex + self.n_ey
        return self.values[start:]

    @property
    def Ex(self) -> np.ndarray:
        """Alias for :attr:`ex`."""
        return self.ex

    @property
    def Ey(self) -> np.ndarray:
        """Alias for :attr:`ey`."""
        return self.ey

    @property
    def Ez(self) -> np.ndarray:
        """Alias for :attr:`ez`."""
        return self.ez


# ---------------------------------------------------------------------------
# Magnetic field
# ---------------------------------------------------------------------------


@dataclass
class FaceMagneticField:
    """
    Magnetic field stored on mesh faces.

    Global ordering:

        [Hx, Hy, Hz]

    where:

        Hx -> Fx faces,
        Hy -> Fy faces,
        Hz -> Fz faces.

    The coordinate convention is

        x = East,
        y = North,
        z = positive depth.
    """

    values: np.ndarray
    n_fx: int
    n_fy: int
    n_fz: int

    def __post_init__(self) -> None:
        self.values = _as_complex_1d(
            self.values,
            name="Magnetic field values",
        )

        self.n_fx = int(self.n_fx)
        self.n_fy = int(self.n_fy)
        self.n_fz = int(self.n_fz)

        if not np.all(np.isfinite(self.values)):
            raise ValueError(
                "Magnetic field values must be finite "
                "(non-finite values detected)."
            )

        _validate_component_sizes(
            self.values,
            n_x=self.n_fx,
            n_y=self.n_fy,
            n_z=self.n_fz,
            field_name="Magnetic field",
        )

    @property
    def shape(self) -> tuple[int]:
        """Shape of the flattened magnetic-field vector."""
        return self.values.shape

    @property
    def real(self) -> np.ndarray:
        """Real part of the magnetic-field values."""
        return self.values.real

    @property
    def imag(self) -> np.ndarray:
        """Imaginary part of the magnetic-field values."""
        return self.values.imag

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Expose the field as a NumPy array for legacy callers.
        """
        return np.asarray(
            self.values,
            dtype=dtype,
        )

    @property
    def n_faces(self) -> int:
        """Total number of magnetic-field face values."""
        return self.values.size

    @property
    def hx(self) -> np.ndarray:
        """Magnetic Hx component."""
        return self.values[: self.n_fx]

    @property
    def hy(self) -> np.ndarray:
        """Magnetic Hy component."""
        start = self.n_fx
        stop = start + self.n_fy
        return self.values[start:stop]

    @property
    def hz(self) -> np.ndarray:
        """Magnetic Hz component."""
        start = self.n_fx + self.n_fy
        return self.values[start:]

    @property
    def Hx(self) -> np.ndarray:
        """Alias for :attr:`hx`."""
        return self.hx

    @property
    def Hy(self) -> np.ndarray:
        """Alias for :attr:`hy`."""
        return self.hy

    @property
    def Hz(self) -> np.ndarray:
        """Alias for :attr:`hz`."""
        return self.hz


# ---------------------------------------------------------------------------
# Field constructors
# ---------------------------------------------------------------------------


def build_edge_electric_field(
    mesh,
    values: np.ndarray,
) -> EdgeElectricField:
    """
    Build an :class:`EdgeElectricField` from a mesh and global values.
    """
    array = _as_complex_1d(
        values,
        name="Electric field values",
    )

    edge_counts = _edge_counts(mesh)
    expected = sum(edge_counts)

    if array.size != expected:
        raise ValueError(
            f"Electric field has an incompatible size: {array.size} "
            f"values provided, but the mesh requires {expected} "
            f"edge values."
        )

    return EdgeElectricField(
        array,
        edge_counts[0],
        edge_counts[1],
        edge_counts[2],
    )


def build_face_magnetic_field(
    mesh,
    values: np.ndarray,
) -> FaceMagneticField:
    """
    Build a :class:`FaceMagneticField` from a mesh and global values.
    """
    array = _as_complex_1d(
        values,
        name="Magnetic field values",
    )

    face_counts = _face_counts(mesh)
    expected = sum(face_counts)

    if array.size != expected:
        raise ValueError(
            f"Magnetic field has an incompatible size: {array.size} "
            f"values provided, but the mesh requires {expected} "
            f"face values."
        )

    return FaceMagneticField(
        array,
        face_counts[0],
        face_counts[1],
        face_counts[2],
    )


# ---------------------------------------------------------------------------
# Mesh topology / geometry helpers
# ---------------------------------------------------------------------------


def _mesh_shape(
    mesh,
) -> tuple[int, int, int]:
    """
    Return the effective mesh shape.

    The returned values are the numbers of cells in x, y and z.
    """
    from geoinvlab.core.mesh.structured import StructuredMesh3D

    if not isinstance(
        mesh,
        StructuredMesh3D,
    ):
        raise TypeError(
            "mesh must be a StructuredMesh3D instance."
        )

    shape = getattr(
        mesh,
        "shape",
        None,
    )

    if shape is None:
        shape = (
            mesh.nx,
            mesh.ny,
            mesh.nz,
        )

    if len(shape) != 3:
        raise ValueError(
            f"Expected a 3D mesh shape; got {shape!r}."
        )

    nx, ny, nz = (
        int(value)
        for value in shape
    )

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(
            "Mesh dimensions must be strictly positive; "
            f"got {shape!r}."
        )

    return nx, ny, nz


def _edge_counts(
    mesh,
) -> tuple[int, int, int]:
    """
    Return the numbers of Ex, Ey and Ez edge unknowns.

    For nx, ny, nz cells:

        nEx = nx * (ny + 1) * (nz + 1)
        nEy = (nx + 1) * ny * (nz + 1)
        nEz = (nx + 1) * (ny + 1) * nz
    """
    nx, ny, nz = _mesh_shape(mesh)

    return (
        nx * (ny + 1) * (nz + 1),
        (nx + 1) * ny * (nz + 1),
        (nx + 1) * (ny + 1) * nz,
    )


def _face_counts(
    mesh,
) -> tuple[int, int, int]:
    """
    Return the numbers of Fx, Fy and Fz face unknowns.

    For nx, ny, nz cells:

        nFx = (nx + 1) * ny * nz
        nFy = nx * (ny + 1) * nz
        nFz = nx * ny * (nz + 1)
    """
    nx, ny, nz = _mesh_shape(mesh)

    return (
        (nx + 1) * ny * nz,
        nx * (ny + 1) * nz,
        nx * ny * (nz + 1),
    )


# ---------------------------------------------------------------------------
# Topological and physical curl
# ---------------------------------------------------------------------------


def electric_curl(
    mesh,
    electric_field: EdgeElectricField | np.ndarray,
) -> np.ndarray:
    """
    Compute the topological edge-to-face curl.

    This returns

        C E

    and therefore represents circulation rather than physical curl.

    For the physical curl, use :func:`physical_electric_curl`.
    """
    if isinstance(
        electric_field,
        EdgeElectricField,
    ):
        values = electric_field.values
    else:
        values = _as_complex_1d(
            electric_field,
            name="Electric field values",
        )

    edge_counts = _edge_counts(mesh)
    expected = sum(edge_counts)

    if values.size != expected:
        raise ValueError(
            f"Electric field has an incompatible size: got {values.size} "
            f"values, but the mesh requires {expected} edge values."
        )

    curl = build_curl(mesh)
    matrix = getattr(
        curl,
        "matrix",
        curl,
    )

    circulation = matrix @ values

    return np.asarray(
        circulation,
        dtype=np.complex128,
    )


def physical_electric_curl(
    mesh,
    electric_field: EdgeElectricField | np.ndarray,
) -> np.ndarray:
    """
    Compute the physical curl of the electric field.

    The geometry-aware operator is

        C_h = D_face^{-1} C D_edge.

    Therefore,

        curl(E) = D_face^{-1} C D_edge E.

    The input ``E`` is represented by edge-integrated electric
    quantities. ``D_edge`` converts those quantities to the
    corresponding edge tangential field values before applying the
    topological circulation operator.
    """
    if isinstance(
        electric_field,
        EdgeElectricField,
    ):
        values = electric_field.values
    else:
        values = _as_complex_1d(
            electric_field,
            name="Electric field values",
        )

    edge_counts = _edge_counts(mesh)
    face_counts = _face_counts(mesh)

    n_edges = sum(edge_counts)
    n_faces = sum(face_counts)

    if values.size != n_edges:
        raise ValueError(
            f"Electric field has {values.size} values, but the mesh "
            f"requires {n_edges} edge values."
        )

    metric = build_metric(mesh)

    edge_lengths = np.asarray(
        metric.edges.lengths,
        dtype=float,
    )

    face_areas = np.asarray(
        metric.faces.areas,
        dtype=float,
    )

    if edge_lengths.size != n_edges:
        raise ValueError(
            "The metric edge-length array is inconsistent with the "
            f"mesh: got {edge_lengths.size}, expected {n_edges}."
        )

    if face_areas.size != n_faces:
        raise ValueError(
            "The metric face-area array is inconsistent with the "
            f"mesh: got {face_areas.size}, expected {n_faces}."
        )

    if (
        np.any(~np.isfinite(edge_lengths))
        or np.any(edge_lengths <= 0.0)
    ):
        raise ValueError(
            "All edge lengths must be finite and strictly positive."
        )

    if (
        np.any(~np.isfinite(face_areas))
        or np.any(face_areas <= 0.0)
    ):
        raise ValueError(
            "All face areas must be finite and strictly positive."
        )

    curl = build_curl(mesh)
    matrix = getattr(
        curl,
        "matrix",
        curl,
    )

    # E is represented by edge-integrated quantities.
    #
    # Convert edge quantities to tangential field values, compute
    # the topological circulation, then divide by face area:
    #
    #     curl(E) = D_face^{-1} C D_edge E.
    circulation = matrix @ (
        edge_lengths * values
    )

    physical_curl = (
        circulation / face_areas
    )

    return np.asarray(
        physical_curl,
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------------
# Magnetic permeability interpolation
# ---------------------------------------------------------------------------


def _cell_to_face_mu_r(
    mesh,
    mu_r: float | np.ndarray,
) -> np.ndarray:
    """
    Interpolate relative permeability from cells to faces.

    Face ordering:

        [Fx, Fy, Fz]

    with

        Fx -> Hx,
        Fy -> Hy,
        Fz -> Hz.

    Arithmetic averaging is used on internal faces. Boundary faces
    receive the value of their adjacent cell.

    The cell ordering follows the GeoInvLab3D convention in which
    z is the fastest index.
    """
    nx, ny, nz = _mesh_shape(mesh)

    n_cells = nx * ny * nz
    n_fx, n_fy, n_fz = _face_counts(mesh)

    mu = np.asarray(
        mu_r,
        dtype=float,
    )

    if mu.ndim == 0:
        value = float(
            mu.item()
        )

        if (
            not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                "Relative permeability must be finite and "
                "strictly positive."
            )

        cell_mu = np.full(
            n_cells,
            value,
            dtype=float,
        )

    else:
        if mu.ndim != 1:
            raise ValueError(
                "mu_r must be a scalar or a one-dimensional array."
            )

        cell_mu = mu.copy()

        if cell_mu.size != n_cells:
            raise ValueError(
                "Relative permeability array has an incompatible "
                f"size: {cell_mu.size} values provided, but the "
                f"mesh contains {n_cells} cells."
            )

        if (
            np.any(~np.isfinite(cell_mu))
            or np.any(cell_mu <= 0.0)
        ):
            raise ValueError(
                "Relative permeability contains non-finite or "
                "non-positive values."
            )

    # z is the fastest cell index.
    cells = cell_mu.reshape(
        (nx, ny, nz),
        order="C",
    )

    # ------------------------------------------------------------------
    # Fx faces
    # ------------------------------------------------------------------

    mu_fx = np.empty(
        (nx + 1, ny, nz),
        dtype=float,
    )

    mu_fx[0, :, :] = cells[0, :, :]
    mu_fx[nx, :, :] = cells[nx - 1, :, :]

    if nx > 1:
        mu_fx[1:nx, :, :] = 0.5 * (
            cells[:-1, :, :]
            + cells[1:, :, :]
        )

    # ------------------------------------------------------------------
    # Fy faces
    # ------------------------------------------------------------------

    mu_fy = np.empty(
        (nx, ny + 1, nz),
        dtype=float,
    )

    mu_fy[:, 0, :] = cells[:, 0, :]
    mu_fy[:, ny, :] = cells[:, ny - 1, :]

    if ny > 1:
        mu_fy[:, 1:ny, :] = 0.5 * (
            cells[:, :-1, :]
            + cells[:, 1:, :]
        )

    # ------------------------------------------------------------------
    # Fz faces
    # ------------------------------------------------------------------

    mu_fz = np.empty(
        (nx, ny, nz + 1),
        dtype=float,
    )

    mu_fz[:, :, 0] = cells[:, :, 0]
    mu_fz[:, :, nz] = cells[:, :, nz - 1]

    if nz > 1:
        mu_fz[:, :, 1:nz] = 0.5 * (
            cells[:, :, :-1]
            + cells[:, :, 1:]
        )

    result = np.concatenate(
        (
            mu_fx.reshape(-1, order="C"),
            mu_fy.reshape(-1, order="C"),
            mu_fz.reshape(-1, order="C"),
        )
    )

    expected = (
        n_fx
        + n_fy
        + n_fz
    )

    if result.size != expected:
        raise RuntimeError(
            "Internal error while constructing face permeability: "
            f"got {result.size} values, expected {expected}."
        )

    return result


# ---------------------------------------------------------------------------
# Physical magnetic field
# ---------------------------------------------------------------------------


def physical_magnetic_field(
    mesh,
    electric_field: EdgeElectricField | np.ndarray,
    omega: float,
    mu_r: float | np.ndarray = 1.0,
) -> np.ndarray:
    """
    Compute the physical magnetic field from the electric field.

    Time convention:

        exp(+i * omega * t)

    Faraday's law:

        curl(E) = -i * omega * mu * H.

    Therefore:

        H = i * curl(E) / (omega * mu).

    The physical curl is

        curl(E) = D_face^{-1} C D_edge E.

    No additional global sign is introduced here.

    In particular, for a downward-propagating X-polarized plane wave

        E_x(z) = E0 exp(-i k z),

    the correctly oriented discrete curl gives

        (curl E)_y = -i k E_x,

    and consequently

        H_y = k E_x / (omega mu).

    This is required for

        Z_xy = E_x / H_y
             = omega mu / k.

    Likewise, a Y-polarized wave gives

        Z_yx = -omega mu / k.
    """
    omega = float(omega)

    if (
        not np.isfinite(omega)
        or omega <= 0.0
    ):
        raise ValueError(
            "omega must be finite and strictly positive."
        )

    if isinstance(
        electric_field,
        EdgeElectricField,
    ):
        values = electric_field.values
    else:
        values = _as_complex_1d(
            electric_field,
            name="Electric field values",
        )

    n_edges = sum(
        _edge_counts(mesh)
    )
    n_faces = sum(
        _face_counts(mesh)
    )

    if values.size != n_edges:
        raise ValueError(
            f"Electric field has {values.size} values, but the mesh "
            f"requires {n_edges} edge values."
        )

    # ---------------------------------------------------------------
    # Physical electric curl
    # ---------------------------------------------------------------

    curl_e = physical_electric_curl(
        mesh,
        values,
    )

    if curl_e.size != n_faces:
        raise RuntimeError(
            "Physical electric curl has an inconsistent size: "
            f"got {curl_e.size}, expected {n_faces}."
        )

    # ---------------------------------------------------------------
    # Interpolate relative permeability to faces
    # ---------------------------------------------------------------

    mu_r_faces = _cell_to_face_mu_r(
        mesh,
        mu_r,
    )

    if mu_r_faces.size != n_faces:
        raise RuntimeError(
            "Face permeability has an inconsistent size: "
            f"got {mu_r_faces.size}, expected {n_faces}."
        )

    mu_faces = (
        MU0
        * mu_r_faces
    )

    if (
        np.any(~np.isfinite(mu_faces))
        or np.any(mu_faces <= 0.0)
    ):
        raise ValueError(
            "All face magnetic permeabilities must be finite "
            "and strictly positive."
        )

    # ---------------------------------------------------------------
    # Faraday law
    #
    #     curl(E) = -i omega mu H
    #
    # Therefore
    #
    #     H = i curl(E) / (omega mu).
    #
    # ---------------------------------------------------------------

    magnetic = (
        1j
        * curl_e
        / (
            omega
            * mu_faces
        )
    )

    return np.asarray(
        magnetic,
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------------
# Backward-compatible magnetic-field API
# ---------------------------------------------------------------------------


def magnetic_field(
    mesh,
    electric_field: EdgeElectricField | np.ndarray,
    omega: float,
    mu_r: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Return the magnetic field using the legacy topological-curl API.

    This compatibility function pairs with :func:`electric_curl`, which
    returns the topological circulation ``C E``.  Thus it satisfies the
    legacy discrete identity

        -i * omega * mu * H = electric_curl(mesh, E).

    For the geometry-aware physical curl ``C_h = D_face^{-1} C D_edge``
    used by the MT Maxwell solver, call :func:`physical_magnetic_field`.
    Both functions preserve the ``exp(+i omega t)`` convention and use
    ``H = i curl(E) / (omega mu)``.
    """
    omega = float(omega)
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and strictly positive.")

    curl_e = electric_curl(mesh, electric_field)
    mu_r_faces = _cell_to_face_mu_r(mesh, mu_r)
    if mu_r_faces.size != curl_e.size:
        raise RuntimeError(
            "Face permeability has an inconsistent size: "
            f"got {mu_r_faces.size}, expected {curl_e.size}."
        )
    mu_faces = MU0 * mu_r_faces
    if np.any(~np.isfinite(mu_faces)) or np.any(mu_faces <= 0.0):
        raise ValueError(
            "All face magnetic permeabilities must be finite "
            "and strictly positive."
        )
    return np.asarray(1j * curl_e / (omega * mu_faces), dtype=np.complex128)


# ---------------------------------------------------------------------------
# Convenience wrapper returning the typed field
# ---------------------------------------------------------------------------


def build_physical_magnetic_field(
    mesh,
    electric_field: EdgeElectricField | np.ndarray,
    omega: float,
    mu_r: float | np.ndarray = 1.0,
) -> FaceMagneticField:
    """
    Compute and wrap the physical magnetic field.
    """
    values = physical_magnetic_field(
        mesh=mesh,
        electric_field=electric_field,
        omega=omega,
        mu_r=mu_r,
    )

    n_fx, n_fy, n_fz = _face_counts(mesh)

    return FaceMagneticField(
        values,
        n_fx,
        n_fy,
        n_fz,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


__all__ = [
    "EdgeElectricField",
    "FaceMagneticField",
    "build_edge_electric_field",
    "build_face_magnetic_field",
    "electric_curl",
    "physical_electric_curl",
    "physical_magnetic_field",
    "magnetic_field",
    "build_physical_magnetic_field",
]


