"""
Structured three-dimensional Cartesian mesh.

This module provides the structured 3D mesh representation
used by GeoInvLab3D.

The mesh supports optional variable-size padding cells,
which is useful for magnetotelluric simulations.
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Mesh3D


@dataclass
class StructuredMesh3D(Mesh3D):
    """
    Structured Cartesian 3D mesh with optional padding.

    Parameters
    ----------
    nx : int
        Number of cells in the main X domain.

    ny : int
        Number of cells in the main Y domain.

    nz : int
        Number of cells in the main Z domain.

    dx : float
        Main cell size along X.

    dy : float
        Main cell size along Y.

    dz : float
        Main cell size along Z.

    origin : tuple of float, optional
        Coordinates of the lower-left-front corner of
        the complete mesh.

    padding_x : tuple of int, optional
        Number of padding cells on the negative and positive
        X sides.

    padding_y : tuple of int, optional
        Number of padding cells on the negative and positive
        Y sides.

    padding_z : tuple of int, optional
        Number of padding cells on the negative and positive
        Z sides.

    padding_factor : float, optional
        Geometric growth factor used for padding cells.

        A value of 1.0 produces padding cells with the same
        size as the main cells.

        A value greater than 1.0 progressively increases
        the padding cell sizes.

    Notes
    -----
    The original ``nx``, ``ny``, ``nz``, ``dx``, ``dy`` and
    ``dz`` parameters describe the main computational domain.

    Padding cells extend the domain around this main region.

    The main cells remain regular and preserve the behaviour
    expected by the existing gravity and magnetic inversions.

    Edge ordering
    -------------
    The MT discretization uses three consecutive edge blocks:

        [Ex, Ey, Ez]

    with z varying fastest inside each component block.

    Face ordering
    --------------
    The corresponding face discretization uses:

        [Fx, Fy, Fz]

    with z varying fastest inside each component block.
    """

    nx: int
    ny: int
    nz: int

    dx: float
    dy: float
    dz: float

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    padding_x: tuple[int, int] = (0, 0)
    padding_y: tuple[int, int] = (0, 0)
    padding_z: tuple[int, int] = (0, 0)

    padding_factor: float = 1.0

    def __post_init__(self) -> None:
        """Validate mesh parameters."""

        if self.nx <= 0:
            raise ValueError("nx must be positive.")

        if self.ny <= 0:
            raise ValueError("ny must be positive.")

        if self.nz <= 0:
            raise ValueError("nz must be positive.")

        if self.dx <= 0:
            raise ValueError("dx must be positive.")

        if self.dy <= 0:
            raise ValueError("dy must be positive.")

        if self.dz <= 0:
            raise ValueError("dz must be positive.")

        if len(self.origin) != 3:
            raise ValueError(
                "origin must contain exactly three coordinates."
            )

        self._validate_padding(
            self.padding_x,
            "padding_x",
        )

        self._validate_padding(
            self.padding_y,
            "padding_y",
        )

        self._validate_padding(
            self.padding_z,
            "padding_z",
        )

        if self.padding_factor < 1.0:
            raise ValueError(
                "padding_factor must be greater than or equal to 1.0."
            )

    # ==============================================================
    # VALIDATION
    # ==============================================================

    @staticmethod
    def _validate_padding(
        padding: tuple[int, int],
        name: str,
    ) -> None:
        """Validate one padding specification."""

        if len(padding) != 2:
            raise ValueError(
                f"{name} must contain exactly two integers."
            )

        if not all(
            isinstance(value, (int, np.integer))
            for value in padding
        ):
            raise TypeError(
                f"{name} values must be integers."
            )

        if any(value < 0 for value in padding):
            raise ValueError(
                f"{name} values cannot be negative."
            )

    # ==============================================================
    # MAIN MESH SIZE
    # ==============================================================

    @property
    def n_cells_main(self) -> int:
        """Return the number of cells in the main domain."""

        return self.nx * self.ny * self.nz

    @property
    def n_cells_x(self) -> int:
        """Return the total number of cells along X."""

        return (
            self.padding_x[0]
            + self.nx
            + self.padding_x[1]
        )

    @property
    def n_cells_y(self) -> int:
        """Return the total number of cells along Y."""

        return (
            self.padding_y[0]
            + self.ny
            + self.padding_y[1]
        )

    @property
    def n_cells_z(self) -> int:
        """Return the total number of cells along Z."""

        return (
            self.padding_z[0]
            + self.nz
            + self.padding_z[1]
        )

    # ==============================================================
    # TOTAL NUMBER OF CELLS
    # ==============================================================

    @property
    def n_cells(self) -> int:
        """Return the total number of cells including padding."""

        return (
            self.n_cells_x
            * self.n_cells_y
            * self.n_cells_z
        )

    # ==============================================================
    # SHAPE
    # ==============================================================

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Return the total mesh shape including padding.
        """

        return (
            self.n_cells_x,
            self.n_cells_y,
            self.n_cells_z,
        )

    @property
    def main_shape(self) -> tuple[int, int, int]:
        """Return the shape of the main computational domain."""

        return (
            self.nx,
            self.ny,
            self.nz,
        )

    # ==============================================================
    # TOPOLOGICAL EDGE COUNTS
    # ==============================================================

    @property
    def n_edges_x(self) -> int:
        """
        Return the number of X-oriented edges.

        X-oriented edges carry the Ex component.

        For a mesh with ``(nx, ny, nz)`` cells:

            n_ex = nx * (ny + 1) * (nz + 1)
        """

        nx, ny, nz = self.shape

        return nx * (ny + 1) * (nz + 1)

    @property
    def n_edges_y(self) -> int:
        """
        Return the number of Y-oriented edges.

        Y-oriented edges carry the Ey component.

            n_ey = (nx + 1) * ny * (nz + 1)
        """

        nx, ny, nz = self.shape

        return (nx + 1) * ny * (nz + 1)

    @property
    def n_edges_z(self) -> int:
        """
        Return the number of Z-oriented edges.

        Z-oriented edges carry the Ez component.

            n_ez = (nx + 1) * (ny + 1) * nz
        """

        nx, ny, nz = self.shape

        return (nx + 1) * (ny + 1) * nz

    @property
    def n_edges(self) -> int:
        """
        Return the total number of mesh edges.

        The global edge ordering is:

            [Ex, Ey, Ez]

        Therefore:

            n_edges = n_edges_x
                     + n_edges_y
                     + n_edges_z
        """

        return (
            self.n_edges_x
            + self.n_edges_y
            + self.n_edges_z
        )

    # ==============================================================
    # TOPOLOGICAL FACE COUNTS
    # ==============================================================

    @property
    def n_faces_x(self) -> int:
        """
        Return the number of X-normal faces.

        These faces carry the Fx / Hx component.
        """

        nx, ny, nz = self.shape

        return (nx + 1) * ny * nz

    @property
    def n_faces_y(self) -> int:
        """
        Return the number of Y-normal faces.

        These faces carry the Fy / Hy component.
        """

        nx, ny, nz = self.shape

        return nx * (ny + 1) * nz

    @property
    def n_faces_z(self) -> int:
        """
        Return the number of Z-normal faces.

        These faces carry the Fz / Hz component.
        """

        nx, ny, nz = self.shape

        return nx * ny * (nz + 1)

    @property
    def n_faces(self) -> int:
        """
        Return the total number of mesh faces.

        The global face ordering is:

            [Fx, Fy, Fz]
        """

        return (
            self.n_faces_x
            + self.n_faces_y
            + self.n_faces_z
        )

    # ==============================================================
    # CELL WIDTHS
    # ==============================================================

    @staticmethod
    def _build_widths(
        n_main: int,
        main_size: float,
        padding_negative: int,
        padding_positive: int,
        factor: float,
    ) -> np.ndarray:
        """
        Build cell widths along one axis.

        Main cells have constant size.

        Padding cells grow geometrically away from the
        main domain.
        """

        negative = np.array(
            [
                main_size * factor ** (
                    padding_negative - index
                )
                for index in range(
                    padding_negative
                )
            ],
            dtype=float,
        )

        if negative.size:
            negative = negative[::-1]

        main = np.full(
            n_main,
            main_size,
            dtype=float,
        )

        positive = np.array(
            [
                main_size * factor ** (
                    index + 1
                )
                for index in range(
                    padding_positive
                )
            ],
            dtype=float,
        )

        return np.concatenate(
            (
                negative,
                main,
                positive,
            )
        )

    @property
    def cell_widths_x(self) -> np.ndarray:
        """Return all cell widths along X."""

        return self._build_widths(
            self.nx,
            self.dx,
            self.padding_x[0],
            self.padding_x[1],
            self.padding_factor,
        )

    @property
    def cell_widths_y(self) -> np.ndarray:
        """Return all cell widths along Y."""

        return self._build_widths(
            self.ny,
            self.dy,
            self.padding_y[0],
            self.padding_y[1],
            self.padding_factor,
        )

    @property
    def cell_widths_z(self) -> np.ndarray:
        """Return all cell widths along Z."""

        return self._build_widths(
            self.nz,
            self.dz,
            self.padding_z[0],
            self.padding_z[1],
            self.padding_factor,
        )

    # ==============================================================
    # CELL EDGES
    # ==============================================================

    @property
    def edges_x(self) -> np.ndarray:
        """Return cell boundary coordinates along X."""

        ox = self.origin[0]

        return ox + np.concatenate(
            (
                [0.0],
                np.cumsum(self.cell_widths_x),
            )
        )

    @property
    def edges_y(self) -> np.ndarray:
        """Return cell boundary coordinates along Y."""

        oy = self.origin[1]

        return oy + np.concatenate(
            (
                [0.0],
                np.cumsum(self.cell_widths_y),
            )
        )

    @property
    def edges_z(self) -> np.ndarray:
        """Return cell boundary coordinates along Z."""

        oz = self.origin[2]

        return oz + np.concatenate(
            (
                [0.0],
                np.cumsum(self.cell_widths_z),
            )
        )

    # ==============================================================
    # CELL CENTERS
    # ==============================================================

    def cell_centers(self) -> np.ndarray:
        """
        Return coordinates of all cell centers.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_cells, 3)``.

        Notes
        -----
        The ordering remains compatible with the previous
        structured mesh implementation:

            x -> y -> z

        Thus z varies fastest.
        """

        x_edges = self.edges_x
        y_edges = self.edges_y
        z_edges = self.edges_z

        x = 0.5 * (
            x_edges[:-1]
            + x_edges[1:]
        )

        y = 0.5 * (
            y_edges[:-1]
            + y_edges[1:]
        )

        z = 0.5 * (
            z_edges[:-1]
            + z_edges[1:]
        )

        xx, yy, zz = np.meshgrid(
            x,
            y,
            z,
            indexing="ij",
        )

        return np.column_stack(
            (
                xx.ravel(),
                yy.ravel(),
                zz.ravel(),
            )
        )

    # ==============================================================
    # CELL VOLUMES
    # ==============================================================

    @property
    def cell_volume(self) -> float:
        """
        Return the volume of one regular main-domain cell.

        The main computational domain is composed of regular
        cells with dimensions ``dx``, ``dy`` and ``dz``.

        Returns
        -------
        float
            Volume of one main-domain cell.
        """

        return self.dx * self.dy * self.dz

    def cell_volumes(self) -> np.ndarray:
        """
        Return the volume of every mesh cell.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_cells,)``.

        Notes
        -----
        This method accounts for variable-size padding cells.
        """

        volumes = (
            self.cell_widths_x[:, None, None]
            * self.cell_widths_y[None, :, None]
            * self.cell_widths_z[None, None, :]
        )

        return volumes.ravel()

    # ==============================================================
    # MAIN DOMAIN
    # ==============================================================

    @property
    def main_origin(self) -> tuple[float, float, float]:
        """
        Return the origin of the main computational domain.
        """

        return (
            self.edges_x[self.padding_x[0]],
            self.edges_y[self.padding_y[0]],
            self.edges_z[self.padding_z[0]],
        )

    @property
    def main_dimensions(self) -> tuple[float, float, float]:
        """Return the dimensions of the main domain."""

        return (
            self.nx * self.dx,
            self.ny * self.dy,
            self.nz * self.dz,
        )

    # ==============================================================
    # TOTAL DIMENSIONS
    # ==============================================================

    @property
    def dimensions(self) -> tuple[float, float, float]:
        """
        Return the total physical dimensions including padding.
        """

        return (
            float(np.sum(self.cell_widths_x)),
            float(np.sum(self.cell_widths_y)),
            float(np.sum(self.cell_widths_z)),
        )

    # ==============================================================
    # COPY
    # ==============================================================

    def copy(self) -> "StructuredMesh3D":
        """Return an independent copy of the mesh."""

        return StructuredMesh3D(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            origin=tuple(self.origin),
            padding_x=tuple(self.padding_x),
            padding_y=tuple(self.padding_y),
            padding_z=tuple(self.padding_z),
            padding_factor=self.padding_factor,
        )


