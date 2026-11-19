"""
MT plane-wave source and boundary excitation.

Coordinate convention
---------------------
GeoInvLab3D uses

    x = East
    y = North
    z = positive depth (downward)

Time convention
---------------
GeoInvLab3D uses

    exp(+i * omega * t)

and therefore

    curl(E) = -i * omega * mu * H

with

    H = i * curl(E) / (omega * mu).

Plane-wave convention
---------------------
The homogeneous reference solution used throughout GeoInvLab3D is

    gamma = sqrt(-i * omega * mu * sigma)

with the physical branch

    Re(gamma) > 0
    Im(gamma) < 0.

The corresponding electric field convention used by the MT reference
kernel is

    E(z) = E0 * exp(-i * gamma * (z - z_surface)).

Because gamma has a negative imaginary part, this expression decays
with positive depth.

For the discrete MT curl convention used by GeoInvLab3D,

    (curl E)_y = dEx/dz

for an X-polarized wave. Therefore,

    dEx/dz = -i * gamma * Ex,

and

    Hy = i * (curl E)_y / (omega * mu)
       = gamma * Ex / (omega * mu).

Consequently,

    Zxy = Ex / Hy
        = omega * mu / gamma.

Likewise,

    Zyx = -omega * mu / gamma.

Thus a homogeneous half-space with the exp(+i*omega*t)
convention has the reference MT phase used by GeoInvLab3D:

    phase(Zxy) = +45 degrees
    phase(Zyx) = -135 degrees.

Electric unknowns
-----------------
The Maxwell system uses physical edge electric-field values in V/m.

Topology convention
-------------------
The actual global electric-edge indexing is defined by
``build_topology(mesh)``.

The source therefore uses

    topology.edges.ex
    topology.edges.ey
    topology.edges.ez

directly instead of reconstructing global edge indices.

This is important because ``StructuredMesh3D.shape`` is the single
source of truth for the topological dimensions.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .boundary import BoundaryCondition, make_dirichlet_boundary
from .topology import build_topology


MU0 = 4.0e-7 * np.pi


class MTSourcePolarization(str, Enum):
    """Horizontal MT source polarization."""

    X = "x"
    Y = "y"


@dataclass(frozen=True)
class MTPlaneWaveSource:
    """Homogeneous plane-wave MT source."""

    polarization: MTSourcePolarization
    amplitude: complex = 1.0 + 0.0j
    phase: float = 0.0
    frequency: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(
            self.polarization,
            MTSourcePolarization,
        ):
            try:
                polarization = MTSourcePolarization(
                    self.polarization
                )
            except ValueError as exc:
                raise ValueError(
                    "polarization must be 'x' or 'y'."
                ) from exc

            object.__setattr__(
                self,
                "polarization",
                polarization,
            )

        amplitude = complex(self.amplitude)
        phase = float(self.phase)
        frequency = float(self.frequency)

        if not np.all(
            np.isfinite(
                [
                    amplitude.real,
                    amplitude.imag,
                ]
            )
        ):
            raise ValueError(
                "amplitude must be finite."
            )

        if not np.isfinite(phase):
            raise ValueError(
                "phase must be finite."
            )

        if (
            not np.isfinite(frequency)
            or frequency <= 0.0
        ):
            raise ValueError(
                "frequency must be finite and strictly positive."
            )

        object.__setattr__(
            self,
            "amplitude",
            amplitude,
        )

        object.__setattr__(
            self,
            "phase",
            phase,
        )

        object.__setattr__(
            self,
            "frequency",
            frequency,
        )

    @property
    def omega(self) -> float:
        """Angular frequency in rad/s."""
        return 2.0 * np.pi * self.frequency

    @property
    def complex_amplitude(self) -> complex:
        """Complex source amplitude including phase."""
        return (
            self.amplitude
            * np.exp(1j * self.phase)
        )

    def propagation_constant(
        self,
        omega: float,
        conductivity: float,
        mu_r: float = 1.0,
    ) -> complex:
        """
        Return the physical MT propagation constant.

        With the exp(+i*omega*t) convention,

            gamma = sqrt(-i * omega * mu * sigma)

        and the physical branch satisfies

            Re(gamma) > 0
            Im(gamma) < 0.
        """
        omega = float(omega)
        conductivity = float(conductivity)
        mu_r = float(mu_r)

        if (
            not np.isfinite(omega)
            or omega <= 0.0
        ):
            raise ValueError(
                "omega must be finite and strictly positive."
            )

        if (
            not np.isfinite(conductivity)
            or conductivity <= 0.0
        ):
            raise ValueError(
                "conductivity must be finite and strictly positive."
            )

        if (
            not np.isfinite(mu_r)
            or mu_r <= 0.0
        ):
            raise ValueError(
                "mu_r must be finite and strictly positive."
            )

        mu = MU0 * mu_r

        gamma = np.sqrt(
            -1j
            * omega
            * mu
            * conductivity
        )

        # Select the physical branch:
        #
        #     Re(gamma) > 0
        #     Im(gamma) < 0
        #
        if gamma.real < 0.0:
            gamma = -gamma

        if gamma.imag > 0.0:
            gamma = -gamma

        if (
            not np.isfinite(gamma.real)
            or not np.isfinite(gamma.imag)
        ):
            raise ValueError(
                "The propagation constant is not finite."
            )

        if gamma.real <= 0.0:
            raise ValueError(
                "The physical propagation constant must have "
                "a strictly positive real part."
            )

        if gamma.imag >= 0.0:
            raise ValueError(
                "The physical propagation constant must have "
                "a strictly negative imaginary part."
            )

        return complex(gamma)

    def plane_wave_amplitude(
        self,
        z: float,
        z_surface: float,
        conductivity: float,
        mu_r: float = 1.0,
    ) -> complex:
        """
        Return the homogeneous plane-wave electric amplitude at depth z.

        The GeoInvLab3D reference convention is

            E(z)
              = E0 exp(-i * gamma * (z-z_surface))

        with

            gamma = sqrt(-i * omega * mu * sigma).

        Since

            gamma = a - i a

        for a homogeneous half-space, one has

            -i * gamma = -a - i a,

        so the real part of the exponent is negative and the field
        decays with increasing positive depth.
        """
        z = float(z)
        z_surface = float(z_surface)

        if not np.isfinite(z):
            raise ValueError(
                "z must be finite."
            )

        if not np.isfinite(z_surface):
            raise ValueError(
                "z_surface must be finite."
            )

        gamma = self.propagation_constant(
            self.omega,
            conductivity,
            mu_r,
        )

        depth = z - z_surface

        if depth < 0.0:
            raise ValueError(
                "z must be greater than or equal to z_surface "
                "for a downward-decaying MT plane wave."
            )

        return (
            self.complex_amplitude
            * np.exp(
                -1j
                * gamma
                * depth
            )
        )

    def boundary_condition(
        self,
        mesh,
        conductivity: float | None = None,
        mu_r: float = 1.0,
        use_depth_dependence: bool = False,
    ) -> BoundaryCondition:
        """
        Build the Dirichlet boundary excitation.

        X polarization:

            Ex = plane-wave excitation
            Ey = 0
            Ez = 0

        Y polarization:

            Ey = plane-wave excitation
            Ex = 0
            Ez = 0

        Only horizontal MT excitation is introduced.
        """

        topology = build_topology(mesh)

        nx, ny, nz = (
            int(mesh.shape[0]),
            int(mesh.shape[1]),
            int(mesh.shape[2]),
        )

        ex_indices = np.asarray(
            topology.edges.ex,
            dtype=np.int64,
        ).reshape(-1)

        ey_indices = np.asarray(
            topology.edges.ey,
            dtype=np.int64,
        ).reshape(-1)

        ez_indices = np.asarray(
            topology.edges.ez,
            dtype=np.int64,
        ).reshape(-1)

        n_ex = ex_indices.size
        n_ey = ey_indices.size
        n_ez = ez_indices.size

        n_edges = int(
            topology.n_edges
        )

        if n_edges != (
            n_ex
            + n_ey
            + n_ez
        ):
            raise ValueError(
                "Inconsistent electric-edge topology: "
                f"topology.n_edges={n_edges}, "
                f"n_ex+n_ey+n_ez="
                f"{n_ex + n_ey + n_ez}."
            )

        expected_n_ex = (
            nx
            * (ny + 1)
            * (nz + 1)
        )

        expected_n_ey = (
            (nx + 1)
            * ny
            * (nz + 1)
        )

        expected_n_ez = (
            (nx + 1)
            * (ny + 1)
            * nz
        )

        if n_ex != expected_n_ex:
            raise ValueError(
                "Invalid Ex topology size: "
                f"got {n_ex}, "
                f"expected {expected_n_ex}."
            )

        if n_ey != expected_n_ey:
            raise ValueError(
                "Invalid Ey topology size: "
                f"got {n_ey}, "
                f"expected {expected_n_ey}."
            )

        if n_ez != expected_n_ez:
            raise ValueError(
                "Invalid Ez topology size: "
                f"got {n_ez}, "
                f"expected {expected_n_ez}."
            )

        # --------------------------------------------------------------
        # Physical parameters.
        # --------------------------------------------------------------

        if use_depth_dependence:
            if conductivity is None:
                raise ValueError(
                    "conductivity is required when "
                    "use_depth_dependence=True."
                )

            conductivity = float(
                conductivity
            )

            if (
                not np.isfinite(conductivity)
                or conductivity <= 0.0
            ):
                raise ValueError(
                    "conductivity must be finite and "
                    "strictly positive."
                )

        mu_r = float(mu_r)

        if (
            not np.isfinite(mu_r)
            or mu_r <= 0.0
        ):
            raise ValueError(
                "mu_r must be finite and "
                "strictly positive."
            )

        # --------------------------------------------------------------
        # Vertical coordinates.
        #
        # z is positive downward, hence the surface is edges_z[0].
        # --------------------------------------------------------------

        z_edges = np.asarray(
            mesh.edges_z,
            dtype=float,
        ).reshape(-1)

        if z_edges.size != nz + 1:
            raise ValueError(
                "mesh.edges_z must contain nz + 1 coordinates."
            )

        if not np.all(
            np.isfinite(z_edges)
        ):
            raise ValueError(
                "mesh.edges_z must contain finite coordinates."
            )

        if np.any(
            np.diff(z_edges) <= 0.0
        ):
            raise ValueError(
                "mesh.edges_z must be strictly increasing "
                "for z positive downward."
            )

        z_surface = float(
            z_edges[0]
        )

        # --------------------------------------------------------------
        # Boundary value.
        # --------------------------------------------------------------

        if use_depth_dependence:

            def boundary_value(
                z: float,
            ) -> complex:
                return self.plane_wave_amplitude(
                    z=z,
                    z_surface=z_surface,
                    conductivity=conductivity,
                    mu_r=mu_r,
                )

        else:

            def boundary_value(
                z: float,
            ) -> complex:
                del z
                return self.complex_amplitude

        values_by_index: dict[int, complex] = {}

        def add_value(
            index: int,
            value: complex,
        ) -> None:
            index = int(index)
            value = complex(value)

            if (
                index < 0
                or index >= n_edges
            ):
                raise ValueError(
                    "Electric-edge boundary index is outside "
                    f"the topology: {index}."
                )

            previous = values_by_index.get(
                index
            )

            if (
                previous is not None
                and not np.isclose(
                    previous,
                    value,
                )
            ):
                raise ValueError(
                    "Inconsistent boundary value for "
                    f"electric edge {index}."
                )

            values_by_index[index] = value

        # --------------------------------------------------------------
        # Recover the logical component grids from the actual topology
        # arrays. No new global indexing is invented here.
        # --------------------------------------------------------------

        ex_grid = ex_indices.reshape(
            nx,
            ny + 1,
            nz + 1,
        )

        ey_grid = ey_indices.reshape(
            nx + 1,
            ny,
            nz + 1,
        )

        ez_grid = ez_indices.reshape(
            nx + 1,
            ny + 1,
            nz,
        )

        # --------------------------------------------------------------
        # Ex boundary edges.
        #
        # Ex is tangential on y boundaries and z boundaries.
        # --------------------------------------------------------------

        for ix in range(nx):
            for iy in range(ny + 1):
                for iz in range(nz + 1):

                    if (
                        iy in (0, ny)
                        or iz in (0, nz)
                    ):
                        index = int(
                            ex_grid[
                                ix,
                                iy,
                                iz,
                            ]
                        )

                        if (
                            self.polarization
                            is MTSourcePolarization.X
                        ):
                            value = boundary_value(
                                float(z_edges[iz])
                            )
                        else:
                            value = 0.0j

                        add_value(
                            index,
                            value,
                        )

        # --------------------------------------------------------------
        # Ey boundary edges.
        #
        # Ey is tangential on x boundaries and z boundaries.
        # --------------------------------------------------------------

        for ix in range(nx + 1):
            for iy in range(ny):
                for iz in range(nz + 1):

                    if (
                        ix in (0, nx)
                        or iz in (0, nz)
                    ):
                        index = int(
                            ey_grid[
                                ix,
                                iy,
                                iz,
                            ]
                        )

                        if (
                            self.polarization
                            is MTSourcePolarization.Y
                        ):
                            value = boundary_value(
                                float(z_edges[iz])
                            )
                        else:
                            value = 0.0j

                        add_value(
                            index,
                            value,
                        )

        # --------------------------------------------------------------
        # Ez is never part of the horizontal MT excitation.
        #
        # Keep Ez zero on the lateral boundaries.
        # --------------------------------------------------------------

        for ix in range(nx + 1):
            for iy in range(ny + 1):
                for iz in range(nz):

                    if (
                        ix in (0, nx)
                        or iy in (0, ny)
                    ):
                        index = int(
                            ez_grid[
                                ix,
                                iy,
                                iz,
                            ]
                        )

                        add_value(
                            index,
                            0.0j,
                        )

        if not values_by_index:
            raise ValueError(
                "No boundary electric edges were identified."
            )

        indices = np.asarray(
            sorted(values_by_index),
            dtype=np.int64,
        )

        values = np.asarray(
            [
                values_by_index[int(index)]
                for index in indices
            ],
            dtype=complex,
        )

        if indices.size != values.size:
            raise RuntimeError(
                "Boundary indices and values have "
                "different sizes."
            )

        if (
            not np.all(
                np.isfinite(values.real)
            )
            or not np.all(
                np.isfinite(values.imag)
            )
        ):
            raise ValueError(
                "Boundary electric-field values must be finite."
            )

        return make_dirichlet_boundary(
            indices=indices,
            values=values,
            n_edges=n_edges,
        )


def build_mt_source(
    mesh,
    source_or_omega,
    conductivity: float | None = None,
    polarization: MTSourcePolarization | str | None = None,
    amplitude: complex = 1.0 + 0.0j,
    mu_r: float = 1.0,
    use_depth_dependence: bool = False,
) -> BoundaryCondition:
    """
    Build a boundary condition for a plane-wave MT source.

    Preferred API::

        source = MTPlaneWaveSource(
            polarization=MTSourcePolarization.X,
            frequency=0.1,
        )

        boundary = build_mt_source(
            mesh,
            source,
            conductivity=0.01,
            use_depth_dependence=True,
        )

    The legacy omega/polarization API is retained for compatibility.
    """

    if isinstance(
        source_or_omega,
        MTPlaneWaveSource,
    ):
        source = source_or_omega

    else:
        if polarization is None:
            raise TypeError(
                "polarization is required when using "
                "the legacy source API."
            )

        omega = float(
            source_or_omega
        )

        if (
            not np.isfinite(omega)
            or omega <= 0.0
        ):
            raise ValueError(
                "omega must be finite and "
                "strictly positive."
            )

        frequency = (
            omega
            / (2.0 * np.pi)
        )

        source = MTPlaneWaveSource(
            polarization=MTSourcePolarization(
                polarization
            ),
            amplitude=amplitude,
            frequency=frequency,
        )

    return source.boundary_condition(
        mesh,
        conductivity=conductivity,
        mu_r=mu_r,
        use_depth_dependence=use_depth_dependence,
    )


__all__ = [
    "MU0",
    "MTSourcePolarization",
    "MTPlaneWaveSource",
    "build_mt_source",
]
 
