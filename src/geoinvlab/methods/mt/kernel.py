"""
1D magnetotelluric forward kernel.

This module provides a physically consistent 1D MT forward solver
used as a reference implementation for the MT method.

Coordinate convention
---------------------
GeoInvLab3D uses the following right-handed coordinate convention:

    x = East
    y = North
    z = positive downward

Time convention
---------------
GeoInvLab3D uses

    exp(+i * omega * t)

and therefore Faraday's law is

    curl(E) = -i * omega * mu * H.

Plane-wave convention
---------------------
For a downward-propagating and downward-decaying MT field,

    E(z) = E0 * exp(-i * k * z)

where

    k^2 = -i * omega * mu * sigma.

The physical branch is chosen as

    Re(k) > 0
    Im(k) < 0.

For a homogeneous half-space,

    k = a - i*a

with

    a = sqrt(omega * mu * sigma / 2).

Therefore

    exp(-i * k * z)
        = exp(-a * z) * exp(-i * a * z),

which decays with increasing positive depth.

Impedance convention
--------------------
GeoInvLab3D uses

    E = Z H.

For X polarization,

    E = (Ex, 0, 0)
    H = (0, Hy, 0)

and

    Zxy = Ex / Hy = omega * mu / k.

Since

    arg(k) = -45 degrees,

the homogeneous half-space has

    arg(Zxy) = +45 degrees.

For Y polarization,

    Zyx = -Zxy,

so its phase is

    arg(Zyx) = -135 degrees

when phases are wrapped to [-180, 180].

Physical units
--------------
frequency         : Hz
angular frequency : rad/s
resistivity       : ohm.m
conductivity      : S/m
permeability      : H/m

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geoinvlab.core.mesh.base import Mesh3D
from geoinvlab.core.model.model import Model3D


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MU0: float = 4.0 * np.pi * 1.0e-7


# ---------------------------------------------------------------------------
# MT kernel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MTKernel:
    """
    1D layered MT forward solver.

    Parameters
    ----------
    mesh:
        Computational 3D mesh.

    frequencies:
        Frequencies in Hz.

    mu_r:
        Relative magnetic permeability.

    Notes
    -----
    The mesh is interpreted as a stack of horizontal layers beneath
    each station column.

    The current reference implementation supports homogeneous
    isotropic media. It is intended as a physically controlled
    reference kernel for validating the full 3D implementation.
    """

    mesh: Mesh3D
    frequencies: np.ndarray
    mu_r: float = 1.0

    # Backward-compatible access through MTKernel.MU0.
    MU0: float = MU0

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, Mesh3D):
            raise TypeError(
                "mesh must be a Mesh3D instance."
            )

        frequencies = np.asarray(
            self.frequencies,
            dtype=float,
        )

        if frequencies.ndim != 1:
            raise ValueError(
                "frequencies must be a 1D array."
            )

        if frequencies.size == 0:
            raise ValueError(
                "frequencies must not be empty."
            )

        if not np.all(np.isfinite(frequencies)):
            raise ValueError(
                "frequencies must contain only finite values."
            )

        if np.any(frequencies <= 0.0):
            raise ValueError(
                "frequencies must be strictly positive."
            )

        if (
            not np.isfinite(self.mu_r)
            or self.mu_r <= 0.0
        ):
            raise ValueError(
                "mu_r must be finite and strictly positive."
            )

        object.__setattr__(
            self,
            "frequencies",
            frequencies,
        )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def n_frequencies(self) -> int:
        """Return the number of frequencies."""
        return int(self.frequencies.size)

    @property
    def angular_frequencies(self) -> np.ndarray:
        """Return angular frequencies in rad/s."""
        return (
            2.0
            * np.pi
            * self.frequencies
        )

    @property
    def permeability(self) -> float:
        """Return magnetic permeability in H/m."""
        return MU0 * float(self.mu_r)

    # ------------------------------------------------------------------
    # Model handling
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_model_name(
        name: str,
    ) -> str:
        """
        Normalize the model parameterization name.
        """
        if not isinstance(name, str):
            raise TypeError(
                "model.name must be a string."
            )

        normalized = name.strip().lower()

        aliases = {
            "resistivity": "resistivity",
            "rho": "resistivity",
            "res": "resistivity",
            "conductivity": "conductivity",
            "sigma": "conductivity",
            "cond": "conductivity",
        }

        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(
                "model.name must identify either "
                "'resistivity' or 'conductivity'."
            ) from exc

    def _validate_model(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """
        Validate a Model3D and return conductivity values.
        """
        if not isinstance(model, Model3D):
            raise TypeError(
                "model must be a Model3D instance."
            )

        if model.mesh != self.mesh:
            raise ValueError(
                "model mesh does not match kernel mesh."
            )

        parameterization = self._normalize_model_name(
            model.name
        )

        values = np.asarray(
            model.values,
            dtype=float,
        ).reshape(-1)

        if values.size != self.mesh.n_cells:
            raise ValueError(
                "model must contain exactly "
                f"{self.mesh.n_cells} cell values."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "model values must be finite."
            )

        if np.any(values <= 0.0):
            raise ValueError(
                "model values must be strictly positive."
            )

        if parameterization == "resistivity":
            conductivity = 1.0 / values
        else:
            conductivity = values

        if not np.all(np.isfinite(conductivity)):
            raise ValueError(
                "computed conductivity values must be finite."
            )

        if np.any(conductivity <= 0.0):
            raise ValueError(
                "computed conductivity values must be strictly positive."
            )

        return conductivity

    # ------------------------------------------------------------------
    # Wavenumber
    # ------------------------------------------------------------------

    @staticmethod
    def physical_wavenumber(
        frequency: float,
        conductivity: float,
        mu_r: float = 1.0,
    ) -> complex:
        """
        Compute the physical MT wavenumber.

        Parameters
        ----------
        frequency:
            Frequency in Hz.

        conductivity:
            Electrical conductivity in S/m.

        mu_r:
            Relative magnetic permeability.

        Returns
        -------
        complex
            Physical wavenumber satisfying

                Re(k) > 0
                Im(k) < 0

            for the exp(+i omega t) convention.
        """
        frequency = float(frequency)
        conductivity = float(conductivity)
        mu_r = float(mu_r)

        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError(
                "frequency must be finite and strictly positive."
            )

        if not np.isfinite(conductivity) or conductivity <= 0.0:
            raise ValueError(
                "conductivity must be finite and strictly positive."
            )

        if not np.isfinite(mu_r) or mu_r <= 0.0:
            raise ValueError(
                "mu_r must be finite and strictly positive."
            )

        omega = (
            2.0
            * np.pi
            * frequency
        )

        mu = MU0 * mu_r

        k = np.sqrt(
            -1j
            * omega
            * mu
            * conductivity
        )

        if (
            k.real < 0.0
            or (
                np.isclose(k.real, 0.0)
                and k.imag > 0.0
            )
        ):
            k = -k

        if not (
            k.real > 0.0
            and k.imag < 0.0
        ):
            raise RuntimeError(
                "Unable to select the physical MT wavenumber branch."
            )

        return complex(k)

    # ------------------------------------------------------------------
    # Homogeneous impedance
    # ------------------------------------------------------------------

    def homogeneous_zxy(
        self,
        frequency: float,
        conductivity: float,
    ) -> complex:
        """
        Compute the homogeneous half-space Zxy impedance.

        With the GeoInvLab3D convention,

            Zxy = omega * mu / k.

        For exp(+i * omega * t), the physical branch gives

            phase(Zxy) = +45 degrees.
        """
        frequency = float(frequency)
        conductivity = float(conductivity)

        omega = (
            2.0
            * np.pi
            * frequency
        )

        k = self.physical_wavenumber(
            frequency=frequency,
            conductivity=conductivity,
            mu_r=self.mu_r,
        )

        return (
            omega
            * self.permeability
            / k
        )

    # ------------------------------------------------------------------
    # Homogeneous half-space reference response
    # ------------------------------------------------------------------

    def homogeneous_half_space(
        self,
        resistivity: float,
    ) -> np.ndarray:
        """
        Compute the analytical homogeneous half-space Zxy response.

        Parameters
        ----------
        resistivity:
            Homogeneous electrical resistivity in ohm.m.

        Returns
        -------
        numpy.ndarray
            Complex Zxy impedance for each frequency, with shape

                (n_frequencies,)

            The returned quantity is specifically Zxy, not the
            complete 2x2 impedance tensor.

        Notes
        -----
        GeoInvLab3D uses

            exp(+i * omega * t)

        together with

            k^2 = -i * omega * mu * sigma

        and

            E = Z H.

        Therefore,

            Zxy = omega * mu / k.

        For a homogeneous half-space,

            phase(Zxy) = +45 degrees

        and

            rho_a = |Zxy|^2 / (mu * omega)

        is exactly equal to the input resistivity.

        The complete tensor

            [[0,   Zxy],
             [-Zxy, 0]]

        is constructed where needed by the 3D forward solver.
        """
        resistivity = float(resistivity)

        if (
            not np.isfinite(resistivity)
            or resistivity <= 0.0
        ):
            raise ValueError(
                "resistivity must be finite and strictly positive."
            )

        conductivity = 1.0 / resistivity

        if (
            not np.isfinite(conductivity)
            or conductivity <= 0.0
        ):
            raise ValueError(
                "computed conductivity must be finite "
                "and strictly positive."
            )

        impedance = np.empty(
            self.n_frequencies,
            dtype=complex,
        )

        for frequency_index, frequency in enumerate(
            self.frequencies
        ):
            impedance[
                frequency_index
            ] = self.homogeneous_zxy(
                frequency=float(frequency),
                conductivity=conductivity,
            )

        return impedance

    # ------------------------------------------------------------------
    # Public forward solve
    # ------------------------------------------------------------------

    def solve(
        self,
        model: Model3D,
        locations: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the MT impedance tensor.

        Parameters
        ----------
        model:
            Resistivity or conductivity model defined on ``self.mesh``.

            ``model.name`` must identify either
            ``"resistivity"`` or ``"conductivity"``.

        locations:
            Station coordinates with shape ``(n_stations, 3)``.

            Coordinates follow the GeoInvLab3D convention:

                x = East
                y = North
                z = positive downward

        Returns
        -------
        numpy.ndarray
            Complex impedance tensor with shape

                (n_stations, n_frequencies, 2, 2)

            and component ordering

                [[Zxx, Zxy],
                 [Zyx, Zyy]]

        Notes
        -----
        For a homogeneous isotropic Earth,

            Zxx = 0
            Zyy = 0
            Zyx = -Zxy.

        The current reference implementation deliberately rejects
        heterogeneous models instead of silently treating them as
        homogeneous.
        """
        conductivity = self._validate_model(
            model
        )

        locations = np.asarray(
            locations,
            dtype=float,
        )

        if (
            locations.ndim != 2
            or locations.shape[1] != 3
        ):
            raise ValueError(
                "locations must have shape (n_stations, 3)."
            )

        if not np.all(np.isfinite(locations)):
            raise ValueError(
                "locations must contain only finite values."
            )

        if conductivity.size == 0:
            raise ValueError(
                "model must contain at least one cell."
            )

        if not np.all(
            conductivity == conductivity[0]
        ):
            raise NotImplementedError(
                "MTKernel currently supports homogeneous "
                "isotropic media only."
            )

        sigma = float(
            conductivity[0]
        )

        n_stations = locations.shape[0]

        impedance = np.zeros(
            (
                n_stations,
                self.n_frequencies,
                2,
                2,
            ),
            dtype=complex,
        )

        for frequency_index, frequency in enumerate(
            self.frequencies
        ):
            zxy = self.homogeneous_zxy(
                frequency=frequency,
                conductivity=sigma,
            )

            impedance[
                :,
                frequency_index,
                0,
                1,
            ] = zxy

            impedance[
                :,
                frequency_index,
                1,
                0,
            ] = -zxy

        return impedance

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    def forward(
        self,
        model: Model3D,
        locations: np.ndarray,
    ) -> np.ndarray:
        """
        Alias for :meth:`solve`.
        """
        return self.solve(
            model=model,
            locations=locations,
        )

    def compute_impedance(
        self,
        model: Model3D,
        locations: np.ndarray,
    ) -> np.ndarray:
        """
        Backward-compatible alias for :meth:`solve`.
        """
        return self.solve(
            model=model,
            locations=locations,
        )
 

