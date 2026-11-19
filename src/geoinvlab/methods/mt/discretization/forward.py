"""
3D magnetotelluric forward solver.

Time convention
---------------
GeoInvLab3D uses

    exp(+i * omega * t)

and therefore

    curl(E) = -i * omega * mu * H

so that

    H = i * curl(E) / (omega * mu).

The 1D physical propagation constant is

    gamma = sqrt(-i * omega * mu * sigma)

with the physical branch

    Re(gamma) > 0
    Im(gamma) < 0

and

    E(z) = E0 * exp(-gamma * z).

The impedance convention is

    E = Z H

with tensor ordering

    [[Zxx, Zxy],
     [Zyx, Zyy]].

Source polarization ordering:

    0 -> X
    1 -> Y
    
     MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.core.model.model import Model3D

from .field import (
    EdgeElectricField,
    FaceMagneticField,
    build_edge_electric_field,
    build_face_magnetic_field,
    physical_magnetic_field,
)

from .impedance import MTImpedance

from .maxwell import (
    MaxwellSystem,
    build_maxwell_system,
)

from .observation import (
    MTObservedFields,
    observe_fields,
)

from .solver_3d import (
    LinearSolveResult,
    solve_maxwell,
)

from .source import (
    MTPlaneWaveSource,
    MTSourcePolarization,
    build_mt_source,
)

from .station import (
    MTStationSet,
    stations_to_observations,
)


MU0 = 4.0e-7 * np.pi


# ============================================================================
# Result
# ============================================================================


@dataclass(frozen=True)
class MTForwardResult:
    """
    Result of a 3D MT forward calculation.
    """

    frequencies: np.ndarray
    stations: MTStationSet
    impedance: np.ndarray

    electric_fields: np.ndarray | None = None
    magnetic_fields: np.ndarray | None = None

    def __post_init__(self) -> None:
        frequencies = np.asarray(
            self.frequencies,
            dtype=float,
        )

        impedance = np.asarray(
            self.impedance,
            dtype=complex,
        )

        if frequencies.ndim != 1:
            raise ValueError(
                "frequencies must be a 1D array."
            )

        if frequencies.size == 0:
            raise ValueError(
                "frequencies must not be empty."
            )

        if not np.all(
            np.isfinite(frequencies)
        ):
            raise ValueError(
                "frequencies must contain finite values."
            )

        if np.any(
            frequencies <= 0.0
        ):
            raise ValueError(
                "frequencies must be strictly positive."
            )

        if not isinstance(
            self.stations,
            MTStationSet,
        ):
            raise TypeError(
                "stations must be an MTStationSet."
            )

        expected_shape = (
            len(self.stations),
            frequencies.size,
            2,
            2,
        )

        if impedance.shape != expected_shape:
            raise ValueError(
                "impedance must have shape "
                f"{expected_shape}, got {impedance.shape}."
            )

        if not np.all(
            np.isfinite(impedance)
        ):
            raise ValueError(
                "impedance must contain finite values."
            )

        if self.electric_fields is not None:
            self._validate_electric_fields(
                self.electric_fields,
                frequencies.size,
            )

        if self.magnetic_fields is not None:
            self._validate_magnetic_fields(
                self.magnetic_fields,
                frequencies.size,
            )

        object.__setattr__(
            self,
            "frequencies",
            frequencies,
        )

        object.__setattr__(
            self,
            "impedance",
            impedance,
        )

    @staticmethod
    def _validate_electric_fields(
        fields: np.ndarray,
        n_frequencies: int,
    ) -> None:
        if not isinstance(
            fields,
            np.ndarray,
        ):
            raise TypeError(
                "electric_fields must be a NumPy array."
            )

        if fields.shape != (
            n_frequencies,
            2,
        ):
            raise ValueError(
                "electric_fields must have shape "
                f"({n_frequencies}, 2)."
            )

        for frequency_index in range(
            n_frequencies
        ):
            for polarization_index in range(2):
                field = fields[
                    frequency_index,
                    polarization_index,
                ]

                if not isinstance(
                    field,
                    EdgeElectricField,
                ):
                    raise TypeError(
                        "electric_fields contains an invalid "
                        "EdgeElectricField."
                    )

    @staticmethod
    def _validate_magnetic_fields(
        fields: np.ndarray,
        n_frequencies: int,
    ) -> None:
        if not isinstance(
            fields,
            np.ndarray,
        ):
            raise TypeError(
                "magnetic_fields must be a NumPy array."
            )

        if fields.shape != (
            n_frequencies,
            2,
        ):
            raise ValueError(
                "magnetic_fields must have shape "
                f"({n_frequencies}, 2)."
            )

        for frequency_index in range(
            n_frequencies
        ):
            for polarization_index in range(2):
                field = fields[
                    frequency_index,
                    polarization_index,
                ]

                if not isinstance(
                    field,
                    FaceMagneticField,
                ):
                    raise TypeError(
                        "magnetic_fields contains an invalid "
                        "FaceMagneticField."
                    )

                values = np.asarray(
                    field.values,
                    dtype=complex,
                )

                if values.ndim != 1:
                    raise ValueError(
                        "magnetic field values must be one-dimensional."
                    )

                if values.size == 0:
                    raise ValueError(
                        "magnetic field values must not be empty."
                    )

                if not np.all(
                    np.isfinite(values)
                ):
                    raise ValueError(
                        "magnetic field values must be finite."
                    )

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_frequencies(self) -> int:
        return self.frequencies.size

    @property
    def zxx(self) -> np.ndarray:
        return self.impedance[:, :, 0, 0]

    @property
    def zxy(self) -> np.ndarray:
        return self.impedance[:, :, 0, 1]

    @property
    def zyx(self) -> np.ndarray:
        return self.impedance[:, :, 1, 0]

    @property
    def zyy(self) -> np.ndarray:
        return self.impedance[:, :, 1, 1]

    @property
    def apparent_resistivity(self) -> np.ndarray:
        """
        Apparent resistivity:

            rho_a = |Z|^2 / (mu0 * omega)
        """
        omega = (
            2.0
            * np.pi
            * self.frequencies
        )

        return (
            np.abs(self.impedance) ** 2
            / (
                MU0
                * omega[None, :, None, None]
            )
        )

    @property
    def resistivity(self) -> np.ndarray:
        return self.apparent_resistivity

    @property
    def rho(self) -> np.ndarray:
        return self.apparent_resistivity

    @property
    def phase(self) -> np.ndarray:
        """Impedance phase in degrees."""
        return np.degrees(
            np.angle(
                self.impedance
            )
        )

    @property
    def phase_degrees(self) -> np.ndarray:
        return self.phase

    def tensor(
        self,
        station_index: int,
        frequency_index: int,
    ) -> MTImpedance:
        tensor = self.impedance[
            station_index,
            frequency_index,
        ]

        return MTImpedance(
            zxx=tensor[0, 0],
            zxy=tensor[0, 1],
            zyx=tensor[1, 0],
            zyy=tensor[1, 1],
        )


# ============================================================================
# Forward solver
# ============================================================================


class MTForwardSolver:
    """
    3D magnetotelluric forward solver.
    """

    def __init__(
        self,
        mesh: StructuredMesh3D,
        frequencies: Sequence[float],
        parameterization: str = "conductivity",
        mu_r: float | np.ndarray = 1.0,
        solver_method: str = "direct",
        solver_tolerance: float = 1.0e-8,
        solver_maxiter: int | None = None,
    ) -> None:

        if not isinstance(
            mesh,
            StructuredMesh3D,
        ):
            raise TypeError(
                "mesh must be a StructuredMesh3D."
            )

        mesh.validate()

        frequencies_array = np.asarray(
            frequencies,
            dtype=float,
        )

        if frequencies_array.ndim != 1:
            raise ValueError(
                "frequencies must be a 1D array."
            )

        if frequencies_array.size == 0:
            raise ValueError(
                "frequencies must not be empty."
            )

        if not np.all(
            np.isfinite(frequencies_array)
        ):
            raise ValueError(
                "frequencies must contain finite values."
            )

        if np.any(
            frequencies_array <= 0.0
        ):
            raise ValueError(
                "frequencies must be strictly positive."
            )

        self.parameterization = (
            self._normalize_parameterization(
                parameterization
            )
        )

        if solver_method not in {
            "direct",
            "gmres",
        }:
            raise ValueError(
                "solver_method must be 'direct' or 'gmres'."
            )

        solver_tolerance = float(
            solver_tolerance
        )

        if (
            not np.isfinite(
                solver_tolerance
            )
            or solver_tolerance <= 0.0
        ):
            raise ValueError(
                "solver_tolerance must be finite "
                "and strictly positive."
            )

        if solver_maxiter is not None:
            if (
                not isinstance(
                    solver_maxiter,
                    (int, np.integer),
                )
                or solver_maxiter <= 0
            ):
                raise ValueError(
                    "solver_maxiter must be a positive integer "
                    "or None."
                )

        self.mesh = mesh
        self.frequencies = frequencies_array
        self.mu_r = mu_r
        self.solver_method = solver_method
        self.solver_tolerance = solver_tolerance
        self.solver_maxiter = solver_maxiter

    # ------------------------------------------------------------------
    # Parameterization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_parameterization(
        parameterization: str,
    ) -> str:

        if not isinstance(
            parameterization,
            str,
        ):
            raise TypeError(
                "parameterization must be a string."
            )

        value = parameterization.strip().lower()

        aliases = {
            "rho": "resistivity",
            "res": "resistivity",
            "resistivity": "resistivity",
            "sigma": "conductivity",
            "cond": "conductivity",
            "conductivity": "conductivity",
        }

        if value not in aliases:
            raise ValueError(
                "parameterization must be one of "
                "'resistivity', 'rho', "
                "'conductivity', or 'sigma'."
            )

        return aliases[value]

    @property
    def n_frequencies(self) -> int:
        return self.frequencies.size

    # ------------------------------------------------------------------
    # Permeability
    # ------------------------------------------------------------------

    def _source_mu_r(self) -> float:
        """
        Return a scalar relative permeability for the plane-wave source.
        """
        values = np.asarray(
            self.mu_r,
            dtype=float,
        )

        if values.ndim == 0:
            value = float(values)

        elif (
            values.ndim == 1
            and values.size == self.mesh.n_cells
        ):
            value = float(
                np.mean(values)
            )

        else:
            raise ValueError(
                "mu_r must be a positive scalar or one value per cell."
            )

        if (
            not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                "mu_r must be finite and strictly positive."
            )

        return value

    # ------------------------------------------------------------------
    # Model -> conductivity
    # ------------------------------------------------------------------

    def conductivity_from_model(
        self,
        model: Model3D | np.ndarray,
        parameterization: str | None = None,
    ) -> np.ndarray:
        """
        Convert a Model3D or array into cell conductivity.

        Model3D stores the actual parameter vector in ``model.values``.
        """

        if isinstance(
            model,
            Model3D,
        ):
            if model.mesh != self.mesh:
                raise ValueError(
                    "model mesh does not match solver mesh."
                )

            values = np.asarray(
                model.values,
                dtype=float,
            ).reshape(-1)

            if parameterization is None:
                parameterization = (
                    self._normalize_parameterization(
                        model.name
                    )
                )
            else:
                parameterization = (
                    self._normalize_parameterization(
                        parameterization
                    )
                )

        else:
            values = np.asarray(
                model,
                dtype=float,
            ).reshape(-1)

            if parameterization is None:
                parameterization = (
                    self.parameterization
                )
            else:
                parameterization = (
                    self._normalize_parameterization(
                        parameterization
                    )
                )

        if values.size != self.mesh.n_cells:
            raise ValueError(
                "model size must match mesh.n_cells: "
                f"received {values.size}, "
                f"expected {self.mesh.n_cells}."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "model values must be finite."
            )

        if np.any(
            values <= 0.0
        ):
            raise ValueError(
                "model values must be strictly positive."
            )

        if parameterization == "resistivity":
            conductivity = 1.0 / values
        else:
            conductivity = values.copy()

        if not np.all(
            np.isfinite(conductivity)
        ):
            raise ValueError(
                "conductivity contains non-finite values."
            )

        if np.any(
            conductivity <= 0.0
        ):
            raise ValueError(
                "conductivity must be strictly positive."
            )

        return conductivity

    # ------------------------------------------------------------------
    # Maxwell system
    # ------------------------------------------------------------------

    def build_system(
        self,
        model: Model3D | np.ndarray,
        parameterization: str | None = None,
    ) -> MaxwellSystem:
        """
        Build the geometry-aware Maxwell system.
        """

        conductivity = self.conductivity_from_model(
            model=model,
            parameterization=parameterization,
        )

        return build_maxwell_system(
            mesh=self.mesh,
            conductivity=conductivity,
            mu_r=self.mu_r,
        )

    # ------------------------------------------------------------------
    # Solve one source
    # ------------------------------------------------------------------

    def solve_source(
        self,
        system: MaxwellSystem,
        frequency: float,
        polarization: MTSourcePolarization,
        conductivity: float | None = None,
    ) -> tuple[
        EdgeElectricField,
        FaceMagneticField,
        LinearSolveResult,
    ]:
        """
        Solve one frequency and one source polarization.

        IMPORTANT
        ---------
        ``build_mt_source()`` returns a BoundaryCondition object.

        That object must be passed to ``solve_maxwell()``. It is NOT a
        numerical RHS vector and must never be passed to ``np.asarray()``.
        """

        frequency = float(
            frequency
        )

        if (
            not np.isfinite(frequency)
            or frequency <= 0.0
        ):
            raise ValueError(
                "frequency must be finite and strictly positive."
            )

        if not isinstance(
            polarization,
            MTSourcePolarization,
        ):
            try:
                polarization = MTSourcePolarization(
                    polarization
                )
            except ValueError as exc:
                raise ValueError(
                    "polarization must be 'x' or 'y'."
                ) from exc

        omega = (
            2.0
            * np.pi
            * frequency
        )

        source = MTPlaneWaveSource(
            polarization=polarization,
            frequency=frequency,
        )

        # --------------------------------------------------------------
        # Conductivity used by the incident plane-wave boundary.
        # --------------------------------------------------------------
        if conductivity is None:
            conductivity = 1.0
        else:
            conductivity = float(
                conductivity
            )

        if (
            not np.isfinite(conductivity)
            or conductivity <= 0.0
        ):
            raise ValueError(
                "conductivity must be finite and strictly positive."
            )

        mu_r_source = self._source_mu_r()

        # --------------------------------------------------------------
        # Build the MT boundary condition.
        #
        # This returns BoundaryCondition.
        # --------------------------------------------------------------
        boundary = build_mt_source(
            mesh=self.mesh,
            source_or_omega=source,
            conductivity=conductivity,
            mu_r=mu_r_source,
            use_depth_dependence=True,
        )

        # --------------------------------------------------------------
        # Maxwell matrix.
        # --------------------------------------------------------------
        matrix = system.system_matrix(
            omega
        )

        rhs = np.zeros(
            system.n_edges,
            dtype=complex,
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # BoundaryCondition is handled by solve_maxwell().
        # Do NOT try:
        #
        #     np.asarray(boundary)
        #
        # and do NOT manually interpret it as a RHS vector.
        # --------------------------------------------------------------
        solve_result = solve_maxwell(
            matrix=matrix,
            rhs=rhs,
            boundary=boundary,
            method=self.solver_method,
            tolerance=self.solver_tolerance,
            maxiter=self.solver_maxiter,
        )

        electric_values = np.asarray(
            solve_result.field,
            dtype=complex,
        ).reshape(-1)

        if not np.all(
            np.isfinite(electric_values)
        ):
            raise RuntimeError(
                "The Maxwell solve returned non-finite electric-field "
                "values."
            )

        # --------------------------------------------------------------
        # Use the topology created by MaxwellSystem as the authoritative
        # edge layout.
        # --------------------------------------------------------------
        # MaxwellSystem exposes the authoritative component sizes.
        # Do not infer them from ``system.curl.topology``: depending on
        # the topology implementation, that attribute may itself be a
        # CurlOperator wrapper rather than a component container.
        n_ex = int(system.n_ex)
        n_ey = int(system.n_ey)
        n_ez = int(system.n_ez)

        expected_edges = (
            n_ex
            + n_ey
            + n_ez
        )

        if electric_values.size != expected_edges:
            raise ValueError(
                "The Maxwell electric solution has an unexpected size: "
                f"{electric_values.size}; expected {expected_edges}."
            )

        electric_field = build_edge_electric_field(
            self.mesh,
            electric_values,
        )

        # --------------------------------------------------------------
        # Physical H field.
        #
        # The field module applies the physical metric-aware curl and
        #
        #     H = i curl(E) / (omega * mu)
        #
        # consistently with exp(+i omega t).
        # --------------------------------------------------------------
        magnetic_values = np.asarray(
            physical_magnetic_field(
                mesh=self.mesh,
                electric_field=electric_field,
                omega=omega,
                mu_r=self.mu_r,
            ),
            dtype=complex,
        ).reshape(-1)

        n_fx = int(system.n_fx)
        n_fy = int(system.n_fy)
        n_fz = int(system.n_fz)

        expected_faces = (
            n_fx
            + n_fy
            + n_fz
        )

        if magnetic_values.size != expected_faces:
            raise ValueError(
                "The physical magnetic field has an unexpected size: "
                f"{magnetic_values.size}; expected {expected_faces}."
            )

        if not np.all(
            np.isfinite(magnetic_values)
        ):
            raise RuntimeError(
                "The physical magnetic field contains non-finite values."
            )

        # IMPORTANT: station Hx/Hy indices are component-local.
        # observe_fields() therefore needs the FaceMagneticField wrapper
        # carrying n_fx/n_fy/n_fz so it can map local indices to the
        # global [Hx | Hy | Hz] vector correctly. Passing the raw ndarray
        # here would make observe_fields() interpret local indices as
        # global indices and corrupt the impedance tensor.
        magnetic_field = build_face_magnetic_field(
            mesh=self.mesh,
            values=magnetic_values,
        )

        return (
            electric_field,
            magnetic_field,
            solve_result,
        )

    # ------------------------------------------------------------------
    # Strict impedance
    # ------------------------------------------------------------------

    @staticmethod
    def _strict_impedance_from_observations(
        observations_x: Sequence[MTObservedFields],
        observations_y: Sequence[MTObservedFields],
        condition_tolerance: float = 1.0e-14,
    ) -> np.ndarray:
        """
        Recover Z from E = Z H.
        """

        n_stations = len(
            observations_x
        )

        if len(
            observations_y
        ) != n_stations:
            raise ValueError(
                "observations_x and observations_y must have "
                "the same length."
            )

        result = np.empty(
            (
                n_stations,
                2,
                2,
            ),
            dtype=complex,
        )

        for index, (
            obs_x,
            obs_y,
        ) in enumerate(
            zip(
                observations_x,
                observations_y,
            )
        ):
            if not isinstance(
                obs_x,
                MTObservedFields,
            ):
                raise TypeError(
                    "observations_x must contain MTObservedFields."
                )

            if not isinstance(
                obs_y,
                MTObservedFields,
            ):
                raise TypeError(
                    "observations_y must contain MTObservedFields."
                )

            E = np.array(
                [
                    [
                        obs_x.ex,
                        obs_y.ex,
                    ],
                    [
                        obs_x.ey,
                        obs_y.ey,
                    ],
                ],
                dtype=complex,
            )

            H = np.array(
                [
                    [
                        obs_x.hx,
                        obs_y.hx,
                    ],
                    [
                        obs_x.hy,
                        obs_y.hy,
                    ],
                ],
                dtype=complex,
            )

            if not np.all(
                np.isfinite(E)
            ):
                raise np.linalg.LinAlgError(
                    "Electric observations contain non-finite values "
                    f"at station {index}."
                )

            if not np.all(
                np.isfinite(H)
            ):
                raise np.linalg.LinAlgError(
                    "Magnetic observations contain non-finite values "
                    f"at station {index}."
                )

            scale = float(
                np.max(
                    np.abs(H)
                )
            )

            if (
                not np.isfinite(scale)
                or scale <= 0.0
            ):
                raise np.linalg.LinAlgError(
                    "Magnetic field is numerically zero at "
                    f"station {index}."
                )

            H_scaled = H / scale

            determinant = np.linalg.det(
                H_scaled
            )

            if (
                not np.isfinite(
                    determinant
                )
                or abs(determinant)
                <= condition_tolerance
            ):
                raise np.linalg.LinAlgError(
                    "Magnetic observation matrix is singular or "
                    "rank deficient at station "
                    f"{index}."
                )

            # E = Z H
            #
            # H.T Z.T = E.T
            #
            # Therefore:
            #
            # Z = solve(H.T, E.T).T
            Z = np.linalg.solve(
                H.T,
                E.T,
            ).T

            result[index] = Z

        return result

    # ------------------------------------------------------------------
    # Pseudoinverse fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _pseudoinverse_impedance_from_observations(
        observations_x: Sequence[MTObservedFields],
        observations_y: Sequence[MTObservedFields],
        rcond: float = 1.0e-12,
    ) -> np.ndarray:
        """
        Recover impedance using a scaled Moore-Penrose pseudoinverse.
        """

        rcond = float(
            rcond
        )

        if (
            not np.isfinite(rcond)
            or rcond <= 0.0
        ):
            raise ValueError(
                "rcond must be finite and strictly positive."
            )

        n_stations = len(
            observations_x
        )

        if len(
            observations_y
        ) != n_stations:
            raise ValueError(
                "observations_x and observations_y must have "
                "the same length."
            )

        result = np.empty(
            (
                n_stations,
                2,
                2,
            ),
            dtype=complex,
        )

        for index, (
            obs_x,
            obs_y,
        ) in enumerate(
            zip(
                observations_x,
                observations_y,
            )
        ):
            E = np.array(
                [
                    [
                        obs_x.ex,
                        obs_y.ex,
                    ],
                    [
                        obs_x.ey,
                        obs_y.ey,
                    ],
                ],
                dtype=complex,
            )

            H = np.array(
                [
                    [
                        obs_x.hx,
                        obs_y.hx,
                    ],
                    [
                        obs_x.hy,
                        obs_y.hy,
                    ],
                ],
                dtype=complex,
            )

            scale = float(
                np.max(
                    np.abs(H)
                )
            )

            if (
                not np.isfinite(scale)
                or scale <= 0.0
            ):
                raise np.linalg.LinAlgError(
                    "Magnetic field is numerically zero at "
                    f"station {index}."
                )

            H_scaled = H / scale

            H_pinv = np.linalg.pinv(
                H_scaled,
                rcond=rcond,
            )

            Z = (
                E
                @ H_pinv
                / scale
            )

            result[index] = Z

        if not np.all(
            np.isfinite(result)
        ):
            raise np.linalg.LinAlgError(
                "Pseudoinverse impedance recovery produced "
                "non-finite values."
            )

        return result

    # ------------------------------------------------------------------
    # Public impedance helper
    # ------------------------------------------------------------------

    @staticmethod
    def impedance_from_observations(
        observations_x: Sequence[MTObservedFields],
        observations_y: Sequence[MTObservedFields],
        condition_tolerance: float = 1.0e-14,
    ) -> np.ndarray:
        """
        Strict recovery of the impedance tensor.
        """

        condition_tolerance = float(
            condition_tolerance
        )

        if (
            not np.isfinite(
                condition_tolerance
            )
            or condition_tolerance <= 0.0
        ):
            raise ValueError(
                "condition_tolerance must be finite and strictly positive."
            )

        result = (
            MTForwardSolver
            ._strict_impedance_from_observations(
                observations_x=observations_x,
                observations_y=observations_y,
                condition_tolerance=condition_tolerance,
            )
        )

        # Public compatibility API: preserve an explicit station axis
        # when exactly one station is supplied.
        if result.ndim == 2:
            result = result[np.newaxis, ...]

        return result

    # ------------------------------------------------------------------
    # Main forward calculation
    # ------------------------------------------------------------------

    def forward(
        self,
        model: Model3D,
        stations: MTStationSet,
        store_fields: bool = False,
        condition_tolerance: float = 1.0e-14,
    ) -> MTForwardResult:
        """
        Run the complete 3D MT forward calculation.
        """

        if not isinstance(
            model,
            Model3D,
        ):
            raise TypeError(
                "model must be a Model3D."
            )

        if model.mesh != self.mesh:
            raise ValueError(
                "model mesh does not match solver mesh."
            )

        if not isinstance(
            stations,
            MTStationSet,
        ):
            if isinstance(stations, (tuple, list)) and len(stations) == 0:
                raise ValueError(
                    "stations must not be empty."
                )
            raise TypeError(
                "stations must be an MTStationSet."
            )

        if len(stations) == 0:
            raise ValueError(
                "stations must not be empty."
            )

        condition_tolerance = float(
            condition_tolerance
        )

        if (
            not np.isfinite(
                condition_tolerance
            )
            or condition_tolerance <= 0.0
        ):
            raise ValueError(
                "condition_tolerance must be finite and strictly positive."
            )

        # --------------------------------------------------------------
        # Model -> conductivity.
        # --------------------------------------------------------------
        conductivity = (
            self.conductivity_from_model(
                model=model,
                parameterization=self.parameterization,
            )
        )

        # --------------------------------------------------------------
        # Incident source conductivity.
        #
        # The source is a plane-wave excitation. For a heterogeneous
        # model, use the geometric mean as the incident background.
        # For a homogeneous model this is exactly the model conductivity.
        # --------------------------------------------------------------
        source_conductivity = float(
            np.exp(
                np.mean(
                    np.log(
                        conductivity
                    )
                )
            )
        )

        # --------------------------------------------------------------
        # Maxwell system.
        #
        # conductivity is already in S/m, so explicitly tell
        # build_system that the supplied array is conductivity.
        # --------------------------------------------------------------
        system = self.build_system(
            conductivity,
            parameterization="conductivity",
        )

        # --------------------------------------------------------------
        # Stations -> local field observations.
        # --------------------------------------------------------------
        station_observations = (
            stations_to_observations(
                mesh=self.mesh,
                stations=stations,
            )
        )

        n_stations = len(
            stations
        )

        n_frequencies = (
            self.n_frequencies
        )

        impedance = np.empty(
            (
                n_stations,
                n_frequencies,
                2,
                2,
            ),
            dtype=complex,
        )

        if store_fields:
            electric_fields_result = np.empty(
                (
                    n_frequencies,
                    2,
                ),
                dtype=object,
            )

            magnetic_fields_result = np.empty(
                (
                    n_frequencies,
                    2,
                ),
                dtype=object,
            )

        else:
            electric_fields_result = None
            magnetic_fields_result = None

        # --------------------------------------------------------------
        # Frequency loop.
        # --------------------------------------------------------------
        for frequency_index, frequency in enumerate(
            self.frequencies
        ):
            # ----------------------------------------------------------
            # X polarization.
            # ----------------------------------------------------------
            (
                electric_x,
                magnetic_x,
                _solve_x,
            ) = self.solve_source(
                system=system,
                frequency=frequency,
                polarization=MTSourcePolarization.X,
                conductivity=source_conductivity,
            )

            # ----------------------------------------------------------
            # Y polarization.
            # ----------------------------------------------------------
            (
                electric_y,
                magnetic_y,
                _solve_y,
            ) = self.solve_source(
                system=system,
                frequency=frequency,
                polarization=MTSourcePolarization.Y,
                conductivity=source_conductivity,
            )

            # ----------------------------------------------------------
            # Store complete fields.
            # ----------------------------------------------------------
            if store_fields:
                electric_fields_result[
                    frequency_index,
                    0,
                ] = electric_x

                electric_fields_result[
                    frequency_index,
                    1,
                ] = electric_y

                topology = system.curl.topology

                magnetic_fields_result[
                    frequency_index,
                    0,
                ] = magnetic_x

                magnetic_fields_result[
                    frequency_index,
                    1,
                ] = magnetic_y

            # ----------------------------------------------------------
            # Observe fields at all stations.
            #
            # stations_to_observations() returns objects containing the
            # actual MTObservation in .observation.
            # ----------------------------------------------------------
            observations_x = []
            observations_y = []

            for station_observation in (
                station_observations
            ):
                observation = (
                    station_observation.observation
                )

                observations_x.append(
                    observe_fields(
                        electric_field=electric_x,
                        magnetic_field_values=magnetic_x,
                        observation=observation,
                    )
                )

                observations_y.append(
                    observe_fields(
                        electric_field=electric_y,
                        magnetic_field_values=magnetic_y,
                        observation=observation,
                    )
                )

            # ----------------------------------------------------------
            # Recover Z.
            # ----------------------------------------------------------
            try:
                frequency_impedance = (
                    self.impedance_from_observations(
                        observations_x=observations_x,
                        observations_y=observations_y,
                        condition_tolerance=condition_tolerance,
                    )
                )

            except np.linalg.LinAlgError:
                # Temporary numerical fallback for rank-deficient
                # observation matrices.
                frequency_impedance = (
                    self._pseudoinverse_impedance_from_observations(
                        observations_x=observations_x,
                        observations_y=observations_y,
                        rcond=1.0e-12,
                    )
                )

            # The public helper may preserve a singleton station axis.
            # The forward result already owns the station axis.
            if frequency_impedance.ndim == 4:
                frequency_impedance = frequency_impedance[0]

            impedance[
                :,
                frequency_index,
                :,
                :,
            ] = frequency_impedance

        if not np.all(
            np.isfinite(impedance)
        ):
            raise RuntimeError(
                "The 3D MT forward calculation produced non-finite "
                "impedance values."
            )

        return MTForwardResult(
            frequencies=self.frequencies.copy(),
            stations=stations,
            impedance=impedance,
            electric_fields=electric_fields_result,
            magnetic_fields=magnetic_fields_result,
        )


# ============================================================================
# Convenience function
# ============================================================================


def forward_mt(
    mesh: StructuredMesh3D,
    model: Model3D,
    frequencies: Sequence[float],
    stations: MTStationSet,
    parameterization: str | None = None,
    mu_r: float | np.ndarray = 1.0,
    solver_method: str = "direct",
    solver_tolerance: float = 1.0e-8,
    solver_maxiter: int | None = None,
    store_fields: bool = False,
) -> MTForwardResult:
    """
    Convenience wrapper around MTForwardSolver.
    """

    if parameterization is None:
        if isinstance(
            model,
            Model3D,
        ):
            parameterization = model.name
        else:
            parameterization = "conductivity"

    solver = MTForwardSolver(
        mesh=mesh,
        frequencies=frequencies,
        parameterization=parameterization,
        mu_r=mu_r,
        solver_method=solver_method,
        solver_tolerance=solver_tolerance,
        solver_maxiter=solver_maxiter,
    )

    return solver.forward(
        model=model,
        stations=stations,
        store_fields=store_fields,
    )


def solve_mt_forward(
    *args,
    **kwargs,
) -> MTForwardResult:
    """
    Backward-compatible alias.
    """
    return forward_mt(
        *args,
        **kwargs,
    )


__all__ = [
    "MU0",
    "MTForwardResult",
    "MTForwardSolver",
    "forward_mt",
    "solve_mt_forward",
]
