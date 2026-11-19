"""
Magnetotelluric (MT) forward operator.

This module connects MT models, the 3D forward solver and MT observations.

Conventions
-----------
Time dependence:
    exp(+i * omega * t)

Maxwell-Faraday equation:
    curl(E) = -i * omega * mu * H

Impedance:
    E = Z H

Impedance tensor:
    [[Zxx, Zxy],
     [Zyx, Zyy]]

Canonical data ordering:
    [rho_xx, rho_xy, rho_yx, rho_yy,
     phase_xx, phase_xy, phase_yx, phase_yy]

Each dataset observation corresponds to one station-frequency pair.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geoinvlab.core.model import Model3D
from geoinvlab.core.mesh import Mesh3D

from .discretization.forward import MTForwardResult, MTForwardSolver
from .discretization.station import MTStation, MTStationSet


MU0 = 4.0e-7 * np.pi


@dataclass
class MTOperator:
    """
    Magnetotelluric forward operator.

    Parameters
    ----------
    mesh:
        Computational mesh.

    frequencies:
        Frequencies used by the forward solver.

    locations:
        Optional station coordinates with shape (n_stations, 3).

    observation_frequencies:
        Optional frequencies associated with individual observations.

    solver_method:
        Linear solver method used by the forward solver.

    solver_tolerance:
        Solver convergence tolerance.

    solver_maxiter:
        Maximum number of solver iterations.

    condition_tolerance:
        Numerical tolerance used when checking degenerate responses.

    jacobian_step:
        Relative finite-difference perturbation for the Jacobian.
    """

    mesh: Mesh3D
    frequencies: np.ndarray
    locations: np.ndarray | None = None
    observation_frequencies: np.ndarray | None = None

    solver_method: str = "direct"
    solver_tolerance: float = 1.0e-10
    solver_maxiter: int | None = 1000
    condition_tolerance: float = 1.0e-12
    jacobian_step: float = 1.0e-5

    forward_solver: MTForwardSolver = field(init=False)

    def __post_init__(self) -> None:
        """Validate the configuration and initialise the solver."""
        self.frequencies = np.asarray(
            self.frequencies,
            dtype=float,
        ).reshape(-1)

        if self.frequencies.size == 0:
            raise ValueError(
                "At least one frequency must be supplied."
            )

        if not np.all(np.isfinite(self.frequencies)):
            raise ValueError(
                "Frequencies must be finite."
            )

        if np.any(self.frequencies <= 0.0):
            raise ValueError(
                "Frequencies must be strictly positive."
            )

        # Preserve the first occurrence of each frequency.
        self.frequencies = np.asarray(
            list(dict.fromkeys(self.frequencies.tolist())),
            dtype=float,
        )

        if self.locations is not None:
            self.locations = self._validate_locations(
                self.locations
            )

        if self.observation_frequencies is not None:
            self.observation_frequencies = np.asarray(
                self.observation_frequencies,
                dtype=float,
            ).reshape(-1)

            if not np.all(
                np.isfinite(self.observation_frequencies)
            ):
                raise ValueError(
                    "Observation frequencies must be finite."
                )

            if np.any(self.observation_frequencies <= 0.0):
                raise ValueError(
                    "Observation frequencies must be positive."
                )

        if self.solver_tolerance <= 0.0:
            raise ValueError(
                "solver_tolerance must be positive."
            )

        if self.condition_tolerance <= 0.0:
            raise ValueError(
                "condition_tolerance must be positive."
            )

        if self.jacobian_step <= 0.0:
            raise ValueError(
                "jacobian_step must be positive."
            )

        self.forward_solver = MTForwardSolver(
            mesh=self.mesh,
            frequencies=self.frequencies,
            solver_method=self.solver_method,
            solver_tolerance=self.solver_tolerance,
            solver_maxiter=self.solver_maxiter,
        )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def n_stations(self) -> int:
        """Number of configured stations."""
        if self.locations is None:
            return 0

        return int(self.locations.shape[0])

    @property
    def n_frequencies(self) -> int:
        """Number of distinct operator frequencies."""
        return int(self.frequencies.size)

    @property
    def n_observations(self) -> int:
        """Number of configured station-frequency observations."""
        if self.observation_frequencies is not None:
            return int(self.observation_frequencies.size)

        if self.locations is None:
            return 0

        return self.n_stations * self.n_frequencies

    @property
    def n_data(self) -> int:
        """Number of scalar MT data values."""
        return 8 * self.n_observations

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_locations(
        locations: np.ndarray,
    ) -> np.ndarray:
        """Validate station coordinates."""
        locations = np.asarray(
            locations,
            dtype=float,
        )

        if locations.ndim != 2 or locations.shape[1] != 3:
            raise ValueError(
                "Station locations must have shape (n_stations, 3)."
            )

        if locations.shape[0] == 0:
            raise ValueError(
                "At least one station location is required."
            )

        if not np.all(np.isfinite(locations)):
            raise ValueError(
                "Station locations must be finite."
            )

        return locations

    @staticmethod
    def _validate_impedance(
        impedance: np.ndarray,
    ) -> np.ndarray:
        """
        Validate a complex impedance tensor.

        Accepted shape:
            (n_stations, n_frequencies, 2, 2)

        Observation-wise predictions may use:
            (n_observations, 1, 2, 2)
        """
        impedance = np.asarray(impedance)

        if impedance.ndim != 4:
            raise ValueError(
                "Impedance must have shape "
                "(n_stations, n_frequencies, 2, 2)."
            )

        if impedance.shape[-2:] != (2, 2):
            raise ValueError(
                "The last two impedance dimensions must be (2, 2)."
            )

        if impedance.shape[0] == 0 or impedance.shape[1] == 0:
            raise ValueError(
                "Impedance must contain at least one station "
                "and one frequency."
            )

        if not np.all(np.isfinite(impedance)):
            raise ValueError(
                "Impedance contains non-finite values."
            )

        return np.asarray(
            impedance,
            dtype=complex,
        )

    # ------------------------------------------------------------------
    # Model parameterisation
    # ------------------------------------------------------------------

    @staticmethod
    def _model_parameterization(
        model: Model3D,
    ) -> str:
        """
        Infer whether model values represent conductivity
        or resistivity.

        Unlabelled models default to resistivity.
        """
        name = str(
            getattr(model, "name", "")
        ).strip().lower()

        if (
            "conductivity" in name
            or "sigma" in name
            or name in {"cond", "conduct"}
        ):
            return "conductivity"

        if (
            "resistivity" in name
            or "rho" in name
            or "res" in name
        ):
            return "resistivity"

        return "resistivity"

    def _model_for_forward_solver(
        self,
        model: Model3D,
    ) -> Model3D:
        """Convert the model to conductivity parameterisation."""
        values = np.asarray(
            model.values,
            dtype=float,
        ).reshape(-1)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Model values must be finite."
            )

        parameterization = self._model_parameterization(
            model
        )

        if parameterization == "resistivity":
            if np.any(values <= 0.0):
                raise ValueError(
                    "Resistivity values must be positive."
                )

            conductivity = 1.0 / values

        else:
            if np.any(values <= 0.0):
                raise ValueError(
                    "Conductivity values must be positive."
                )

            conductivity = values.copy()

        if conductivity.size != self.mesh.n_cells:
            raise ValueError(
                "The model size does not match the mesh: "
                f"received {conductivity.size} values, "
                f"expected {self.mesh.n_cells}."
            )

        return Model3D(
            mesh=self.mesh,
            values=conductivity,
            name="conductivity",
        )

    # ------------------------------------------------------------------
    # Stations
    # ------------------------------------------------------------------

    @staticmethod
    def _make_stations(
        locations: np.ndarray,
    ) -> MTStationSet:
        """Create a station set from an array of coordinates."""
        locations = MTOperator._validate_locations(
            locations
        )

        stations = [
            MTStation(
                x=float(location[0]),
                y=float(location[1]),
                z=float(location[2]),
                name=f"STA{index + 1:03d}",
            )
            for index, location in enumerate(locations)
        ]

        return MTStationSet(stations=stations)

    @staticmethod
    def _unique_locations(
        locations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return unique locations and the inverse mapping."""
        locations = MTOperator._validate_locations(
            locations
        )

        unique_locations, inverse = np.unique(
            locations,
            axis=0,
            return_inverse=True,
        )

        return unique_locations, inverse

    # ------------------------------------------------------------------
    # Forward modelling
    # ------------------------------------------------------------------

    def forward_result(
        self,
        model: Model3D,
        locations: np.ndarray | None = None,
        store_fields: bool = False,
    ) -> MTForwardResult:
        """Compute the complete MT forward result."""
        if locations is None:
            locations = self.locations

        if locations is None:
            raise ValueError(
                "Station locations must be supplied either to the "
                "operator or to forward_result()."
            )

        locations = self._validate_locations(
            locations
        )

        conductivity_model = self._model_for_forward_solver(
            model
        )

        stations = self._make_stations(
            locations
        )

        return self.forward_solver.forward(
            model=conductivity_model,
            stations=stations,
            store_fields=store_fields,
        )

    def forward(
        self,
        model: Model3D,
        locations: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the complex impedance tensor.

        Returns
        -------
        numpy.ndarray
            Shape (n_stations, n_frequencies, 2, 2).
        """
        result = self.forward_result(
            model=model,
            locations=locations,
            store_fields=False,
        )

        return self._validate_impedance(
            result.impedance
        )

    # ------------------------------------------------------------------
    # MT responses
    # ------------------------------------------------------------------

    def apparent_resistivity(
        self,
        impedance: np.ndarray,
        frequencies: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute component-wise apparent resistivity.

        rho_ij = |Z_ij|**2 / (mu0 * omega)

        Observation-wise impedance with shape
        (n_observations, 1, 2, 2) can be paired with one frequency
        per observation.
        """
        impedance = self._validate_impedance(
            impedance
        )

        if frequencies is None:
            frequencies = self.frequencies

        frequencies = np.asarray(
            frequencies,
            dtype=float,
        ).reshape(-1)

        if frequencies.size == 0:
            raise ValueError(
                "At least one frequency is required."
            )

        if not np.all(np.isfinite(frequencies)):
            raise ValueError(
                "Frequencies must be finite."
            )

        if np.any(frequencies <= 0.0):
            raise ValueError(
                "Frequencies must be strictly positive."
            )

        n_stations = impedance.shape[0]
        n_frequency_axis = impedance.shape[1]

        if frequencies.size == n_frequency_axis:
            effective_frequencies = frequencies

        elif (
            n_frequency_axis == 1
            and frequencies.size == n_stations
        ):
            effective_frequencies = frequencies.reshape(
                n_stations,
                1,
            )

        elif frequencies.size == 1:
            effective_frequencies = frequencies

        else:
            raise ValueError(
                "The frequency dimension of impedance does not "
                "match the supplied frequencies. For observation-wise "
                "impedance, supply one frequency per observation."
            )

        omega = 2.0 * np.pi * effective_frequencies

        return (
            np.abs(impedance) ** 2
            / (MU0 * omega[..., None, None])
        )

    @staticmethod
    def impedance_phase(
        impedance: np.ndarray,
    ) -> np.ndarray:
        """Compute impedance phases in degrees."""
        impedance = MTOperator._validate_impedance(
            impedance
        )

        return np.degrees(
            np.angle(impedance)
        )

    @staticmethod
    def phase_tensor(
        impedance: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the phase tensor Phi = X^{-1} Y, where Z = X + iY.

        Singular real parts produce NaN phase tensors for the
        affected station-frequency entries.
        """
        impedance = MTOperator._validate_impedance(
            impedance
        )

        real_part = impedance.real
        imag_part = impedance.imag

        output = np.full(
            impedance.shape,
            np.nan,
            dtype=float,
        )

        for station_index in range(impedance.shape[0]):
            for frequency_index in range(impedance.shape[1]):
                x = real_part[
                    station_index,
                    frequency_index,
                ]
                y = imag_part[
                    station_index,
                    frequency_index,
                ]

                try:
                    output[
                        station_index,
                        frequency_index,
                    ] = np.linalg.solve(x, y)
                except np.linalg.LinAlgError:
                    continue

        return output

    def response_from_impedance(
        self,
        impedance: np.ndarray,
        frequencies: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Calculate the MT responses from a complex impedance tensor."""
        impedance = self._validate_impedance(
            impedance
        )

        if frequencies is None:
            frequencies = self.frequencies

        frequencies = np.asarray(
            frequencies,
            dtype=float,
        ).reshape(-1)

        n_stations = impedance.shape[0]
        n_frequency_axis = impedance.shape[1]

        if frequencies.size == n_frequency_axis:
            response_frequencies = frequencies

        elif (
            n_frequency_axis == 1
            and frequencies.size == n_stations
        ):
            response_frequencies = frequencies

        elif frequencies.size == 1:
            response_frequencies = frequencies

        else:
            raise ValueError(
                "The supplied frequencies are incompatible with "
                "the impedance shape. Supply one frequency per "
                "frequency-axis entry or one per observation."
            )

        rho = self.apparent_resistivity(
            impedance,
            frequencies=response_frequencies,
        )

        phase = self.impedance_phase(
            impedance
        )

        # Phase is undefined for effectively zero impedance.
        rho_scale = max(
            float(np.max(rho)) if rho.size else 0.0,
            1.0,
        )

        zero_threshold = max(
            1.0e-12 * rho_scale,
            1.0e-14,
        )

        phase = np.asarray(
            phase,
            dtype=float,
        ).copy()

        phase[rho <= zero_threshold] = 0.0

        return {
            "impedance": impedance,
            "apparent_resistivity": rho,
            "phase_tensor": self.phase_tensor(
                impedance
            ),
            "impedance_phase": phase,
        }

    # ------------------------------------------------------------------
    # Observation-wise frequency selection
    # ------------------------------------------------------------------

    def _frequency_index(
        self,
        frequency: float,
    ) -> int:
        """Find the operator frequency matching a requested value."""
        matches = np.flatnonzero(
            np.isclose(
                self.frequencies,
                float(frequency),
                rtol=1.0e-10,
                atol=0.0,
            )
        )

        if matches.size == 0:
            raise ValueError(
                f"Observation frequency {frequency!r} is not present "
                "in the operator frequency array."
            )

        return int(matches[0])

    def _predict_observations(
        self,
        model: Model3D,
        locations: np.ndarray,
        observation_frequencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict one impedance tensor for each observation.

        Returns
        -------
        impedance:
            Shape (n_observations, 1, 2, 2).

        frequencies:
            Shape (n_observations, 1).
        """
        locations = self._validate_locations(
            locations
        )

        observation_frequencies = np.asarray(
            observation_frequencies,
            dtype=float,
        ).reshape(-1)

        if locations.shape[0] != observation_frequencies.size:
            raise ValueError(
                "The number of observation locations must match "
                "the number of observation frequencies."
            )

        if not np.all(np.isfinite(observation_frequencies)):
            raise ValueError(
                "Observation frequencies must be finite."
            )

        if np.any(observation_frequencies <= 0.0):
            raise ValueError(
                "Observation frequencies must be positive."
            )

        unique_locations, inverse = self._unique_locations(
            locations
        )

        full_impedance = self.forward(
            model=model,
            locations=unique_locations,
        )

        selected = np.empty(
            (
                locations.shape[0],
                1,
                2,
                2,
            ),
            dtype=complex,
        )

        for observation_index, frequency in enumerate(
            observation_frequencies
        ):
            frequency_index = self._frequency_index(
                float(frequency)
            )

            station_index = int(
                inverse[observation_index]
            )

            selected[
                observation_index,
                0,
            ] = full_impedance[
                station_index,
                frequency_index,
            ]

        selected_frequencies = observation_frequencies.reshape(
            -1,
            1,
        )

        return selected, selected_frequencies

    # ------------------------------------------------------------------
    # Response vector
    # ------------------------------------------------------------------

    @staticmethod
    def _response_to_vector(
        response: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Convert responses into the canonical data vector.

        Ordering per observation:
            rho_xx, rho_xy, rho_yx, rho_yy,
            phase_xx, phase_xy, phase_yx, phase_yy
        """
        rho = np.asarray(
            response["apparent_resistivity"],
            dtype=float,
        )

        phase = np.asarray(
            response["impedance_phase"],
            dtype=float,
        )

        if rho.shape != phase.shape:
            raise ValueError(
                "Apparent resistivity and impedance phase "
                "must have identical shapes."
            )

        if rho.ndim != 4 or rho.shape[-2:] != (2, 2):
            raise ValueError(
                "Response arrays must have shape "
                "(n_stations, n_frequencies, 2, 2)."
            )

        rho_flat = rho.reshape(
            rho.shape[0],
            rho.shape[1],
            4,
        )

        phase_flat = phase.reshape(
            phase.shape[0],
            phase.shape[1],
            4,
        )

        combined = np.concatenate(
            (
                rho_flat,
                phase_flat,
            ),
            axis=-1,
        )

        return combined.reshape(-1)
        # ------------------------------------------------------------------
    # Data residuals
    # ------------------------------------------------------------------

    @staticmethod
    def data_residual(
        predicted: np.ndarray,
        observed: np.ndarray,
    ) -> np.ndarray:
        """
        Compute MT data residuals with circular phase differences.

        Canonical data ordering per observation:

            rho_xx, rho_xy, rho_yx, rho_yy,
            phase_xx, phase_xy, phase_yx, phase_yy

        Apparent-resistivity residuals use ordinary subtraction:

            r_rho = rho_predicted - rho_observed

        Phase residuals use the shortest angular difference:

            r_phase = ((phase_predicted - phase_observed + 180)
                       % 360) - 180

        This avoids artificial residuals close to 360 degrees
        when predicted and observed phases lie on opposite sides
        of the -180/180 degree boundary.

        Parameters
        ----------
        predicted : ndarray
            Predicted flattened MT data vector.

        observed : ndarray
            Observed flattened MT data vector.

        Returns
        -------
        ndarray
            Flattened MT residual vector with the canonical ordering.
        """

        predicted = np.asarray(
            predicted,
            dtype=float,
        ).reshape(-1)

        observed = np.asarray(
            observed,
            dtype=float,
        ).reshape(-1)

        if predicted.size != observed.size:
            raise ValueError(
                "Predicted/observed dimension mismatch: "
                f"{predicted.size} predicted values but "
                f"{observed.size} observed values."
            )

        if predicted.size == 0 or predicted.size % 8 != 0:
            raise ValueError(
                "MT data vectors must contain exactly eight "
                "scalar values per observation."
            )

        if not np.all(np.isfinite(predicted)):
            raise ValueError(
                "Predicted MT data must contain only finite values."
            )

        if not np.all(np.isfinite(observed)):
            raise ValueError(
                "Observed MT data must contain only finite values."
            )

        residual = predicted - observed

        # The first four components are apparent resistivities.
        # The last four components are impedance phases in degrees.
        phase_mask = (
            np.arange(residual.size) % 8
        ) >= 4

        residual[phase_mask] = (
            (
                residual[phase_mask] + 180.0
            ) % 360.0
        ) - 180.0

        return residual
    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def predict(
        self,
        model: Model3D,
        data: Any | None = None,
    ) -> np.ndarray:
        """
        Predict the flattened MT data vector.

        Each dataset row is interpreted as one station-frequency
        observation and produces eight scalar values.
        """
        if data is not None:
            locations = self._validate_locations(
                data.locations
            )

            observation_frequencies = np.asarray(
                data.frequencies,
                dtype=float,
            ).reshape(-1)

            if locations.shape[0] != observation_frequencies.size:
                raise ValueError(
                    "Dataset locations and frequencies must contain "
                    "the same number of observations."
                )

            impedance, selected_frequencies = (
                self._predict_observations(
                    model=model,
                    locations=locations,
                    observation_frequencies=observation_frequencies,
                )
            )

            response = self.response_from_impedance(
                impedance,
                frequencies=selected_frequencies,
            )

            return self._response_to_vector(
                response
            )

        if (
            self.locations is not None
            and self.observation_frequencies is not None
        ):
            impedance, selected_frequencies = (
                self._predict_observations(
                    model=model,
                    locations=self.locations,
                    observation_frequencies=self.observation_frequencies,
                )
            )

            response = self.response_from_impedance(
                impedance,
                frequencies=selected_frequencies,
            )

            return self._response_to_vector(
                response
            )

        if self.locations is None:
            raise ValueError(
                "Station locations must be configured or supplied "
                "through a dataset."
            )

        impedance = self.forward(
            model=model,
            locations=self.locations,
        )

        response = self.response_from_impedance(
            impedance,
            frequencies=self.frequencies,
        )

        return self._response_to_vector(
            response
        )

    def predict_dataset(
        self,
        model: Model3D,
        dataset: Any,
    ) -> np.ndarray:
        """Predict observations for a dataset."""
        return self.predict(
            model=model,
            data=dataset,
        )

    def check_prediction(
        self,
        model: Model3D,
        dataset: Any,
    ) -> None:
        """
        Validate the predicted data against a dataset.

        This method accepts a model and a dataset, computes the
        prediction and checks its dimensions and finite values.

        This signature is compatible with the inverse-problem
        validation contract:
            operator.check_prediction(model, dataset)
        """
        prediction = np.asarray(
            self.predict(
                model=model,
                data=dataset,
            ),
            dtype=float,
        ).reshape(-1)

        expected = int(dataset.n_values)

        if prediction.size != expected:
            raise ValueError(
                f"Prediction contains {prediction.size} values; "
                f"dataset expects {expected}."
            )

        if not np.all(np.isfinite(prediction)):
            raise ValueError(
                "Prediction contains non-finite values."
            )

    # ------------------------------------------------------------------
    # Numerical Jacobian
    # ------------------------------------------------------------------

    def jacobian(
        self,
        model: Model3D,
        data: Any | None = None,
        step: float | None = None,
    ) -> np.ndarray:
        """
        Compute a numerical Jacobian using central differences.

        The returned matrix contains derivatives of predicted data
        with respect to the model parameters.
        """
        if step is None:
            step = self.jacobian_step

        step = float(step)

        if not np.isfinite(step) or step <= 0.0:
            raise ValueError(
                "The Jacobian step must be finite and positive."
            )

        values = np.asarray(
            model.values,
            dtype=float,
        ).reshape(-1)

        if values.size != self.mesh.n_cells:
            raise ValueError(
                "The model size does not match the mesh: "
                f"received {values.size} values, "
                f"expected {self.mesh.n_cells}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Model values must be finite."
            )

        if np.any(values <= 0.0):
            raise ValueError(
                "Model values must be positive."
            )

        baseline = np.asarray(
            self.predict(
                model=model,
                data=data,
            ),
            dtype=float,
        ).reshape(-1)

        if baseline.size == 0 or baseline.size % 8 != 0:
            raise ValueError(
                "The MT prediction vector must contain eight values "
                "per observation."
            )

        if not np.all(np.isfinite(baseline)):
            raise ValueError(
                "The baseline prediction contains non-finite values."
            )

        jacobian = np.empty(
            (
                baseline.size,
                values.size,
            ),
            dtype=float,
        )

        # In the canonical MT data vector, each observation contains
        # four apparent-resistivity values followed by four phases.
        phase_mask = (
            np.arange(baseline.size) % 8
        ) >= 4

        for parameter_index in range(values.size):
            value = values[parameter_index]

            # Use a genuinely relative perturbation. The previous
            # max(abs(value), 1) rule produced an excessively large
            # relative step for small conductivity values.
            delta = step * abs(value)

            if not np.isfinite(delta) or delta <= 0.0:
                raise ValueError(
                    "The finite-difference perturbation is invalid "
                    f"for model parameter {parameter_index}."
                )

            plus_values = values.copy()
            minus_values = values.copy()

            plus_values[parameter_index] += delta
            minus_values[parameter_index] -= delta

            # Use a central difference whenever the negative
            # perturbation remains inside the positive model domain.
            if minus_values[parameter_index] > 0.0:
                denominator = 2.0 * delta
                use_central_difference = True
            else:
                # Fall back to a forward difference near the
                # positivity boundary.
                minus_values[parameter_index] = value
                denominator = delta
                use_central_difference = False

            plus_model = Model3D(
                mesh=self.mesh,
                values=plus_values,
                name=model.name,
            )

            plus_prediction = np.asarray(
                self.predict(
                    model=plus_model,
                    data=data,
                ),
                dtype=float,
            ).reshape(-1)

            if plus_prediction.size != baseline.size:
                raise ValueError(
                    "The prediction size changed while computing "
                    f"the Jacobian column {parameter_index}."
                )

            if use_central_difference:
                minus_model = Model3D(
                    mesh=self.mesh,
                    values=minus_values,
                    name=model.name,
                )

                minus_prediction = np.asarray(
                    self.predict(
                        model=minus_model,
                        data=data,
                    ),
                    dtype=float,
                ).reshape(-1)

                if minus_prediction.size != baseline.size:
                    raise ValueError(
                        "The prediction size changed while computing "
                        f"the Jacobian column {parameter_index}."
                    )

                # Apparent-resistivity differences are ordinary
                # differences. Phase differences must use the
                # shortest angular displacement to avoid artificial
                # jumps of approximately 360 degrees at the wrap.
                difference = plus_prediction - minus_prediction
                difference[phase_mask] = (
                    (
                        difference[phase_mask] + 180.0
                    ) % 360.0
                ) - 180.0
            else:
                difference = plus_prediction - baseline
                difference[phase_mask] = (
                    (
                        difference[phase_mask] + 180.0
                    ) % 360.0
                ) - 180.0

            if not np.all(np.isfinite(difference)):
                raise ValueError(
                    "A non-finite finite-difference response was "
                    f"encountered for model parameter {parameter_index}."
                )

            jacobian[:, parameter_index] = (
                difference / denominator
            )

        return jacobian

    # ------------------------------------------------------------------
    # Compatibility aliases
    # ------------------------------------------------------------------

    def compute_response(
        self,
        model: Model3D,
        locations: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Return the response dictionary for a model."""
        impedance = self.forward(
            model=model,
            locations=locations,
        )

        return self.response_from_impedance(
            impedance,
            frequencies=self.frequencies,
        )

    def compute_apparent_resistivity(
        self,
        impedance: np.ndarray,
        frequencies: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compatibility alias for apparent_resistivity()."""
        return self.apparent_resistivity(
            impedance,
            frequencies=frequencies,
        )

    def compute_impedance_phase(
        self,
        impedance: np.ndarray,
    ) -> np.ndarray:
        """Compatibility alias for impedance_phase()."""
        return self.impedance_phase(
            impedance
        )
 
