"""
3D magnetotelluric inversion method for GeoInvLab3D.

Data convention:
    rho_xx, rho_xy, rho_yx, rho_yy,
    phase_xx, phase_xy, phase_yx, phase_yy

Each MTDataSet row represents one station-frequency observation
and contributes eight scalar values to the inversion vector.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

import numpy as np

from geoinvlab.core.data.mt_dataset import MTDataSet
from geoinvlab.core.mesh import Mesh3D
from geoinvlab.core.model import Model3D

from .discretization.station import MTStation, MTStationSet
from .operator import MTOperator


class MTMethod:
    """MT method interface connecting the dataset and the forward operator."""

    name = "MT"

    def __init__(self, dataset: MTDataSet, mesh: Mesh3D) -> None:
        if not isinstance(dataset, MTDataSet):
            raise TypeError("dataset must be an instance of MTDataSet.")
        if not isinstance(mesh, Mesh3D):
            raise TypeError("mesh must be an instance of Mesh3D.")

        self.dataset = dataset
        self.mesh = mesh

        self.operator = MTOperator(
            mesh=mesh,
            frequencies=dataset.frequencies,
            locations=dataset.locations,
            observation_frequencies=dataset.frequencies,
        )

    # ------------------------------------------------------------------
    # Data dimensions and properties
    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        """Number of station-frequency rows in the dataset."""
        return int(self.dataset.n_observations)

    @property
    def n_data(self) -> int:
        """Number of scalar values used by inversion: 8N."""
        return int(self.dataset.n_data)

    @property
    def n_values(self) -> int:
        """Explicit alias for the number of scalar MT values."""
        return int(self.dataset.n_values)

    @property
    def n_frequencies(self) -> int:
        """Number of unique frequencies."""
        return int(self.dataset.n_frequencies)

    @property
    def frequencies(self) -> np.ndarray:
        """Frequency associated with each observation."""
        return np.asarray(self.dataset.frequencies, dtype=float)

    @property
    def observed_values(self) -> np.ndarray:
        """Flattened observed vector of length 8N."""
        return np.asarray(self.dataset.values, dtype=float).reshape(-1)

    @property
    def observed_resistivity(self) -> np.ndarray:
        """Observed apparent resistivities, shape (N, 2, 2)."""
        return self.dataset.resistivity

    @property
    def observed_phase(self) -> np.ndarray:
        """Observed impedance phases, shape (N, 2, 2), in degrees."""
        return self.dataset.phase

    @property
    def locations(self) -> np.ndarray:
        """
        Return the unique physical locations of MT stations.

        The dataset contains one row per station-frequency pair, so the
        same station may appear multiple times. Coordinates are deduplicated
        while preserving their order of first appearance.
        """
        raw_locations = np.asarray(self.dataset.locations, dtype=float)

        if raw_locations.size == 0:
            return np.empty((0, 3), dtype=float)

        if raw_locations.ndim != 2 or raw_locations.shape[1] != 3:
            raise ValueError(
                "dataset.locations must have shape "
                "(n_observations, 3)."
            )

        unique: list[np.ndarray] = []
        for point in raw_locations:
            if not any(np.array_equal(point, other) for other in unique):
                unique.append(point.copy())

        return (
            np.vstack(unique)
            if unique
            else np.empty((0, 3), dtype=float)
        )

    @property
    def unique_locations(self) -> np.ndarray:
        """Explicit alias for locations, retained for compatibility."""
        return self.locations

    def weights(self) -> np.ndarray:
        """Return scalar weights supplied by MTDataSet."""
        weights = np.asarray(
            self.dataset.weights(),
            dtype=float,
        ).reshape(-1)

        if weights.size != self.n_data:
            raise ValueError(
                f"dataset.weights() returned {weights.size} weights, "
                f"but n_data={self.n_data}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("Dataset weights must be finite.")
        if np.any(weights < 0.0):
            raise ValueError("Dataset weights must be non-negative.")

        return weights

    # ------------------------------------------------------------------
    # Station geometry
    # ------------------------------------------------------------------

    def _station_set(self) -> MTStationSet:
        """Build unique stations using the MTStation API."""
        stations = tuple(
            MTStation(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]),
                name=f"STA{index + 1:02d}",
            )
            for index, point in enumerate(self.unique_locations)
        )

        if not stations:
            raise ValueError("The dataset contains no stations.")

        return MTStationSet(stations=stations)

    # ------------------------------------------------------------------
    # Forward modelling
    # ------------------------------------------------------------------

    def forward(self, model: Model3D) -> np.ndarray:
        """
        Compute the predicted MT response as a vector of length 8N.

        The operator returns values in the same order as MTDataSet.values.
        """
        self.validate_model(model)

        predicted = np.asarray(
            self.operator.predict(model=model, data=self.dataset),
            dtype=float,
        ).reshape(-1)

        if predicted.size != self.n_data:
            raise ValueError(
                "Invalid MT prediction size: "
                f"{predicted.size} values returned, "
                f"{self.n_data} expected."
            )
        if not np.all(np.isfinite(predicted)):
            raise ValueError(
                "The MT prediction contains NaN or infinite values."
            )

        return predicted

    def forward_impedance(self, model: Model3D) -> np.ndarray:
        """
        Compute the complex impedance tensor at unique stations.

        Expected shape: (n_unique_stations, n_unique_frequencies, 2, 2).
        """
        self.validate_model(model)

        impedance = np.asarray(
            self.operator.forward(
                model=model,
                locations=self.unique_locations,
            ),
            dtype=complex,
        )

        expected = (
            self.unique_locations.shape[0],
            self.n_frequencies,
            2,
            2,
        )

        if impedance.shape != expected:
            raise ValueError(
                "Invalid impedance tensor shape: "
                f"{impedance.shape}, expected shape {expected}."
            )
        if not np.all(np.isfinite(impedance)):
            raise ValueError(
                "The impedance tensor contains NaN or infinite values."
            )

        return impedance

    def response_from_impedance(
        self,
        impedance: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Convert impedances to MT observables using the operator."""
        return self.operator.response_from_impedance(impedance)

    # ------------------------------------------------------------------
    # Convert observables to a vector
    # ------------------------------------------------------------------

    @staticmethod
    def _phase_with_zero_detection(
        rho: float,
        phase: float,
        rho_scale: float,
    ) -> float:
        """Avoid using the arbitrary phase of a nearly zero component."""
        threshold = max(
            1.0e-12 * max(abs(float(rho_scale)), 1.0),
            1.0e-14,
        )

        if abs(float(rho)) <= threshold:
            return 0.0

        return float(phase)

    @classmethod
    def _response_to_vector(
        cls,
        response: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Flatten apparent resistivity and impedance phase.

        Per observation:
            rho_xx, rho_xy, rho_yx, rho_yy,
            phase_xx, phase_xy, phase_yx, phase_yy.
        """
        if "apparent_resistivity" not in response:
            raise KeyError(
                "The MT response does not contain 'apparent_resistivity'."
            )
        if "impedance_phase" not in response:
            raise KeyError(
                "The MT response does not contain 'impedance_phase'."
            )

        rho_all = np.asarray(
            response["apparent_resistivity"],
            dtype=float,
        )
        phase_all = np.asarray(
            response["impedance_phase"],
            dtype=float,
        )

        if rho_all.ndim != 4 or rho_all.shape[2:] != (2, 2):
            raise ValueError(
                "apparent_resistivity must have shape "
                "(n_stations, n_frequencies, 2, 2)."
            )
        if phase_all.shape != rho_all.shape:
            raise ValueError(
                "impedance_phase and apparent_resistivity must have "
                "identical shapes."
            )
        if (
            not np.all(np.isfinite(rho_all))
            or not np.all(np.isfinite(phase_all))
        ):
            raise ValueError(
                "The MT response contains NaN or infinite values."
            )

        rows: list[list[float]] = []

        for station_index in range(rho_all.shape[0]):
            for frequency_index in range(rho_all.shape[1]):
                rho = rho_all[station_index, frequency_index]
                phase = phase_all[station_index, frequency_index]
                rho_scale = float(np.max(np.abs(rho)))

                rows.append(
                    [
                        float(rho[0, 0]),
                        float(rho[0, 1]),
                        float(rho[1, 0]),
                        float(rho[1, 1]),
                        cls._phase_with_zero_detection(
                            rho[0, 0], phase[0, 0], rho_scale
                        ),
                        cls._phase_with_zero_detection(
                            rho[0, 1], phase[0, 1], rho_scale
                        ),
                        cls._phase_with_zero_detection(
                            rho[1, 0], phase[1, 0], rho_scale
                        ),
                        cls._phase_with_zero_detection(
                            rho[1, 1], phase[1, 1], rho_scale
                        ),
                    ]
                )

        return np.asarray(rows, dtype=float).reshape(-1)

    # ------------------------------------------------------------------
    # Residuals and validation
    # ------------------------------------------------------------------

    @staticmethod
    def wrapped_phase_difference(
        predicted: np.ndarray,
        observed: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the shortest signed angular difference in degrees.

        The result belongs to [-180, 180), so phases near +180 and -180
        degrees are correctly treated as close to one another.
        """
        predicted = np.asarray(predicted, dtype=float)
        observed = np.asarray(observed, dtype=float)

        if predicted.shape != observed.shape:
            raise ValueError(
                "Predicted and observed phase arrays must have "
                "identical shapes."
            )
        if (
            not np.all(np.isfinite(predicted))
            or not np.all(np.isfinite(observed))
        ):
            raise ValueError(
                "Predicted and observed phases must be finite."
            )

        return (predicted - observed + 180.0) % 360.0 - 180.0

    def residual(
        self,
        predicted: np.ndarray,
        weighted: bool = True,
    ) -> np.ndarray:
        """
        Compute residuals in the canonical 8-values-per-observation order.

        Apparent-resistivity residuals use ordinary subtraction. Phase
        residuals use the wrapped angular difference in [-180, 180).
        Weights, when requested, are applied after phase wrapping.
        """
        predicted = np.asarray(
            predicted,
            dtype=float,
        ).reshape(-1)
        observed = self.observed_values

        if predicted.size != observed.size:
            raise ValueError(
                f"Incompatible prediction size: {predicted.size}; "
                f"{observed.size} values expected."
            )
        if predicted.size % 8 != 0:
            raise ValueError(
                "The MT data vector must contain eight values "
                "per station-frequency observation."
            )
        if (
            not np.all(np.isfinite(predicted))
            or not np.all(np.isfinite(observed))
        ):
            raise ValueError(
                "Predicted and observed MT data must be finite."
            )

        predicted_rows = predicted.reshape(-1, 8)
        observed_rows = observed.reshape(-1, 8)

        residual_rows = predicted_rows - observed_rows

        # First four columns: apparent resistivity.
        # Last four columns: impedance phase (degrees).
        residual_rows[:, 4:8] = self.wrapped_phase_difference(
            predicted_rows[:, 4:8],
            observed_rows[:, 4:8],
        )

        residual = residual_rows.reshape(-1)

        if weighted:
            residual *= self.weights()

        return residual

    def misfit(
        self,
        predicted: np.ndarray,
        weighted: bool = True,
    ) -> float:
        """Return the sum of squared MT residuals."""
        residual = self.residual(
            predicted,
            weighted=weighted,
        )
        return float(np.dot(residual, residual))

    def validate_model(self, model: Model3D) -> None:
        """Check the model type and dimensional compatibility."""
        if not isinstance(model, Model3D):
            raise TypeError("model must be an instance of Model3D.")

        if model.mesh.n_cells != self.mesh.n_cells:
            raise ValueError(
                "The model mesh is incompatible with the MT method: "
                f"{model.mesh.n_cells} cells versus {self.mesh.n_cells}."
            )

        model_values = np.asarray(
            model.values,
            dtype=float,
        ).reshape(-1)

        if model_values.size != self.mesh.n_cells:
            raise ValueError(
                "The model parameter count does not match the MT mesh: "
                f"{model_values.size} values versus {self.mesh.n_cells}."
            )
        if not np.all(np.isfinite(model_values)):
            raise ValueError(
                "The model contains NaN or infinite values."
            )

    def __call__(self, model: Model3D) -> np.ndarray:
        """Alias for forward(model)."""
        return self.forward(model)

    def __repr__(self) -> str:
        return (
            "MTMethod("
            f"n_observations={self.n_observations}, "
            f"n_data={self.n_data}, "
            f"n_frequencies={self.n_frequencies}, "
            f"n_cells={self.mesh.n_cells})"
        )


__all__ = ["MTMethod"]
