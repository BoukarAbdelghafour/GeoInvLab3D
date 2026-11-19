"""
Magnetotelluric data containers.

This module defines the MTDataSet class used to store observed
magnetotelluric responses for inversion.

For each station-frequency observation, eight scalar quantities are
stored:

    rho_xx, rho_xy, rho_yx, rho_yy
    phase_xx, phase_xy, phase_yx, phase_yy

The phase components are impedance phases in degrees.

The data vector therefore has length

    n_data = 8 * n_observations
    
    MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MTDataSet:
    """
    Magnetotelluric observed data set.

    Parameters
    ----------
    locations :
        Observation coordinates with shape ``(N, 3)``.
    frequencies :
        Frequency associated with each observation, in Hz.
        Shape ``(N,)``.
    rho_xx, rho_xy, rho_yx, rho_yy :
        Apparent resistivity tensor components, in ohm.m.
    phase_xx, phase_xy, phase_yx, phase_yy :
        Impedance phase tensor components, in degrees.
    uncertainties :
        Optional uncertainties with shape ``(N, 8)``.
        The eight columns follow the same order as ``values``.
    name :
        Optional data-set name.
    """

    locations: np.ndarray
    frequencies: np.ndarray

    rho_xx: np.ndarray
    rho_xy: np.ndarray
    rho_yx: np.ndarray
    rho_yy: np.ndarray

    phase_xx: np.ndarray
    phase_xy: np.ndarray
    phase_yx: np.ndarray
    phase_yy: np.ndarray

    uncertainties: np.ndarray | None = None
    name: str = "MTDataSet"

    def __post_init__(self) -> None:
        """Convert inputs to arrays and validate the data set."""

        self.locations = np.asarray(self.locations, dtype=float)
        self.frequencies = np.asarray(self.frequencies, dtype=float)

        self.rho_xx = np.asarray(self.rho_xx, dtype=float)
        self.rho_xy = np.asarray(self.rho_xy, dtype=float)
        self.rho_yx = np.asarray(self.rho_yx, dtype=float)
        self.rho_yy = np.asarray(self.rho_yy, dtype=float)

        self.phase_xx = np.asarray(self.phase_xx, dtype=float)
        self.phase_xy = np.asarray(self.phase_xy, dtype=float)
        self.phase_yx = np.asarray(self.phase_yx, dtype=float)
        self.phase_yy = np.asarray(self.phase_yy, dtype=float)

        if self.uncertainties is not None:
            self.uncertainties = np.asarray(
                self.uncertainties,
                dtype=float,
            )

        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validate all dimensions and numerical values."""

        if self.locations.ndim != 2 or self.locations.shape[1] != 3:
            raise ValueError(
                "locations must have shape (N, 3)"
            )

        n = self.locations.shape[0]

        if self.frequencies.shape != (n,):
            raise ValueError(
                "frequencies must have shape (N,)"
            )

        if not np.all(np.isfinite(self.locations)):
            raise ValueError(
                "locations must contain only finite values"
            )

        if not np.all(np.isfinite(self.frequencies)):
            raise ValueError(
                "frequencies must contain only finite values"
            )

        if np.any(self.frequencies <= 0.0):
            raise ValueError(
                "frequencies must be strictly positive"
            )

        response_arrays = (
            self.rho_xx,
            self.rho_xy,
            self.rho_yx,
            self.rho_yy,
            self.phase_xx,
            self.phase_xy,
            self.phase_yx,
            self.phase_yy,
        )

        response_names = (
            "rho_xx",
            "rho_xy",
            "rho_yx",
            "rho_yy",
            "phase_xx",
            "phase_xy",
            "phase_yx",
            "phase_yy",
        )

        for name, values in zip(response_names, response_arrays):
            if values.shape != (n,):
                raise ValueError(
                    f"{name} must have shape (N,)"
                )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name} must contain only finite values"
                )

        rho_arrays = (
            self.rho_xx,
            self.rho_xy,
            self.rho_yx,
            self.rho_yy,
        )

        for name, values in zip(
            (
                "rho_xx",
                "rho_xy",
                "rho_yx",
                "rho_yy",
            ),
            rho_arrays,
        ):
            if np.any(values < 0.0):
                raise ValueError(
                    f"{name} must contain non-negative values"
                )

        if self.uncertainties is not None:
            uncertainties = np.asarray(
                self.uncertainties,
                dtype=float,
            )

            # A scalar uncertainty is expanded to one value per MT
            # component and observation. Zero means unspecified.
            if uncertainties.ndim == 0:
                value = float(uncertainties)

                if not np.isfinite(value):
                    raise ValueError(
                        "Dataset scalar uncertainty must be finite."
                    )

                if value < 0.0:
                    raise ValueError(
                        "Dataset scalar uncertainty must be non-negative."
                    )

                if value == 0.0:
                    self.uncertainties = None
                    return

                self.uncertainties = np.full((n, 8), value, dtype=float)
                return

            # Canonical form: one row per station-frequency observation.
            # The larger legacy form remains accepted for compatibility;
            # weights() uses only its first n physical-observation rows.
            valid_shapes = (
                (n, 8),
                (self.n_data, 8),
            )

            if uncertainties.shape not in valid_shapes:
                raise ValueError(
                    "uncertainties must have shape "
                    f"({n}, 8) or ({self.n_data}, 8)"
                )

            if not np.all(np.isfinite(uncertainties)):
                raise ValueError(
                    "uncertainties must contain only finite values"
                )

            if np.any(uncertainties <= 0.0):
                raise ValueError(
                    "uncertainties must be strictly positive"
                )

            self.uncertainties = uncertainties

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        """
        Number of station-frequency observations.

        Each row corresponds to one station-frequency pair.
        """
        return int(self.locations.shape[0])

    @property
    def n_values(self) -> int:
        """
        Total number of scalar data values.

        There are eight scalar MT quantities per observation.
        """
        return 8 * self.n_observations

    @property
    def n_data(self) -> int:
        """
        Number of scalar data values used by inversion.

        This is intentionally equal to ``n_values`` rather than
        ``n_observations`` because the inverse-problem framework
        operates on scalar data vectors.
        """
        return self.n_values

    @property
    def n_frequencies(self) -> int:
        """Number of distinct frequencies."""
        return int(np.unique(self.frequencies).size)

    @property
    def unique_frequencies(self) -> np.ndarray:
        """Sorted unique frequencies."""
        return np.unique(self.frequencies)

    @property
    def frequencies_unique(self) -> np.ndarray:
        """Alias for :attr:`unique_frequencies`."""
        return self.unique_frequencies

    # ------------------------------------------------------------------
    # Tensor representations
    # ------------------------------------------------------------------

    @property
    def resistivity(self) -> np.ndarray:
        """
        Apparent resistivity tensor.

        Returns
        -------
        ndarray
            Array with shape ``(N, 2, 2)``.
        """

        return np.stack(
            (
                np.stack(
                    (self.rho_xx, self.rho_xy),
                    axis=-1,
                ),
                np.stack(
                    (self.rho_yx, self.rho_yy),
                    axis=-1,
                ),
            ),
            axis=-2,
        )

    @property
    def phase(self) -> np.ndarray:
        """
        Impedance phase tensor.

        Returns
        -------
        ndarray
            Array with shape ``(N, 2, 2)`` in degrees.
        """

        return np.stack(
            (
                np.stack(
                    (self.phase_xx, self.phase_xy),
                    axis=-1,
                ),
                np.stack(
                    (self.phase_yx, self.phase_yy),
                    axis=-1,
                ),
            ),
            axis=-2,
        )

    # ------------------------------------------------------------------
    # Flattened inversion vector
    # ------------------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """
        Flattened scalar data vector.

        Ordering for each observation is

            rho_xx
            rho_xy
            rho_yx
            rho_yy
            phase_xx
            phase_xy
            phase_yx
            phase_yy

        Returns
        -------
        ndarray
            One-dimensional vector with length ``8*N``.
        """

        return np.column_stack(
            (
                self.rho_xx,
                self.rho_xy,
                self.rho_yx,
                self.rho_yy,
                self.phase_xx,
                self.phase_xy,
                self.phase_yx,
                self.phase_yy,
            )
        ).ravel()

    # ------------------------------------------------------------------
    # Data weights
    # ------------------------------------------------------------------

    def weights(self) -> np.ndarray:
        """
        Return inverse-uncertainty weights.

        If no uncertainties are supplied, unit weights are returned.

        Returns
        -------
        ndarray
            One-dimensional array with shape ``(n_data,)``.
        """

        if self.uncertainties is None:
            return np.ones(self.n_data, dtype=float)

        # ``n_data`` is the scalar-data length.  The canonical uncertainty
        # table has one row per station-frequency observation and eight
        # columns.  Some inversion callers historically allocate ``n_data``
        # rows instead; in that case only the rows corresponding to the
        # physical observations contribute to the scalar weight vector.
        n = self.n_observations
        uncertainty_table = self.uncertainties[:n, :]

        weights = (1.0 / uncertainty_table).ravel()

        if weights.size != self.n_data:
            raise RuntimeError(
                "Internal uncertainty/weight size mismatch: "
                f"got {weights.size}, expected {self.n_data}."
            )

        return weights

    @property
    def weight_matrix(self) -> np.ndarray:
        """Return the diagonal weight matrix."""

        return np.diag(self.weights())

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> dict[str, float]:
        """Return basic statistics of the scalar data vector."""

        values = self.values

        return {
            "n_observations": float(self.n_observations),
            "n_data": float(self.n_data),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_array(self) -> np.ndarray:
        """
        Serialize the data set to a two-dimensional array.

        Columns are

            x, y, z, frequency,
            rho_xx, rho_xy, rho_yx, rho_yy,
            phase_xx, phase_xy, phase_yx, phase_yy

        followed optionally by the eight uncertainty columns.
        """

        base = np.column_stack(
            (
                self.locations,
                self.frequencies,
                self.rho_xx,
                self.rho_xy,
                self.rho_yx,
                self.rho_yy,
                self.phase_xx,
                self.phase_xy,
                self.phase_yx,
                self.phase_yy,
            )
        )

        if self.uncertainties is not None:
            return np.column_stack(
                (base, self.uncertainties)
            )

        return base

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        name: str = "MTDataSet",
    ) -> "MTDataSet":
        """
        Build a data set from a serialized array.

        Accepted formats contain either 12 or 20 columns.

        12 columns:
            coordinates + frequency + 8 responses

        20 columns:
            coordinates + frequency + 8 responses + 8 uncertainties
        """

        array = np.asarray(array, dtype=float)

        if array.ndim != 2:
            raise ValueError(
                "array must be two-dimensional"
            )

        if array.shape[1] not in (12, 20):
            raise ValueError(
                "array must have either 12 or 20 columns"
            )

        locations = array[:, 0:3]
        frequencies = array[:, 3]

        responses = array[:, 4:12]

        uncertainties = None
        if array.shape[1] == 20:
            uncertainties = array[:, 12:20]

        return cls(
            locations=locations,
            frequencies=frequencies,
            rho_xx=responses[:, 0],
            rho_xy=responses[:, 1],
            rho_yx=responses[:, 2],
            rho_yy=responses[:, 3],
            phase_xx=responses[:, 4],
            phase_xy=responses[:, 5],
            phase_yx=responses[:, 6],
            phase_yy=responses[:, 7],
            uncertainties=uncertainties,
            name=name,
        )

    # ------------------------------------------------------------------
    # Copying
    # ------------------------------------------------------------------

    def copy(
        self,
        *,
        name: str | None = None,
    ) -> "MTDataSet":
        """Return an independent copy of the data set."""

        return MTDataSet(
            locations=self.locations.copy(),
            frequencies=self.frequencies.copy(),
            rho_xx=self.rho_xx.copy(),
            rho_xy=self.rho_xy.copy(),
            rho_yx=self.rho_yx.copy(),
            rho_yy=self.rho_yy.copy(),
            phase_xx=self.phase_xx.copy(),
            phase_xy=self.phase_xy.copy(),
            phase_yx=self.phase_yx.copy(),
            phase_yy=self.phase_yy.copy(),
            uncertainties=(
                None
                if self.uncertainties is None
                else self.uncertainties.copy()
            ),
            name=self.name if name is None else name,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise representation."""

        return (
            f"MTDataSet("
            f"name={self.name!r}, "
            f"n_observations={self.n_observations}, "
            f"n_data={self.n_data}, "
            f"n_frequencies={self.n_frequencies})"
        )
