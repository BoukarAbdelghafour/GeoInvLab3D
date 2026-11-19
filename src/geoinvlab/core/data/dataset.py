"""
Geophysical data representation.

This module provides the DataSet class used to represent
observations and associated information.

For compatibility, ``Dataset`` is also provided as an alias
of ``DataSet``.

MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DataSet:
    """
    Container for geophysical observations.

    Parameters
    ----------
    values : numpy.ndarray
        Observed data values.

    locations : numpy.ndarray
        Observation coordinates with shape (n_data, 3).

    uncertainties : numpy.ndarray or None, optional
        Standard deviation associated with each observation.

    name : str, optional
        Name of the dataset.

    units : str, optional
        Physical units of the observations.
    """

    values: np.ndarray
    locations: np.ndarray

    uncertainties: np.ndarray | None = None

    name: str = "data"
    units: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize the dataset."""

        self.values = np.asarray(
            self.values,
            dtype=float,
        )

        self.locations = np.asarray(
            self.locations,
            dtype=float,
        )

        # ----------------------------------------------------------
        # VALUES
        # ----------------------------------------------------------

        if self.values.ndim != 1:
            raise ValueError(
                "Data values must be a one-dimensional array."
            )

        # ----------------------------------------------------------
        # LOCATIONS
        # ----------------------------------------------------------

        if self.locations.ndim != 2:
            raise ValueError(
                "Locations must be a two-dimensional array."
            )

        if self.locations.shape[1] != 3:
            raise ValueError(
                "Locations must have shape (n_data, 3)."
            )

        if self.locations.shape[0] != self.values.size:
            raise ValueError(
                "Number of locations must match "
                "number of data values."
            )

        # ----------------------------------------------------------
        # UNCERTAINTIES
        # ----------------------------------------------------------

        if self.uncertainties is not None:

            self.uncertainties = np.asarray(
                self.uncertainties,
                dtype=float,
            )

            if self.uncertainties.ndim != 1:
                raise ValueError(
                    "Uncertainties must be one-dimensional."
                )

            if self.uncertainties.size != self.values.size:
                raise ValueError(
                    "Number of uncertainties must match "
                    "number of data."
                )

            if not np.all(
                np.isfinite(self.uncertainties)
            ):
                raise ValueError(
                    "Uncertainties must contain only finite values."
                )

            if np.any(self.uncertainties <= 0.0):
                raise ValueError(
                    "Uncertainties must be strictly positive."
                )

        # ----------------------------------------------------------
        # FINITE VALUES
        # ----------------------------------------------------------

        if not np.all(np.isfinite(self.values)):
            raise ValueError(
                "Data values must contain only finite values."
            )

        if not np.all(np.isfinite(self.locations)):
            raise ValueError(
                "Observation locations must contain "
                "only finite values."
            )

    # ==============================================================
    # PROPERTIES
    # ==============================================================

    @property
    def n_data(self) -> int:
        """Return the number of observations."""

        return int(self.values.size)

    @property
    def dimension(self) -> int:
        """Return the spatial dimension."""

        return int(self.locations.shape[1])

    # ==============================================================
    # WEIGHTS
    # ==============================================================

    def weights(self) -> np.ndarray:
        """
        Return data weighting coefficients.

        If uncertainties are available:

            w_i = 1 / sigma_i

        Otherwise:

            w_i = 1

        Returns
        -------
        numpy.ndarray
            Weight vector with shape ``(n_data,)``.
        """

        if self.uncertainties is None:
            return np.ones(
                self.n_data,
                dtype=float,
            )

        return 1.0 / self.uncertainties

    # ==============================================================
    # WEIGHT MATRIX
    # ==============================================================

    def weight_matrix(self) -> np.ndarray:
        """
        Return the diagonal data-weighting matrix.

        Notes
        -----
        For large datasets, constructing this matrix is generally
        less efficient than using :meth:`weights` directly.
        """

        return np.diag(
            self.weights()
        )

    # ==============================================================
    # STATISTICS
    # ==============================================================

    def minimum(self) -> float:
        """Return the minimum observed value."""

        return float(np.min(self.values))

    def maximum(self) -> float:
        """Return the maximum observed value."""

        return float(np.max(self.values))

    def mean(self) -> float:
        """Return the mean observed value."""

        return float(np.mean(self.values))

    def rms(self) -> float:
        """Return the root-mean-square observed value."""

        return float(
            np.sqrt(
                np.mean(
                    self.values ** 2
                )
            )
        )

    # ==============================================================
    # COPY
    # ==============================================================

    def copy(self) -> "DataSet":
        """Return an independent copy of the dataset."""

        uncertainties = None

        if self.uncertainties is not None:
            uncertainties = (
                self.uncertainties.copy()
            )

        return DataSet(
            values=self.values.copy(),
            locations=self.locations.copy(),
            uncertainties=uncertainties,
            name=self.name,
            units=self.units,
        )


# ------------------------------------------------------------------
# COMPATIBILITY ALIAS
# ------------------------------------------------------------------
#
# Some parts of GeoInvLab3D use ``Dataset`` while the original
# implementation used ``DataSet``.
#
# Providing the alias here keeps both APIs compatible:
#
#     from geoinvlab.core.data.dataset import DataSet
#
# and:
#
#     from geoinvlab.core.data.dataset import Dataset
#
# refer to exactly the same class.
# ------------------------------------------------------------------

Dataset = DataSet

