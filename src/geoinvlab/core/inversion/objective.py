"""
Objective functions for geophysical inversion.

This module implements weighted least-squares objective functions
for geophysical inverse problems, including magnetotellurics (MT).
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

import numpy as np

from geoinvlab.core.data import DataSet
from geoinvlab.core.model import Model3D
from geoinvlab.core.operators import ForwardOperator
from geoinvlab.core.inversion.regularization import Regularization


class LeastSquaresObjective:
    """
    Weighted least-squares objective function.

    The data misfit is

        Phi_d(m) = || Wd * r(m) ||²

    where:

        F(m) : predicted scalar data vector
        d    : observed scalar data vector
        Wd   : inverse-uncertainty weights
        r(m) : data residual

    For standard geophysical data, the residual is

        r(m) = F(m) - d

    Operators may optionally provide a ``data_residual`` method
    to implement a problem-specific residual definition.

    This is required for magnetotelluric phase data, where residuals
    must account for the circular nature of phase angles.

    Optionally, the total objective is

        Phi(m) = Phi_d(m) + lambda * R(m)

    The objective operates on flattened scalar data vectors.

    For MT data, each station-frequency observation contributes
    eight scalar values:

        rho_xx, rho_xy, rho_yx, rho_yy,
        phase_xx, phase_xy, phase_yx, phase_yy.
    """

    def __init__(
        self,
        operator: ForwardOperator,
        data: DataSet,
        regularization: Regularization | None = None,
        regularization_parameter: float = 0.0,
    ) -> None:
        """Initialize the least-squares objective."""

        if not np.isfinite(regularization_parameter):
            raise ValueError(
                "Regularization parameter must be finite."
            )

        if regularization_parameter < 0.0:
            raise ValueError(
                "Regularization parameter must be non-negative."
            )

        self.operator = operator
        self.data = data
        self.regularization = regularization
        self.regularization_parameter = float(
            regularization_parameter
        )

        # Validate that observed values and weights describe
        # the same scalar data vector.
        observed = self._observed_values()
        weights = self._data_weights()

        if observed.size == 0:
            raise ValueError(
                "The data set must contain at least one scalar value."
            )

        if weights.size != observed.size:
            raise ValueError(
                "Data/weight dimension mismatch: "
                f"{observed.size} observed values but "
                f"{weights.size} weights."
            )

        if not np.all(np.isfinite(observed)):
            raise ValueError(
                "Observed data must contain only finite values."
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "Data weights must contain only finite values."
            )

        if np.any(weights < 0.0):
            raise ValueError(
                "Data weights must be non-negative."
            )

    # ------------------------------------------------------------------
    # Data vectors
    # ------------------------------------------------------------------

    def _observed_values(self) -> np.ndarray:
        """
        Return observed data as a one-dimensional scalar vector.

        The data container determines the component ordering.
        For MTDataSet, this is the eight-component ordering documented
        in the MT data container.
        """

        values = np.asarray(
            self.data.values,
            dtype=float,
        ).reshape(-1)

        return values

    def _data_weights(self) -> np.ndarray:
        """
        Return the weights as a one-dimensional scalar vector.

        MTDataSet.weights() returns one weight for every
        scalar MT datum.
        """

        weights = np.asarray(
            self.data.weights(),
            dtype=float,
        ).reshape(-1)

        return weights

    def _validate_prediction(
        self,
        predicted: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and return a flattened predicted data vector.
        """

        predicted = np.asarray(
            predicted,
            dtype=float,
        ).reshape(-1)

        observed = self._observed_values()

        if predicted.size != observed.size:
            raise ValueError(
                "Prediction/data dimension mismatch: "
                f"the forward operator returned {predicted.size} "
                f"scalar values, but the data set contains "
                f"{observed.size} scalar values."
            )

        if not np.all(np.isfinite(predicted)):
            raise ValueError(
                "Predicted data must contain only finite values."
            )

        return predicted

    def _predicted_values(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """
        Calculate and validate the predicted scalar data vector.
        """

        predicted = self.operator.predict(model)

        return self._validate_prediction(predicted)

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    def residual_from_prediction(
        self,
        predicted: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the residual from an existing prediction.

        The default residual convention is

            r(m) = F(m) - d

        If the forward operator implements

            data_residual(predicted, observed)

        that method is used instead.

        This optional operator-specific interface allows MT to use
        circular phase residuals without introducing MT-specific
        assumptions into the generic objective function.

        Parameters
        ----------
        predicted : ndarray
            Predicted scalar data vector.

        Returns
        -------
        ndarray
            One-dimensional residual vector.
        """

        predicted = self._validate_prediction(predicted)
        observed = self._observed_values()

        residual_method = getattr(
            self.operator,
            "data_residual",
            None,
        )

        if callable(residual_method):
            residual = residual_method(
                predicted,
                observed,
            )
        else:
            residual = predicted - observed

        residual = np.asarray(
            residual,
            dtype=float,
        ).reshape(-1)

        if residual.size != observed.size:
            raise ValueError(
                "Residual/data dimension mismatch: "
                f"{residual.size} residuals but "
                f"{observed.size} observed values."
            )

        if not np.all(np.isfinite(residual)):
            raise ValueError(
                "Data residuals must contain only finite values."
            )

        return residual

    def weighted_residual_from_prediction(
        self,
        predicted: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the weighted residual from an existing prediction.

        The weighted residual is

            rw(m) = Wd * r(m)

        where r(m) is obtained from residual_from_prediction().
        """

        residual = self.residual_from_prediction(predicted)
        weights = self._data_weights()

        if residual.size != weights.size:
            raise ValueError(
                "Residual/weight dimension mismatch: "
                f"{residual.size} residuals but "
                f"{weights.size} weights."
            )

        return residual * weights

    def residual(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """
        Return the unweighted data residual.

        The residual convention is

            r(m) = F(m) - d

        or the operator-specific equivalent when available.
        """

        predicted = self._predicted_values(model)

        return self.residual_from_prediction(predicted)

    def weighted_residual(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """
        Return the weighted data residual.

        The weighted residual is

            rw(m) = Wd * r(m)

        where Wd contains the data weights.
        """

        predicted = self._predicted_values(model)

        return self.weighted_residual_from_prediction(
            predicted
        )

    # ------------------------------------------------------------------
    # Objective contributions
    # ------------------------------------------------------------------

    def data_misfit_from_prediction(
        self,
        predicted: np.ndarray,
    ) -> float:
        """
        Calculate the data misfit from an existing prediction.

        Phi_d(m) = || Wd * r(m) ||²
        """

        residual = self.weighted_residual_from_prediction(
            predicted
        )

        return float(
            np.dot(
                residual,
                residual,
            )
        )

    def data_misfit(
        self,
        model: Model3D,
    ) -> float:
        """
        Return the weighted least-squares data misfit.

        Phi_d(m) = || Wd * r(m) ||²
        """

        predicted = self._predicted_values(model)

        return self.data_misfit_from_prediction(
            predicted
        )

    def regularization_value(
        self,
        model: Model3D,
    ) -> float:
        """
        Return the regularization contribution before scaling
        by the regularization parameter.
        """

        if self.regularization is None:
            return 0.0

        value = float(
            self.regularization.value(model)
        )

        if not np.isfinite(value):
            raise ValueError(
                "Regularization returned a non-finite value."
            )

        if value < 0.0:
            raise ValueError(
                "Regularization value must be non-negative."
            )

        return value

    def value_from_prediction(
        self,
        model: Model3D,
        predicted: np.ndarray,
    ) -> float:
        """
        Calculate the total objective from an existing prediction.

        This method avoids an additional forward calculation when
        the predicted data are already available.

        Phi(m) = Phi_d(m) + lambda * R(m)
        """

        misfit = self.data_misfit_from_prediction(
            predicted
        )

        regularization = (
            self.regularization_parameter
            * self.regularization_value(model)
        )

        result = misfit + regularization

        if not np.isfinite(result):
            raise ValueError(
                "The objective function returned a non-finite value."
            )

        return float(result)

    def value(
        self,
        model: Model3D,
    ) -> float:
        """
        Return the total objective value.

        Phi(m) = Phi_d(m) + lambda * R(m)
        """

        predicted = self._predicted_values(model)

        return self.value_from_prediction(
            model,
            predicted,
        )
