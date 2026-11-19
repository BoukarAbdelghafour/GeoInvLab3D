"""
Linear forward operator.

This module provides a linear forward operator defined by
a sensitivity matrix G.

The forward problem is:

    d = G m
    MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""
from dataclasses import dataclass

import numpy as np

from geoinvlab.core.model import Model3D
from geoinvlab.core.operators.base import ForwardOperator


@dataclass
class LinearOperator(ForwardOperator):
    """
    Linear forward operator.

    Parameters
    ----------
    matrix : numpy.ndarray
        Sensitivity matrix G with shape
        ``(n_data, n_parameters)``.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        """Validate the sensitivity matrix."""

        self.matrix = np.asarray(
            self.matrix,
            dtype=float,
        )

        if self.matrix.ndim != 2:
            raise ValueError(
                "Sensitivity matrix must be two-dimensional."
            )

        if not np.all(np.isfinite(self.matrix)):
            raise ValueError(
                "Sensitivity matrix must contain only finite values."
            )

    @property
    def n_data(self) -> int:
        """Return the number of observations."""

        return self.matrix.shape[0]

    @property
    def n_parameters(self) -> int:
        """Return the number of model parameters."""

        return self.matrix.shape[1]

    def predict(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """
        Compute predicted observations.

        Parameters
        ----------
        model : Model3D
            Physical model.

        Returns
        -------
        numpy.ndarray
            Predicted observations.
        """

        if model.n_parameters != self.n_parameters:
            raise ValueError(
                "Number of model parameters does not match "
                "the number of columns of the sensitivity matrix."
            )

        return self.matrix @ model.values

    def jacobian(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """
        Return the Jacobian matrix.

        Parameters
        ----------
        model : Model3D
            Physical model.

        Returns
        -------
        numpy.ndarray
            Sensitivity matrix.
        """

        if model.n_parameters != self.n_parameters:
            raise ValueError(
                "Number of model parameters does not match "
                "the number of columns of the sensitivity matrix."
            )

        return self.matrix.copy()

