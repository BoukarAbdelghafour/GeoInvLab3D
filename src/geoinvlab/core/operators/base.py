"""
Base interfaces for forward operators.
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from abc import ABC, abstractmethod

import numpy as np

from geoinvlab.core.data import DataSet
from geoinvlab.core.model import Model3D


class ForwardOperator(ABC):
    """
    Abstract base class for geophysical forward operators.

    A forward operator maps a physical model to predicted
    observations:

        d_pred = F(m)
    """

    @property
    @abstractmethod
    def n_data(self) -> int:
        """Return the number of predicted observations."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, model: Model3D) -> np.ndarray:
        """Compute predicted observations."""
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, model: Model3D) -> np.ndarray:
        """
        Compute the Jacobian matrix.

        Returns
        -------
        numpy.ndarray
            Matrix with shape ``(n_data, n_parameters)``.
        """
        raise NotImplementedError

    def check_prediction(
        self,
        model: Model3D,
        data: DataSet,
    ) -> None:
        """Check that predictions are compatible with data."""

        if self.n_data != data.n_data:
            raise ValueError(
                "Number of predicted data does not match "
                "the observation dataset."
            )

        predicted = np.asarray(
            self.predict(model),
            dtype=float,
        )

        if predicted.ndim != 1:
            raise ValueError(
                "Predicted data must be one-dimensional."
            )

        if predicted.size != data.n_data:
            raise ValueError(
                "Number of predicted data does not match "
                "the observation dataset."
            )

