"""
Regularization terms for GeoInvLab3D.

MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]

This module provides regularization operators for geophysical
inverse problems.

The general Tikhonov formulation is

    Phi(m) =
        || W (G m - d) ||²
        + lambda || L (m - m_ref) ||²

where

    G       : forward/Jacobian operator
    d       : observed data
    W       : data weighting matrix
    L       : regularization operator
    m       : model parameters
    m_ref   : reference model
    lambda  : regularization parameter

For zero-order Tikhonov regularization,

    L = I

and therefore

    R(m) = ||m - m_ref||².
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from geoinvlab.core.model import Model3D


class Regularization(ABC):
    """
    Abstract base class for regularization terms.

    A regularization term defines

        R(m) = ||L (m - m_ref)||²

    where

        L     : regularization operator
        m_ref : reference model

    If no reference model is provided, the reference model is
    assumed to be the zero model.
    """

    def __init__(
        self,
        reference_model: Model3D | None = None,
    ) -> None:
        """Create a regularization term."""

        if reference_model is not None:
            if not isinstance(reference_model, Model3D):
                raise TypeError(
                    "reference_model must be a Model3D instance "
                    "or None."
                )

        self.reference_model = reference_model

    # ------------------------------------------------------------------
    # Reference model
    # ------------------------------------------------------------------

    @property
    def has_reference_model(self) -> bool:
        """Return True if an explicit reference model is configured."""

        return self.reference_model is not None

    def _reference_values(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """
        Return reference-model values compatible with ``model``.

        If no explicit reference model is provided, a zero vector
        is returned.
        """

        if not isinstance(model, Model3D):
            raise TypeError(
                "model must be a Model3D instance."
            )

        if self.reference_model is None:
            return np.zeros(
                model.n_parameters,
                dtype=float,
            )

        reference = self.reference_model

        if reference.n_parameters != model.n_parameters:
            raise ValueError(
                "Reference model must have the same number "
                "of parameters as the current model."
            )

        if reference.mesh is not model.mesh:
            raise ValueError(
                "Reference model must use the same mesh "
                "as the current model."
            )

        values = np.asarray(
            reference.values,
            dtype=float,
        )

        if values.shape != model.values.shape:
            raise ValueError(
                "Reference model values have an incompatible shape."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Reference model contains NaN or infinite values."
            )

        return values

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def value(
        self,
        model: Model3D,
    ) -> float:
        """
        Compute the regularization functional.

        The general functional is

            R(m) = ||L (m - m_ref)||².
        """

        raise NotImplementedError

    @abstractmethod
    def gradient(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """
        Compute the regularization gradient.

        Returns
        -------
        numpy.ndarray
            Gradient with respect to model parameters.
        """

        raise NotImplementedError

    @abstractmethod
    def matrix(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """
        Return the regularization operator matrix L.
        """

        raise NotImplementedError


class L2Regularization(Regularization):
    """
    Zero-order Tikhonov regularization.

    The functional is

        R(m) = ||m - m_ref||²

    with

        L = I

    and

        grad R(m) = 2 (m - m_ref).

    Parameters
    ----------
    reference_model : Model3D or None, optional
        Reference model.

        If None, the reference model is the zero model.
    """

    def __init__(
        self,
        reference_model: Model3D | None = None,
    ) -> None:
        """Create zero-order Tikhonov regularization."""

        super().__init__(
            reference_model=reference_model,
        )

    # ------------------------------------------------------------------
    # Value
    # ------------------------------------------------------------------

    def value(
        self,
        model: Model3D,
    ) -> float:
        """
        Return

            ||m - m_ref||².
        """

        reference_values = self._reference_values(
            model,
        )

        model_values = np.asarray(
            model.values,
            dtype=float,
        )

        difference = (
            model_values - reference_values
        )

        return float(
            np.dot(
                difference,
                difference,
            )
        )

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------

    def gradient(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """
        Return

            2 (m - m_ref).
        """

        reference_values = self._reference_values(
            model,
        )

        model_values = np.asarray(
            model.values,
            dtype=float,
        )

        return 2.0 * (
            model_values - reference_values
        )

    # ------------------------------------------------------------------
    # Matrix
    # ------------------------------------------------------------------

    def matrix(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """
        Return the zero-order Tikhonov matrix.

        For L2 regularization,

            L = I.
        """

        return np.eye(
            model.n_parameters,
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Reference vector
    # ------------------------------------------------------------------

    def reference_values(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """
        Return the reference model vector.

        This method is used by the linear solver when constructing
        the augmented Tikhonov least-squares system.
        """

        return self._reference_values(
            model,
        )
