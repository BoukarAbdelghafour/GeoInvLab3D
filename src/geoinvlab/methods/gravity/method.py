"""
Gravity method implementation.

MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from geoinvlab.methods.base import GeophysicalMethod
from geoinvlab.core.operators import ForwardOperator

from geoinvlab.methods.gravity.operator import GravityOperator


class GravityMethod(GeophysicalMethod):
    """
    Gravimetric inversion method.

    The model parameter represents density contrast.
    """

    @property
    def name(self) -> str:
        """Return the human-readable method name."""

        return "Gravity"

    @property
    def identifier(self) -> str:
        """Return the unique method identifier."""

        return "gravity"

    def create_operator(
        self,
        observation_locations,
    ) -> ForwardOperator:
        """
        Create a gravity forward operator.

        Parameters
        ----------
        observation_locations : numpy.ndarray
            Observation coordinates with shape (n_data, 3).

        Returns
        -------
        GravityOperator
            Gravity forward operator.
        """

        return GravityOperator(
            observation_locations=observation_locations,
        )

