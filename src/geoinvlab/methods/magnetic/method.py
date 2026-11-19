"""
Magnetic method implementation.

MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from dataclasses import dataclass

import numpy as np

from geoinvlab.core.operators import ForwardOperator
from geoinvlab.methods.base import GeophysicalMethod
from geoinvlab.methods.magnetic.operator import MagneticOperator


@dataclass
class MagneticMethod(GeophysicalMethod):
    """
    Magnetic inversion method.

    The model parameter represents magnetic susceptibility.

    The magnetic forward problem is linear with respect
    to the susceptibility when the inducing magnetic
    field and the mesh geometry are fixed.

    Parameters
    ----------
    inclination : float
        Inclination of the inducing magnetic field in degrees.

    declination : float
        Declination of the inducing magnetic field in degrees.

    field_intensity : float
        Intensity of the inducing magnetic field in nT.
    """

    inclination: float
    declination: float
    field_intensity: float

    def __post_init__(self) -> None:
        """
        Validate magnetic field parameters.
        """

        self.inclination = float(
            self.inclination
        )

        self.declination = float(
            self.declination
        )

        self.field_intensity = float(
            self.field_intensity
        )

        # --------------------------------------------------------------
        # Validate inclination.
        # --------------------------------------------------------------

        if not np.isfinite(
            self.inclination
        ):
            raise ValueError(
                "Inclination must be finite."
            )

        # --------------------------------------------------------------
        # Validate declination.
        # --------------------------------------------------------------

        if not np.isfinite(
            self.declination
        ):
            raise ValueError(
                "Declination must be finite."
            )

        # --------------------------------------------------------------
        # Validate field intensity.
        # --------------------------------------------------------------

        if not np.isfinite(
            self.field_intensity
        ):
            raise ValueError(
                "Field intensity must be finite."
            )

        if self.field_intensity < 0.0:
            raise ValueError(
                "Field intensity must be non-negative."
            )

    @property
    def name(self) -> str:
        """
        Return the human-readable method name.
        """

        return "Magnetic"

    @property
    def identifier(self) -> str:
        """
        Return the unique method identifier.
        """

        return "magnetic"

    def create_operator(
        self,
        observation_locations: np.ndarray,
    ) -> ForwardOperator:
        """
        Create a magnetic forward operator.

        Parameters
        ----------
        observation_locations : numpy.ndarray
            Observation coordinates with shape
            ``(n_data, 3)``.

        Returns
        -------
        MagneticOperator
            Magnetic forward operator.
        """

        return MagneticOperator(
            observation_locations=observation_locations,
            inclination=self.inclination,
            declination=self.declination,
            field_intensity=self.field_intensity,
        )

