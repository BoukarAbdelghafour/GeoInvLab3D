"""
Base interfaces for geophysical methods.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from abc import ABC, abstractmethod


class GeophysicalMethod(ABC):
    """
    Abstract base class for a geophysical method.

    A geophysical method defines the physics-specific part
    of a GeoInvLab3D workflow.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable method name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Return the unique method identifier."""
        raise NotImplementedError

