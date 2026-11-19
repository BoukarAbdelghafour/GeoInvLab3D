"""
Registry for available geophysical methods.

The registry provides a central mechanism for discovering and
creating geophysical methods such as:

    - gravity
    - magnetic
    - mt
    - electric

The registry itself does not perform any inversion. It only
manages the available physical methods.

Typical workflow
----------------

    registry = create_default_registry()

    method = registry.get("gravity")

    operator = method.create_operator(
        observation_locations
    )

The inversion engine remains completely independent from the
specific geophysical method.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from geoinvlab.methods.base import GeophysicalMethod


class MethodRegistry:
    """Registry of available geophysical methods."""

    def __init__(self) -> None:
        """Create an empty method registry."""

        self._methods: dict[
            str,
            GeophysicalMethod,
        ] = {}

    # ==============================================================
    # REGISTRATION
    # ==============================================================

    def register(
        self,
        method: GeophysicalMethod,
        *,
        replace: bool = False,
    ) -> None:
        """Register a geophysical method."""

        if not isinstance(
            method,
            GeophysicalMethod,
        ):
            raise TypeError(
                "method must be a GeophysicalMethod instance."
            )

        identifier = method.identifier

        if not isinstance(identifier, str):
            raise TypeError(
                "Method identifier must be a string."
            )

        identifier = identifier.strip().lower()

        if not identifier:
            raise ValueError(
                "Method identifier must not be empty."
            )

        if (
            identifier in self._methods
            and not replace
        ):
            raise ValueError(
                f"Method '{identifier}' "
                "is already registered."
            )

        self._methods[identifier] = method

    # ==============================================================
    # UNREGISTER
    # ==============================================================

    def unregister(
        self,
        identifier: str,
    ) -> None:
        """Remove a method from the registry."""

        identifier = self._normalize_identifier(
            identifier
        )

        try:
            del self._methods[identifier]

        except KeyError as exc:
            raise KeyError(
                f"Unknown geophysical method: "
                f"'{identifier}'."
            ) from exc

    # ==============================================================
    # GET
    # ==============================================================

    def get(
        self,
        identifier: str,
    ) -> GeophysicalMethod:
        """Return a registered method."""

        identifier = self._normalize_identifier(
            identifier
        )

        try:
            return self._methods[identifier]

        except KeyError as exc:
            raise KeyError(
                f"Unknown geophysical method: "
                f"'{identifier}'. "
                f"Available methods: "
                f"{', '.join(self.identifiers())}"
            ) from exc

    # ==============================================================
    # OPTIONAL GET
    # ==============================================================

    def find(
        self,
        identifier: str,
    ) -> GeophysicalMethod | None:
        """Return a method if registered, otherwise None."""

        identifier = self._normalize_identifier(
            identifier
        )

        return self._methods.get(
            identifier
        )

    # ==============================================================
    # AVAILABLE
    # ==============================================================

    def available(
        self,
    ) -> list[GeophysicalMethod]:
        """Return all registered methods."""

        return list(
            self._methods.values()
        )

    # ==============================================================
    # IDENTIFIERS
    # ==============================================================

    def identifiers(self) -> list[str]:
        """Return identifiers of all registered methods."""

        return list(
            self._methods.keys()
        )

    # ==============================================================
    # DISPLAY NAMES
    # ==============================================================

    def names(self) -> list[str]:
        """Return human-readable names."""

        return [
            method.name
            for method in self._methods.values()
        ]

    # ==============================================================
    # CONTAINS
    # ==============================================================

    def __contains__(
        self,
        identifier: str,
    ) -> bool:
        """Return whether a method is registered."""

        identifier = self._normalize_identifier(
            identifier
        )

        return identifier in self._methods

    # ==============================================================
    # LENGTH
    # ==============================================================

    def __len__(self) -> int:
        """Return the number of registered methods."""

        return len(
            self._methods
        )

    # ==============================================================
    # ITERATION
    # ==============================================================

    def __iter__(self):
        """Iterate over registered methods."""

        return iter(
            self._methods.values()
        )

    # ==============================================================
    # REPRESENTATION
    # ==============================================================

    def __repr__(self) -> str:
        """Return a readable representation."""

        identifiers = ", ".join(
            self.identifiers()
        )

        return (
            f"MethodRegistry("
            f"methods=[{identifiers}]"
            f")"
        )

    # ==============================================================
    # NORMALIZATION
    # ==============================================================

    @staticmethod
    def _normalize_identifier(
        identifier: str,
    ) -> str:
        """Normalize a method identifier."""

        if not isinstance(
            identifier,
            str,
        ):
            raise TypeError(
                "Method identifier must be a string."
            )

        identifier = identifier.strip().lower()

        if not identifier:
            raise ValueError(
                "Method identifier must not be empty."
            )

        return identifier


# ==================================================================
# DEFAULT REGISTRY
# ==================================================================

def create_default_registry() -> MethodRegistry:
    """
    Create the default registry.

    Currently available:

        - Gravity
        - Magnetic
        - MT

    MT is registered so that the application can recognize and
    configure the method. Its actual forward modelling and
    inversion will be implemented separately.
    """

    registry = MethodRegistry()

    # --------------------------------------------------------------
    # Gravity
    # --------------------------------------------------------------

    from geoinvlab.methods.gravity import GravityMethod

    registry.register(
        GravityMethod()
    )

    # --------------------------------------------------------------
    # Magnetic
    # --------------------------------------------------------------

    from geoinvlab.methods.magnetic import MagneticMethod

    magnetic = MagneticMethod(
    inclination=60.0,
    declination=0.0,
    field_intensity=50000.0,
    )

    registry.register(
    magnetic
    )

    # --------------------------------------------------------------
    # Magnetotelluric (MT)
    # --------------------------------------------------------------

    from geoinvlab.methods.mt import MTMethod

    registry.register(
        MTMethod()
    )

    return registry


# ==================================================================
# GLOBAL DEFAULT REGISTRY
# ==================================================================

_default_registry: MethodRegistry | None = None


def get_default_registry() -> MethodRegistry:
    """
    Return the global default method registry.

    The registry is created lazily on the first call.
    """

    global _default_registry

    if _default_registry is None:
        _default_registry = (
            create_default_registry()
        )

    return _default_registry


def reset_default_registry() -> None:
    """
    Reset the global default registry.

    Mainly useful for tests and application
    reinitialization.
    """

    global _default_registry

    _default_registry = None
