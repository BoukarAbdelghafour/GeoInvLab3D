"""
MT station field observation and extraction.

This module maps discrete electric and magnetic fields onto explicit
MT observations.

Station indices are COMPONENT-LOCAL:

    Ex -> [0, n_ex)
    Ey -> [0, n_ey)
    Hx -> [0, n_fx)
    Hy -> [0, n_fy)

The global vectors are ordered as

    E = [Ex, Ey, Ez]
    H = [Hx, Hy, Hz]

Therefore, local component indices are converted to global indices
only when values are extracted from the concatenated field vectors.

Time convention
---------------
The MT implementation uses

    exp(+i * omega * t)

and therefore

    curl(E) = -i * omega * mu * H.
     MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from .field import EdgeElectricField, FaceMagneticField


# ============================================================================
# MT observation
# ============================================================================


@dataclass(frozen=True)
class MTObservation:
    """
    Explicit mapping of one MT station onto discrete field degrees
    of freedom.

    All indices are LOCAL to their corresponding field component.

    Parameters
    ----------
    ex_index:
        Local index in the Ex edge component.

    ey_index:
        Local index in the Ey edge component.

    hx_index:
        Local index in the Hx/Fx face component.

    hy_index:
        Local index in the Hy/Fy face component.
    """

    ex_index: int
    ey_index: int
    hx_index: int
    hy_index: int

    def __post_init__(self) -> None:
        indices = (
            ("ex_index", self.ex_index),
            ("ey_index", self.ey_index),
            ("hx_index", self.hx_index),
            ("hy_index", self.hy_index),
        )

        for name, index in indices:
            if not isinstance(index, (int, np.integer)):
                raise ValueError(
                    f"{name} must be an integer."
                )

            index = int(index)

            if index < 0:
                raise ValueError(
                    f"{name} must be non-negative."
                )

        object.__setattr__(
            self,
            "ex_index",
            int(self.ex_index),
        )
        object.__setattr__(
            self,
            "ey_index",
            int(self.ey_index),
        )
        object.__setattr__(
            self,
            "hx_index",
            int(self.hx_index),
        )
        object.__setattr__(
            self,
            "hy_index",
            int(self.hy_index),
        )


# ============================================================================
# Observation set
# ============================================================================


@dataclass(frozen=True)
class MTObservationSet:
    """
    Immutable collection of MT observations.
    """

    observations: tuple[MTObservation, ...]

    def __init__(
        self,
        observations: Sequence[MTObservation]
        | tuple[MTObservation, ...],
    ) -> None:
        try:
            values = tuple(observations)
        except TypeError as exc:
            raise TypeError(
                "observations must be an iterable of "
                "MTObservation objects."
            ) from exc

        if len(values) == 0:
            raise ValueError(
                "MTObservationSet must not be empty."
            )

        for observation in values:
            if not isinstance(
                observation,
                MTObservation,
            ):
                raise TypeError(
                    "MTObservationSet can contain only "
                    "MTObservation objects."
                )

        object.__setattr__(
            self,
            "observations",
            values,
        )

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self) -> Iterator[MTObservation]:
        return iter(self.observations)

    def __getitem__(
        self,
        index: int | slice,
    ) -> MTObservation | tuple[MTObservation, ...]:
        return self.observations[index]

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    @property
    def size(self) -> int:
        return len(self.observations)

    def __bool__(self) -> bool:
        return True


# ============================================================================
# Observed fields
# ============================================================================


@dataclass(frozen=True)
class MTObservedFields:
    """
    Observed MT field components at one station.
    """

    ex: complex
    ey: complex
    hx: complex
    hy: complex

    def __post_init__(self) -> None:
        values = (
            ("ex", self.ex),
            ("ey", self.ey),
            ("hx", self.hx),
            ("hy", self.hy),
        )

        converted: dict[str, complex] = {}

        for name, value in values:
            value = complex(value)

            if not (
                np.isfinite(value.real)
                and np.isfinite(value.imag)
            ):
                raise ValueError(
                    f"{name} must be finite."
                )

            converted[name] = value

        object.__setattr__(
            self,
            "ex",
            converted["ex"],
        )
        object.__setattr__(
            self,
            "ey",
            converted["ey"],
        )
        object.__setattr__(
            self,
            "hx",
            converted["hx"],
        )
        object.__setattr__(
            self,
            "hy",
            converted["hy"],
        )

    @property
    def electric(self) -> np.ndarray:
        """
        Return the horizontal electric field:

            [Ex, Ey]
        """
        return np.asarray(
            [self.ex, self.ey],
            dtype=complex,
        )

    @property
    def magnetic(self) -> np.ndarray:
        """
        Return the horizontal magnetic field:

            [Hx, Hy]
        """
        return np.asarray(
            [self.hx, self.hy],
            dtype=complex,
        )


# ============================================================================
# Constructors
# ============================================================================


def make_observation(
    ex_index: int,
    ey_index: int,
    hx_index: int,
    hy_index: int,
) -> MTObservation:
    """
    Construct one MT observation.

    All indices are component-local.
    """
    return MTObservation(
        ex_index=ex_index,
        ey_index=ey_index,
        hx_index=hx_index,
        hy_index=hy_index,
    )


def make_observation_set(
    observations: Sequence[MTObservation],
) -> MTObservationSet:
    """
    Construct an MT observation set.
    """
    return MTObservationSet(observations)


# ============================================================================
# Validation
# ============================================================================


def validate_observation(
    observation: MTObservation,
    n_edges: int,
    n_faces: int,
    *,
    n_ex: int | None = None,
    n_ey: int | None = None,
    n_fx: int | None = None,
    n_fy: int | None = None,
) -> MTObservation:
    """
    Validate an MT observation.

    Parameters
    ----------
    observation:
        Observation to validate.

    n_edges:
        Total number of electric-field edges.

    n_faces:
        Total number of magnetic-field faces.

    n_ex, n_ey:
        Number of Ex and Ey degrees of freedom.

    n_fx, n_fy:
        Number of Hx/Fx and Hy/Fy degrees of freedom.

    Notes
    -----
    When component sizes are supplied, indices are interpreted as
    LOCAL component indices.

    Without component sizes, the function retains the historical
    global-index validation mode for backward compatibility.
    """
    if not isinstance(
        observation,
        MTObservation,
    ):
        raise TypeError(
            "observation must be an MTObservation."
        )

    n_edges = int(n_edges)
    n_faces = int(n_faces)

    if n_edges < 0:
        raise ValueError(
            "n_edges must be non-negative."
        )

    if n_faces < 0:
        raise ValueError(
            "n_faces must be non-negative."
        )

    component_mode = any(
        value is not None
        for value in (
            n_ex,
            n_ey,
            n_fx,
            n_fy,
        )
    )

    # ------------------------------------------------------------------
    # Component-local validation.
    # ------------------------------------------------------------------
    if component_mode:
        if n_ex is None or n_ey is None:
            raise ValueError(
                "n_ex and n_ey must be supplied together."
            )

        if n_fx is None or n_fy is None:
            raise ValueError(
                "n_fx and n_fy must be supplied together."
            )

        n_ex = int(n_ex)
        n_ey = int(n_ey)
        n_fx = int(n_fx)
        n_fy = int(n_fy)

        if min(
            n_ex,
            n_ey,
            n_fx,
            n_fy,
        ) < 0:
            raise ValueError(
                "Component sizes must be non-negative."
            )

        if n_ex + n_ey > n_edges:
            raise ValueError(
                "n_ex + n_ey exceeds the total number "
                "of electric-field edges."
            )

        if n_fx + n_fy > n_faces:
            raise ValueError(
                "n_fx + n_fy exceeds the total number "
                "of magnetic-field faces."
            )

        component_indices = (
            (
                "ex_index",
                observation.ex_index,
                n_ex,
            ),
            (
                "ey_index",
                observation.ey_index,
                n_ey,
            ),
            (
                "hx_index",
                observation.hx_index,
                n_fx,
            ),
            (
                "hy_index",
                observation.hy_index,
                n_fy,
            ),
        )

        for name, index, size in component_indices:
            if index >= size:
                raise ValueError(
                    f"{name}={index} is outside its "
                    f"field component with {size} "
                    "degrees of freedom."
                )

        return observation

    # ------------------------------------------------------------------
    # Backward-compatible global-index validation.
    # ------------------------------------------------------------------
    for name, index in (
        ("ex_index", observation.ex_index),
        ("ey_index", observation.ey_index),
    ):
        if index >= n_edges:
            raise ValueError(
                f"{name}={index} is outside the "
                f"electric field with {n_edges} edges."
            )

    for name, index in (
        ("hx_index", observation.hx_index),
        ("hy_index", observation.hy_index),
    ):
        if index >= n_faces:
            raise ValueError(
                f"{name}={index} is outside the "
                f"magnetic field with {n_faces} faces."
            )

    return observation


# ============================================================================
# Field normalization
# ============================================================================


def _electric_values(
    electric_field: EdgeElectricField,
) -> np.ndarray:
    """
    Return electric-field values as a one-dimensional complex array.
    """
    if not isinstance(
        electric_field,
        EdgeElectricField,
    ):
        raise TypeError(
            "electric_field must be an EdgeElectricField."
        )

    electric = np.asarray(
        electric_field.values,
        dtype=complex,
    )

    if electric.ndim != 1:
        raise ValueError(
            "electric_field.values must be one-dimensional."
        )

    if not np.all(np.isfinite(electric)):
        raise ValueError(
            "electric_field.values must contain finite values."
        )

    return electric


def _magnetic_values(
    magnetic_field: FaceMagneticField | np.ndarray,
) -> np.ndarray:
    """
    Return magnetic-field values as a one-dimensional complex array.

    A FaceMagneticField preserves component metadata.

    A raw NumPy array is accepted for backward compatibility, but
    then its indices are interpreted as global indices.
    """
    if isinstance(
        magnetic_field,
        FaceMagneticField,
    ):
        magnetic = np.asarray(
            magnetic_field.values,
            dtype=complex,
        )

    elif isinstance(
        magnetic_field,
        np.ndarray,
    ):
        magnetic = np.asarray(
            magnetic_field,
            dtype=complex,
        )

    else:
        raise TypeError(
            "magnetic_field must be a FaceMagneticField "
            "or a NumPy array."
        )

    if magnetic.ndim != 1:
        raise ValueError(
            "magnetic field values must be one-dimensional."
        )

    if not np.all(np.isfinite(magnetic)):
        raise ValueError(
            "magnetic field values must contain finite values."
        )

    return magnetic


# ============================================================================
# Component sizes
# ============================================================================


def _electric_component_sizes(
    electric_field: EdgeElectricField,
) -> tuple[int, int, int]:
    """
    Return:

        (n_ex, n_ey, n_ez)
    """
    if not isinstance(
        electric_field,
        EdgeElectricField,
    ):
        raise TypeError(
            "electric_field must be an EdgeElectricField."
        )

    try:
        n_ex = int(electric_field.n_ex)
        n_ey = int(electric_field.n_ey)
        n_ez = int(electric_field.n_ez)
    except AttributeError as exc:
        raise TypeError(
            "EdgeElectricField must expose "
            "n_ex, n_ey and n_ez."
        ) from exc

    if min(
        n_ex,
        n_ey,
        n_ez,
    ) < 0:
        raise ValueError(
            "Electric-field component sizes must "
            "be non-negative."
        )

    expected = n_ex + n_ey + n_ez

    if expected != electric_field.values.size:
        raise ValueError(
            "Electric-field component sizes do not "
            "match electric_field.values."
        )

    return n_ex, n_ey, n_ez


def _magnetic_component_sizes(
    magnetic_field: FaceMagneticField,
) -> tuple[int, int, int]:
    """
    Return:

        (n_fx, n_fy, n_fz)
    """
    if not isinstance(
        magnetic_field,
        FaceMagneticField,
    ):
        raise TypeError(
            "magnetic_field must be a FaceMagneticField."
        )

    try:
        n_fx = int(magnetic_field.n_fx)
        n_fy = int(magnetic_field.n_fy)
        n_fz = int(magnetic_field.n_fz)
    except AttributeError as exc:
        raise TypeError(
            "FaceMagneticField must expose "
            "n_fx, n_fy and n_fz."
        ) from exc

    if min(
        n_fx,
        n_fy,
        n_fz,
    ) < 0:
        raise ValueError(
            "Magnetic-field component sizes must "
            "be non-negative."
        )

    expected = n_fx + n_fy + n_fz

    if expected != magnetic_field.values.size:
        raise ValueError(
            "Magnetic-field component sizes do not "
            "match magnetic_field.values."
        )

    return n_fx, n_fy, n_fz


# ============================================================================
# Local -> global index conversion
# ============================================================================


def _global_electric_indices(
    observation: MTObservation,
    n_ex: int,
) -> tuple[int, int]:
    """
    Convert local Ex/Ey indices to global E indices.

    Global ordering:

        E = [Ex | Ey | Ez]
    """
    ex_global = observation.ex_index

    ey_global = n_ex + observation.ey_index

    return ex_global, ey_global


def _global_magnetic_indices(
    observation: MTObservation,
    n_fx: int,
) -> tuple[int, int]:
    """
    Convert local Hx/Hy indices to global H indices.

    Global ordering:

        H = [Hx | Hy | Hz]
    """
    hx_global = observation.hx_index

    hy_global = n_fx + observation.hy_index

    return hx_global, hy_global


# ============================================================================
# Field observation
# ============================================================================


def observe_fields(
    electric_field: EdgeElectricField,
    magnetic_field_values: FaceMagneticField | np.ndarray,
    observation: MTObservation,
) -> MTObservedFields:
    """
    Extract Ex, Ey, Hx and Hy at one MT station.

    For structured field wrappers, observation indices are LOCAL
    component indices and are converted to GLOBAL indices here.

    Electric ordering:

        E = [Ex | Ey | Ez]

    Magnetic ordering:

        H = [Hx | Hy | Hz]

    Parameters
    ----------
    electric_field:
        Structured edge electric field.

    magnetic_field_values:
        Structured face magnetic field or, for backward compatibility,
        a raw one-dimensional NumPy array.

    observation:
        MT station observation using component-local indices.
    """
    if not isinstance(
        electric_field,
        EdgeElectricField,
    ):
        raise TypeError(
            "electric_field must be an EdgeElectricField."
        )

    if not isinstance(
        observation,
        MTObservation,
    ):
        raise TypeError(
            "observation must be an MTObservation."
        )

    electric = _electric_values(
        electric_field
    )

    n_ex, n_ey, _ = _electric_component_sizes(
        electric_field
    )

    # ------------------------------------------------------------------
    # Preferred path: structured magnetic field.
    #
    # This is the path that MUST be used by the 3-D MT forward solver,
    # because it retains n_fx/n_fy/n_fz metadata.
    # ------------------------------------------------------------------
    if isinstance(
        magnetic_field_values,
        FaceMagneticField,
    ):
        magnetic = _magnetic_values(
            magnetic_field_values
        )

        n_fx, n_fy, _ = _magnetic_component_sizes(
            magnetic_field_values
        )

        validate_observation(
            observation=observation,
            n_edges=electric.size,
            n_faces=magnetic.size,
            n_ex=n_ex,
            n_ey=n_ey,
            n_fx=n_fx,
            n_fy=n_fy,
        )

        ex_global, ey_global = _global_electric_indices(
            observation=observation,
            n_ex=n_ex,
        )

        hx_global, hy_global = _global_magnetic_indices(
            observation=observation,
            n_fx=n_fx,
        )

    # ------------------------------------------------------------------
    # Backward-compatible raw-array path.
    #
    # A raw magnetic array contains no component-size metadata.
    # Therefore the observation indices are interpreted as GLOBAL
    # indices in this case.
    # ------------------------------------------------------------------
    else:
        magnetic = _magnetic_values(
            magnetic_field_values
        )

        validate_observation(
            observation=observation,
            n_edges=electric.size,
            n_faces=magnetic.size,
        )

        ex_global = observation.ex_index
        ey_global = observation.ey_index

        hx_global = observation.hx_index
        hy_global = observation.hy_index

    # ------------------------------------------------------------------
    # Final global bounds checks.
    # ------------------------------------------------------------------
    global_indices = (
        ("Ex", ex_global, electric.size),
        ("Ey", ey_global, electric.size),
        ("Hx", hx_global, magnetic.size),
        ("Hy", hy_global, magnetic.size),
    )

    for name, index, size in global_indices:
        if not (
            0 <= index < size
        ):
            raise ValueError(
                f"{name} global index {index} is "
                f"outside the corresponding field "
                f"of size {size}."
            )

    # ------------------------------------------------------------------
    # Extract physical field components.
    # ------------------------------------------------------------------
    return MTObservedFields(
        ex=electric[ex_global],
        ey=electric[ey_global],
        hx=magnetic[hx_global],
        hy=magnetic[hy_global],
    )


# ============================================================================
# Multiple observations
# ============================================================================


def observe_multiple(
    electric_field: EdgeElectricField,
    magnetic_field_values: FaceMagneticField | np.ndarray,
    observations: MTObservationSet
    | Sequence[MTObservation],
) -> tuple[MTObservedFields, ...]:
    """
    Extract MT fields for all stations.
    """
    if isinstance(
        observations,
        MTObservationSet,
    ):
        observation_set = observations
    else:
        observation_set = MTObservationSet(
            observations
        )

    return tuple(
        observe_fields(
            electric_field=electric_field,
            magnetic_field_values=magnetic_field_values,
            observation=observation,
        )
        for observation in observation_set
    )


def observe_field_set(
    electric_field: EdgeElectricField,
    magnetic_field_values: FaceMagneticField | np.ndarray,
    observations: MTObservationSet
    | Sequence[MTObservation],
) -> tuple[MTObservedFields, ...]:
    """
    Backward-compatible alias for :func:`observe_multiple`.
    """
    return observe_multiple(
        electric_field=electric_field,
        magnetic_field_values=magnetic_field_values,
        observations=observations,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "MTObservation",
    "MTObservationSet",
    "MTObservedFields",
    "make_observation",
    "make_observation_set",
    "validate_observation",
    "observe_fields",
    "observe_multiple",
    "observe_field_set",
]
