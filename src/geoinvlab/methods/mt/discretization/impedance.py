"""Magnetotelluric impedance utilities.

GeoInvLab3D uses the ``exp(+i * omega * t)`` convention and the tensor
ordering ``[[Zxx, Zxy], [Zyx, Zyy]]``. Apparent resistivity is
``rho_a = abs(Z)**2 / (MU0 * omega)``.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MU0 = 4.0e-7 * np.pi


def _finite_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=complex)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite; NaN and infinity are not allowed.")
    return arr


def _check_tolerance(value: float, name: str = "denominator_tolerance") -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


def _safe_ratio(numerator: Any, denominator: Any, tolerance: float, name: str) -> np.ndarray:
    num = _finite_array(numerator, "Electric field")
    den = _finite_array(denominator, "Magnetic field")
    try:
        num, den = np.broadcast_arrays(num, den)
    except ValueError as exc:
        raise ValueError("Field components must have broadcast-compatible shapes.") from exc
    scale = np.maximum(1.0, np.abs(den))
    if np.any(np.abs(den) <= tolerance * scale):
        raise ValueError(f"{name} is too small to compute impedance safely (zero or below denominator_tolerance).")
    return num / den


@dataclass(frozen=True)
class MTImpedance:
    """One finite complex 2 x 2 MT impedance tensor."""
    zxx: complex
    zxy: complex
    zyx: complex
    zyy: complex

    def __post_init__(self) -> None:
        vals = tuple(complex(v) for v in (self.zxx, self.zxy, self.zyx, self.zyy))
        if not all(np.isfinite(v.real) and np.isfinite(v.imag) for v in vals):
            raise ValueError("All impedance components must be finite.")
        for name, value in zip(("zxx", "zxy", "zyx", "zyy"), vals):
            object.__setattr__(self, name, value)

    @property
    def tensor(self) -> np.ndarray:
        return np.array([[self.zxx, self.zxy], [self.zyx, self.zyy]], dtype=complex)

    @property
    def matrix(self) -> np.ndarray:
        return self.tensor

    @property
    def phase(self) -> np.ndarray:
        return np.degrees(np.angle(self.tensor))

    @property
    def phase_tensor(self) -> np.ndarray:
        x, y = self.tensor.real, self.tensor.imag
        scale = max(float(np.linalg.norm(x, ord=np.inf)), 1.0)
        if abs(np.linalg.det(x)) <= 100.0 * np.finfo(float).eps * scale:
            raise ValueError("Cannot compute the MT phase tensor: the real part of the impedance is singular or ill-conditioned.")
        return np.linalg.solve(x, y)

    def as_array(self) -> np.ndarray:
        return self.tensor.copy()

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.tensor, dtype=dtype)


@dataclass(frozen=True, init=False)
class MTResponse:
    """Response for one impedance tensor and frequency."""
    impedance: MTImpedance
    frequency: float

    def __init__(self, impedance: MTImpedance, frequency: float) -> None:
        if not isinstance(impedance, MTImpedance):
            raise TypeError("impedance must be an MTImpedance instance.")
        frequency = float(frequency)
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("frequency must be finite and strictly positive.")
        object.__setattr__(self, "impedance", impedance)
        object.__setattr__(self, "frequency", frequency)

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.frequency

    @property
    def apparent_resistivity(self) -> MTImpedance:
        """Apparent-resistivity tensor derived componentwise from impedance."""
        result = apparent_resistivity(self.impedance.tensor, self.omega)
        return MTImpedance(result[0, 0], result[0, 1], result[1, 0], result[1, 1])

    @property
    def resistivity(self) -> MTImpedance:
        return self.apparent_resistivity

    @property
    def rho(self) -> MTImpedance:
        return self.apparent_resistivity

    @property
    def apparent_resistivity_tensor(self) -> MTImpedance:
        return self.apparent_resistivity

    @property
    def phase(self) -> np.ndarray:
        return self.impedance.phase

    @property
    def phase_degrees(self) -> np.ndarray:
        return self.phase

    def phase_tensor(self) -> np.ndarray:
        return self.impedance.phase_tensor


def impedance_from_fields(
    ex: Any,
    ey: Any,
    hx: Any | None = None,
    hy: Any | None = None,
    *,
    denominator_tolerance: float = 1.0e-12,
    condition_tolerance: float | None = None,
) -> Any:
    """Calculate impedance from fields.

    Four field arguments calculate component ratios ``Zxx=Ex/Hx``,
    ``Zxy=Ex/Hy``, ``Zyx=Ey/Hx``, ``Zyy=Ey/Hy``. Scalar inputs return an
    :class:`MTImpedance`; array inputs return an array with trailing shape
    ``(2, 2)``. For compatibility, two observation objects (each exposing
    ``ex, ey, hx, hy``) are also accepted and solved as ``Z = E H**-1``.
    """
    tol = _check_tolerance(denominator_tolerance)
    if condition_tolerance is not None:
        tol = _check_tolerance(condition_tolerance, "condition_tolerance")

    # Legacy API: impedance_from_fields(observation_x, observation_y).
    if hx is None and hy is None and all(hasattr(ex, n) for n in ("ex", "ey", "hx", "hy")) and all(hasattr(ey, n) for n in ("ex", "ey", "hx", "hy")):
        ox, oy = ex, ey
        e = np.array([[ox.ex, oy.ex], [ox.ey, oy.ey]], dtype=complex)
        h = np.array([[ox.hx, oy.hx], [ox.hy, oy.hy]], dtype=complex)
        if not np.all(np.isfinite(e)) or not np.all(np.isfinite(h)):
            raise ValueError("Electric and magnetic observations must be finite (no NaN or infinity).")
        scale = float(np.max(np.abs(h)))
        if not np.isfinite(scale) or scale <= 0.0:
            raise np.linalg.LinAlgError("Magnetic field is numerically zero.")
        hs = h / scale
        if abs(np.linalg.det(hs)) <= tol * max(float(np.linalg.norm(hs, ord=np.inf)), 1.0) ** 2:
            raise np.linalg.LinAlgError("Magnetic observation matrix is singular or rank deficient.")
        z = np.linalg.solve(hs.T, e.T).T / scale
        if not np.all(np.isfinite(z)):
            raise np.linalg.LinAlgError("Impedance recovery produced non-finite values.")
        return z

    if hx is None or hy is None:
        raise TypeError("Provide ex, ey, hx and hy, or two observation objects.")
    # For the full-tensor convenience API, a zero field component leaves
    # the corresponding component ratio undefined. Preserve a finite tensor
    # by returning zero for those entries; the stricter off-diagonal helper
    # still raises for zero/small magnetic denominators.
    def _ratio_allow_zero(num: Any, den: Any) -> np.ndarray:
        numerator = _finite_array(num, "Electric field")
        denominator = _finite_array(den, "Magnetic field")
        try:
            numerator, denominator = np.broadcast_arrays(numerator, denominator)
        except ValueError as exc:
            raise ValueError("Field components must have broadcast-compatible shapes.") from exc
        result = np.zeros(numerator.shape, dtype=complex)
        valid = np.abs(denominator) > tol * np.maximum(1.0, np.abs(denominator))
        np.divide(numerator, denominator, out=result, where=valid)
        return result

    zxx = _ratio_allow_zero(ex, hx)
    zxy = _ratio_allow_zero(ex, hy)
    zyx = _ratio_allow_zero(ey, hx)
    zyy = _ratio_allow_zero(ey, hy)
    z = np.empty(np.broadcast_shapes(zxx.shape, zxy.shape, zyx.shape, zyy.shape) + (2, 2), dtype=complex)
    z[..., 0, 0], z[..., 0, 1] = zxx, zxy
    z[..., 1, 0], z[..., 1, 1] = zyx, zyy
    if z.ndim == 2:
        return MTImpedance(z[0, 0], z[0, 1], z[1, 0], z[1, 1])
    return z


def off_diagonal_impedance(
    ex: Any,
    ey: Any | None = None,
    hx: Any | None = None,
    hy: Any | None = None,
    *,
    denominator_tolerance: float = 1.0e-12,
) -> tuple[Any, Any]:
    """Return ``(Zxy, Zyx)`` from fields or from an impedance tensor.

    Accepts ``(ex, ey, hx, hy)``, an :class:`MTImpedance`, a tensor ending
    in ``(2, 2)``, or explicit ``(zxy, zyx)`` as two arguments.
    """
    tol = _check_tolerance(denominator_tolerance)
    if isinstance(ex, MTImpedance):
        if any(v is not None for v in (ey, hx, hy)):
            raise ValueError("Field arguments must not accompany an MTImpedance.")
        return ex.zxy, ex.zyx
    if hx is not None or hy is not None:
        if ey is None or hx is None or hy is None:
            raise TypeError("Provide all four field components ex, ey, hx and hy.")
        return _safe_ratio(ex, hy, tol, "Hy"), _safe_ratio(ey, hx, tol, "Hx")
    if ey is not None:
        zxy, zyx = _finite_array(ex, "Zxy"), _finite_array(ey, "Zyx")
        return zxy.item() if zxy.ndim == 0 else zxy, zyx.item() if zyx.ndim == 0 else zyx
    values = _finite_array(ex, "impedance")
    if values.ndim < 2 or values.shape[-2:] != (2, 2):
        raise ValueError("impedance must have shape (..., 2, 2).")
    return values[..., 0, 1], values[..., 1, 0]


def apparent_resistivity(impedance: Any, omega: float | np.ndarray) -> Any:
    """Compute ``abs(Z)**2 / (MU0 * omega)`` for scalar, vector or tensor Z."""
    w = np.asarray(omega, dtype=float)
    if w.ndim > 1 or w.size == 0 or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("omega must be a finite, strictly positive scalar or one-dimensional array.")
    if isinstance(impedance, MTImpedance):
        if w.ndim != 0:
            raise ValueError("A scalar omega is required when impedance is an MTImpedance.")
        f = MU0 * float(w)
        return MTImpedance(*(abs(v) ** 2 / f for v in (impedance.zxx, impedance.zxy, impedance.zyx, impedance.zyy)))

    values = np.asarray(impedance, dtype=complex)
    if not np.all(np.isfinite(values)):
        raise ValueError("impedance must contain only finite values (no NaN or infinity).")
    if w.ndim == 0:
        return np.abs(values) ** 2 / (MU0 * float(w))
    # A 1-D impedance vector naturally corresponds elementwise to frequencies.
    if values.ndim == 1:
        if values.size != w.size:
            raise ValueError(f"omega has length {w.size}, but impedance has {values.size} values.")
        return np.abs(values) ** 2 / (MU0 * w)
    if values.ndim < 3 or values.shape[-3] != w.size or values.shape[-2:] != (2, 2):
        raise ValueError("A frequency vector requires impedance shaped (..., n_frequencies, 2, 2).")
    wf = w.reshape((1,) * (values.ndim - 3) + (w.size, 1, 1))
    return np.abs(values) ** 2 / (MU0 * wf)


def impedance_phase(impedance: Any) -> Any:
    """Return impedance phase in degrees, rejecting NaN and infinities."""
    if isinstance(impedance, MTImpedance):
        return impedance.phase
    values = np.asarray(impedance, dtype=complex)
    if not np.all(np.isfinite(values)):
        raise ValueError("impedance phase cannot be computed for non-finite values; impedance must be finite.")
    return np.degrees(np.angle(values))


def phase_tensor(impedance: Any) -> np.ndarray:
    """Compute the MT phase tensor ``Re(Z)**-1 Im(Z)``."""
    if isinstance(impedance, MTImpedance):
        return impedance.phase_tensor
    values = np.asarray(impedance, dtype=complex)
    if values.ndim != 4 or values.shape[-2:] != (2, 2):
        raise ValueError("impedance must have shape (n_stations, n_frequencies, 2, 2).")
    if not np.all(np.isfinite(values)):
        raise ValueError("impedance must contain only finite values.")
    out = np.empty(values.shape, dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            x, y = values[i, j].real, values[i, j].imag
            scale = max(float(np.linalg.norm(x, ord=np.inf)), 1.0)
            if abs(np.linalg.det(x)) <= 100.0 * np.finfo(float).eps * scale:
                raise ValueError("Cannot compute the MT phase tensor: the real part of the impedance is singular or ill-conditioned.")
            out[i, j] = np.linalg.solve(x, y)
    return out


def response_from_impedance(impedance: MTImpedance, frequency: float) -> MTResponse:
    """Construct an :class:`MTResponse` from an impedance and frequency."""
    return MTResponse(impedance, frequency)


__all__ = ["MU0", "MTImpedance", "MTResponse", "impedance_from_fields", "off_diagonal_impedance", "apparent_resistivity", "impedance_phase", "phase_tensor", "response_from_impedance"]

