"""
Gravitational kernel for rectangular 3D cells.

Coordinate convention
---------------------
x : East
y : North
z : Up

The returned value is the vertical gravitational
acceleration per unit density.

MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

import numpy as np


# Gravitational constant [m^3 kg^-1 s^-2].
G = 6.67430e-11


def _safe_log_argument(value: float) -> float:
    """
    Return a strictly positive argument for a logarithm.

    This protects the analytical kernel from log(0) at
    geometrically singular configurations.
    """

    eps = np.finfo(float).eps

    return max(abs(value), eps)


def gravity_kernel(
    observation: np.ndarray,
    cell_center: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> float:
    """
    Compute the vertical gravitational response of a
    homogeneous rectangular prism per unit density.

    Parameters
    ----------
    observation : numpy.ndarray
        Observation coordinates with shape (3,).

    cell_center : numpy.ndarray
        Cell center coordinates with shape (3,).

    dx : float
        Cell dimension along x.

    dy : float
        Cell dimension along y.

    dz : float
        Cell dimension along z.

    Returns
    -------
    float
        Vertical gravitational acceleration per unit density.

    Notes
    -----
    The analytical expression is the rectangular-prism
    gravitational kernel:

        Kz = G * sum[
            s * (
                x log(y + r)
                + y log(x + r)
                - z atan2(x y, z r)
            )
        ]

    where

        r = sqrt(x² + y² + z²)

    and the summation is performed over the eight
    cell corners.

    The result corresponds to a unit density contrast.
    """

    observation = np.asarray(
        observation,
        dtype=float,
    )

    cell_center = np.asarray(
        cell_center,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validate coordinates.
    # ------------------------------------------------------------------

    if observation.shape != (3,):
        raise ValueError(
            "Observation must have shape (3,)."
        )

    if cell_center.shape != (3,):
        raise ValueError(
            "Cell center must have shape (3,)."
        )

    # ------------------------------------------------------------------
    # Validate cell dimensions.
    # ------------------------------------------------------------------

    if dx <= 0:
        raise ValueError(
            "dx must be positive."
        )

    if dy <= 0:
        raise ValueError(
            "dy must be positive."
        )

    if dz <= 0:
        raise ValueError(
            "dz must be positive."
        )

    # ------------------------------------------------------------------
    # Cell boundaries.
    # ------------------------------------------------------------------

    x_bounds = np.array(
        [
            cell_center[0] - dx / 2.0,
            cell_center[0] + dx / 2.0,
        ],
        dtype=float,
    )

    y_bounds = np.array(
        [
            cell_center[1] - dy / 2.0,
            cell_center[1] + dy / 2.0,
        ],
        dtype=float,
    )

    z_bounds = np.array(
        [
            cell_center[2] - dz / 2.0,
            cell_center[2] + dz / 2.0,
        ],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Sum the eight corner contributions.
    # ------------------------------------------------------------------

    total = 0.0

    for ix, x_corner in enumerate(x_bounds):

        for iy, y_corner in enumerate(y_bounds):

            for iz, z_corner in enumerate(z_bounds):

                # Coordinates from the corner to the observation.
                x = observation[0] - x_corner
                y = observation[1] - y_corner
                z = observation[2] - z_corner

                r = np.sqrt(
                    x * x
                    + y * y
                    + z * z
                )

                if r == 0.0:
                    raise ValueError(
                        "Observation coincides with a cell corner."
                    )

                # ------------------------------------------------------
                # Alternating corner sign.
                # ------------------------------------------------------

                sign = (-1) ** (
                    ix + iy + iz
                )

                # ------------------------------------------------------
                # Logarithmic terms.
                # ------------------------------------------------------

                term_x = (
                    x
                    * np.log(
                        _safe_log_argument(
                            y + r
                        )
                    )
                )

                term_y = (
                    y
                    * np.log(
                        _safe_log_argument(
                            x + r
                        )
                    )
                )

                # ------------------------------------------------------
                # Arctangent term.
                #
                # atan2 is used instead of atan(x*y/(z*r))
                # because it remains well-defined when z is
                # close to zero.
                # ------------------------------------------------------

                if abs(z) < np.finfo(float).eps:

                    term_z = 0.0

                else:

                    term_z = (
                        z
                        * np.arctan2(
                            x * y,
                            z * r,
                        )
                    )

                # ------------------------------------------------------
                # Complete rectangular-prism contribution.
                # ------------------------------------------------------

                contribution = (
                    term_x
                    + term_y
                    - term_z
                )

                total += sign * contribution

    # ------------------------------------------------------------------
    # Convert the geometrical kernel to acceleration per unit density.
    # ------------------------------------------------------------------

    return G * total
