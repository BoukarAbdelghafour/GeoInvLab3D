"""
Magnetic forward kernel for rectangular 3D cells.

Coordinate convention
---------------------
x : East
y : North
z : Up

Magnetic field convention
-------------------------
D : declination [degrees]
    Positive clockwise from North toward East.

I : inclination [degrees]
    Positive downward.

F : field intensity [nT]

The kernel computes the total-field magnetic anomaly
produced by a rectangular prism with unit magnetic
susceptibility.

The forward relation is linear with respect to
magnetic susceptibility:

    d = G @ chi

where G is the magnetic sensitivity matrix.

MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

import numpy as np


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

NT_TO_T = 1.0e-9
T_TO_NT = 1.0e9


# ----------------------------------------------------------------------
# Magnetic field direction
# ----------------------------------------------------------------------


def _field_direction(
    inclination: float,
    declination: float,
) -> np.ndarray:
    """
    Compute the unit vector of the regional magnetic field.

    Parameters
    ----------
    inclination : float
        Magnetic inclination in degrees.

        Positive values point downward.

    declination : float
        Magnetic declination in degrees.

        Measured clockwise from North toward East.

    Returns
    -------
    numpy.ndarray
        Unit vector [East, North, Up].

    Notes
    -----
    Coordinate convention:

        x = East
        y = North
        z = Up

    Therefore:

        hx = cos(I) sin(D)
        hy = cos(I) cos(D)
        hz = -sin(I)

    Examples
    --------
    D = 0, I = 0
        -> North

    D = 90, I = 0
        -> East

    I = 90
        -> Down

    I = -90
        -> Up
    """

    inclination_rad = np.deg2rad(
        inclination
    )

    declination_rad = np.deg2rad(
        declination
    )

    cos_i = np.cos(
        inclination_rad
    )

    hx = (
        cos_i
        * np.sin(declination_rad)
    )

    hy = (
        cos_i
        * np.cos(declination_rad)
    )

    hz = -np.sin(
        inclination_rad
    )

    direction = np.array(
        [
            hx,
            hy,
            hz,
        ],
        dtype=float,
    )

    # Normalize to remove tiny floating-point errors.
    norm = np.linalg.norm(direction)

    if norm == 0.0:
        raise ValueError(
            "Magnetic field direction has zero norm."
        )

    return direction / norm


# ----------------------------------------------------------------------
# Magnetic tensor
# ----------------------------------------------------------------------


def _magnetic_tensor(
    observation: np.ndarray,
    cell_center: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """
    Compute the magnetic tensor of a rectangular prism.

    Parameters
    ----------
    observation : numpy.ndarray
        Observation coordinates with shape (3,).

    cell_center : numpy.ndarray
        Cell center coordinates with shape (3,).

    dx : float
        Cell dimension along East.

    dy : float
        Cell dimension along North.

    dz : float
        Cell dimension along Up.

    Returns
    -------
    numpy.ndarray
        Symmetric magnetic tensor with shape (3, 3).

    Notes
    -----
    The tensor contains the second spatial derivatives
    of the Newtonian potential, integrated over the
    rectangular prism.
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

    if not np.all(
        np.isfinite(observation)
    ):
        raise ValueError(
            "Observation must contain only finite values."
        )

    if not np.all(
        np.isfinite(cell_center)
    ):
        raise ValueError(
            "Cell center must contain only finite values."
        )

    # ------------------------------------------------------------------
    # Validate dimensions.
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

    if not np.isfinite(dx):
        raise ValueError(
            "dx must be finite."
        )

    if not np.isfinite(dy):
        raise ValueError(
            "dy must be finite."
        )

    if not np.isfinite(dz):
        raise ValueError(
            "dz must be finite."
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
    # Tensor components.
    # ------------------------------------------------------------------

    xx = 0.0
    yy = 0.0
    zz = 0.0

    xy = 0.0
    xz = 0.0
    yz = 0.0

    # ------------------------------------------------------------------
    # Sum contributions from the eight corners.
    # ------------------------------------------------------------------

    for ix, x_corner in enumerate(x_bounds):

        for iy, y_corner in enumerate(y_bounds):

            for iz, z_corner in enumerate(z_bounds):

                x = (
                    observation[0]
                    - x_corner
                )

                y = (
                    observation[1]
                    - y_corner
                )

                z = (
                    observation[2]
                    - z_corner
                )

                r = np.sqrt(
                    x * x
                    + y * y
                    + z * z
                )

                if r == 0.0:
                    raise ValueError(
                        "Observation coincides with a cell corner."
                    )

                sign = (-1) ** (
                    ix + iy + iz
                )

                # ------------------------------------------------------
                # Diagonal terms.
                # ------------------------------------------------------

                xx_term = np.arctan2(
                    y * z,
                    x * r,
                )

                yy_term = np.arctan2(
                    x * z,
                    y * r,
                )

                zz_term = np.arctan2(
                    x * y,
                    z * r,
                )

                # ------------------------------------------------------
                # Off-diagonal terms.
                # ------------------------------------------------------

                xy_term = np.log(
                    max(
                        abs(z + r),
                        np.finfo(float).eps,
                    )
                )

                xz_term = np.log(
                    max(
                        abs(y + r),
                        np.finfo(float).eps,
                    )
                )

                yz_term = np.log(
                    max(
                        abs(x + r),
                        np.finfo(float).eps,
                    )
                )

                # ------------------------------------------------------
                # Accumulate.
                # ------------------------------------------------------

                xx += sign * xx_term
                yy += sign * yy_term
                zz += sign * zz_term

                xy += sign * xy_term
                xz += sign * xz_term
                yz += sign * yz_term

    # ------------------------------------------------------------------
    # Sign convention for mixed derivatives.
    # ------------------------------------------------------------------

    xy = -xy
    xz = -xz
    yz = -yz

    # ------------------------------------------------------------------
    # Assemble tensor.
    # ------------------------------------------------------------------

    tensor = np.array(
        [
            [
                xx,
                xy,
                xz,
            ],
            [
                xy,
                yy,
                yz,
            ],
            [
                xz,
                yz,
                zz,
            ],
        ],
        dtype=float,
    )

    return tensor


# ----------------------------------------------------------------------
# Magnetic kernel
# ----------------------------------------------------------------------


def magnetic_kernel(
    observation: np.ndarray,
    cell_center: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    inclination: float,
    declination: float,
    field_intensity: float,
) -> float:
    """
    Compute total-field magnetic response per unit susceptibility.

    Parameters
    ----------
    observation : numpy.ndarray
        Observation coordinates with shape (3,).

    cell_center : numpy.ndarray
        Cell center coordinates with shape (3,).

    dx : float
        Cell dimension along East.

    dy : float
        Cell dimension along North.

    dz : float
        Cell dimension along Up.

    inclination : float
        Regional magnetic field inclination [degrees].

        Positive inclination is downward.

    declination : float
        Regional magnetic field declination [degrees].

        Positive clockwise from North toward East.

    field_intensity : float
        Regional magnetic field intensity [nT].

    Returns
    -------
    float
        Total-field magnetic response in nT per unit
        magnetic susceptibility.

    Notes
    -----
    The induced magnetization is assumed parallel
    to the regional magnetic field.

    The magnetic anomaly is:

        dT = F * h^T T h

    apart from the unit conversion and the conventional
    magnetic normalization factor.

    Since F, geometry, D and I are fixed, the forward
    operator is linear in susceptibility.
    """

    # ------------------------------------------------------------------
    # Validate field intensity.
    # ------------------------------------------------------------------

    if not np.isfinite(
        field_intensity
    ):
        raise ValueError(
            "field_intensity must be finite."
        )

    if field_intensity < 0:
        raise ValueError(
            "field_intensity must be non-negative."
        )

    # ------------------------------------------------------------------
    # Field direction.
    # ------------------------------------------------------------------

    field_direction = _field_direction(
        inclination=inclination,
        declination=declination,
    )

    # ------------------------------------------------------------------
    # Magnetic tensor.
    # ------------------------------------------------------------------

    tensor = _magnetic_tensor(
        observation=observation,
        cell_center=cell_center,
        dx=dx,
        dy=dy,
        dz=dz,
    )

    # ------------------------------------------------------------------
    # Convert field intensity from nT to T.
    # ------------------------------------------------------------------

    field_tesla = (
        field_intensity
        * NT_TO_T
    )

    # ------------------------------------------------------------------
    # Project tensor along the field direction.
    # ------------------------------------------------------------------

    response = (
        field_direction
        @ tensor
        @ field_direction
    )

    # ------------------------------------------------------------------
    # Convert back to nT.
    #
    # The factor 1/(4*pi) corresponds to the magnetic
    # normalization used in the tensor formulation.
    # ------------------------------------------------------------------

    anomaly_tesla = (
        field_tesla
        * response
        / (4.0 * np.pi)
    )

    anomaly_nt = (
        anomaly_tesla
        * T_TO_NT
    )

    return float(
        anomaly_nt
    )

