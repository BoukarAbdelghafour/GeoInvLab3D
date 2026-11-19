#!/usr/bin/env python3
"""
Generate a complete synthetic 3-D MT dataset for GeoInvLab3D.

Run from the repository root:
    python examples/mt/generate_mt_synthetic.py

Model:
    Mesh: 6 x 6 x 4 cells
    Cell dimensions: 100 x 100 x 100 m
    Origin: (-300, -300, 0) m
    Background resistivity: 100 ohm.m
    Conductive anomaly: 10 ohm.m
    Resistive anomaly: 1000 ohm.m

Acquisition:
    10 surface stations
    10 logarithmically spaced frequencies from 0.01 to 100 Hz

Noise:
    10% multiplicative lognormal apparent-resistivity noise
    3 degree Gaussian impedance-phase noise

All output files are ASCII .dat files.

Boukar Abdelghafour  MIT License Copyright (c) 2026 
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


# =====================================================================
# PATHS AND CONFIGURATION
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SRC_DIR = REPO_ROOT / "src"

if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = SCRIPT_DIR

RANDOM_SEED = 20261018

# Mesh
NX, NY, NZ = 6, 6, 4
DX = DY = DZ = 100.0
ORIGIN = (-300.0, -300.0, 0.0)

# Resistivity model
BACKGROUND_RESISTIVITY = 100.0
CONDUCTIVE_RESISTIVITY = 10.0
RESISTIVE_RESISTIVITY = 1000.0

# MT acquisition
N_FREQUENCIES = 10
FREQUENCY_MIN = 1.0e-2
FREQUENCY_MAX = 1.0e2

# Noise
RHO_NOISE_FRACTION = 0.1
PHASE_NOISE_STD_DEG = 3.0

COMPONENTS = ("xx", "xy", "yx", "yy")


# =====================================================================
# GEOINVLAB3D IMPORTS
# =====================================================================

try:
    from geoinvlab.core.mesh.structured import StructuredMesh3D
    from geoinvlab.core.model import Model3D
    from geoinvlab.methods.mt.operator import MTOperator

except ImportError as exc:
    raise ImportError(
        "Could not import GeoInvLab3D components. "
        "Check the Python environment and repository src/ directory."
    ) from exc


# =====================================================================
# ASCII FILE UTILITIES
# =====================================================================

def save_table(
    path: Path,
    data: np.ndarray,
    header: str,
) -> None:
    """Write a numerical table as an ASCII file."""

    path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        path,
        np.asarray(data, dtype=float),
        fmt="%.12e",
        header=header,
        comments="# ",
    )


def save_mesh(path: Path) -> None:
    """Save structured mesh dimensions and origin."""

    row = np.array(
        [[
            NX,
            NY,
            NZ,
            DX,
            DY,
            DZ,
            ORIGIN[0],
            ORIGIN[1],
            ORIGIN[2],
        ]],
        dtype=float,
    )

    save_table(
        path,
        row,
        "nx ny nz dx_m dy_m dz_m origin_x_m origin_y_m origin_z_m",
    )


def save_model(
    path: Path,
    mesh: StructuredMesh3D,
    values: np.ndarray,
) -> None:
    """Save cell indices, coordinates, and resistivity."""

    centers = np.asarray(
        mesh.cell_centers(),
        dtype=float,
    )

    if centers.shape != (mesh.n_cells, 3):
        raise ValueError(
            f"Unexpected cell-center shape: {centers.shape}"
        )

    indices = np.indices(
        (NX, NY, NZ)
    ).reshape(3, -1).T

    table = np.column_stack(
        (
            indices,
            centers,
            values,
        )
    )

    save_table(
        path,
        table,
        "ix iy iz x_center_m y_center_m z_center_m resistivity_ohm_m",
    )


# =====================================================================
# MESH GENERATION
# =====================================================================

def create_mesh() -> StructuredMesh3D:
    """Create the structured 3-D mesh."""

    mesh = StructuredMesh3D(
        nx=NX,
        ny=NY,
        nz=NZ,
        dx=DX,
        dy=DY,
        dz=DZ,
        origin=ORIGIN,
    )

    return mesh


# =====================================================================
# RESISTIVITY MODELS
# =====================================================================

def create_models(
    mesh: StructuredMesh3D,
) -> tuple[Model3D, Model3D]:
    """Create true and initial resistivity models."""

    initial_values = np.full(
        mesh.n_cells,
        BACKGROUND_RESISTIVITY,
        dtype=float,
    )

    true_values = initial_values.copy()

    centers = np.asarray(
        mesh.cell_centers(),
        dtype=float,
    )

    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]

    # Conductive anomaly.
    conductive_mask = (
        (x >= -200.0)
        & (x < -100.0)
        & (y >= -200.0)
        & (y < -100.0)
        & (z >= 100.0)
        & (z < 300.0)
    )

    # Resistive anomaly.
    resistive_mask = (
        (x >= -200.0)
        & (x < -100.0)
        & (y >= 100.0)
        & (y < 200.0)
        & (z >= 100.0)
        & (z < 300.0)
    )

    if np.count_nonzero(conductive_mask) != 2:
        raise RuntimeError(
            "The conductive anomaly must occupy exactly one cell."
        )

    if np.count_nonzero(resistive_mask) != 2:
        raise RuntimeError(
            "The resistive anomaly must occupy exactly one cell."
        )

    true_values[conductive_mask] = CONDUCTIVE_RESISTIVITY
    true_values[resistive_mask] = RESISTIVE_RESISTIVITY

    true_model = Model3D(
        mesh=mesh,
        values=true_values,
        name="resistivity",
    )

    initial_model = Model3D(
        mesh=mesh,
        values=initial_values,
        name="resistivity",
    )

    return true_model, initial_model


# =====================================================================
# MT STATIONS
# =====================================================================

def create_stations() -> np.ndarray:
    """Generate ten surface MT stations."""

    x_coordinates = np.array(
        [-250.0, -150.0, -50.0, 50.0, 150.0]
    )

    y_coordinates = np.array(
        [-200.0, 200.0]
    )

    stations = np.array(
        [
            (x, y, 0.0)
            for y in y_coordinates
            for x in x_coordinates
        ],
        dtype=float,
    )

    return stations


# =====================================================================
# MT FREQUENCIES
# =====================================================================

def create_frequencies() -> np.ndarray:
    """Generate logarithmically spaced frequencies."""

    return np.logspace(
        np.log10(FREQUENCY_MIN),
        np.log10(FREQUENCY_MAX),
        N_FREQUENCIES,
    )


# =====================================================================
# MT FORWARD MODELLING
# =====================================================================

def compute_clean_response(
    mesh: StructuredMesh3D,
    true_model: Model3D,
    stations: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute clean MT apparent resistivity and impedance phase.

    The expected output shape is:

        (n_stations, n_frequencies, 2, 2)
    """

    operator = MTOperator(
        mesh=mesh,
        frequencies=frequencies,
        locations=stations,
        observation_frequencies=frequencies,
        solver_method="direct",
        solver_tolerance=1.0e-10,
        solver_maxiter=1000,
    )

    print("Computing complex MT impedance tensor...")

    impedance = np.asarray(
        operator.forward(
            model=true_model,
            locations=stations,
        ),
        dtype=complex,
    )

    expected_shape = (
        stations.shape[0],
        frequencies.size,
        2,
        2,
    )

    if impedance.shape != expected_shape:
        raise ValueError(
            f"Unexpected impedance shape: {impedance.shape}. "
            f"Expected: {expected_shape}."
        )

    if not np.all(np.isfinite(impedance)):
        raise ValueError(
            "The impedance tensor contains NaN or infinite values."
        )

    response = operator.response_from_impedance(
        impedance
    )

    required_keys = (
        "apparent_resistivity",
        "impedance_phase",
    )

    for key in required_keys:
        if key not in response:
            raise KeyError(
                f"Missing MT response key: {key}. "
                f"Available keys: {list(response.keys())}"
            )

    rho = np.asarray(
        response["apparent_resistivity"],
        dtype=float,
    )

    phase = np.asarray(
        response["impedance_phase"],
        dtype=float,
    )

    if rho.shape != expected_shape:
        raise ValueError(
            f"Invalid apparent resistivity shape: {rho.shape}. "
            f"Expected: {expected_shape}."
        )

    if phase.shape != expected_shape:
        raise ValueError(
            f"Invalid impedance phase shape: {phase.shape}. "
            f"Expected: {expected_shape}."
        )

    if not np.all(np.isfinite(rho)):
        raise ValueError(
            "Apparent resistivity contains NaN or infinite values."
        )

    if not np.all(np.isfinite(phase)):
        raise ValueError(
            "Impedance phase contains NaN or infinite values."
        )

    # Apparent resistivity must not be negative.
    # Exact zero values are allowed and are diagnosed below.
    if np.any(rho < 0.0):
        raise ValueError(
            "Negative apparent resistivity detected. "
            "Check the MT response calculation."
        )

    # --------------------------------------------------------------
    # Diagnostic information
    # --------------------------------------------------------------

    print("\nApparent-resistivity diagnostics:")

    print(
        f"Minimum: {np.min(rho):.6e} ohm.m"
    )

    print(
        f"Maximum: {np.max(rho):.6e} ohm.m"
    )

    print(
        "Number of zero values:",
        np.count_nonzero(rho == 0.0),
    )

    print(
        "Number of negative values:",
        np.count_nonzero(rho < 0.0),
    )

    print("\nTensor components:")

    for index, component in enumerate(COMPONENTS):

        values = rho[
            :,
            :,
            index // 2,
            index % 2,
        ]

        print(
            f"rho_{component}: "
            f"min={values.min():.6e}, "
            f"max={values.max():.6e}, "
            f"zeros={np.count_nonzero(values == 0.0)}, "
            f"negative={np.count_nonzero(values < 0.0)}"
        )

    return rho, phase


# =====================================================================
# NOISE GENERATION
# =====================================================================

def add_noise(
    clean_rho: np.ndarray,
    clean_phase: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add lognormal apparent-resistivity noise and Gaussian phase noise.

    Multiplicative noise preserves exact zero resistivity values.
    """

    rng = np.random.default_rng(seed)

    rho_noisy = np.zeros_like(
        clean_rho,
        dtype=float,
    )

    positive_mask = clean_rho > 0.0

    sigma_log = RHO_NOISE_FRACTION

    noise = rng.standard_normal(
        np.count_nonzero(positive_mask)
    )

    rho_noisy[positive_mask] = (
        clean_rho[positive_mask]
        * np.exp(
            sigma_log * noise
            - 0.5 * sigma_log**2
        )
    )

    phase_noisy = clean_phase + rng.normal(
        loc=0.0,
        scale=PHASE_NOISE_STD_DEG,
        size=clean_phase.shape,
    )

    # The phase of an exactly zero impedance component is undefined.
    phase_noisy[~positive_mask] = 0.0

    return rho_noisy, phase_noisy


# =====================================================================
# OBSERVATION FORMATTING
# =====================================================================

def flatten_response(
    rho: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    """Flatten the MT response using the GeoInvLab3D data convention."""

    n_stations, n_frequencies = rho.shape[:2]

    rows = []

    for station_index in range(n_stations):

        for frequency_index in range(n_frequencies):

            rho_row = rho[
                station_index,
                frequency_index,
            ].reshape(4)

            phase_row = phase[
                station_index,
                frequency_index,
            ].reshape(4)

            rows.append(
                np.concatenate(
                    (
                        rho_row,
                        phase_row,
                    )
                )
            )

    return np.asarray(rows, dtype=float)


def save_observations(
    path: Path,
    stations: np.ndarray,
    frequencies: np.ndarray,
    rho: np.ndarray,
    phase: np.ndarray,
) -> None:
    """Save one row per station-frequency observation."""

    response = flatten_response(
        rho,
        phase,
    )

    rows = []
    row_index = 0

    for station_id, location in enumerate(
        stations,
        start=1,
    ):

        for frequency in frequencies:

            metadata = np.array(
                [
                    float(station_id),
                    float(frequency),
                    location[0],
                    location[1],
                    location[2],
                ],
                dtype=float,
            )

            rows.append(
                np.concatenate(
                    (
                        metadata,
                        response[row_index],
                    )
                )
            )

            row_index += 1

    save_table(
        path,
        np.asarray(rows),
        (
            "station_id frequency_Hz x_m y_m z_m "
            "rho_xx rho_xy rho_yx rho_yy "
            "phase_xx_deg phase_xy_deg phase_yx_deg phase_yy_deg"
        ),
    )


# =====================================================================
# UNCERTAINTIES
# =====================================================================

def save_uncertainties(path: Path) -> None:
    """Save the configured MT uncertainties."""

    values = np.array(
        [[
            RHO_NOISE_FRACTION,
            RHO_NOISE_FRACTION,
            RHO_NOISE_FRACTION,
            RHO_NOISE_FRACTION,
            PHASE_NOISE_STD_DEG,
            PHASE_NOISE_STD_DEG,
            PHASE_NOISE_STD_DEG,
            PHASE_NOISE_STD_DEG,
        ]],
        dtype=float,
    )

    save_table(
        path,
        values,
        (
            "relative_sigma_rho_xx relative_sigma_rho_xy "
            "relative_sigma_rho_yx relative_sigma_rho_yy "
            "sigma_phase_xx_deg sigma_phase_xy_deg "
            "sigma_phase_yx_deg sigma_phase_yy_deg"
        ),
    )


# =====================================================================
# README
# =====================================================================

def save_readme(
    stations: np.ndarray,
    frequencies: np.ndarray,
) -> None:
    """Write a description of the generated synthetic dataset."""

    readme = f"""
GeoInvLab3D - Synthetic 3-D MT model
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
=======================================

MESH
----
Dimensions: {NX} x {NY} x {NZ}
Number of cells: {NX * NY * NZ}
Cell dimensions: {DX} x {DY} x {DZ} m
Origin: {ORIGIN}

Coordinates:
    x = East
    y = North
    z = positive downward
    Surface = z = 0 m

RESISTIVITY MODEL
-----------------
Background: {BACKGROUND_RESISTIVITY} ohm.m
Conductive anomaly: {CONDUCTIVE_RESISTIVITY} ohm.m
Resistive anomaly: {RESISTIVE_RESISTIVITY} ohm.m

Initial model:
    Homogeneous {BACKGROUND_RESISTIVITY} ohm.m

MT ACQUISITION
--------------
Number of stations: {stations.shape[0]}
Number of frequencies: {frequencies.size}
Number of observations: {stations.shape[0] * frequencies.size}

Frequency range:
    {frequencies.min():.6g} to {frequencies.max():.6g} Hz

Station elevation:
    z = 0 m

NOISE
-----
Apparent resistivity:
    {100 * RHO_NOISE_FRACTION:.1f}% multiplicative lognormal noise

Impedance phase:
    {PHASE_NOISE_STD_DEG:.1f} degree Gaussian noise

Random seed:
    {RANDOM_SEED}

OUTPUT FILES
------------
mesh.dat
model_true.dat
model_initial.dat
stations.dat
frequencies.dat
mt_observations_clean.dat
mt_observations_noisy.dat
mt_uncertainties.dat

OBSERVATION COLUMN ORDER
------------------------
station_id frequency_Hz x_m y_m z_m
rho_xx rho_xy rho_yx rho_yy
phase_xx_deg phase_xy_deg phase_yx_deg phase_yy_deg
"""

    (OUTPUT_DIR / "README.txt").write_text(
        readme.strip() + "\n",
        encoding="utf-8",
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print("=" * 68)
    print("GeoInvLab3D - MT SYNTHETIC DATA GENERATOR ** MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]")
    print("=" * 68)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Mesh
    print("\n[1/7] Creating structured mesh...")

    mesh = create_mesh()

    print(f"Mesh dimensions: {NX} x {NY} x {NZ}")
    print(f"Number of cells: {mesh.n_cells}")
    print(f"Cell dimensions: {DX} x {DY} x {DZ} m")
    print(f"Origin: {ORIGIN}")

    # 2. Models
    print("\n[2/7] Creating resistivity models...")

    true_model, initial_model = create_models(mesh)

    true_values = np.asarray(
        true_model.values,
        dtype=float,
    )

    initial_values = np.asarray(
        initial_model.values,
        dtype=float,
    )

    print(
        f"Background resistivity: "
        f"{BACKGROUND_RESISTIVITY} ohm.m"
    )

    print(
        f"Conductive anomaly: "
        f"{CONDUCTIVE_RESISTIVITY} ohm.m"
    )

    print(
        f"Resistive anomaly: "
        f"{RESISTIVE_RESISTIVITY} ohm.m"
    )

    print(
        "Conductive cells:",
        np.count_nonzero(
            true_values == CONDUCTIVE_RESISTIVITY
        ),
    )

    print(
        "Resistive cells:",
        np.count_nonzero(
            true_values == RESISTIVE_RESISTIVITY
        ),
    )

    # 3. Stations
    print("\n[3/7] Generating MT stations...")

    stations = create_stations()

    print(f"Number of stations: {stations.shape[0]}")
    print("Station elevation: z = 0 m")

    # 4. Frequencies
    print("\n[4/7] Generating MT frequencies...")

    frequencies = create_frequencies()

    print(f"Number of frequencies: {frequencies.size}")

    print(
        f"Frequency range: "
        f"{frequencies.min():.6g} to "
        f"{frequencies.max():.6g} Hz"
    )

    # 5. Forward modelling
    print("\n[5/7] Running MT forward modelling...")

    clean_rho, clean_phase = compute_clean_response(
        mesh=mesh,
        true_model=true_model,
        stations=stations,
        frequencies=frequencies,
    )

    # 6. Noise
    print("\n[6/7] Adding noise...")

    noisy_rho, noisy_phase = add_noise(
        clean_rho=clean_rho,
        clean_phase=clean_phase,
        seed=RANDOM_SEED,
    )

    # 7. Save files
    print("\n[7/7] Writing ASCII data files...")

    save_mesh(
        OUTPUT_DIR / "mesh.dat"
    )

    save_model(
        OUTPUT_DIR / "model_true.dat",
        mesh,
        true_values,
    )

    save_model(
        OUTPUT_DIR / "model_initial.dat",
        mesh,
        initial_values,
    )

    save_table(
        OUTPUT_DIR / "stations.dat",
        stations,
        "x_m y_m z_m",
    )

    save_table(
        OUTPUT_DIR / "frequencies.dat",
        frequencies.reshape(-1, 1),
        "frequency_Hz",
    )

    save_observations(
        OUTPUT_DIR / "mt_observations_clean.dat",
        stations,
        frequencies,
        clean_rho,
        clean_phase,
    )

    save_observations(
        OUTPUT_DIR / "mt_observations_noisy.dat",
        stations,
        frequencies,
        noisy_rho,
        noisy_phase,
    )

    save_uncertainties(
        OUTPUT_DIR / "mt_uncertainties.dat"
    )

    save_readme(
        stations,
        frequencies,
    )

    # Summary
    print("\n" + "=" * 68)
    print("GENERATION COMPLETED")
    print("=" * 68)

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Mesh cells: {mesh.n_cells}")
    print(f"Stations: {stations.shape[0]}")
    print(f"Frequencies: {frequencies.size}")
    print(
        "Station-frequency observations:",
        stations.shape[0] * frequencies.size,
    )

    print(
        "Clean apparent-resistivity range: "
        f"{clean_rho.min():.6e} to "
        f"{clean_rho.max():.6e} ohm.m"
    )

    print(
        "Noisy apparent-resistivity range: "
        f"{noisy_rho.min():.6e} to "
        f"{noisy_rho.max():.6e} ohm.m"
    )

    print("\nGenerated files:")

    for path in sorted(OUTPUT_DIR.glob("*.dat")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
