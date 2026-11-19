#!/usr/bin/env python3
"""
Generate noisy synthetic magnetic observations for GeoInvLab3D.

Three noise levels are generated:
    - 1%
    - 3%
    - 5%

The noise is independent, zero-mean Gaussian noise.

The standard deviation is defined as:

    sigma_noise = noise_level * max(abs(TMI_observations))

The same standard normal realization is used for all noise levels,
allowing a consistent comparison between experiments.

Input files:
    examples/magnetic/synthetic/data/
        mesh.dat
        model_true.dat
        magnetic_observations.dat

Output files:
    examples/magnetic/noisy/
        1pct/data/
        3pct/data/
        5pct/data/

GeoInvLab3D conventions:
    - X: East
    - Y: North
    - Z: positive downward
    - Magnetic observations are expressed in nT.

MIT License
Copyright (c) 2026 Boukar Abdelghafour
"""

from pathlib import Path
import shutil

import numpy as np


# ======================================================================
# PATHS
# ======================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

SOURCE_DIR = (
    SCRIPT_DIR.parent
    / "synthetic"
    / "data"
)

OUTPUT_DIR = SCRIPT_DIR


# ======================================================================
# CONFIGURATION
# ======================================================================

NOISE_LEVELS = (1, 3, 5)

RANDOM_SEED = 20261016

OBSERVATIONS_FILENAME = "magnetic_observations.dat"
MESH_FILENAME = "mesh.dat"
MODEL_FILENAME = "model_true.dat"

OBSERVATION_HEADER = (
    "x_m y_m z_m TMI_anomaly_nT"
)

# Numerical precision used when writing ASCII files.
OUTPUT_FORMAT = "%.10e"


# ======================================================================
# FILE VALIDATION
# ======================================================================

def validate_input_files():
    """Check that all required input files exist."""

    required_files = (
        SOURCE_DIR / OBSERVATIONS_FILENAME,
        SOURCE_DIR / MESH_FILENAME,
        SOURCE_DIR / MODEL_FILENAME,
    )

    missing_files = [
        path for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        message = "\n".join(
            f"  - {path}" for path in missing_files
        )

        raise FileNotFoundError(
            "The following input files are missing:\n"
            f"{message}"
        )


# ======================================================================
# LOAD OBSERVATIONS
# ======================================================================

def load_observations():
    """
    Load magnetic observations.

    Expected columns:
        x_m
        y_m
        z_m
        TMI_anomaly_nT
    """

    filepath = SOURCE_DIR / OBSERVATIONS_FILENAME

    observations = np.loadtxt(
        filepath,
        comments="#",
        dtype=float,
    )

    if observations.ndim == 1:
        observations = observations.reshape(1, -1)

    if observations.shape[1] != 4:
        raise ValueError(
            "Invalid magnetic observation file. "
            "Expected exactly four columns: "
            "x_m, y_m, z_m, TMI_anomaly_nT."
        )

    if not np.all(np.isfinite(observations)):
        raise ValueError(
            "Magnetic observations contain NaN or infinite values."
        )

    return observations


# ======================================================================
# WRITE OBSERVATIONS
# ======================================================================

def write_observations(filepath, observations):
    """Write magnetic observations in GeoInvLab3D ASCII format."""

    np.savetxt(
        filepath,
        observations,
        fmt=OUTPUT_FORMAT,
        header=OBSERVATION_HEADER,
        comments="# ",
    )


# ======================================================================
# COPY MODEL AND MESH
# ======================================================================

def copy_model_files(output_data_dir):
    """
    Copy the original mesh and true model into an experiment folder.

    The files are copied without modifying their contents.
    """

    files_to_copy = (
        MESH_FILENAME,
        MODEL_FILENAME,
    )

    for filename in files_to_copy:

        source = SOURCE_DIR / filename
        destination = output_data_dir / filename

        shutil.copy2(source, destination)


# ======================================================================
# GENERATE NOISY DATA
# ======================================================================

def generate_noisy_datasets():
    """Generate all noisy magnetic datasets."""

    print("=" * 72)
    print("GeoInvLab3D - MAGNETIC NOISY DATA GENERATION")
    print("=" * 72)

    print(f"\nInput directory : {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seed     : {RANDOM_SEED}")

    validate_input_files()

    # ------------------------------------------------------------------
    # Load original observations
    # ------------------------------------------------------------------

    print("\n[1/4] Loading original magnetic observations...")

    observations = load_observations()

    coordinates = observations[:, :3]
    original_tmi = observations[:, 3]

    n_observations = original_tmi.size

    reference_amplitude = np.max(np.abs(original_tmi))

    if reference_amplitude <= 0.0:
        raise ValueError(
            "The maximum absolute magnetic anomaly must be positive."
        )

    print(f"Number of observations : {n_observations}")
    print(f"TMI minimum            : {original_tmi.min():.6f} nT")
    print(f"TMI maximum            : {original_tmi.max():.6f} nT")
    print(f"TMI RMS                : "
          f"{np.sqrt(np.mean(original_tmi**2)):.6f} nT")
    print(f"Reference amplitude    : {reference_amplitude:.6f} nT")

    # ------------------------------------------------------------------
    # Generate one standard normal realization
    # ------------------------------------------------------------------

    print("\n[2/4] Generating Gaussian noise...")

    rng = np.random.default_rng(RANDOM_SEED)

    standard_normal_noise = rng.standard_normal(
        n_observations
    )

    print("Noise distribution: Gaussian")
    print("Noise mean         : 0")
    print("Noise realization : reproducible")

    # ------------------------------------------------------------------
    # Generate datasets
    # ------------------------------------------------------------------

    print("\n[3/4] Generating noisy datasets...")

    for percentage in NOISE_LEVELS:

        noise_fraction = percentage / 100.0

        sigma_noise = (
            noise_fraction * reference_amplitude
        )

        noise = sigma_noise * standard_normal_noise

        noisy_tmi = original_tmi + noise

        noisy_observations = np.column_stack(
            (
                coordinates,
                noisy_tmi,
            )
        )

        # Output directory
        experiment_name = f"{percentage}pct"

        experiment_dir = (
            OUTPUT_DIR
            / experiment_name
            / "data"
        )

        experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Write noisy observations
        observation_path = (
            experiment_dir
            / OBSERVATIONS_FILENAME
        )

        write_observations(
            observation_path,
            noisy_observations,
        )

        # Copy mesh and true model
        copy_model_files(experiment_dir)

        # Statistics
        noise_rms = np.sqrt(
            np.mean(noise**2)
        )

        noisy_rms = np.sqrt(
            np.mean(noisy_tmi**2)
        )

        print(f"\n--- Noise level: {percentage}% ---")
        print(f"Noise standard deviation : {sigma_noise:.6f} nT")
        print(f"Actual noise RMS          : {noise_rms:.6f} nT")
        print(f"Noisy TMI minimum         : {noisy_tmi.min():.6f} nT")
        print(f"Noisy TMI maximum         : {noisy_tmi.max():.6f} nT")
        print(f"Noisy TMI RMS             : {noisy_rms:.6f} nT")
        print(f"Output directory          : {experiment_dir}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n[4/4] Generation completed.")

    print("\nGenerated experiments:")

    for percentage in NOISE_LEVELS:

        experiment_dir = (
            OUTPUT_DIR
            / f"{percentage}pct"
            / "data"
        )

        print(f"\n{percentage}% noise:")
        print(f"  {experiment_dir / MESH_FILENAME}")
        print(f"  {experiment_dir / MODEL_FILENAME}")
        print(f"  {experiment_dir / OBSERVATIONS_FILENAME}")

    print("\n" + "=" * 72)
    print("NOISY MAGNETIC DATA GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 72)


# ======================================================================
# MAIN
# ======================================================================

def main():
    """Main entry point."""

    generate_noisy_datasets()


if __name__ == "__main__":
    main()
