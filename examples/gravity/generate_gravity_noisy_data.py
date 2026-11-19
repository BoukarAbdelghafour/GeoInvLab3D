#!/usr/bin/env python3
"""
Generate a synthetic 3D gravity model and noisy gravity observations.

The model contains:
    - A homogeneous background with zero density contrast.
    - A positive density-contrast body (+500 kg/m3).
    - A negative density-contrast body (-300 kg/m3).

The script generates:
    - The 3D mesh.
    - The true density-contrast model.
    - A homogeneous initial model.
    - Surface gravity stations.
    - Noise-free gravity observations.
    - Noisy gravity observations for 1%, 3%, and 5% noise levels.

Gravity observations are stored in mGal.

Noise is additive Gaussian noise with standard deviation defined as
a percentage of the RMS amplitude of the noise-free gravity response.

All output files are ASCII text files.

Boukar Abdelghafour  GeoInvLab3D   MIT License Copyright (c) 2026
"""

from pathlib import Path

import numpy as np

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.core.model import Model3D
from geoinvlab.methods.gravity.operator import GravityOperator


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR / "data"
NOISY_DIR = SCRIPT_DIR / "noisy"

# Mesh dimensions
NX = 20
NY = 20
NZ = 15

DX = 100.0
DY = 100.0
DZ = 100.0

ORIGIN = (0.0, 0.0, 0.0)

# Density contrasts (kg/m3)
BACKGROUND = 0.0
DENSE_BODY = 500.0
LIGHT_BODY = -300.0

# Dense body boundaries (m)
DENSE_X = (500.0, 900.0)
DENSE_Y = (500.0, 900.0)
DENSE_Z = (300.0, 700.0)

# Low-density body boundaries (m)
LIGHT_X = (1100.0, 1500.0)
LIGHT_Y = (1100.0, 1500.0)
LIGHT_Z = (500.0, 900.0)

# Gravity stations
N_STATIONS_X = 20
N_STATIONS_Y = 20
STATION_HEIGHT = 0.0

# Noise levels
NOISE_LEVELS = {
    "noise_1pct": 0.01,
    "noise_3pct": 0.03,
    "noise_5pct": 0.05,
}

# Reproducible random seeds
BASE_RANDOM_SEED = 20261015

# Unit conversion
SI_TO_MGAL = 1.0e5


# =============================================================================
# Mesh and model generation
# =============================================================================

def create_mesh():
    """Create the structured 3D mesh."""

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


def create_true_model(mesh):
    """
    Create the true density-contrast model.

    The model contains a positive and a negative density anomaly
    embedded in a homogeneous zero-contrast background.
    """

    centers = np.asarray(mesh.cell_centers(), dtype=float)

    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]

    values = np.full(centers.shape[0], BACKGROUND, dtype=float)

    dense_mask = (
        (x >= DENSE_X[0])
        & (x < DENSE_X[1])
        & (y >= DENSE_Y[0])
        & (y < DENSE_Y[1])
        & (z >= DENSE_Z[0])
        & (z < DENSE_Z[1])
    )

    light_mask = (
        (x >= LIGHT_X[0])
        & (x < LIGHT_X[1])
        & (y >= LIGHT_Y[0])
        & (y < LIGHT_Y[1])
        & (z >= LIGHT_Z[0])
        & (z < LIGHT_Z[1])
    )

    values[dense_mask] = DENSE_BODY
    values[light_mask] = LIGHT_BODY

    model = Model3D(
        mesh=mesh,
        values=values,
        name="gravity_true_density_contrast",
    )

    return model, centers, values


def create_initial_model(mesh):
    """Create a homogeneous zero-contrast initial model."""

    values = np.full(NX * NY * NZ, BACKGROUND, dtype=float)

    return Model3D(
        mesh=mesh,
        values=values,
        name="gravity_initial_density_contrast",
    )


# =============================================================================
# Gravity stations
# =============================================================================

def create_stations():
    """Create a regular grid of gravity stations at the surface."""

    x_values = np.linspace(
        DX / 2.0,
        NX * DX - DX / 2.0,
        N_STATIONS_X,
    )

    y_values = np.linspace(
        DY / 2.0,
        NY * DY - DY / 2.0,
        N_STATIONS_Y,
    )

    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")

    zz = np.full_like(xx, STATION_HEIGHT, dtype=float)

    stations = np.column_stack(
        (
            xx.ravel(),
            yy.ravel(),
            zz.ravel(),
        )
    )

    return stations


# =============================================================================
# File output
# =============================================================================

def save_mesh(mesh, centers, output_path):
    """Save mesh cell centers and cell dimensions."""

    nx, ny, nz = NX, NY, NZ

    indices = np.indices((nx, ny, nz)).reshape(3, -1).T

    table = np.column_stack(
        (
            indices,
            centers,
            np.full(len(centers), DX),
            np.full(len(centers), DY),
            np.full(len(centers), DZ),
        )
    )

    np.savetxt(
        output_path,
        table,
        fmt="%.10e",
        header=(
            "ix iy iz "
            "x_center_m y_center_m z_center_m "
            "dx_m dy_m dz_m"
        ),
        comments="# ",
    )


def save_model(centers, values, output_path):
    """Save a density-contrast model."""

    indices = np.indices((NX, NY, NZ)).reshape(3, -1).T

    table = np.column_stack(
        (
            indices,
            centers,
            values,
        )
    )

    np.savetxt(
        output_path,
        table,
        fmt="%.10e",
        header=(
            "ix iy iz "
            "x_center_m y_center_m z_center_m "
            "density_contrast_kg_m3"
        ),
        comments="# ",
    )


def save_stations(stations, output_path):
    """Save gravity station coordinates."""

    np.savetxt(
        output_path,
        stations,
        fmt="%.10e",
        header="x_m y_m z_m",
        comments="# ",
    )


def save_gravity_data(stations, values, output_path):
    """Save gravity observations in mGal."""

    table = np.column_stack(
        (
            stations,
            values,
        )
    )

    np.savetxt(
        output_path,
        table,
        fmt="%.10e",
        header="x_m y_m z_m gravity_mGal",
        comments="# ",
    )


# =============================================================================
# Gravity forward modelling
# =============================================================================

def compute_gravity_response(model, stations):
    """
    Compute the gravity response using the GeoInvLab3D operator.

    The operator returns gravity values in m/s2.
    Results are converted to mGal.
    """

    operator = GravityOperator(
        observation_locations=stations
    )

    gravity_si = np.asarray(
        operator.predict(model),
        dtype=float,
    ).ravel()

    gravity_mgal = gravity_si * SI_TO_MGAL

    if gravity_mgal.size != stations.shape[0]:
        raise ValueError(
            "The number of predicted gravity values does not match "
            "the number of observation stations."
        )

    if not np.all(np.isfinite(gravity_mgal)):
        raise ValueError(
            "The gravity forward response contains non-finite values."
        )

    return gravity_mgal


# =============================================================================
# Noise generation
# =============================================================================

def generate_noisy_data(clean_data, noise_fraction, seed):
    """
    Add reproducible Gaussian noise to gravity observations.

    The noise standard deviation is:

        sigma = noise_fraction * RMS(clean_data)

    where RMS is the root-mean-square amplitude of the noise-free
    gravity response.
    """

    clean_data = np.asarray(clean_data, dtype=float)

    rms = np.sqrt(np.mean(clean_data**2))

    sigma = noise_fraction * rms

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=clean_data.shape,
    )

    noisy_data = clean_data + noise

    return noisy_data, noise, sigma


def generate_all_noisy_datasets(stations, clean_data):
    """Generate and save the 1%, 3%, and 5% noisy datasets."""

    print("\nGenerating noisy gravity datasets")
    print("---------------------------------")

    for index, (folder_name, noise_fraction) in enumerate(
        NOISE_LEVELS.items()
    ):

        output_dir = NOISY_DIR / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        seed = BASE_RANDOM_SEED + index

        noisy_data, noise, sigma = generate_noisy_data(
            clean_data=clean_data,
            noise_fraction=noise_fraction,
            seed=seed,
        )

        save_gravity_data(
            stations,
            noisy_data,
            output_dir / "gravity_observations_noisy.dat",
        )

        save_gravity_data(
            stations,
            noise,
            output_dir / "gravity_noise_realization.dat",
        )

        uncertainties = np.column_stack(
            (
                stations,
                np.full(stations.shape[0], sigma),
            )
        )

        np.savetxt(
            output_dir / "gravity_uncertainties.dat",
            uncertainties,
            fmt="%.10e",
            header="x_m y_m z_m sigma_mGal",
            comments="# ",
        )

        print(f"\n{folder_name}")
        print(f"  Noise level       : {100.0 * noise_fraction:.1f}%")
        print(f"  Random seed       : {seed}")
        print(f"  Noise standard dev: {sigma:.6e} mGal")
        print(
            f"  Noisy data range  : "
            f"{noisy_data.min():.6e} to {noisy_data.max():.6e} mGal"
        )
        print(f"  Output directory  : {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate the gravity model and all synthetic observations."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NOISY_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("GeoInvLab3D - Synthetic gravity data generation")
    print("=" * 65)

    # 1. Create mesh
    print("\nCreating structured mesh...")

    mesh = create_mesh()

    centers = np.asarray(
        mesh.cell_centers(),
        dtype=float,
    )

    print(f"  Mesh dimensions : {NX} x {NY} x {NZ}")
    print(f"  Number of cells : {len(centers)}")
    print(f"  Cell dimensions : {DX:.1f} x {DY:.1f} x {DZ:.1f} m")

    # 2. Create true and initial models
    print("\nCreating density-contrast models...")

    true_model, centers, true_values = create_true_model(mesh)

    initial_model = create_initial_model(mesh)

    print(f"  Background contrast : {BACKGROUND:.1f} kg/m3")
    print(f"  Dense body contrast : {DENSE_BODY:.1f} kg/m3")
    print(f"  Light body contrast : {LIGHT_BODY:.1f} kg/m3")
    print(f"  Dense body cells    : {np.count_nonzero(true_values == DENSE_BODY)}")
    print(f"  Light body cells    : {np.count_nonzero(true_values == LIGHT_BODY)}")

    # 3. Create observation stations
    print("\nCreating gravity stations...")

    stations = create_stations()

    print(f"  Number of stations : {len(stations)}")
    print(f"  Station elevation  : {STATION_HEIGHT:.1f} m")

    # 4. Save mesh, models, and stations
    print("\nSaving model and geometry files...")

    save_mesh(
        mesh,
        centers,
        DATA_DIR / "mesh.dat",
    )

    save_model(
        centers,
        true_values,
        DATA_DIR / "model_true.dat",
    )

    save_model(
        centers,
        np.asarray(initial_model.values, dtype=float).ravel(),
        DATA_DIR / "model_initial.dat",
    )

    save_stations(
        stations,
        DATA_DIR / "stations.dat",
    )

    # 5. Compute noise-free gravity observations
    print("\nComputing noise-free gravity response...")

    clean_data = compute_gravity_response(
        true_model,
        stations,
    )

    save_gravity_data(
        stations,
        clean_data,
        DATA_DIR / "gravity_observations.dat",
    )

    clean_rms = np.sqrt(np.mean(clean_data**2))

    print(f"  Gravity range : {clean_data.min():.6e} to {clean_data.max():.6e} mGal")
    print(f"  Gravity RMS   : {clean_rms:.6e} mGal")

    # 6. Generate noisy datasets
    generate_all_noisy_datasets(
        stations,
        clean_data,
    )

    # 7. Summary
    print("\n" + "=" * 65)
    print("Synthetic gravity data generation completed.")
    print("=" * 65)

    print(f"\nNoise-free data : {DATA_DIR / 'gravity_observations.dat'}")
    print(f"Noisy datasets  : {NOISY_DIR}")

    print("\nGenerated directory structure:")
    print("examples/gravity/")
    print("├── create_gravity_true_model.py")
    print("├── data/")
    print("│   ├── mesh.dat")
    print("│   ├── model_true.dat")
    print("│   ├── model_initial.dat")
    print("│   ├── stations.dat")
    print("│   └── gravity_observations.dat")
    print("└── noisy/")
    print("    ├── noise_1pct/")
    print("    │   ├── gravity_observations_noisy.dat")
    print("    │   ├── gravity_noise_realization.dat")
    print("    │   └── gravity_uncertainties.dat")
    print("    ├── noise_3pct/")
    print("    │   └── ...")
    print("    └── noise_5pct/")
    print("        └── ...")


if __name__ == "__main__":
    main()
