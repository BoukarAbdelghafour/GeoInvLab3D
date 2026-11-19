#!/usr/bin/env python3
"""
Generate a noiseless 3-D synthetic magnetic (TMI) dataset for GeoInvLab3D.

The synthetic model consists of:
- A homogeneous background with moderate susceptibility.
- A highly magnetic rectangular prism.
- A weakly magnetic rectangular prism.

Run from the repository root:
    python examples/magnetic/synthetic/generate_magnetic_synthetic.py

Coordinates used by GeoInvLab3D:
    x = East
    y = North
    z = positive downward
    Surface = z = 0

The MagneticOperator converts coordinates to the z-up convention
required by the analytical magnetic kernel. Do not flip coordinates here.

MIT License
Copyright (c) Boukar 2026
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


# ---------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"

if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


try:
    from geoinvlab.core.mesh.structured import StructuredMesh3D
except ImportError as exc:
    raise ImportError(
        "Could not import StructuredMesh3D from "
        "'geoinvlab.core.mesh.structured'."
    ) from exc


try:
    from geoinvlab.core.model import Model3D
except ImportError as exc:
    raise ImportError(
        "Could not import Model3D from 'geoinvlab.core.model'."
    ) from exc


try:
    from geoinvlab.methods.magnetic.operator import MagneticOperator
except ImportError as exc:
    raise ImportError(
        "Could not import MagneticOperator from "
        "'geoinvlab.methods.magnetic.operator'."
    ) from exc


# ---------------------------------------------------------------------
# Synthetic model configuration
# ---------------------------------------------------------------------

# Mesh dimensions
NX = 20
NY = 20
NZ = 15

DX = 100.0
DY = 100.0
DZ = 100.0

ORIGIN = (0.0, 0.0, 0.0)


# Magnetic susceptibility (SI)
SUSCEPTIBILITY_BACKGROUND = 0.010

SUSCEPTIBILITY_HIGH = 0.050
SUSCEPTIBILITY_LOW = 0.002


# Highly magnetic body
HIGH_BODY_BOUNDS = {
    "x": (700.0, 1300.0),
    "y": (700.0, 1300.0),
    "z": (300.0, 900.0),
}


# Weakly magnetic body
LOW_BODY_BOUNDS = {
    "x": (200.0, 700.0),
    "y": (1200.0, 1700.0),
    "z": (400.0, 900.0),
}


# Regional magnetic field
INCLINATION_DEG = 60.0
DECLINATION_DEG = 0.0
FIELD_INTENSITY_NT = 50000.0


# Surface stations
STATION_SPACING = 100.0
STATION_OFFSET = 50.0


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def save_table(
    path: Path,
    header: str,
    data: np.ndarray,
) -> None:
    """Save an ASCII table with a commented column header."""

    path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        path,
        data,
        fmt="%.10e",
        header=header,
        comments="# ",
    )


def create_body_mask(
    centers: np.ndarray,
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Create a rectangular-prism mask using cell-center coordinates."""

    x_min, x_max = bounds["x"]
    y_min, y_max = bounds["y"]
    z_min, z_max = bounds["z"]

    return (
        (centers[:, 0] >= x_min)
        & (centers[:, 0] < x_max)
        & (centers[:, 1] >= y_min)
        & (centers[:, 1] < y_max)
        & (centers[:, 2] >= z_min)
        & (centers[:, 2] < z_max)
    )


# ---------------------------------------------------------------------
# Main generation routine
# ---------------------------------------------------------------------

def main() -> None:

    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GeoInvLab3D - 3D MAGNETIC SYNTHETIC MODEL GENERATION")
    print("=" * 72)

    # -------------------------------------------------------------
    # 1. Create structured mesh
    # -------------------------------------------------------------

    print("\n[1/6] Creating mesh...")

    mesh = StructuredMesh3D(
        nx=NX,
        ny=NY,
        nz=NZ,
        dx=DX,
        dy=DY,
        dz=DZ,
        origin=ORIGIN,
    )

    centers = np.asarray(mesh.cell_centers(), dtype=float)
    n_cells = centers.shape[0]

    print(f"Mesh dimensions : {NX} x {NY} x {NZ}")
    print(f"Number of cells : {n_cells}")
    print(f"Cell size       : {DX:g} x {DY:g} x {DZ:g} m")

    # -------------------------------------------------------------
    # 2. Define the two anomalous bodies
    # -------------------------------------------------------------

    print("\n[2/6] Building true susceptibility model...")

    high_body = create_body_mask(
        centers,
        HIGH_BODY_BOUNDS,
    )

    low_body = create_body_mask(
        centers,
        LOW_BODY_BOUNDS,
    )

    if np.any(high_body & low_body):
        raise RuntimeError(
            "The two anomalous bodies overlap. "
            "Check their geometrical bounds."
        )

    if not np.any(high_body):
        raise RuntimeError(
            "The highly magnetic body contains no mesh cells."
        )

    if not np.any(low_body):
        raise RuntimeError(
            "The weakly magnetic body contains no mesh cells."
        )

    # -------------------------------------------------------------
    # 3. Build true and initial models
    # -------------------------------------------------------------

    true_values = np.full(
        n_cells,
        SUSCEPTIBILITY_BACKGROUND,
        dtype=float,
    )

    true_values[high_body] = SUSCEPTIBILITY_HIGH
    true_values[low_body] = SUSCEPTIBILITY_LOW

    # Homogeneous initial reference model
    initial_values = np.full(
        n_cells,
        SUSCEPTIBILITY_BACKGROUND,
        dtype=float,
    )

    true_model = Model3D(
        mesh=mesh,
        values=true_values,
        name="magnetic_true_susceptibility",
    )

    initial_model = Model3D(
        mesh=mesh,
        values=initial_values,
        name="magnetic_initial_susceptibility",
    )

    print("\nTrue model:")
    print(
        f"  Background susceptibility : "
        f"{SUSCEPTIBILITY_BACKGROUND:.4f} SI"
    )
    print(
        f"  Highly magnetic body      : "
        f"{SUSCEPTIBILITY_HIGH:.4f} SI"
    )
    print(
        f"  Weakly magnetic body      : "
        f"{SUSCEPTIBILITY_LOW:.4f} SI"
    )

    print("\nAnomalous body geometry:")

    print(
        "  High susceptibility body : "
        f"X={HIGH_BODY_BOUNDS['x']}, "
        f"Y={HIGH_BODY_BOUNDS['y']}, "
        f"Z={HIGH_BODY_BOUNDS['z']}"
    )

    print(
        "  Low susceptibility body  : "
        f"X={LOW_BODY_BOUNDS['x']}, "
        f"Y={LOW_BODY_BOUNDS['y']}, "
        f"Z={LOW_BODY_BOUNDS['z']}"
    )

    print("\nCell counts:")
    print(f"  Background cells : {int(n_cells - high_body.sum() - low_body.sum())}")
    print(f"  High body cells  : {int(high_body.sum())}")
    print(f"  Low body cells   : {int(low_body.sum())}")

    # -------------------------------------------------------------
    # 4. Generate surface observation stations
    # -------------------------------------------------------------

    print("\n[3/6] Generating observation stations...")

    station_axis = np.arange(
        STATION_OFFSET,
        NX * DX,
        STATION_SPACING,
    )

    xx, yy = np.meshgrid(
        station_axis,
        station_axis,
        indexing="ij",
    )

    stations = np.column_stack(
        (
            xx.ravel(),
            yy.ravel(),
            np.zeros(xx.size),
        )
    )

    print(f"Number of stations : {stations.shape[0]}")
    print(f"Station spacing    : {STATION_SPACING:g} m")
    print("Station elevation  : 0 m")

    # -------------------------------------------------------------
    # 5. Calculate synthetic magnetic observations
    # -------------------------------------------------------------

    print("\n[4/6] Computing synthetic TMI observations...")

    operator = MagneticOperator(
        observation_locations=stations,
        inclination=INCLINATION_DEG,
        declination=DECLINATION_DEG,
        field_intensity=FIELD_INTENSITY_NT,
    )

    predicted_nt = np.asarray(
        operator.predict(true_model),
        dtype=float,
    ).reshape(-1)

    if predicted_nt.size != stations.shape[0]:
        raise RuntimeError(
            f"MagneticOperator returned {predicted_nt.size} values "
            f"for {stations.shape[0]} stations."
        )

    if not np.all(np.isfinite(predicted_nt)):
        raise RuntimeError(
            "Magnetic predictions contain NaN or Inf."
        )

    print(f"TMI minimum : {predicted_nt.min():.6f} nT")
    print(f"TMI maximum : {predicted_nt.max():.6f} nT")

    # -------------------------------------------------------------
    # 6. Save synthetic dataset
    # -------------------------------------------------------------

    print("\n[5/6] Saving synthetic dataset...")

    save_table(
        output_dir / "mesh.dat",
        (
            "nx ny nz dx_m dy_m dz_m "
            "origin_x_m origin_y_m origin_z_m"
        ),
        np.array(
            [[NX, NY, NZ, DX, DY, DZ, *ORIGIN]],
            dtype=float,
        ),
    )

    save_table(
        output_dir / "model_true.dat",
        "x_center_m y_center_m z_center_m susceptibility_SI",
        np.column_stack((centers, true_values)),
    )

    save_table(
        output_dir / "model_initial.dat",
        "x_center_m y_center_m z_center_m susceptibility_SI",
        np.column_stack((centers, initial_values)),
    )

    save_table(
        output_dir / "stations.dat",
        "x_m y_m z_m",
        stations,
    )

    save_table(
        output_dir / "magnetic_observations.dat",
        "x_m y_m z_m TMI_anomaly_nT",
        np.column_stack((stations, predicted_nt)),
    )

    # -------------------------------------------------------------
    # README
    # -------------------------------------------------------------

    readme = f"""GeoInvLab3D - Synthetic 3-D Magnetic Example
================================================

Generated by:
generate_magnetic_synthetic.py

Coordinate convention
---------------------
x: East (m)
y: North (m)
z: positive downward (m)
Surface: z = 0.

The magnetic kernel uses z positive upward internally.
MagneticOperator performs the coordinate conversion.
Do not flip coordinates in these files.

Mesh
----
Dimensions: {NX} x {NY} x {NZ} cells
Number of cells: {n_cells}
Cell size: {DX:g} x {DY:g} x {DZ:g} m
Domain: 2000 x 2000 x 1500 m
Padding: none

Synthetic model
---------------
The model consists of a homogeneous background containing
two rectangular-prism susceptibility anomalies.

Background
----------
Susceptibility: {SUSCEPTIBILITY_BACKGROUND:.6f} SI

Highly magnetic body
--------------------
Susceptibility: {SUSCEPTIBILITY_HIGH:.6f} SI
X: {HIGH_BODY_BOUNDS['x'][0]:g} to {HIGH_BODY_BOUNDS['x'][1]:g} m
Y: {HIGH_BODY_BOUNDS['y'][0]:g} to {HIGH_BODY_BOUNDS['y'][1]:g} m
Z: {HIGH_BODY_BOUNDS['z'][0]:g} to {HIGH_BODY_BOUNDS['z'][1]:g} m
Number of cells: {int(high_body.sum())}

Weakly magnetic body
--------------------
Susceptibility: {SUSCEPTIBILITY_LOW:.6f} SI
X: {LOW_BODY_BOUNDS['x'][0]:g} to {LOW_BODY_BOUNDS['x'][1]:g} m
Y: {LOW_BODY_BOUNDS['y'][0]:g} to {LOW_BODY_BOUNDS['y'][1]:g} m
Z: {LOW_BODY_BOUNDS['z'][0]:g} to {LOW_BODY_BOUNDS['z'][1]:g} m
Number of cells: {int(low_body.sum())}

Initial model
-------------
Homogeneous susceptibility: {SUSCEPTIBILITY_BACKGROUND:.6f} SI

Regional magnetic field
-----------------------
Inclination: {INCLINATION_DEG:g} degrees
Declination: {DECLINATION_DEG:g} degrees
Intensity: {FIELD_INTENSITY_NT:g} nT

Observations
------------
Number of stations: {stations.shape[0]}
Station spacing: {STATION_SPACING:g} m
Station elevation: 0 m

Magnetic observations represent the total-field magnetic anomaly (TMI).
No noise has been added.

Files
-----
mesh.dat
    Mesh dimensions and geometry.

model_true.dat
    True susceptibility model.

model_initial.dat
    Homogeneous initial susceptibility model.

stations.dat
    Surface station coordinates.

magnetic_observations.dat
    Synthetic TMI observations in nT.

All files are plain ASCII text with commented column headers.
"""

    (output_dir / "README.txt").write_text(
        readme,
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------

    print("\n[6/6] Generation completed.")

    print("\n" + "=" * 72)
    print("SYNTHETIC MAGNETIC DATASET GENERATED SUCCESSFULLY")
    print("=" * 72)

    print(f"Output directory : {output_dir}")
    print(f"Number of cells  : {n_cells}")
    print(f"Number of stations : {stations.shape[0]}")

    print("\nSusceptibility values:")
    print(f"  Background : {SUSCEPTIBILITY_BACKGROUND:.4f} SI")
    print(f"  High body  : {SUSCEPTIBILITY_HIGH:.4f} SI")
    print(f"  Low body   : {SUSCEPTIBILITY_LOW:.4f} SI")

    print("\nAnomalous cells:")
    print(f"  High body : {int(high_body.sum())}")
    print(f"  Low body  : {int(low_body.sum())}")

    print("\nTMI anomaly:")
    print(f"  Minimum : {predicted_nt.min():.6f} nT")
    print(f"  Maximum : {predicted_nt.max():.6f} nT")

    print("\nGenerated files:")
    for filename in (
        "mesh.dat",
        "model_true.dat",
        "model_initial.dat",
        "stations.dat",
        "magnetic_observations.dat",
        "README.txt",
    ):
        print(f"  {filename}")

    print("=" * 72)


if __name__ == "__main__":
    main()

