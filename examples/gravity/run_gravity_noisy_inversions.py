#!/usr/bin/env python3
"""
Run three independent 3-D gravity inversions using noisy synthetic data.

Experiments:
    - 1% Gaussian noise
    - 3% Gaussian noise
    - 5% Gaussian noise

Run from the repository root:

    python examples/gravity/run_gravity_noisy_inversions.py

Change the regularization parameter:

    python examples/gravity/run_gravity_noisy_inversions.py --lambda 1e-16

Coordinates:
    x = East
    y = North
    z = positive downward
    surface z = 0

Units:
    Observations on disk: mGal
    GravityOperator predictions: m/s²
    Model parameter: density contrast in kg/m³

The inversion uses L2 regularization toward a zero reference model.

Outputs:
    - Inverted density-contrast models
    - Predicted gravity data
    - Model differences
    - Inversion reports
    - Iteration histories (one file per noise experiment)
    - Model and data comparison figures
    
    Boukar Abdelghafour  GeoInvLab3D   MIT License Copyright (c) 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


# =============================================================================
# Paths and imports
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC = REPO_ROOT / "src"

if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from geoinvlab.core.mesh.structured import StructuredMesh3D
    from geoinvlab.core.model import Model3D
    from geoinvlab.core.data.dataset import DataSet

    from geoinvlab.core.inversion.problem import InverseProblem
    from geoinvlab.core.inversion.engine import InversionEngine
    from geoinvlab.core.inversion.optimizer import GaussNewtonSolver
    from geoinvlab.core.inversion.regularization import L2Regularization

    from geoinvlab.methods.gravity.method import GravityMethod

except ImportError as exc:
    raise ImportError(
        "Could not import GeoInvLab3D components. "
        "Run this script from the repository root using the "
        "project Python environment."
    ) from exc


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = SCRIPT_DIR / "data"
NOISY_DIR = SCRIPT_DIR / "noisy"
RESULTS_DIR = SCRIPT_DIR / "inversion_results"

NOISE_EXPERIMENTS = {
    "noise_1pct": 0.01,
    "noise_3pct": 0.03,
    "noise_5pct": 0.05,
}

SI_TO_MGAL = 1.0e5
MGAL_TO_SI = 1.0e-5

HISTORY_FILENAME = "inversion_history.dat"


# =============================================================================
# Utility functions
# =============================================================================

def load_table(path: Path, expected_columns: int) -> np.ndarray:
    """Load an ASCII table, ignoring its commented header."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    values = np.loadtxt(path, comments="#", dtype=float)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2 or values.shape[1] != expected_columns:
        raise ValueError(
            f"Invalid format in {path.name}: "
            f"expected {expected_columns} columns, "
            f"received {values.shape[1] if values.ndim == 2 else 'unknown'}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(f"NaN or Inf values found in {path}.")

    return values


def save_table(
    path: Path,
    header: str,
    values: np.ndarray,
) -> None:
    """Save an ASCII table with a commented header."""

    path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        path,
        np.asarray(values, dtype=float),
        fmt="%.10e",
        header=header,
        comments="# ",
    )


def calculate_rms(values: np.ndarray) -> float:
    """Calculate the root mean square."""

    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def validate_history_file(path: Path) -> None:
    """
    Validate the Gauss-Newton iteration history.

    Expected columns:
        iteration
        data_rms
        normalized_rms
        phi_d
        phi_m
        phi
        model_min
        model_max
        elapsed_s
        alpha
        relative_update
    """

    if not path.exists():
        raise RuntimeError(
            "The inversion completed, but the history file was not created: "
            f"{path}"
        )

    history = np.loadtxt(path, comments="#", dtype=float)

    if history.ndim == 1:
        history = history.reshape(1, -1)

    if history.ndim != 2 or history.shape[1] != 11:
        raise RuntimeError(
            f"Invalid inversion history in {path}. "
            f"Expected 11 columns, received "
            f"{history.shape[1] if history.ndim == 2 else 'unknown'}."
        )

    if not np.all(np.isfinite(history)):
        raise RuntimeError(
            f"Non-finite values detected in inversion history: {path}"
        )

    print(f"  History file       : {path}")
    print(f"  Recorded iterations: {history.shape[0]}")
    print(f"  Last iteration     : {int(history[-1, 0])}")


# =============================================================================
# Mesh and model loading
# =============================================================================

def load_mesh(data_dir: Path) -> StructuredMesh3D:
    """
    Reconstruct the structured mesh from mesh.dat.

    The generated mesh file contains one row per cell:

        ix iy iz x_center y_center z_center dx dy dz

    Mesh dimensions are inferred from the cell indices.
    """

    table = load_table(
        data_dir / "mesh.dat",
        expected_columns=9,
    )

    ix = table[:, 0].astype(int)
    iy = table[:, 1].astype(int)
    iz = table[:, 2].astype(int)

    nx = len(np.unique(ix))
    ny = len(np.unique(iy))
    nz = len(np.unique(iz))

    dx = float(np.median(table[:, 6]))
    dy = float(np.median(table[:, 7]))
    dz = float(np.median(table[:, 8]))

    x_origin = float(np.min(table[:, 3]) - dx / 2.0)
    y_origin = float(np.min(table[:, 4]) - dy / 2.0)
    z_origin = float(np.min(table[:, 5]) - dz / 2.0)

    mesh = StructuredMesh3D(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        dz=dz,
        origin=(x_origin, y_origin, z_origin),
    )

    centers = np.asarray(mesh.cell_centers(), dtype=float)

    if centers.shape[0] != table.shape[0]:
        raise ValueError(
            "The number of cells reconstructed from mesh.dat "
            "does not match the mesh file."
        )

    if not np.allclose(centers, table[:, 3:6]):
        raise ValueError(
            "The reconstructed mesh cell centers do not match "
            "the coordinates stored in mesh.dat. Check the cell ordering."
        )

    print(f"  Mesh dimensions: {nx} x {ny} x {nz}")
    print(f"  Number of cells: {mesh.n_cells}")
    print(f"  Cell dimensions: {dx:g} x {dy:g} x {dz:g} m")

    return mesh


def load_model_values(
    path: Path,
    centers: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """
    Load a density-contrast model.

    Supported formats:
        4 columns: x y z density_contrast
        7 columns: ix iy iz x y z density_contrast
    """

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    table = np.loadtxt(path, comments="#", dtype=float)

    if table.ndim == 1:
        table = table.reshape(1, -1)

    if table.shape[0] != n_cells:
        raise ValueError(
            f"{path.name}: expected {n_cells} cells, "
            f"received {table.shape[0]}."
        )

    if table.shape[1] == 4:
        model_centers = table[:, :3]
        values = table[:, 3]

    elif table.shape[1] == 7:
        model_centers = table[:, 3:6]
        values = table[:, 6]

    else:
        raise ValueError(
            f"Unsupported model format in {path.name}. "
            "Expected 4 or 7 columns."
        )

    if not np.allclose(model_centers, centers):
        raise ValueError(
            f"Cell coordinates in {path.name} do not match the mesh."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid model values in {path}.")

    return values


def load_stations(data_dir: Path) -> np.ndarray:
    """Load gravity station coordinates."""

    return load_table(
        data_dir / "stations.dat",
        expected_columns=3,
    )


def load_noisy_observations(
    experiment_dir: Path,
    stations: np.ndarray,
) -> np.ndarray:
    """Load noisy gravity observations and verify station coordinates."""

    table = load_table(
        experiment_dir / "gravity_observations_noisy.dat",
        expected_columns=4,
    )

    observation_locations = table[:, :3]
    observed_mgal = table[:, 3]

    if observation_locations.shape != stations.shape:
        raise ValueError(
            "The number of observation locations does not match "
            "the number of gravity stations."
        )

    if not np.allclose(observation_locations, stations):
        raise ValueError(
            "Observation coordinates do not match stations.dat."
        )

    return observed_mgal


# =============================================================================
# Plotting
# =============================================================================

def create_figures(
    output_dir: Path,
    mesh: StructuredMesh3D,
    centers: np.ndarray,
    true_values: np.ndarray,
    inverted_values: np.ndarray,
    stations: np.ndarray,
    observed_mgal: np.ndarray,
    predicted_mgal: np.ndarray,
    noise_label: str,
) -> None:
    """Generate model and gravity data comparison figures."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is unavailable. Figures were not generated.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Central Y section
    y_values = np.unique(centers[:, 1])
    central_y = y_values[len(y_values) // 2]

    selection = np.isclose(centers[:, 1], central_y)

    x_section = centers[selection, 0]
    z_section = centers[selection, 2]

    true_section = true_values[selection]
    inverted_section = inverted_values[selection]

    # -------------------------------------------------------------
    # Figure 1: true and inverted models
    # -------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    model_plots = [
        (axes[0], true_section, "True density contrast"),
        (axes[1], inverted_section, "Inverted density contrast"),
    ]

    for ax, values, title in model_plots:

        scatter = ax.scatter(
            x_section,
            z_section,
            c=values,
            s=45,
            marker="s",
            cmap="RdBu_r",
        )

        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Depth Z (m)")
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")

        fig.colorbar(
            scatter,
            ax=ax,
            label="Density contrast (kg/m³)",
        )

    fig.suptitle(
        f"Gravity inversion — {noise_label}, "
        f"central section Y = {central_y:.1f} m"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "gravity_model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 2: observed, predicted and residual gravity maps
    # -------------------------------------------------------------

    residual_mgal = predicted_mgal - observed_mgal

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    maps = [
        (observed_mgal, "Observed gravity"),
        (predicted_mgal, "Predicted gravity"),
        (residual_mgal, "Residual"),
    ]

    for ax, (values, title) in zip(axes, maps):

        scatter = ax.scatter(
            stations[:, 0],
            stations[:, 1],
            c=values,
            s=35,
            marker="s",
        )

        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal", adjustable="box")

        fig.colorbar(
            scatter,
            ax=ax,
            label="Gravity (mGal)",
        )

    fig.suptitle(f"Gravity data fit — {noise_label}")

    fig.tight_layout()

    fig.savefig(
        output_dir / "gravity_data_fit.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 3: model error
    # -------------------------------------------------------------

    difference = inverted_values - true_values

    fig, ax = plt.subplots(figsize=(8, 5))

    scatter = ax.scatter(
        x_section,
        z_section,
        c=difference[selection],
        s=45,
        marker="s",
        cmap="RdBu_r",
    )

    ax.set_title(f"Inverted model - true model ({noise_label})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Depth Z (m)")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    fig.colorbar(
        scatter,
        ax=ax,
        label="Density difference (kg/m³)",
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "gravity_model_error.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Figures saved in: {output_dir}")


# =============================================================================
# Single inversion
# =============================================================================

def run_single_inversion(
    noise_label: str,
    noise_fraction: float,
    lambda_value: float,
    mesh: StructuredMesh3D,
    centers: np.ndarray,
    true_values: np.ndarray,
    initial_values: np.ndarray,
    stations: np.ndarray,
) -> dict:
    """Run one gravity inversion for a specified noise level."""

    print("\n" + "=" * 72)
    print(f"GRAVITY INVERSION — {noise_label.upper()}")
    print("=" * 72)

    experiment_dir = NOISY_DIR / noise_label
    output_dir = RESULTS_DIR / noise_label
    figures_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    # History file specific to this noise experiment.
    history_file = output_dir / HISTORY_FILENAME

    # -------------------------------------------------------------
    # 1. Load noisy observations
    # -------------------------------------------------------------

    print("\n[1/7] Loading noisy observations...")

    observed_mgal = load_noisy_observations(
        experiment_dir,
        stations,
    )

    observed_si = observed_mgal * MGAL_TO_SI

    print(f"  Noise level       : {100.0 * noise_fraction:.1f}%")
    print(f"  Number of stations: {len(stations)}")
    print(
        f"  Observed range    : "
        f"{observed_mgal.min():.6e} to "
        f"{observed_mgal.max():.6e} mGal"
    )

    # -------------------------------------------------------------
    # 2. Construct models
    # -------------------------------------------------------------

    print("\n[2/7] Constructing inversion models...")

    initial_model = Model3D(
        mesh=mesh,
        values=initial_values.copy(),
        name="gravity_initial_density_contrast",
    )

    reference_model = Model3D(
        mesh=mesh,
        values=np.zeros(mesh.n_cells, dtype=float),
        name="gravity_zero_reference",
    )

    # -------------------------------------------------------------
    # 3. Create dataset
    # -------------------------------------------------------------

    print("\n[3/7] Creating gravity dataset...")

    dataset = DataSet(
        values=observed_si,
        locations=stations,
        uncertainties=None,
        name=f"Synthetic gravity observations ({noise_label})",
        units="m/s²",
    )

    # -------------------------------------------------------------
    # 4. Create gravity operator
    # -------------------------------------------------------------

    print("\n[4/7] Creating gravity operator...")

    gravity_method = GravityMethod()

    operator = gravity_method.create_operator(
        observation_locations=stations,
    )

    print(f"  Operator: {operator.__class__.__name__}")

    # -------------------------------------------------------------
    # 5. Construct inverse problem
    # -------------------------------------------------------------

    print("\n[5/7] Constructing inverse problem...")

    regularization = L2Regularization(
        reference_model=reference_model,
    )

    problem = InverseProblem(
        model=initial_model,
        data=dataset,
        operator=operator,
        regularization=regularization,
        regularization_parameter=lambda_value,
    )

    print(f"  Number of data      : {problem.n_data}")
    print(f"  Number of parameters: {problem.n_parameters}")
    print("  Regularization      : L2")
    print(f"  Lambda              : {lambda_value:.6e}")

    # -------------------------------------------------------------
    # 6. Run Gauss-Newton inversion and save iteration history
    # -------------------------------------------------------------

    print("\n[6/7] Running Gauss-Newton inversion...")

    print(f"  History output     : {history_file}")

    start_time = time.perf_counter()

    # Initialize Gauss-Newton with an individual history file.
    # The solver records iteration 0 and every accepted iteration.
    solver = GaussNewtonSolver(
        history_file=history_file,
    )

    # InversionEngine uses the supplied Gauss-Newton solver.
    engine = InversionEngine(
        solver=solver,
    )

    result = engine.solve(problem)

    elapsed_time = time.perf_counter() - start_time

    inverted_values = np.asarray(
        result.values,
        dtype=float,
    ).reshape(-1)

    if inverted_values.size != mesh.n_cells:
        raise RuntimeError(
            "The inverted model size does not match the number of cells."
        )

    inverted_model = result.model

    # Verify that the iteration history was saved correctly.
    validate_history_file(history_file)

    print("  Inversion completed.")

    # -------------------------------------------------------------
    # 7. Calculate predictions and diagnostics
    # -------------------------------------------------------------

    print("\n[7/7] Calculating predictions and diagnostics...")

    predicted_si = np.asarray(
        operator.predict(inverted_model),
        dtype=float,
    ).reshape(-1)

    predicted_mgal = predicted_si * SI_TO_MGAL

    residual_mgal = predicted_mgal - observed_mgal

    observed_rms = calculate_rms(observed_mgal)
    residual_rms = calculate_rms(residual_mgal)

    model_error = inverted_values - true_values

    model_error_rms = calculate_rms(model_error)

    relative_model_error = (
        np.linalg.norm(model_error)
        / max(
            np.linalg.norm(true_values),
            np.finfo(float).eps,
        )
    ) * 100.0

    # -------------------------------------------------------------
    # Save model and data results
    # -------------------------------------------------------------

    save_table(
        output_dir / "model_inverted.dat",
        "x_center_m y_center_m z_center_m density_contrast_kg_m3",
        np.column_stack((centers, inverted_values)),
    )

    save_table(
        output_dir / "gravity_predictions.dat",
        "x_m y_m z_m observed_mGal predicted_mGal residual_mGal",
        np.column_stack(
            (
                stations,
                observed_mgal,
                predicted_mgal,
                residual_mgal,
            )
        ),
    )

    save_table(
        output_dir / "model_difference.dat",
        "x_center_m y_center_m z_center_m inverted_minus_true_kg_m3",
        np.column_stack((centers, model_error)),
    )

    # -------------------------------------------------------------
    # Save inversion report
    # -------------------------------------------------------------

    report = f"""GeoInvLab3D — 3-D gravity inversion report
================================================

Experiment
----------
Noise level: {100.0 * noise_fraction:.1f}%
Noise directory: {experiment_dir.name}

Mesh
----
Dimensions: {mesh.nx} x {mesh.ny} x {mesh.nz}
Number of cells: {mesh.n_cells}

Acquisition
-----------
Number of stations: {len(stations)}

Inversion
---------
Solver: {engine.solver.__class__.__name__}
Regularization: L2
Lambda: {lambda_value:.10e}
Elapsed time: {elapsed_time:.6f} s

Iteration history
-----------------
File: {history_file.name}
Number of recorded iterations: {len(np.loadtxt(history_file, comments="#", ndmin=2))}
Columns:
iteration data_rms normalized_rms phi_d phi_m phi
model_min model_max elapsed_s alpha relative_update

Objective diagnostics
---------------------
Objective value: {result.objective_value:.10e}
Data misfit: {result.data_misfit:.10e}
Regularization value: {result.regularization_value:.10e}

Data fit
--------
Observed RMS: {observed_rms:.10e} mGal
Residual RMS: {residual_rms:.10e} mGal
Maximum absolute residual: {np.max(np.abs(residual_mgal)):.10e} mGal

Model
-----
True minimum: {true_values.min():.10e} kg/m3
True maximum: {true_values.max():.10e} kg/m3

Inverted minimum: {inverted_values.min():.10e} kg/m3
Inverted maximum: {inverted_values.max():.10e} kg/m3

Model error RMS: {model_error_rms:.10e} kg/m3
Relative model error: {relative_model_error:.6f} %

Notes
-----
The inversion uses an unconstrained L2 formulation.
Negative density contrasts are mathematically possible.

The data are supplied without explicit uncertainty weighting,
consistent with the reference inversion script.
"""

    (output_dir / "inversion_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------------

    create_figures(
        output_dir=figures_dir,
        mesh=mesh,
        centers=centers,
        true_values=true_values,
        inverted_values=inverted_values,
        stations=stations,
        observed_mgal=observed_mgal,
        predicted_mgal=predicted_mgal,
        noise_label=f"{100.0 * noise_fraction:.0f}% noise",
    )

    # -------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------

    print("\n" + "-" * 65)
    print(f"SUMMARY — {noise_label.upper()}")
    print("-" * 65)

    print(f"Objective value      : {result.objective_value:.6e}")
    print(f"Data misfit          : {result.data_misfit:.6e}")
    print(f"Regularization value : {result.regularization_value:.6e}")

    print(f"Observed RMS         : {observed_rms:.6e} mGal")
    print(f"Residual RMS         : {residual_rms:.6e} mGal")

    print(
        f"Inverted model range : "
        f"{inverted_values.min():.4f} to "
        f"{inverted_values.max():.4f} kg/m³"
    )

    print(f"Model error RMS      : {model_error_rms:.6e} kg/m³")
    print(f"Relative model error : {relative_model_error:.4f}%")
    print(f"Elapsed time         : {elapsed_time:.2f} s")

    print(f"History saved        : {history_file}")

    return {
        "noise_level_percent": 100.0 * noise_fraction,
        "objective_value": float(result.objective_value),
        "data_misfit": float(result.data_misfit),
        "regularization_value": float(result.regularization_value),
        "observed_rms_mgal": observed_rms,
        "residual_rms_mgal": residual_rms,
        "model_error_rms": model_error_rms,
        "relative_model_error_percent": relative_model_error,
        "model_min": float(inverted_values.min()),
        "model_max": float(inverted_values.max()),
        "elapsed_time_s": elapsed_time,
        "history_file": str(history_file),
        "output_dir": str(output_dir),
    }


# =============================================================================
# Run all experiments
# =============================================================================

def run_all_inversions(lambda_value: float) -> None:
    """Run the three gravity inversions and compare their diagnostics."""

    print("\n" + "=" * 72)
    print("GeoInvLab3D — 3-D GRAVITY INVERSIONS WITH NOISY DATA")
    print("=" * 72)

    # -------------------------------------------------------------
    # 1. Load common mesh and models
    # -------------------------------------------------------------

    print("\nLoading common mesh and models...")

    mesh = load_mesh(DATA_DIR)

    centers = np.asarray(
        mesh.cell_centers(),
        dtype=float,
    )

    true_values = load_model_values(
        DATA_DIR / "model_true.dat",
        centers,
        mesh.n_cells,
    )

    initial_values = load_model_values(
        DATA_DIR / "model_initial.dat",
        centers,
        mesh.n_cells,
    )

    stations = load_stations(DATA_DIR)

    print(f"  Number of cells   : {mesh.n_cells}")
    print(f"  Number of stations: {len(stations)}")

    print(
        f"  True model range  : "
        f"{true_values.min():.4f} to {true_values.max():.4f} kg/m³"
    )

    print(
        f"  Initial model range: "
        f"{initial_values.min():.4f} to {initial_values.max():.4f} kg/m³"
    )

    # -------------------------------------------------------------
    # 2. Run each noise experiment
    # -------------------------------------------------------------

    summaries = []

    for noise_label, noise_fraction in NOISE_EXPERIMENTS.items():

        experiment_dir = NOISY_DIR / noise_label

        if not experiment_dir.exists():
            raise FileNotFoundError(
                f"Missing noisy dataset directory: {experiment_dir}"
            )

        summary = run_single_inversion(
            noise_label=noise_label,
            noise_fraction=noise_fraction,
            lambda_value=lambda_value,
            mesh=mesh,
            centers=centers,
            true_values=true_values,
            initial_values=initial_values,
            stations=stations,
        )

        summaries.append(summary)

    # -------------------------------------------------------------
    # 3. Save comparison table
    # -------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_columns = [
        "noise_level_percent",
        "objective_value",
        "data_misfit",
        "regularization_value",
        "observed_rms_mgal",
        "residual_rms_mgal",
        "model_error_rms",
        "relative_model_error_percent",
        "model_min",
        "model_max",
        "elapsed_time_s",
    ]

    comparison_table = np.array(
        [
            [summary[column] for column in comparison_columns]
            for summary in summaries
        ],
        dtype=float,
    )

    save_table(
        RESULTS_DIR / "gravity_noise_inversion_comparison.dat",
        " ".join(comparison_columns),
        comparison_table,
    )

    # -------------------------------------------------------------
    # 4. Print final comparison
    # -------------------------------------------------------------

    print("\n" + "=" * 100)
    print("COMPARISON OF GRAVITY INVERSIONS")
    print("=" * 100)

    print(
        f"{'Noise':>8} "
        f"{'Residual RMS (mGal)':>22} "
        f"{'Model error RMS':>20} "
        f"{'Relative error (%)':>21} "
        f"{'Time (s)':>12}"
    )

    print("-" * 100)

    for summary in summaries:
        print(
            f"{summary['noise_level_percent']:7.1f}% "
            f"{summary['residual_rms_mgal']:22.6e} "
            f"{summary['model_error_rms']:20.6e} "
            f"{summary['relative_model_error_percent']:21.4f} "
            f"{summary['elapsed_time_s']:12.2f}"
        )

    print("-" * 100)

    # -------------------------------------------------------------
    # 5. Final history file summary
    # -------------------------------------------------------------

    print("\nINVERSION HISTORY FILES")
    print("-" * 72)

    for summary in summaries:
        print(
            f"{summary['noise_level_percent']:4.0f}% noise : "
            f"{summary['history_file']}"
        )

    print("\nAll inversions completed.")
    print(f"Results directory: {RESULTS_DIR}")


# =============================================================================
# Command-line interface
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run 3-D gravity inversions for 1%, 3%, and 5% noisy datasets."
        )
    )

    parser.add_argument(
        "--lambda",
        dest="lambda_value",
        type=float,
        default=1.0e-16,
        help="L2 regularization parameter (default: 1e-16).",
    )

    args = parser.parse_args()

    if (
        not np.isfinite(args.lambda_value)
        or args.lambda_value < 0.0
    ):
        parser.error("--lambda must be a finite non-negative number.")

    run_all_inversions(args.lambda_value)


if __name__ == "__main__":
    main()
