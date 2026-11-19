#!/usr/bin/env python3
"""
Run a positive-constrained 3-D magnetic inversion with GeoInvLab3D.

The inversion minimizes:

    ||W(G m - d)||² + lambda ||m - m_ref||²

subject to:

    susceptibility >= 0

The initial susceptibility model is also used as the L2 reference model.

Coordinate convention:
    x = East
    y = North
    z = positive downward
    surface = z = 0

Run from the repository root:

    python examples/magnetic/synthetic/run_magnetic_inversion_positive.py

Optional parameters:

    --lambda 1.0
    --tolerance 1e-8
    --max-iterations 200
    --verbose

Input files:
    mesh.dat
    stations.dat
    magnetic_observations.dat
    model_initial.dat
    model_true.dat

The script searches for these files in:
    1. examples/magnetic/synthetic/data/
    2. examples/magnetic/synthetic/
    3. examples/magnetic/data/

Results are written to:
    examples/magnetic/synthetic/magnetic_inversion_positive_results/
    
    Boukar Abdelghafour  GeoInvLab3D   MIT License Copyright (c) 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geoinvlab.core.data.dataset import DataSet
from geoinvlab.core.inversion.engine import InversionEngine
from geoinvlab.core.inversion.optimizer import (
    PositiveBoundedLeastSquaresSolver,
)
from geoinvlab.core.inversion.problem import InverseProblem
from geoinvlab.core.inversion.regularization import L2Regularization
from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.core.model.model import Model3D
from geoinvlab.methods.magnetic.method import MagneticMethod


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

RESULTS_DIR = SCRIPT_DIR / "magnetic_inversion_positive_results"

INPUT_FILENAMES = (
    "mesh.dat",
    "stations.dat",
    "magnetic_observations.dat",
    "model_initial.dat",
    "model_true.dat",
)


def resolve_data_directory() -> Path:
    """Find the directory containing all required synthetic input files."""

    candidates = (
        SCRIPT_DIR / "data",
        SCRIPT_DIR,
        SCRIPT_DIR.parent / "data",
    )

    for directory in candidates:
        if all((directory / filename).is_file()
               for filename in INPUT_FILENAMES):
            return directory

    searched = "\n".join(f"  - {path}" for path in candidates)
    missing = "\n".join(f"  - {name}" for name in INPUT_FILENAMES)

    raise FileNotFoundError(
        "Could not locate all magnetic synthetic input files.\n"
        f"Searched directories:\n{searched}\n"
        f"Required files:\n{missing}"
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mesh(path: Path) -> StructuredMesh3D:
    """Load structured mesh geometry from mesh.dat."""

    if not path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    values = np.asarray(np.loadtxt(path, comments="#"), dtype=float).ravel()

    if values.size != 9:
        raise ValueError(
            f"Expected 9 values in {path}, got {values.size}."
        )

    nx, ny, nz = values[:3].astype(int)
    dx, dy, dz = values[3:6]
    origin = tuple(float(value) for value in values[6:9])

    if min(nx, ny, nz) < 1:
        raise ValueError("Mesh dimensions must be positive.")
    if min(dx, dy, dz) <= 0.0:
        raise ValueError("Mesh cell dimensions must be positive.")

    return StructuredMesh3D(
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        dx=float(dx),
        dy=float(dy),
        dz=float(dz),
        origin=origin,
    )


def load_stations(path: Path) -> np.ndarray:
    """Load magnetic observation coordinates."""

    if not path.is_file():
        raise FileNotFoundError(f"Station file not found: {path}")

    stations = np.loadtxt(path, comments="#", ndmin=2)

    if stations.shape[1] != 3:
        raise ValueError(
            "stations.dat must contain three columns: x, y, z."
        )
    if not np.all(np.isfinite(stations)):
        raise ValueError("Station coordinates contain non-finite values.")

    return stations


def load_observations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load observation coordinates and total-field magnetic anomaly."""

    if not path.is_file():
        raise FileNotFoundError(f"Observation file not found: {path}")

    table = np.loadtxt(path, comments="#", ndmin=2)

    if table.shape[1] != 4:
        raise ValueError(
            "magnetic_observations.dat must contain four columns: "
            "x, y, z, TMI."
        )
    if not np.all(np.isfinite(table)):
        raise ValueError("Magnetic observations contain non-finite values.")

    return table[:, :3], table[:, 3]


def load_model(path: Path, mesh: StructuredMesh3D, name: str) -> Model3D:
    """Load a cell-centred susceptibility model from an ASCII file."""

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    table = np.loadtxt(path, comments="#", ndmin=2)

    if table.shape[1] != 4:
        raise ValueError(
            f"{path.name} must contain four columns: "
            "x, y, z, susceptibility."
        )
    if table.shape[0] != mesh.n_cells:
        raise ValueError(
            f"{path.name} contains {table.shape[0]} cells, "
            f"but the mesh has {mesh.n_cells} cells."
        )
    if not np.all(np.isfinite(table)):
        raise ValueError(f"{path.name} contains non-finite values.")

    return Model3D(
        mesh=mesh,
        values=table[:, 3],
        name=name,
    )


# =============================================================================
# STATISTICS
# =============================================================================

def calculate_data_statistics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate magnetic data-fit statistics."""

    residual = predicted - observed

    return {
        "rms": float(np.sqrt(np.mean(residual**2))),
        "mean_absolute_error": float(np.mean(np.abs(residual))),
        "maximum_absolute_error": float(np.max(np.abs(residual))),
        "relative_l2": float(
            np.linalg.norm(residual) / max(np.linalg.norm(observed), 1e-15)
        ),
    }


def calculate_model_statistics(
    inverted: Model3D,
    true_model: Model3D,
) -> dict[str, float]:
    """Calculate statistics for the inverted susceptibility model."""

    values = np.asarray(inverted.values, dtype=float)
    difference = values - np.asarray(true_model.values, dtype=float)

    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "model_rmse": float(np.sqrt(np.mean(difference**2))),
        "negative_cells": int(np.count_nonzero(values < 0.0)),
        "positive_cells": int(np.count_nonzero(values > 0.0)),
        "zero_cells": int(np.count_nonzero(values == 0.0)),
    }


# =============================================================================
# RESULTS SAVING
# =============================================================================

def save_model(path: Path, model: Model3D) -> None:
    """Save the inverted susceptibility model as an ASCII table."""

    centers = model.mesh.cell_centers()
    table = np.column_stack((centers, model.values))

    np.savetxt(
        path,
        table,
        fmt="%.10e",
        header="x_center_m y_center_m z_center_m susceptibility_SI",
    )


def save_data_fit(
    path: Path,
    locations: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """Save observed, predicted and residual magnetic data."""

    residual = predicted - observed
    table = np.column_stack((locations, observed, predicted, residual))

    np.savetxt(
        path,
        table,
        fmt="%.10e",
        header=(
            "x_m y_m z_m observed_TMI_nT "
            "predicted_TMI_nT residual_nT"
        ),
    )


def save_statistics(
    path: Path,
    data_statistics: dict[str, float],
    model_statistics: dict[str, float],
    regularization_parameter: float,
) -> None:
    """Save inversion statistics in a plain ASCII file."""

    values = np.array(
        [
            data_statistics["rms"],
            data_statistics["mean_absolute_error"],
            data_statistics["maximum_absolute_error"],
            data_statistics["relative_l2"],
            model_statistics["minimum"],
            model_statistics["maximum"],
            model_statistics["mean"],
            model_statistics["rms"],
            model_statistics["model_rmse"],
            model_statistics["negative_cells"],
            model_statistics["positive_cells"],
            model_statistics["zero_cells"],
            regularization_parameter,
        ],
        dtype=float,
    )

    header = (
        "data_rms_nT\n"
        "mean_absolute_error_nT\n"
        "maximum_absolute_error_nT\n"
        "relative_l2\n"
        "susceptibility_minimum_SI\n"
        "susceptibility_maximum_SI\n"
        "susceptibility_mean_SI\n"
        "susceptibility_rms_SI\n"
        "model_rmse_SI\n"
        "negative_cells\n"
        "positive_cells\n"
        "zero_cells\n"
        "regularization_lambda"
    )

    np.savetxt(path, values, fmt="%.12e", header=header)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GeoInvLab3D positive-constrained 3-D magnetic inversion."
        )
    )

    parser.add_argument(
        "--lambda",
        dest="regularization_parameter",
        type=float,
        default=1.0,
        help="L2 regularization parameter (default: 1.0).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Optimization tolerance (default: 1e-8).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Maximum solver iterations (default: 200).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display solver convergence information.",
    )

    args = parser.parse_args()

    lam = args.regularization_parameter

    if not np.isfinite(lam) or lam <= 0.0:
        parser.error("--lambda must be finite and strictly positive.")
    if not np.isfinite(args.tolerance) or args.tolerance <= 0.0:
        parser.error("--tolerance must be finite and strictly positive.")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1.")

    data_dir = resolve_data_directory()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GeoInvLab3D - POSITIVE-CONSTRAINED MAGNETIC INVERSION")
    print("=" * 72)
    print(f"Script directory : {SCRIPT_DIR}")
    print(f"Input directory  : {data_dir}")
    print(f"Results directory: {RESULTS_DIR}")

    # -------------------------------------------------------------------------
    # 1. Mesh
    # -------------------------------------------------------------------------

    print("\n[1/9] Loading mesh...")

    mesh = load_mesh(data_dir / "mesh.dat")

    print(f"Mesh dimensions: {mesh.nx} x {mesh.ny} x {mesh.nz}")
    print(f"Number of cells: {mesh.n_cells}")
    print(f"Cell size      : {mesh.dx} x {mesh.dy} x {mesh.dz} m")
    print(f"Origin         : {mesh.origin}")

    # -------------------------------------------------------------------------
    # 2. Stations and observations
    # -------------------------------------------------------------------------

    print("\n[2/9] Loading stations and observations...")

    stations = load_stations(data_dir / "stations.dat")
    locations, observed = load_observations(
        data_dir / "magnetic_observations.dat"
    )

    if stations.shape != locations.shape:
        raise ValueError(
            "Station coordinates and observation coordinates "
            "have different shapes."
        )
    if not np.allclose(stations, locations):
        raise ValueError(
            "Station coordinates do not match observation coordinates."
        )
    if not np.allclose(stations[:, 2], 0.0):
        raise ValueError("Expected all observation stations at z = 0.")

    print(f"Number of stations    : {stations.shape[0]}")
    print(f"Number of observations: {observed.size}")
    print(
        f"Observation range     : {observed.min():.6f} to "
        f"{observed.max():.6f} nT"
    )
    print(
        f"Observation RMS       : "
        f"{np.sqrt(np.mean(observed**2)):.6f} nT"
    )

    # -------------------------------------------------------------------------
    # 3. Initial and true models
    # -------------------------------------------------------------------------

    print("\n[3/9] Loading susceptibility models...")

    initial_model = load_model(
        data_dir / "model_initial.dat",
        mesh,
        name="initial_susceptibility",
    )
    true_model = load_model(
        data_dir / "model_true.dat",
        mesh,
        name="true_susceptibility",
    )

    if np.any(initial_model.values < 0.0):
        raise ValueError(
            "The initial susceptibility model contains negative values."
        )

    print(
        f"Initial model range: {initial_model.values.min():.6e} to "
        f"{initial_model.values.max():.6e} SI"
    )
    print(
        f"True model range   : {true_model.values.min():.6e} to "
        f"{true_model.values.max():.6e} SI"
    )

    # -------------------------------------------------------------------------
    # 4. Dataset
    # -------------------------------------------------------------------------

    print("\n[4/9] Building magnetic dataset...")

    dataset = DataSet(
        values=observed,
        locations=locations,
        name="synthetic_magnetic_TMI",
        units="nT",
    )

    # -------------------------------------------------------------------------
    # 5. Magnetic operator
    # -------------------------------------------------------------------------

    print("\n[5/9] Creating magnetic operator...")

    magnetic_method = MagneticMethod(
        inclination=60.0,
        declination=0.0,
        field_intensity=50000.0,
    )

    operator = magnetic_method.create_operator(
        observation_locations=stations,
    )

    # -------------------------------------------------------------------------
    # 6. Inverse problem
    # -------------------------------------------------------------------------

    print("\n[6/9] Building inverse problem...")

    regularization = L2Regularization(
        reference_model=initial_model,
    )

    problem = InverseProblem(
        model=initial_model,
        data=dataset,
        operator=operator,
        regularization=regularization,
        regularization_parameter=lam,
    )

    # -------------------------------------------------------------------------
    # 7. Positive bounded least-squares solver
    # -------------------------------------------------------------------------

    print("\n[7/9] Configuring positive bounded solver...")

    solver = PositiveBoundedLeastSquaresSolver(
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        lsmr_tolerance="auto",
        verbose=args.verbose,
    )

    engine = InversionEngine(solver=solver)

    print("Physical constraint : susceptibility >= 0 SI")
    print(f"Regularization lambda: {lam:g}")
    print(f"Solver tolerance     : {args.tolerance:g}")
    print(f"Maximum iterations   : {args.max_iterations}")

    # -------------------------------------------------------------------------
    # 8. Run inversion and calculate predicted data
    # -------------------------------------------------------------------------

    print("\n[8/9] Running positive magnetic inversion...")

    result = engine.solve(problem)
    inverted_model = result.model

    print("\nCalculating predicted magnetic data...")

    predicted = np.asarray(operator.predict(inverted_model), dtype=float)

    if predicted.shape != observed.shape:
        raise ValueError(
            "Predicted data have an invalid shape: "
            f"{predicted.shape}, expected {observed.shape}."
        )
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Predicted data contain non-finite values.")

    data_statistics = calculate_data_statistics(observed, predicted)
    model_statistics = calculate_model_statistics(
        inverted_model,
        true_model,
    )

    # -------------------------------------------------------------------------
    # 9. Display and save results
    # -------------------------------------------------------------------------

    print("\nData-fit statistics")
    print("-" * 56)
    for key, value in data_statistics.items():
        print(f"{key:30s}: {value:.8e}")

    print("\nInverted susceptibility statistics")
    print("-" * 56)
    print(f"Minimum susceptibility : {model_statistics['minimum']:.8e} SI")
    print(f"Maximum susceptibility : {model_statistics['maximum']:.8e} SI")
    print(f"Mean susceptibility    : {model_statistics['mean']:.8e} SI")
    print(f"RMS susceptibility     : {model_statistics['rms']:.8e} SI")
    print(f"Model RMSE             : {model_statistics['model_rmse']:.8e} SI")
    print(f"Negative cells         : {model_statistics['negative_cells']}")
    print(f"Positive cells         : {model_statistics['positive_cells']}")
    print(f"Zero cells             : {model_statistics['zero_cells']}")

    # A small numerical tolerance is used only for reporting the constraint.
    if model_statistics["minimum"] < -1e-12:
        print(
            "\nWARNING: the returned model contains values below "
            "the non-negativity tolerance."
        )

    print("\nSaving inversion results...")

    save_model(
        RESULTS_DIR / "model_inverted_positive.dat",
        inverted_model,
    )
    save_data_fit(
        RESULTS_DIR / "data_fit_positive.dat",
        locations,
        observed,
        predicted,
    )
    save_statistics(
        RESULTS_DIR / "inversion_statistics_positive.dat",
        data_statistics,
        model_statistics,
        lam,
    )

    print(f"\nResults directory: {RESULTS_DIR}")
    print("Generated files:")
    print("  - model_inverted_positive.dat")
    print("  - data_fit_positive.dat")
    print("  - inversion_statistics_positive.dat")

    print("\n" + "=" * 72)
    print("POSITIVE MAGNETIC INVERSION FINISHED")
    print("=" * 72)


if __name__ == "__main__":
    main()
