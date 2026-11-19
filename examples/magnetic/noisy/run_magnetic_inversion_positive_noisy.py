#!/usr/bin/env python3
"""
Run a positive-constrained 3-D magnetic inversion using noisy synthetic data.

The inversion minimizes:

    ||G m - d_noisy||² + lambda ||m - m_ref||²

subject to:

    susceptibility >= 0

The initial susceptibility model is also used as the L2 reference model.

Coordinate convention:
    x = East
    y = North
    z = positive downward
    surface = z = 0

Run from the repository root:

    python examples/magnetic/noisy/run_magnetic_inversion_positive_noisy.py

Select the noise level:

    --noise-level 1
    --noise-level 3
    --noise-level 5

Optional parameters:

    --lambda 1.0
    --tolerance 1e-8
    --max-iterations 200
    --verbose

Shared input files:
    examples/magnetic/synthetic/data/
        mesh.dat
        stations.dat
        model_initial.dat
        model_true.dat

Noisy observation files:
    examples/magnetic/noisy/

    The script supports the following directory layouts:
        1pct/data/magnetic_observations.dat
        3pct/data/magnetic_observations.dat
        5pct/data/magnetic_observations.dat

    It also supports noise_1pct, noise_3pct and noise_5pct
    directory names and magnetic_observations_noisy.dat.

Results:
    examples/magnetic/noisy/inversion_results/
        noise_1pct/
        noise_3pct/
        noise_5pct/
        
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

SHARED_DATA_DIR = (
    PROJECT_ROOT / "examples" / "magnetic" / "synthetic" / "data"
)

RESULTS_ROOT = SCRIPT_DIR / "inversion_results"

NOISE_LEVELS = (1, 3, 5)


# =============================================================================
# INPUT DIRECTORY RESOLUTION
# =============================================================================

def resolve_observation_file(noise_level: int) -> Path:
    """
    Locate the noisy magnetic observation file.

    Several directory layouts are supported to accommodate
    different noisy-data generation arrangements.
    """

    filenames = (
        "magnetic_observations.dat",
        "magnetic_observations_noisy.dat",
    )

    directory_names = (
        f"{noise_level}pct",
        f"noise_{noise_level}pct",
        f"{noise_level}%",
    )

    candidates = []

    for directory_name in directory_names:

        candidates.extend(
            [
                SCRIPT_DIR / directory_name / "data",
                SCRIPT_DIR / directory_name,
            ]
        )

    for directory_name in directory_names:

        candidates.extend(
            [
                SCRIPT_DIR / "data" / directory_name / "data",
                SCRIPT_DIR / "data" / directory_name,
            ]
        )

    for directory in candidates:

        for filename in filenames:

            path = directory / filename

            if path.is_file():
                return path

    searched = "\n".join(
        f"  - {directory / filename}"
        for directory in candidates
        for filename in filenames
    )

    raise FileNotFoundError(
        f"Could not locate noisy observations for {noise_level}% noise.\n"
        f"Searched locations:\n{searched}"
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mesh(path: Path) -> StructuredMesh3D:
    """Load structured mesh geometry."""

    if not path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    values = np.asarray(
        np.loadtxt(path, comments="#"),
        dtype=float,
    ).ravel()

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
    """Load magnetic observation station coordinates."""

    if not path.is_file():
        raise FileNotFoundError(f"Station file not found: {path}")

    stations = np.loadtxt(
        path,
        comments="#",
        ndmin=2,
    )

    if stations.shape[1] != 3:
        raise ValueError(
            "stations.dat must contain three columns: x, y, z."
        )

    if not np.all(np.isfinite(stations)):
        raise ValueError("Station coordinates contain non-finite values.")

    return stations


def load_observations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load noisy observation coordinates and TMI anomalies."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Magnetic observation file not found: {path}"
        )

    table = np.loadtxt(
        path,
        comments="#",
        ndmin=2,
    )

    if table.shape[1] != 4:
        raise ValueError(
            f"{path.name} must contain four columns: "
            "x, y, z, TMI."
        )

    if not np.all(np.isfinite(table)):
        raise ValueError(
            "Magnetic observations contain non-finite values."
        )

    return table[:, :3], table[:, 3]


def load_model(
    path: Path,
    mesh: StructuredMesh3D,
    name: str,
) -> Model3D:
    """Load a cell-centred susceptibility model."""

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    table = np.loadtxt(
        path,
        comments="#",
        ndmin=2,
    )

    if table.shape[1] != 4:
        raise ValueError(
            f"{path.name} must contain four columns: "
            "x, y, z, susceptibility."
        )

    if table.shape[0] != mesh.n_cells:
        raise ValueError(
            f"{path.name} contains {table.shape[0]} cells, "
            f"but the mesh has {mesh.n_cells}."
        )

    if not np.all(np.isfinite(table)):
        raise ValueError(
            f"{path.name} contains non-finite values."
        )

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
        "rms": float(
            np.sqrt(np.mean(residual**2))
        ),
        "mean_absolute_error": float(
            np.mean(np.abs(residual))
        ),
        "maximum_absolute_error": float(
            np.max(np.abs(residual))
        ),
        "relative_l2": float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(observed), 1e-15)
        ),
    }


def calculate_model_statistics(
    inverted: Model3D,
    true_model: Model3D,
) -> dict[str, float]:
    """Calculate inverted susceptibility model statistics."""

    values = np.asarray(
        inverted.values,
        dtype=float,
    )

    true_values = np.asarray(
        true_model.values,
        dtype=float,
    )

    difference = values - true_values

    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "model_rmse": float(
            np.sqrt(np.mean(difference**2))
        ),
        "negative_cells": int(
            np.count_nonzero(values < 0.0)
        ),
        "positive_cells": int(
            np.count_nonzero(values > 0.0)
        ),
        "zero_cells": int(
            np.count_nonzero(values == 0.0)
        ),
    }


# =============================================================================
# RESULTS SAVING
# =============================================================================

def save_model(
    path: Path,
    model: Model3D,
) -> None:
    """Save inverted susceptibility model."""

    centers = model.mesh.cell_centers()

    table = np.column_stack(
        (
            centers,
            model.values,
        )
    )

    np.savetxt(
        path,
        table,
        fmt="%.10e",
        header=(
            "x_center_m y_center_m z_center_m "
            "susceptibility_SI"
        ),
    )


def save_data_fit(
    path: Path,
    locations: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """Save observed, predicted and residual magnetic data."""

    residual = predicted - observed

    table = np.column_stack(
        (
            locations,
            observed,
            predicted,
            residual,
        )
    )

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
    noise_level: int,
) -> None:
    """Save inversion statistics."""

    values = np.array(
        [
            noise_level,
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
        "noise_level_percent\n"
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

    np.savetxt(
        path,
        values,
        fmt="%.12e",
        header=header,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "GeoInvLab3D positive-constrained 3-D magnetic "
            "inversion using noisy synthetic observations."
        )
    )

    parser.add_argument(
        "--noise-level",
        type=int,
        choices=NOISE_LEVELS,
        default=3,
        help=(
            "Gaussian noise level in percent: "
            "1, 3 or 5 (default: 3)."
        ),
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
    noise_level = args.noise_level

    if not np.isfinite(lam) or lam <= 0.0:
        parser.error("--lambda must be finite and strictly positive.")

    if not np.isfinite(args.tolerance) or args.tolerance <= 0.0:
        parser.error(
            "--tolerance must be finite and strictly positive."
        )

    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1.")

    # -------------------------------------------------------------------------
    # Resolve input and output directories
    # -------------------------------------------------------------------------

    observation_file = resolve_observation_file(noise_level)

    results_dir = (
        RESULTS_ROOT / f"noise_{noise_level}pct"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("GeoInvLab3D - POSITIVE-CONSTRAINED MAGNETIC INVERSION")
    print("NOISY SYNTHETIC DATA")
    print("=" * 76)

    print(f"Noise level       : {noise_level}%")
    print(f"Script directory  : {SCRIPT_DIR}")
    print(f"Shared data       : {SHARED_DATA_DIR}")
    print(f"Observation file  : {observation_file}")
    print(f"Results directory : {results_dir}")

    # -------------------------------------------------------------------------
    # 1. Mesh
    # -------------------------------------------------------------------------

    print("\n[1/9] Loading mesh...")

    mesh = load_mesh(
        SHARED_DATA_DIR / "mesh.dat"
    )

    print(f"Mesh dimensions: {mesh.nx} x {mesh.ny} x {mesh.nz}")
    print(f"Number of cells: {mesh.n_cells}")
    print(f"Cell size      : {mesh.dx} x {mesh.dy} x {mesh.dz} m")
    print(f"Origin         : {mesh.origin}")

    # -------------------------------------------------------------------------
    # 2. Stations and noisy observations
    # -------------------------------------------------------------------------

    print("\n[2/9] Loading stations and noisy observations...")

    stations = load_stations(
        SHARED_DATA_DIR / "stations.dat"
    )

    locations, observed = load_observations(
        observation_file
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
        raise ValueError(
            "Expected all observation stations at z = 0."
        )

    print(f"Number of stations    : {stations.shape[0]}")
    print(f"Number of observations: {observed.size}")

    print(
        f"Noisy observation range: "
        f"{observed.min():.6f} to {observed.max():.6f} nT"
    )

    print(
        f"Noisy observation RMS  : "
        f"{np.sqrt(np.mean(observed**2)):.6f} nT"
    )

    # -------------------------------------------------------------------------
    # 3. Initial and true models
    # -------------------------------------------------------------------------

    print("\n[3/9] Loading initial and true susceptibility models...")

    initial_model = load_model(
        SHARED_DATA_DIR / "model_initial.dat",
        mesh,
        name="initial_susceptibility",
    )

    true_model = load_model(
        SHARED_DATA_DIR / "model_true.dat",
        mesh,
        name="true_susceptibility",
    )

    if np.any(initial_model.values < 0.0):
        raise ValueError(
            "The initial model contains negative susceptibility."
        )

    print(
        f"Initial model range: "
        f"{initial_model.values.min():.6e} to "
        f"{initial_model.values.max():.6e} SI"
    )

    print(
        f"True model range   : "
        f"{true_model.values.min():.6e} to "
        f"{true_model.values.max():.6e} SI"
    )

    # -------------------------------------------------------------------------
    # 4. Dataset
    # -------------------------------------------------------------------------

    print("\n[4/9] Building noisy magnetic dataset...")

    dataset = DataSet(
        values=observed,
        locations=locations,
        name=f"synthetic_magnetic_TMI_noise_{noise_level}pct",
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
    # 7. Positive bounded solver
    # -------------------------------------------------------------------------

    print("\n[7/9] Configuring positive bounded solver...")

    solver = PositiveBoundedLeastSquaresSolver(
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        lsmr_tolerance="auto",
        verbose=args.verbose,
        history_file=(
            results_dir / "inversion_history_positive.dat"
        ),
        model_history_file=(
            results_dir / "inversion_model_history_positive.dat"
        ),
        parameters_file=(
            results_dir / "inversion_parameters_positive.dat"
        ),
    )

    engine = InversionEngine(
        solver=solver,
    )

    print("Physical constraint : susceptibility >= 0 SI")
    print(f"Noise level         : {noise_level}%")
    print(f"Regularization lambda: {lam:g}")
    print(f"Solver tolerance     : {args.tolerance:g}")
    print(f"Maximum iterations   : {args.max_iterations}")

    # -------------------------------------------------------------------------
    # 8. Run inversion
    # -------------------------------------------------------------------------

    print("\n[8/9] Running positive-constrained magnetic inversion...")

    result = engine.solve(problem)

    inverted_model = result.model

    print("\nCalculating predicted magnetic data...")

    predicted = np.asarray(
        operator.predict(inverted_model),
        dtype=float,
    )

    if predicted.shape != observed.shape:
        raise ValueError(
            f"Invalid predicted data shape: {predicted.shape}; "
            f"expected {observed.shape}."
        )

    if not np.all(np.isfinite(predicted)):
        raise ValueError(
            "Predicted magnetic data contain non-finite values."
        )

    data_statistics = calculate_data_statistics(
        observed,
        predicted,
    )

    model_statistics = calculate_model_statistics(
        inverted_model,
        true_model,
    )

    # -------------------------------------------------------------------------
    # 9. Display and save results
    # -------------------------------------------------------------------------

    print("\nData-fit statistics")
    print("-" * 60)

    for key, value in data_statistics.items():
        print(f"{key:30s}: {value:.8e}")

    print("\nInverted susceptibility statistics")
    print("-" * 60)

    print(
        f"Minimum susceptibility : "
        f"{model_statistics['minimum']:.8e} SI"
    )

    print(
        f"Maximum susceptibility : "
        f"{model_statistics['maximum']:.8e} SI"
    )

    print(
        f"Mean susceptibility    : "
        f"{model_statistics['mean']:.8e} SI"
    )

    print(
        f"RMS susceptibility     : "
        f"{model_statistics['rms']:.8e} SI"
    )

    print(
        f"Model RMSE             : "
        f"{model_statistics['model_rmse']:.8e} SI"
    )

    print(
        f"Negative cells         : "
        f"{model_statistics['negative_cells']}"
    )

    print(
        f"Positive cells         : "
        f"{model_statistics['positive_cells']}"
    )

    print(
        f"Zero cells             : "
        f"{model_statistics['zero_cells']}"
    )

    if model_statistics["minimum"] < -1e-12:
        print(
            "\nWARNING: the returned model contains values below "
            "the non-negativity tolerance."
        )

    print("\nSaving inversion results...")

    save_model(
        results_dir / "model_inverted_positive.dat",
        inverted_model,
    )

    save_data_fit(
        results_dir / "data_fit_positive.dat",
        locations,
        observed,
        predicted,
    )

    save_statistics(
        results_dir / "inversion_statistics_positive.dat",
        data_statistics,
        model_statistics,
        lam,
        noise_level,
    )

    print("\nGenerated files:")

    print("  - model_inverted_positive.dat")
    print("  - data_fit_positive.dat")
    print("  - inversion_statistics_positive.dat")
    print("  - inversion_history_positive.dat")
    print("  - inversion_model_history_positive.dat")
    print("  - inversion_parameters_positive.dat")

    print(f"\nResults directory: {results_dir}")

    print("\n" + "=" * 76)
    print("POSITIVE MAGNETIC INVERSION WITH NOISY DATA FINISHED")
    print("=" * 76)


if __name__ == "__main__":
    main()
