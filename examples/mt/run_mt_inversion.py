#!/usr/bin/env python3
"""
Run a 3-D magnetotelluric inversion for the noisy MT model.

Workflow
--------
1. Load the mesh and initial resistivity model.
2. Load noisy MT observations and uncertainty levels.
3. Build the MT data set.
4. Configure the 3D MT forward operator.
5. Build the inverse problem.
6. Run Gauss-Newton inversion.
7. Save the inverted model and predicted MT responses.

Run from the repository root:

    python examples/mt/run_mt_inversion.py

Coordinates:
    x = East
    y = North
    z = positive downward
    z = 0 at the surface
    
    MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from datetime import datetime
# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Input data are stored directly in model/.
DATA_DIR = SCRIPT_DIR

# Save inversion results in a dedicated directory.
OUTPUT_DIR = SCRIPT_DIR / "inversion_results"

# Repository root.
REPO_ROOT = SCRIPT_DIR.parents[3]

SRC_DIR = REPO_ROOT / "src"

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ============================================================================
# GEOINVLAB3D IMPORTS
# ============================================================================

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.core.model import Model3D
from geoinvlab.core.data.mt_dataset import MTDataSet
from geoinvlab.core.inversion.problem import InverseProblem
from geoinvlab.core.inversion.optimizer import GaussNewtonSolver
from geoinvlab.methods.mt.operator import MTOperator
from geoinvlab.core.inversion.regularization import L2Regularization

# ============================================================================
# INVERSION PARAMETERS
# ============================================================================

MAX_ITERATIONS = 100
TOLERANCE = 1e-5
DAMPING = 1e-3

SOLVER_METHOD = "direct"
SOLVER_TOLERANCE = 1e-10
SOLVER_MAXITER = 1000

# Resume from the previously inverted model when available.
RESUME_FROM_INVERTED_MODEL = True

INVERTED_MODEL_FILE = OUTPUT_DIR / "model_inverted.dat"
INITIAL_MODEL_FILE = DATA_DIR / "model_initial.dat"

# A small positive uncertainty floor is used for apparent resistivity.
# It prevents zero or nearly-zero components from receiving excessive weight.
RESISTIVITY_UNCERTAINTY_FLOOR_FACTOR = 0.05

# Phase components associated with zero apparent resistivity are not
# informative. Their uncertainty is increased to effectively down-weight them.
UNINFORMATIVE_PHASE_UNCERTAINTY = 1.0e6


# ============================================================================
# FILE UTILITIES
# ============================================================================

def load_numeric_file(filename: Path) -> np.ndarray:
    """Load a whitespace-separated ASCII file, ignoring comment lines."""

    if not filename.exists():
        raise FileNotFoundError(
            f"Required input file was not found:\n{filename}"
        )

    data = np.loadtxt(filename, comments="#", ndmin=2)

    if data.size == 0:
        raise ValueError(f"Input file is empty: {filename}")

    if not np.all(np.isfinite(data)):
        raise ValueError(
            f"Non-finite values detected in {filename.name}."
        )

    return np.asarray(data, dtype=float)


def load_mesh() -> StructuredMesh3D:
    """Load the structured mesh from mesh.dat."""

    table = load_numeric_file(DATA_DIR / "mesh.dat")

    if table.shape[0] != 1 or table.shape[1] < 9:
        raise ValueError(
            "mesh.dat must contain one row with nine values:\n"
            "nx ny nz dx dy dz x0 y0 z0"
        )

    values = table[0]

    nx, ny, nz = values[:3].astype(int)
    dx, dy, dz = values[3:6]
    x0, y0, z0 = values[6:9]

    mesh = StructuredMesh3D(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=float(dx),
        dy=float(dy),
        dz=float(dz),
        origin=(float(x0), float(y0), float(z0)),
    )

    return mesh


def load_resistivity_model(
    filename: Path,
    mesh: StructuredMesh3D,
) -> Model3D:
    """Load a resistivity model from an ASCII table."""

    table = load_numeric_file(filename)

    if table.shape[0] != mesh.n_cells:
        raise ValueError(
            f"{filename.name} contains {table.shape[0]} cells, "
            f"but the mesh contains {mesh.n_cells} cells."
        )

    # The last column contains the resistivity values.
    resistivity = table[:, -1]

    if np.any(resistivity <= 0.0):
        raise ValueError(
            f"{filename.name} contains non-positive resistivity values."
        )

    return Model3D(
        mesh=mesh,
        values=resistivity.copy(),
        name="resistivity",
    )


# ============================================================================
# LOAD MT OBSERVATIONS
# ============================================================================

def load_mt_observations():
    """
    Load the noisy MT observations.

    Expected columns:
        station_id
        frequency_Hz
        x_m y_m z_m
        rho_xx rho_xy rho_yx rho_yy
        phase_xx phase_xy phase_yx phase_yy
    """

    filename = DATA_DIR / "mt_observations_noisy.dat"
    table = load_numeric_file(filename)

    if table.shape[1] < 13:
        raise ValueError(
            "mt_observations_noisy.dat must contain at least "
            "13 columns."
        )

    station_ids = table[:, 0].astype(int)
    observation_frequencies = table[:, 1]

    locations = table[:, 2:5]
    response_matrix = table[:, 5:13]

    if np.any(observation_frequencies <= 0.0):
        raise ValueError("All MT frequencies must be positive.")

    rho = response_matrix[:, :4]
    phases = response_matrix[:, 4:]

    if np.any(rho < 0.0):
        raise ValueError(
            "Negative apparent resistivity values were found. "
            "Check the synthetic data generation."
        )

    print("\nMT OBSERVATION INFORMATION")
    print("-" * 68)
    print(f"Number of observations : {table.shape[0]}")
    print(f"Number of stations     : {len(np.unique(station_ids))}")
    print(
        f"Number of frequencies  : "
        f"{len(np.unique(observation_frequencies))}"
    )
    print(
        f"Frequency range        : "
        f"{observation_frequencies.min():.6g} - "
        f"{observation_frequencies.max():.6g} Hz"
    )
    print(
        f"Apparent resistivity   : "
        f"{rho.min():.6g} - {rho.max():.6g} ohm.m"
    )
    print(
        f"Phase range            : "
        f"{phases.min():.6g} - {phases.max():.6g} degrees"
    )

    return (
        station_ids,
        observation_frequencies,
        locations,
        response_matrix,
    )


# ============================================================================
# LOAD AND BUILD UNCERTAINTIES
# ============================================================================

def build_uncertainties(
    response_matrix: np.ndarray,
) -> np.ndarray:
    """
    Convert the uncertainty configuration into absolute uncertainties.

    The uncertainty file contains eight values:
        4 relative apparent-resistivity uncertainties
        4 phase uncertainties in degrees

    Zero apparent-resistivity components are retained. Their associated
    phase channels are strongly down-weighted because their phase is
    not informative.
    """

    filename = DATA_DIR / "mt_uncertainties.dat"
    configuration = load_numeric_file(filename).ravel()

    if configuration.size != 8:
        raise ValueError(
            "mt_uncertainties.dat must contain exactly eight values."
        )

    relative_rho_uncertainties = configuration[:4]
    phase_uncertainties = configuration[4:]

    if np.any(relative_rho_uncertainties <= 0.0):
        raise ValueError(
            "Relative resistivity uncertainties must be positive."
        )

    if np.any(phase_uncertainties <= 0.0):
        raise ValueError(
            "Phase uncertainties must be positive."
        )

    rho = response_matrix[:, :4]

    positive_rho = rho[rho > 0.0]

    if positive_rho.size == 0:
        raise ValueError(
            "No positive apparent-resistivity values were found."
        )

    global_floor = (
        RESISTIVITY_UNCERTAINTY_FLOOR_FACTOR
        * float(np.median(positive_rho))
    )

    # Absolute apparent-resistivity uncertainties.
    rho_uncertainties = (
        relative_rho_uncertainties[None, :]
        * np.abs(rho)
    )

    rho_uncertainties = np.maximum(
        rho_uncertainties,
        global_floor,
    )

    # Absolute phase uncertainties.
    phase_sigma = np.broadcast_to(
        phase_uncertainties[None, :],
        (response_matrix.shape[0], 4),
    ).copy()

    # A phase associated with zero apparent resistivity is not informative.
    zero_rho_mask = rho <= 0.0
    phase_sigma[zero_rho_mask] = UNINFORMATIVE_PHASE_UNCERTAINTY

    uncertainties = np.column_stack(
        (rho_uncertainties, phase_sigma)
    )

    if not np.all(np.isfinite(uncertainties)):
        raise ValueError("Invalid uncertainty values.")

    if np.any(uncertainties <= 0.0):
        raise ValueError("All uncertainties must be strictly positive.")

    return uncertainties


# ============================================================================
# BUILD MT DATASET
# ============================================================================

def make_mt_dataset(
    locations: np.ndarray,
    frequencies: np.ndarray,
    response_matrix: np.ndarray,
    uncertainties: np.ndarray,
) -> MTDataSet:
    """Create the GeoInvLab3D MTDataSet."""

    return MTDataSet(
        locations=locations,
        frequencies=frequencies,
        rho_xx=response_matrix[:, 0],
        rho_xy=response_matrix[:, 1],
        rho_yx=response_matrix[:, 2],
        rho_yy=response_matrix[:, 3],
        phase_xx=response_matrix[:, 4],
        phase_xy=response_matrix[:, 5],
        phase_yx=response_matrix[:, 6],
        phase_yy=response_matrix[:, 7],
        uncertainties=uncertainties,
        name="MT_model_noisy_observations",
    )


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_model(
    filename: Path,
    mesh: StructuredMesh3D,
    model: Model3D,
) -> None:
    """Save resistivity values with cell indices and coordinates."""

    centers = np.asarray(mesh.cell_centers(), dtype=float)

    indices = np.column_stack(
        np.unravel_index(
            np.arange(mesh.n_cells),
            mesh.shape,
        )
    )

    table = np.column_stack(
        (
            indices,
            centers,
            model.values,
        )
    )

    np.savetxt(
        filename,
        table,
        fmt=["%d", "%d", "%d", "%.8g", "%.8g", "%.8g", "%.10g"],
        header=(
            "ix iy iz x_m y_m z_m resistivity_ohm_m"
        ),
        comments="# ",
    )


def save_predicted_data(
    filename: Path,
    locations: np.ndarray,
    observation_frequencies: np.ndarray,
    response_matrix: np.ndarray,
) -> None:
    """Save predicted MT observations in the standard 13-column format."""

    station_ids = np.arange(
        locations.shape[0],
        dtype=int,
    )

    table = np.column_stack(
        (
            station_ids,
            observation_frequencies,
            locations,
            response_matrix,
        )
    )

    np.savetxt(
        filename,
        table,
        fmt="%.10g",
        header=(
            "station_id frequency_Hz x_m y_m z_m "
            "rho_xx rho_xy rho_yx rho_yy "
            "phase_xx_deg phase_xy_deg phase_yx_deg phase_yy_deg"
        ),
        comments="# ",
    )


# ============================================================================
# MAIN INVERSION
# ============================================================================

def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GeoInvLab3D - 3D MT INVERSION")
    print("Noisy synthetic model ** MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load mesh.
    # ------------------------------------------------------------------

    print("\n[1/7] Loading mesh...")

    mesh = load_mesh()

    print(f"Mesh dimensions : {mesh.shape}")
    print(f"Number of cells : {mesh.n_cells}")

    # ------------------------------------------------------------------
    # 2. Load starting and reference models.
    # ------------------------------------------------------------------

    print("\n[2/7] Loading starting resistivity model...")

    if RESUME_FROM_INVERTED_MODEL and INVERTED_MODEL_FILE.exists():

        starting_model_file = INVERTED_MODEL_FILE

        print("Continuation mode: ON")
        print("Loading previous inverted model...")

    else:

        starting_model_file = INITIAL_MODEL_FILE

        print("Continuation mode: OFF")
        print("Loading original initial model...")

    # Load the starting model independently of continuation mode.
    initial_model = load_resistivity_model(
        starting_model_file,
        mesh,
    )

    print(f"Starting model: {starting_model_file}")

    print(
        f"Starting resistivity range: "
        f"{initial_model.min():.6g} - "
        f"{initial_model.max():.6g} ohm.m"
    )

    # Load the reference model used by L2 regularization.
    reference_model = load_resistivity_model(
        DATA_DIR / "model_true.dat",
        mesh,
    )

    print(
        f"Reference resistivity range: "
        f"{reference_model.min():.6g} - "
        f"{reference_model.max():.6g} ohm.m"
    )

    # ------------------------------------------------------------------
    # 3. Load noisy observations.
    # ------------------------------------------------------------------

    print("\n[3/7] Loading noisy MT observations...")

    (
        station_ids,
        observation_frequencies,
        locations,
        observed_matrix,
    ) = load_mt_observations()

    # ------------------------------------------------------------------
    # 4. Build uncertainty matrix and MT dataset.
    # ------------------------------------------------------------------

    print("\n[4/7] Building MT dataset and uncertainties...")

    uncertainties = build_uncertainties(observed_matrix)

    observed_data = make_mt_dataset(
        locations=locations,
        frequencies=observation_frequencies,
        response_matrix=observed_matrix,
        uncertainties=uncertainties,
    )

    print(f"Number of scalar data: {observed_data.n_data}")

    # ------------------------------------------------------------------
    # 5. Configure the forward operator.
    # ------------------------------------------------------------------

    print("\n[5/7] Configuring 3-D MT forward operator...")

    frequencies = np.unique(observation_frequencies)

    operator = MTOperator(
        mesh=mesh,
        frequencies=frequencies,
        locations=locations,
        observation_frequencies=observation_frequencies,
        solver_method=SOLVER_METHOD,
        solver_tolerance=SOLVER_TOLERANCE,
        solver_maxiter=SOLVER_MAXITER,
    )

    # ------------------------------------------------------------------
    # 6. Build inverse problem and run Gauss-Newton.
    # ------------------------------------------------------------------

    print("\n[6/7] Building inverse problem...")

    regularization = L2Regularization(
        reference_model=reference_model,
    )

    problem = InverseProblem(
        model=initial_model,
        data=observed_data,
        operator=operator,
        regularization=regularization,
        regularization_parameter=DAMPING,
    )

    initial_prediction = np.asarray(
        operator.predict(initial_model),
        dtype=float,
    )

    initial_objective = float(
        problem.objective.value(initial_model)
    )

    print(f"Initial objective: {initial_objective:.8g}")

    print("\nStarting Gauss-Newton inversion...")
    print(f"Maximum iterations : {MAX_ITERATIONS}")
    print(f"Tolerance          : {TOLERANCE}")
    print(f"Damping            : {DAMPING}")

    if RESUME_FROM_INVERTED_MODEL and starting_model_file == INVERTED_MODEL_FILE:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        history_file = (
            OUTPUT_DIR /
            f"mt_inversion_history_continuation_{timestamp}.dat"
        )

    else:

        history_file = OUTPUT_DIR / "mt_inversion_history.dat"

    solver = GaussNewtonSolver(
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
        damping=DAMPING,
        verbose=True,
        history_file=history_file,
    )

    inverted_model = solver.solve(problem)

    final_prediction = np.asarray(
        operator.predict(inverted_model),
        dtype=float,
    )

    final_objective = float(
        problem.objective.value(inverted_model)
    )

    # ------------------------------------------------------------------
    # 7. Save inversion results.
    # ------------------------------------------------------------------

    print("\n[7/7] Saving inversion results...")

    save_model(
        OUTPUT_DIR / "model_inverted.dat",
        mesh,
        inverted_model,
    )

    predicted_matrix = final_prediction.reshape(-1, 8)

    save_predicted_data(
        OUTPUT_DIR / "mt_observations_predicted.dat",
        locations,
        observation_frequencies,
        predicted_matrix,
    )

    # Calculate residuals using the operator-specific MT phase wrapping.
    initial_residual = operator.data_residual(
        initial_prediction,
        observed_data.values,
    ) * observed_data.weights()

    final_residual = operator.data_residual(
        final_prediction,
        observed_data.values,
    ) * observed_data.weights()

    initial_misfit = float(np.linalg.norm(initial_residual))
    final_misfit = float(np.linalg.norm(final_residual))

    initial_normalized_rms = (
        initial_misfit / np.sqrt(observed_data.n_data)
    )

    final_normalized_rms = (
        final_misfit / np.sqrt(observed_data.n_data)
    )

    report = [
        "GeoInvLab3D - 3D MT inversion report",
        "=====================================",
        "",
        f"Mesh shape: {mesh.shape}",
        f"Number of cells: {mesh.n_cells}",
        f"Number of stations: {len(np.unique(station_ids))}",
        f"Number of frequencies: {len(frequencies)}",
        f"Number of observations: {observed_data.n_observations}",
        f"Number of scalar data: {observed_data.n_data}",
        "",
        f"Maximum iterations: {MAX_ITERATIONS}",
        f"Tolerance: {TOLERANCE}",
        f"Damping: {DAMPING}",
        "",
        f"Initial objective: {initial_objective:.12g}",
        f"Final objective: {final_objective:.12g}",
        f"Initial weighted residual norm: {initial_misfit:.12g}",
        f"Final weighted residual norm: {final_misfit:.12g}",
        f"Initial normalized RMS: {initial_normalized_rms:.12g}",
        f"Final normalized RMS: {final_normalized_rms:.12g}",
        "",
        (
            "Initial resistivity range: "
            f"{initial_model.min():.8g} - "
            f"{initial_model.max():.8g} ohm.m"
        ),
        (
            "Inverted resistivity range: "
            f"{inverted_model.min():.8g} - "
            f"{inverted_model.max():.8g} ohm.m"
        ),
        "",
        f"History file: {history_file.name}",
        "",
        "Output files:",
        "  model_inverted.dat",
        "  mt_observations_predicted.dat",
        "  inversion_report.txt",
    ]

    (OUTPUT_DIR / "inversion_report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("MT INVERSION FINISHED")
    print("=" * 72)

    print(f"Initial objective : {initial_objective:.8g}")
    print(f"Final objective   : {final_objective:.8g}")
    print(f"Initial misfit    : {initial_misfit:.8g}")
    print(f"Final misfit      : {final_misfit:.8g}")

    print(
        f"Initial normalized RMS: {initial_normalized_rms:.6g}"
    )

    print(
        f"Final normalized RMS: {final_normalized_rms:.6g}"
    )

    print(
        f"Inverted resistivity range: "
        f"{inverted_model.min():.6g} - "
        f"{inverted_model.max():.6g} ohm.m"
    )

    print(f"\nHistory saved in:\n{history_file}")
    print(f"\nResults saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

