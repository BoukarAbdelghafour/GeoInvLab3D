#!/usr/bin/env python3
"""
Plot the true 3D magnetic susceptibility model.

The visualization highlights magnetic susceptibility contrasts
relative to the homogeneous background.

Features
--------
- Three-dimensional visualization of anomalous bodies.
- High-susceptibility anomaly displayed in red.
- Low-susceptibility anomaly displayed in blue.
- Homogeneous background hidden.
- Black cell edges.
- Transparent computational domain with external outline.
- Depth axis positive downward.
- Coordinate-based reconstruction of the model.
- Validation of model coordinates against the structured mesh.
- Publication-quality PNG and PDF output.

GeoInvLab3D conventions
-----------------------
- Structured Cartesian mesh.
- One susceptibility value per cell.
- Model coordinates: x_center, y_center, z_center.
- Vertical coordinate positive downward.

Boukar Abdelghafour  GeoInvLab3D   MIT License Copyright (c) 2026
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "examples"
    / "magnetic"
    / "synthetic"
    / "data"
)

MESH_FILE = DATA_DIR / "mesh.dat"
TRUE_MODEL_FILE = DATA_DIR / "model_true.dat"

FIGURE_DIR = SCRIPT_DIR / "figures"

OUTPUT_FILE = FIGURE_DIR / "magnetic_true_model_3d.png"
OUTPUT_PDF = FIGURE_DIR / "magnetic_true_model_3d.pdf"


# ============================================================
# MODEL PARAMETERS
# ============================================================

BACKGROUND_SUSCEPTIBILITY = 0.01
SUSCEPTIBILITY_TOLERANCE = 1.0e-10
COORDINATE_TOLERANCE = 1.0e-6

HIGH_SUSCEPTIBILITY_VALUE = 0.05
LOW_SUSCEPTIBILITY_VALUE = 0.002

HIGH_SUSCEPTIBILITY_COLOR = "firebrick"
LOW_SUSCEPTIBILITY_COLOR = "royalblue"

CELL_EDGE_COLOR = "black"
CELL_EDGE_WIDTH = 0.35

FIGURE_DPI = 350

VIEW_ELEVATION = 25
VIEW_AZIMUTH = -55


# ============================================================
# LOAD MESH
# ============================================================

def load_mesh(filename: Path):
    """
    Read the GeoInvLab3D structured mesh definition.

    Expected format
    ---------------
    nx ny nz dx dy dz ox oy oz
    """
    if not filename.exists():
        raise FileNotFoundError(f"Mesh file not found: {filename}")

    values = np.asarray(
        np.loadtxt(filename, comments="#")
    ).ravel()

    if values.size != 9:
        raise ValueError(
            "The mesh file must contain exactly "
            "9 numerical values: nx ny nz dx dy dz ox oy oz."
        )

    nx, ny, nz = map(int, values[:3])
    dx, dy, dz = values[3:6]
    ox, oy, oz = values[6:9]

    if min(nx, ny, nz) <= 0:
        raise ValueError("Mesh dimensions must be positive.")

    if min(dx, dy, dz) <= 0:
        raise ValueError("Cell dimensions must be positive.")

    return nx, ny, nz, dx, dy, dz, ox, oy, oz


# ============================================================
# LOAD AND RECONSTRUCT MAGNETIC MODEL
# ============================================================

def load_magnetic_model(
    filename: Path,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    ox: float,
    oy: float,
    oz: float,
):
    """
    Load the true magnetic susceptibility model.

    The model is reconstructed using the actual cell-center
    coordinates stored in the model file.

    Expected columns
    ----------------
    x_center_m y_center_m z_center_m susceptibility_SI
    """
    if not filename.exists():
        raise FileNotFoundError(
            f"Magnetic model file not found: {filename}"
        )

    data = np.loadtxt(
        filename,
        comments="#",
        ndmin=2,
    )

    expected_cells = nx * ny * nz

    if data.shape[0] != expected_cells:
        raise ValueError(
            "Incorrect number of model cells.\n"
            f"Expected: {expected_cells}\n"
            f"Found: {data.shape[0]}"
        )

    if data.shape[1] < 4:
        raise ValueError(
            "The model file must contain at least "
            "four numerical columns:\n"
            "x_center y_center z_center susceptibility."
        )

    if not np.all(np.isfinite(data[:, :4])):
        raise ValueError(
            "Non-finite values detected in the model file."
        )

    coordinates = data[:, :3]
    susceptibility_values = data[:, 3]

    # Initialize the structured model.
    # Array ordering: model[ix, iy, iz], with iz varying fastest.
    model = np.full(
        (nx, ny, nz),
        np.nan,
        dtype=float,
    )

    occupied = np.zeros(
        (nx, ny, nz),
        dtype=bool,
    )

    # Map coordinates to mesh indices.
    indices_float = np.column_stack(
        (
            (coordinates[:, 0] - (ox + 0.5 * dx)) / dx,
            (coordinates[:, 1] - (oy + 0.5 * dy)) / dy,
            (coordinates[:, 2] - (oz + 0.5 * dz)) / dz,
        )
    )

    indices = np.rint(indices_float).astype(int)

    # Check that coordinates correspond to cell centers.
    expected_coordinates = np.column_stack(
        (
            ox + (indices[:, 0] + 0.5) * dx,
            oy + (indices[:, 1] + 0.5) * dy,
            oz + (indices[:, 2] + 0.5) * dz,
        )
    )

    coordinate_errors = np.abs(
        coordinates - expected_coordinates
    )

    if np.any(
        coordinate_errors > COORDINATE_TOLERANCE
    ):
        bad_rows = np.where(
            np.any(
                coordinate_errors > COORDINATE_TOLERANCE,
                axis=1,
            )
        )[0]

        first_bad = bad_rows[0]

        raise ValueError(
            "Some model coordinates do not correspond "
            "to the mesh cell centers.\n"
            f"Number of incorrect rows: {len(bad_rows)}\n"
            f"First incorrect row: {first_bad + 1}\n"
            f"Model coordinates: {coordinates[first_bad]}\n"
            f"Expected coordinates: "
            f"{expected_coordinates[first_bad]}"
        )

    # Check index limits.
    valid_indices = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < nx)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < ny)
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < nz)
    )

    if not np.all(valid_indices):
        bad_rows = np.where(~valid_indices)[0]
        first_bad = bad_rows[0]

        raise ValueError(
            "Some model cells are outside the computational "
            "mesh.\n"
            f"First incorrect row: {first_bad + 1}\n"
            f"Indices: {indices[first_bad]}"
        )

    # Fill the model using the actual cell indices.
    for row, value in zip(
        indices,
        susceptibility_values,
    ):
        ix, iy, iz = row

        if occupied[ix, iy, iz]:
            raise ValueError(
                "Duplicate cell coordinates detected:\n"
                f"ix={ix}, iy={iy}, iz={iz}"
            )

        model[ix, iy, iz] = value
        occupied[ix, iy, iz] = True

    # Check for missing cells.
    if not np.all(occupied):
        missing = np.argwhere(~occupied)

        raise ValueError(
            "Some mesh cells are missing from the model file.\n"
            f"Missing cells: {len(missing)}\n"
            f"First missing index: {missing[0]}"
        )

    return model


# ============================================================
# CREATE 3D COORDINATES
# ============================================================

def create_mesh_coordinates(
    nx, ny, nz,
    dx, dy, dz,
    ox, oy, oz,
):
    """
    Generate 3D vertex coordinate arrays.

    Matplotlib voxels requires coordinate arrays with
    dimensions (nx+1, ny+1, nz+1).
    """
    x_edges = ox + np.arange(nx + 1) * dx
    y_edges = oy + np.arange(ny + 1) * dy
    z_edges = oz + np.arange(nz + 1) * dz

    X, Y, Z = np.meshgrid(
        x_edges,
        y_edges,
        z_edges,
        indexing="ij",
    )

    return (
        X, Y, Z,
        x_edges,
        y_edges,
        z_edges,
    )


# ============================================================
# DRAW DOMAIN OUTLINE
# ============================================================

def draw_domain_outline(
    ax,
    x_edges,
    y_edges,
    z_edges,
):
    """Draw the twelve external edges of the computational domain."""
    xmin, xmax = x_edges[0], x_edges[-1]
    ymin, ymax = y_edges[0], y_edges[-1]
    zmin, zmax = z_edges[0], z_edges[-1]

    corners = [
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmin, ymax, zmin),
        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for start, end in edges:
        p1 = corners[start]
        p2 = corners[end]

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color="black",
            linewidth=0.9,
            alpha=0.8,
        )


# ============================================================
# PLOT TRUE MAGNETIC MODEL
# ============================================================

def plot_magnetic_model():
    """Plot and save the true 3D magnetic susceptibility model."""
    print("=" * 70)
    print("GeoInvLab3D - TRUE MAGNETIC MODEL 3D")
    print("=" * 70)

    # Load mesh.
    (
        nx, ny, nz,
        dx, dy, dz,
        ox, oy, oz,
    ) = load_mesh(MESH_FILE)

    print(f"Mesh dimensions : {nx} x {ny} x {nz}")
    print(f"Total cells     : {nx * ny * nz}")
    print(f"Cell dimensions : {dx} x {dy} x {dz} m")
    print(f"Mesh origin     : {ox} x {oy} x {oz} m")

    # Load true model using cell coordinates.
    model = load_magnetic_model(
        TRUE_MODEL_FILE,
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        ox,
        oy,
        oz,
    )

    print("\nModel successfully reconstructed.")
    print(
        f"Susceptibility range : "
        f"{model.min():.6f} - {model.max():.6f} SI"
    )
    print(
        f"Background susceptibility : "
        f"{BACKGROUND_SUSCEPTIBILITY:.6f} SI"
    )

    # Identify anomalous cells.
    high_anomaly = (
        model
        > BACKGROUND_SUSCEPTIBILITY + SUSCEPTIBILITY_TOLERANCE
    )

    low_anomaly = (
        model
        < BACKGROUND_SUSCEPTIBILITY - SUSCEPTIBILITY_TOLERANCE
    )

    anomalous_cells = high_anomaly | low_anomaly

    n_high = int(np.count_nonzero(high_anomaly))
    n_low = int(np.count_nonzero(low_anomaly))
    n_background = nx * ny * nz - n_high - n_low

    print("\nSusceptibility distribution")
    print("-" * 70)

    unique_values, counts = np.unique(
        model,
        return_counts=True,
    )

    for value, count in zip(
        unique_values,
        counts,
    ):
        print(f"{value:.6f} SI : {count:6d} cells")

    print("-" * 70)
    print(f"High-susceptibility cells : {n_high}")
    print(f"Low-susceptibility cells  : {n_low}")
    print(f"Background cells          : {n_background}")

    if n_high == 0 and n_low == 0:
        raise ValueError("No anomalous cells detected.")

    # Report anomaly spatial extents.
    print("\nAnomaly spatial distribution")
    print("-" * 70)

    x_centers = ox + (np.arange(nx) + 0.5) * dx
    y_centers = oy + (np.arange(ny) + 0.5) * dy
    z_centers = oz + (np.arange(nz) + 0.5) * dz

    for name, mask in (
        ("High-susceptibility anomaly", high_anomaly),
        ("Low-susceptibility anomaly", low_anomaly),
    ):
        positions = np.argwhere(mask)

        if len(positions) == 0:
            continue

        ix = positions[:, 0]
        iy = positions[:, 1]
        iz = positions[:, 2]

        print(f"\n{name}")
        print(
            f"  East  : {x_centers[ix].min():.1f} "
            f"to {x_centers[ix].max():.1f} m"
        )
        print(
            f"  North : {y_centers[iy].min():.1f} "
            f"to {y_centers[iy].max():.1f} m"
        )
        print(
            f"  Depth : {z_centers[iz].min():.1f} "
            f"to {z_centers[iz].max():.1f} m"
        )

    # Generate 3D coordinates.
    (
        X, Y, Z,
        x_edges,
        y_edges,
        z_edges,
    ) = create_mesh_coordinates(
        nx, ny, nz,
        dx, dy, dz,
        ox, oy, oz,
    )

    # Create face colors.
    facecolors = np.empty(
        (nx, ny, nz, 4),
        dtype=float,
    )
    facecolors[:] = (0.0, 0.0, 0.0, 0.0)

    high_rgba = plt.matplotlib.colors.to_rgba(
        HIGH_SUSCEPTIBILITY_COLOR
    )
    low_rgba = plt.matplotlib.colors.to_rgba(
        LOW_SUSCEPTIBILITY_COLOR
    )

    facecolors[high_anomaly] = high_rgba
    facecolors[low_anomaly] = low_rgba

    # Create figure.
    fig = plt.figure(
        figsize=(11, 9),
        facecolor="white",
        constrained_layout=True,
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )
    ax.set_facecolor("white")

    # Draw all anomalous cells in one voxel operation.
    if np.any(anomalous_cells):
        ax.voxels(
            X,
            Y,
            Z,
            anomalous_cells,
            facecolors=facecolors,
            edgecolor=CELL_EDGE_COLOR,
            linewidth=CELL_EDGE_WIDTH,
            shade=True,
        )

    # Computational domain outline.
    draw_domain_outline(
        ax,
        x_edges,
        y_edges,
        z_edges,
    )

    # Axis labels.
    ax.set_xlabel(
        "East (m)",
        fontsize=12,
        labelpad=12,
    )
    ax.set_ylabel(
        "North (m)",
        fontsize=12,
        labelpad=12,
    )
    ax.set_zlabel(
        "Depth (m)",
        fontsize=12,
        labelpad=10,
    )

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    # Depth positive downward.
    ax.set_zlim(z_edges[-1], z_edges[0])

    ax.set_box_aspect(
        (
            x_edges[-1] - x_edges[0],
            y_edges[-1] - y_edges[0],
            z_edges[-1] - z_edges[0],
        )
    )

    ax.tick_params(
        axis="both",
        labelsize=9,
        pad=2,
    )
    ax.tick_params(
        axis="z",
        labelsize=9,
        pad=2,
    )

    # Camera.
    ax.view_init(
        elev=VIEW_ELEVATION,
        azim=VIEW_AZIMUTH,
    )

    # Legend.
    legend_elements = []

    if n_high > 0:
        legend_elements.append(
            Patch(
                facecolor=HIGH_SUSCEPTIBILITY_COLOR,
                edgecolor="black",
                label=(
                    "High susceptibility "
                    f"({HIGH_SUSCEPTIBILITY_VALUE:.3f} SI)"
                ),
            )
        )

    if n_low > 0:
        legend_elements.append(
            Patch(
                facecolor=LOW_SUSCEPTIBILITY_COLOR,
                edgecolor="black",
                label=(
                    "Low susceptibility "
                    f"({LOW_SUSCEPTIBILITY_VALUE:.3f} SI)"
                ),
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        borderaxespad=0.0,
    )

    # Title.
    ax.set_title(
        "True 3D Magnetic Susceptibility Model",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Remove unnecessary panes.
    ax.grid(False)

    for axis in (
        ax.xaxis,
        ax.yaxis,
        ax.zaxis,
    ):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")

    # Save figure.
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # High-resolution PNG output.
    fig.savefig(
        OUTPUT_FILE,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    # PDF output for scientific publication.
    fig.savefig(
        OUTPUT_PDF,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
    )

    print("\n" + "=" * 70)
    print(f"PNG figure saved: {OUTPUT_FILE}")
    print(f"PDF figure saved: {OUTPUT_PDF}")
    print("=" * 70)

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    plot_magnetic_model()
