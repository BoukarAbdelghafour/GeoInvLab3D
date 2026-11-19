#!/usr/bin/env python3
"""Create a comparison of true, initial and inverted MT models.

Figure layout
-------------
Rows:
    1. True model
    2. Initial model
    3. Inverted model

Columns:
    1. Horizontal X-Y section
    2. Vertical X-Z section
    3. Vertical Y-Z section
    4. Dedicated shared logarithmic colour bar

The horizontal section is selected automatically at the shallowest depth
containing an anomaly in the true model. The vertical sections are selected
near Y = +150 m (resistive anomaly) and X = -150 m (both anomalies).

All model values are plotted without threshold filtering or modification.
The same logarithmic normalization is used in all nine panels.

GeoInvLab3D conventions:
    X: East; Y: North; Z: positive downward; surface z = 0.
    Model ordering: index = i * (ny * nz) + j * nz + k (z fastest).

Associated MT data uncertainties: 5% apparent resistivity and 3 degrees phase.

Boukar Abdelghafour  MIT License Copyright (c) 2026
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
RESULTS_DIR = SCRIPT_DIR / "inversion_results"
FIGURE_DIR = RESULTS_DIR / "figures"

MESH_FILE = DATA_DIR / "mesh.dat"
TRUE_MODEL_FILE = DATA_DIR / "model_true.dat"
INITIAL_MODEL_FILE = DATA_DIR / "model_initial.dat"
INVERTED_MODEL_CANDIDATES = (
    RESULTS_DIR / "model_inverted.dat",
    RESULTS_DIR / "inverted_model.dat",
    RESULTS_DIR / "model_final.dat",
)
OUTPUT_FILE = FIGURE_DIR / "plot_mt_model_comparison.png"


# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

ANOMALY_CONDUCTIVE_MAX = 20.0
ANOMALY_RESISTIVE_MIN = 500.0

# Requested vertical-section coordinates. The nearest cell centre is used.
TARGET_XZ_Y = 150.0
TARGET_YZ_X = -150.0

CMAP = "RdYlBu_r"
FIGSIZE = (16.0, 11.5)
DPI = 600
TITLE_SIZE = 10
LABEL_SIZE = 9
TICK_SIZE = 8


# =============================================================================
# INPUT
# =============================================================================

def load_numeric_table(path: Path) -> np.ndarray:
    """Load a finite numerical ASCII table, ignoring comment lines."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    data = np.loadtxt(path, comments="#", dtype=float, ndmin=2)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError(f"Empty or non-finite numerical data in: {path}")
    return data


def load_mesh(path: Path) -> dict:
    """Load mesh dimensions, cell sizes, origin, edges and cell centres."""
    data = load_numeric_table(path)
    if data.shape[0] != 1 or data.shape[1] < 9:
        raise ValueError(
            "mesh.dat must contain one row with: "
            "nx ny nz dx dy dz origin_x origin_y origin_z"
        )

    row = data[0]
    nx, ny, nz = (int(round(v)) for v in row[:3])
    dx, dy, dz = (float(v) for v in row[3:6])
    x0, y0, z0 = (float(v) for v in row[6:9])

    if min(nx, ny, nz) <= 0 or min(dx, dy, dz) <= 0.0:
        raise ValueError("Mesh dimensions and cell sizes must be positive.")

    x_edges = x0 + np.arange(nx + 1) * dx
    y_edges = y0 + np.arange(ny + 1) * dy
    z_edges = z0 + np.arange(nz + 1) * dz

    return {
        "shape": (nx, ny, nz),
        "nx": nx, "ny": ny, "nz": nz,
        "x_edges": x_edges, "y_edges": y_edges, "z_edges": z_edges,
        "x_centers": 0.5 * (x_edges[:-1] + x_edges[1:]),
        "y_centers": 0.5 * (y_edges[:-1] + y_edges[1:]),
        "z_centers": 0.5 * (z_edges[:-1] + z_edges[1:]),
    }


def load_model(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    """Load the last column of a model file using z-fastest C ordering."""
    data = load_numeric_table(path)
    expected = int(np.prod(shape))
    if data.shape[0] != expected:
        raise ValueError(
            f"Invalid number of cells in {path}: expected {expected}, "
            f"found {data.shape[0]}."
        )

    values = data[:, -1]
    if np.any(values <= 0.0):
        raise ValueError(f"Resistivity must be strictly positive in {path}.")
    return values.reshape(shape, order="C")


def find_inverted_model() -> Path:
    """Return the first existing inverted-model candidate."""
    for path in INVERTED_MODEL_CANDIDATES:
        if path.is_file():
            return path
    candidates = "\n".join(str(p) for p in INVERTED_MODEL_CANDIDATES)
    raise FileNotFoundError(
        "No inverted model was found. Checked the following files:\n" + candidates
    )


# =============================================================================
# SECTION SELECTION AND COLOUR NORMALIZATION
# =============================================================================

def true_anomaly_mask(true_model: np.ndarray) -> np.ndarray:
    """Identify conductive and resistive cells in the true model."""
    return (
        (true_model <= ANOMALY_CONDUCTIVE_MAX)
        | (true_model >= ANOMALY_RESISTIVE_MIN)
    )


def nearest_index(coordinates: np.ndarray, target: float) -> int:
    """Return the index of the cell centre nearest to a target coordinate."""
    return int(np.argmin(np.abs(coordinates - target)))


def select_section_indices(true_model: np.ndarray, mesh: dict) -> tuple[int, int, int]:
    """Select a shallow anomalous XY layer and requested XZ/YZ coordinates."""
    anomaly = true_anomaly_mask(true_model)
    if not np.any(anomaly):
        raise ValueError(
            "No anomalies were detected in the true model. Check the anomaly "
            "thresholds or the true-model resistivities."
        )

    anomalous_depths = np.flatnonzero(np.any(anomaly, axis=(0, 1)))
    k_xy = int(anomalous_depths[0])

    # The X-Z section is deliberately placed through the resistive body.
    j_xz = nearest_index(mesh["y_centers"], TARGET_XZ_Y)

    # The Y-Z section at this X coordinate intersects the two bodies.
    i_yz = nearest_index(mesh["x_centers"], TARGET_YZ_X)

    return k_xy, j_xz, i_yz


def create_shared_log_norm(models: list[np.ndarray]) -> LogNorm:
    """Create one logarithmic colour normalization shared by all panels."""
    values = np.concatenate([model.ravel() for model in models])
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError("No positive finite resistivity values are available.")

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        vmax = vmin * 1.01
    return LogNorm(vmin=vmin, vmax=vmax)


# =============================================================================
# FIGURE
# =============================================================================

def plot_model_comparison(models: list[np.ndarray], mesh: dict) -> None:
    """Plot true, initial and inverted resistivity sections in a 3 x 3 layout."""
    names = ("True model", "Initial model", "Inverted model")
    k_xy, j_xz, i_yz = select_section_indices(models[0], mesh)
    norm = create_shared_log_norm(models)

    x_edges = mesh["x_edges"]
    y_edges = mesh["y_edges"]
    z_edges = mesh["z_edges"]
    x_centres = mesh["x_centers"]
    y_centres = mesh["y_centers"]
    z_centres = mesh["z_centers"]

    depth_xy = float(z_centres[k_xy])
    y_xz = float(y_centres[j_xz])
    x_yz = float(x_centres[i_yz])

    print("\nSelected section locations")
    print("-" * 60)
    print(f"X-Y section: z = {depth_xy:g} m (shallowest anomalous layer)")
    print(f"X-Z section: y = {y_xz:g} m (target {TARGET_XZ_Y:g} m)")
    print(f"Y-Z section: x = {x_yz:g} m (target {TARGET_YZ_X:g} m)")
    print(f"Shared resistivity range: {norm.vmin:.5g} to {norm.vmax:.5g} ohm.m")
    print("-" * 60)

    # A dedicated fourth GridSpec column reserves space for the colour bar.
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=3,
        ncols=4,
        width_ratios=(1.0, 1.0, 1.0, 0.055),
        left=0.065,
        right=0.94,
        bottom=0.075,
        top=0.88,
        wspace=0.34,
        hspace=0.36,
    )

    axes = np.empty((3, 3), dtype=object)
    for row in range(3):
        for col in range(3):
            axes[row, col] = fig.add_subplot(grid[row, col])

    cax = fig.add_subplot(grid[:, 3])
    panel_labels = "abcdefghi"
    last_mesh = None

    for row, (name, model) in enumerate(zip(names, models)):
        # Horizontal X-Y section: array transposed to (Y, X).
        ax = axes[row, 0]
        last_mesh = ax.pcolormesh(
            x_edges, y_edges, model[:, :, k_xy].T,
            cmap=CMAP, norm=norm, shading="flat",
            edgecolors="none", rasterized=True,
        )
        ax.set_title(f"{name}: X-Y, z = {depth_xy:g} m", fontsize=TITLE_SIZE)
        ax.set_xlabel("X East (m)", fontsize=LABEL_SIZE)
        ax.set_ylabel("Y North (m)", fontsize=LABEL_SIZE)
        ax.set_aspect("equal", adjustable="box")

        # Vertical X-Z section through the resistive anomaly.
        ax = axes[row, 1]
        ax.pcolormesh(
            x_edges, z_edges, model[:, j_xz, :].T,
            cmap=CMAP, norm=norm, shading="flat",
            edgecolors="none", rasterized=True,
        )
        ax.set_title(f"{name}: X-Z, y = {y_xz:g} m", fontsize=TITLE_SIZE)
        ax.set_xlabel("X East (m)", fontsize=LABEL_SIZE)
        ax.set_ylabel("Z Depth (m)", fontsize=LABEL_SIZE)
        ax.set_ylim(z_edges[-1], z_edges[0])

        # Vertical Y-Z section through both anomalies.
        ax = axes[row, 2]
        ax.pcolormesh(
            y_edges, z_edges, model[i_yz, :, :].T,
            cmap=CMAP, norm=norm, shading="flat",
            edgecolors="none", rasterized=True,
        )
        ax.set_title(f"{name}: Y-Z, x = {x_yz:g} m", fontsize=TITLE_SIZE)
        ax.set_xlabel("Y North (m)", fontsize=LABEL_SIZE)
        ax.set_ylabel("Z Depth (m)", fontsize=LABEL_SIZE)
        ax.set_ylim(z_edges[-1], z_edges[0])

    # Consistent panel labels and axis formatting.
    for index, ax in enumerate(axes.ravel()):
        ax.text(
            0.025, 0.97, f"({panel_labels[index]})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10, fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
                "pad": 2.0,
            },
        )
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.grid(False)

    # One colour bar in its own column: it cannot overlap any data panel.
    colorbar = fig.colorbar(last_mesh, cax=cax)
    colorbar.set_label(
        r"Electrical resistivity ($\Omega\,\mathrm{m}$)",
        fontsize=LABEL_SIZE,
        labelpad=9,
    )
    colorbar.ax.tick_params(labelsize=TICK_SIZE)
    colorbar.minorticks_off()

    fig.suptitle(
        "MT resistivity model comparison",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5, 0.925,
        r"Data uncertainties: apparent resistivity = 5%; phase = 3$^\circ$",
        ha="center",
        va="center",
        fontsize=10,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    # Save the figure in both high-resolution PNG and publication-ready PDF.
    OUTPUT_PDF_FILE = OUTPUT_FILE.with_suffix(".pdf")

    fig.savefig(
        OUTPUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_PDF_FILE,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"\nPNG figure saved: {OUTPUT_FILE}")
    print(f"PDF figure saved: {OUTPUT_PDF_FILE}")
    plt.show()


def print_model_statistics(models: list[np.ndarray]) -> None:
    """Print basic resistivity statistics for each model."""
    print("\nMODEL STATISTICS")
    print("=" * 65)
    for name, model in zip(
        ("True model", "Initial model", "Inverted model"), models
    ):
        print(f"\n{name}")
        print("-" * 45)
        print(f"Minimum resistivity : {np.min(model):.6g} ohm.m")
        print(f"Maximum resistivity : {np.max(model):.6g} ohm.m")
        print(f"Mean resistivity    : {np.mean(model):.6g} ohm.m")
        print(f"Median resistivity  : {np.median(model):.6g} ohm.m")
    print("=" * 65)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Load the MT models and generate the comparison figure."""
    print("\n" + "=" * 70)
    print("GeoInvLab3D - MT MODEL COMPARISON")
    print("=" * 70)

    mesh = load_mesh(MESH_FILE)
    inverted_path = find_inverted_model()

    print(f"Mesh file       : {MESH_FILE}")
    print(f"True model      : {TRUE_MODEL_FILE}")
    print(f"Initial model   : {INITIAL_MODEL_FILE}")
    print(f"Inverted model  : {inverted_path}")

    true_model = load_model(TRUE_MODEL_FILE, mesh["shape"])
    initial_model = load_model(INITIAL_MODEL_FILE, mesh["shape"])
    inverted_model = load_model(inverted_path, mesh["shape"])
    models = [true_model, initial_model, inverted_model]

    print_model_statistics(models)
    plot_model_comparison(models, mesh)
    print("\nVisualization completed.")


if __name__ == "__main__":
    main()
