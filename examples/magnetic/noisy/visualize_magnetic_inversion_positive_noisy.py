#!/usr/bin/env python3
"""
Visualize positive-constrained 3-D magnetic inversion results.

The script generates:
    - 3-D visualization of the true and inverted models.
    - 2-D horizontal and vertical model sections.
    - True versus inverted model comparison.
    - Observed versus predicted magnetic data.
    - Inversion parameter summary.

Only anomalous cells are displayed in the 3-D views. The background
(host medium) remains transparent, while the external mesh outline
is retained for spatial reference.

Coordinate convention:
    x = East
    y = North
    z = positive downward
    surface = z = 0

Run from the project root:

    python examples/magnetic/noisy/visualize_magnetic_inversion_positive_noisy.py \
        --noise-level 1

Optional arguments:

    --threshold-factor 0.10
    --section-depth 150
    --no-show
    --interactive
    
    Boukar Abdelghafour  GeoInvLab3D   MIT License Copyright (c) 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

DATA_DIR = PROJECT_ROOT / "examples" / "magnetic" / "synthetic" / "data"
RESULTS_ROOT = SCRIPT_DIR / "inversion_results"


# ---------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------

DPI = 300

FIGURE_DIR_NAME = "figures"

DEFAULT_THRESHOLD_FACTOR = 0.10

# Coordinates and model values are expected in SI units.
MODEL_COLUMN_NAMES = (
    "x_center",
    "y_center",
    "z_center",
    "susceptibility",
)


# ---------------------------------------------------------------------
# Console utilities
# ---------------------------------------------------------------------

def print_header(title: str) -> None:
    """Print a formatted console heading."""
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def print_section(title: str) -> None:
    """Print a formatted section heading."""
    print()
    print("-" * 76)
    print(title)
    print("-" * 76)


def stop(message: str) -> None:
    """Stop execution with a readable error message."""
    print()
    print(f"ERROR: {message}")
    sys.exit(1)


# ---------------------------------------------------------------------
# ASCII data loading
# ---------------------------------------------------------------------

def load_mesh(mesh_file: Path) -> dict:
    """
    Load structured mesh metadata.

    Expected format:
        nx ny nz dx dy dz ox oy oz
    """
    if not mesh_file.is_file():
        stop(f"Mesh file not found: {mesh_file}")

    values = np.loadtxt(mesh_file, comments="#", dtype=float)
    values = np.asarray(values).ravel()

    if values.size < 9:
        stop(
            "Invalid mesh.dat format. Expected at least nine values: "
            "nx ny nz dx dy dz ox oy oz."
        )

    nx, ny, nz = (int(round(v)) for v in values[:3])
    dx, dy, dz = values[3:6]
    ox, oy, oz = values[6:9]

    if min(nx, ny, nz) <= 0:
        stop("Mesh dimensions must be strictly positive.")

    if min(dx, dy, dz) <= 0:
        stop("Cell dimensions must be strictly positive.")

    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "ox": float(ox),
        "oy": float(oy),
        "oz": float(oz),
        "x_edges": ox + np.arange(nx + 1) * dx,
        "y_edges": oy + np.arange(ny + 1) * dy,
        "z_edges": oz + np.arange(nz + 1) * dz,
        "x_centers": ox + (np.arange(nx) + 0.5) * dx,
        "y_centers": oy + (np.arange(ny) + 0.5) * dy,
        "z_centers": oz + (np.arange(nz) + 0.5) * dz,
    }


def load_model(model_file: Path, mesh: dict, label: str) -> np.ndarray:
    """
    Load a model file and return a structured 3-D array.

    Expected columns:
        x_center y_center z_center susceptibility

    The model is reshaped using C ordering because z is the
    fastest-varying index in the ASCII model files.
    """
    if not model_file.is_file():
        stop(f"{label} model file not found: {model_file}")

    data = np.loadtxt(model_file, comments="#", dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 4:
        stop(
            f"Invalid {label} model file. Expected four columns: "
            "x_center y_center z_center susceptibility."
        )

    nx, ny, nz = mesh["nx"], mesh["ny"], mesh["nz"]
    expected_cells = nx * ny * nz

    if data.shape[0] != expected_cells:
        stop(
            f"{label} model contains {data.shape[0]} cells, "
            f"but the mesh requires {expected_cells} cells."
        )

    coordinates = data[:, :3]
    expected_x, expected_y, expected_z = np.meshgrid(
        mesh["x_centers"],
        mesh["y_centers"],
        mesh["z_centers"],
        indexing="ij",
    )

    expected_coordinates = np.column_stack(
        (
            expected_x.ravel(order="C"),
            expected_y.ravel(order="C"),
            expected_z.ravel(order="C"),
        )
    )

    if not np.allclose(
        coordinates,
        expected_coordinates,
        rtol=1e-10,
        atol=1e-8,
    ):
        print(
            f"WARNING: Coordinates in {label} model do not exactly "
            "match the expected mesh ordering."
        )

    values = data[:, 3]

    return values.reshape((nx, ny, nz), order="C")


def load_data_fit(data_file: Path) -> np.ndarray | None:
    """
    Load observed and predicted magnetic data.

    The first two columns are interpreted as observed and predicted
    values, respectively.
    """
    if not data_file.is_file():
        print(f"WARNING: Data-fit file not found: {data_file}")
        return None

    data = np.loadtxt(data_file, comments="#", dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        print(
            "WARNING: Data-fit file must contain at least two columns."
        )
        return None

    return data


def load_statistics(statistics_file: Path) -> tuple[list[str], np.ndarray] | None:
    """
    Load numerical inversion statistics.

    The function supports numeric ASCII tables with optional comments.
    """
    if not statistics_file.is_file():
        print(f"WARNING: Statistics file not found: {statistics_file}")
        return None

    try:
        values = np.loadtxt(statistics_file, comments="#", dtype=float)
    except ValueError:
        print(
            "WARNING: Statistics file is not a purely numeric table. "
            "It will not be plotted automatically."
        )
        return None

    if values.ndim == 1:
        values = values.reshape(1, -1)

    return [], values


# ---------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------

def print_model_information(
    mesh: dict,
    true_model: np.ndarray,
    inverted_model: np.ndarray,
) -> None:
    """Print mesh and model information."""
    print_section("MESH INFORMATION")

    print(f"Dimensions       : {mesh['nx']} x {mesh['ny']} x {mesh['nz']}")
    print(f"Number of cells  : {mesh['nx'] * mesh['ny'] * mesh['nz']}")
    print(
        f"Cell dimensions  : "
        f"{mesh['dx']:g} x {mesh['dy']:g} x {mesh['dz']:g} m"
    )
    print(
        f"X extent         : "
        f"{mesh['x_edges'][0]:g} to {mesh['x_edges'][-1]:g} m"
    )
    print(
        f"Y extent         : "
        f"{mesh['y_edges'][0]:g} to {mesh['y_edges'][-1]:g} m"
    )
    print(
        f"Z extent         : "
        f"{mesh['z_edges'][0]:g} to {mesh['z_edges'][-1]:g} m"
    )

    print_section("MODEL INFORMATION")

    for name, model in (
        ("True model", true_model),
        ("Inverted model", inverted_model),
    ):
        print(f"{name}:")
        print(f"    Minimum susceptibility : {np.min(model):.6e} SI")
        print(f"    Maximum susceptibility : {np.max(model):.6e} SI")
        print(f"    Mean susceptibility    : {np.mean(model):.6e} SI")
        print(f"    RMS susceptibility     : {np.sqrt(np.mean(model**2)):.6e} SI")


# ---------------------------------------------------------------------
# PyVista grid construction
# ---------------------------------------------------------------------

def create_pyvista_grid(mesh: dict, model: np.ndarray):
    """
    Create a PyVista StructuredGrid.

    The model array uses C ordering, with z varying fastest in the
    input model file. PyVista cell data are assigned in VTK ordering.
    """
    if pv is None:
        raise ImportError(
            "PyVista is required for 3-D visualization. "
            "Install it with: pip install pyvista"
        )

    X, Y, Z = np.meshgrid(
        mesh["x_edges"],
        mesh["y_edges"],
        mesh["z_edges"],
        indexing="ij",
    )

    grid = pv.StructuredGrid(X, Y, Z)

    # Convert the (nx, ny, nz) model array to VTK cell ordering.
    grid.cell_data["Susceptibility"] = model.ravel(order="F")

    return grid


# ---------------------------------------------------------------------
# 3-D visualization
# ---------------------------------------------------------------------

def plot_model_3d(
    mesh: dict,
    model: np.ndarray,
    title: str,
    output_file: Path,
    threshold_factor: float = DEFAULT_THRESHOLD_FACTOR,
    absolute_threshold: float | None = None,
    interactive: bool = False,
) -> None:
    """
    Plot anomalous cells in 3-D.

    The host medium is not rendered, so it remains transparent.
    Only cells above the selected threshold are displayed.

    If absolute_threshold is provided, it is used directly.
    Otherwise, the threshold is calculated as a fraction of the
    maximum model susceptibility.
    """
    if pv is None:
        print("WARNING: PyVista is not installed. Skipping 3-D plot.")
        return

    grid = create_pyvista_grid(mesh, model)

    model_max = float(np.max(model))
    model_min = float(np.min(model))

    if absolute_threshold is not None:
        threshold = float(absolute_threshold)
    else:
        threshold = threshold_factor * model_max

    if threshold < model_min:
        threshold = model_min

    print(f"{title}")
    print(f"    Threshold : {threshold:.6e} SI")
    print(f"    Maximum   : {model_max:.6e} SI")

    # Keep only cells above the threshold.
    anomalous_grid = grid.threshold(
        value=(threshold, np.inf),
        scalars="Susceptibility",
        preference="cell",
    )

    plotter = pv.Plotter(
        window_size=(1400, 1000),
        off_screen=not interactive,
    )

    if anomalous_grid.n_cells > 0:
        plotter.add_mesh(
            anomalous_grid,
            scalars="Susceptibility",
            cmap="viridis",
            show_edges=True,
            edge_color="black",
            line_width=0.5,
            opacity=1.0,
            scalar_bar_args={
                "title": "Susceptibility (SI)",
                "vertical": True,
                "title_font_size": 18,
                "label_font_size": 14,
            },
        )
    else:
        print("WARNING: No cells satisfy the selected threshold.")

    # Display only the external mesh outline.
    plotter.add_mesh(
        grid.outline(),
        color="black",
        line_width=2.0,
    )

    plotter.add_axes(
        xlabel="East (m)",
        ylabel="North (m)",
        zlabel="Depth (m)",
    )

    plotter.show_bounds(
        grid="back",
        location="outer",
        all_edges=True,
        xlabel="East (m)",
        ylabel="North (m)",
        zlabel="Depth (m)",
        font_size=12,
    )

    plotter.add_text(
        (
            f"{title}\n"
            f"Threshold = {threshold:.3e} SI"
        ),
        position="upper_left",
        font_size=12,
    )

    plotter.camera_position = "iso"
    plotter.camera.zoom(1.15)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if interactive:
        plotter.show()
    else:
        plotter.show(
            screenshot=str(output_file),
            auto_close=True,
        )

    print(f"    Saved: {output_file}")


# ---------------------------------------------------------------------
# 2-D visualization
# ---------------------------------------------------------------------

def plot_horizontal_section(
    mesh: dict,
    model: np.ndarray,
    depth: float,
    title: str,
    output_file: Path,
) -> None:
    """Plot a horizontal susceptibility section at the nearest depth."""
    iz = int(np.argmin(np.abs(mesh["z_centers"] - depth)))
    actual_depth = mesh["z_centers"][iz]

    section = model[:, :, iz].T

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        section,
        origin="lower",
        extent=(
            mesh["x_edges"][0],
            mesh["x_edges"][-1],
            mesh["y_edges"][0],
            mesh["y_edges"][-1],
        ),
        aspect="equal",
        cmap="viridis",
        interpolation="nearest",
    )

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"{title}\nDepth = {actual_depth:g} m")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Susceptibility (SI)")

    fig.tight_layout()
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_vertical_sections(
    mesh: dict,
    model: np.ndarray,
    title: str,
    output_file: Path,
) -> None:
    """Plot central East-West and North-South vertical sections."""
    ix = mesh["nx"] // 2
    iy = mesh["ny"] // 2

    east_west = model[:, iy, :].T
    north_south = model[ix, :, :].T

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    image1 = axes[0].imshow(
        east_west,
        origin="upper",
        extent=(
            mesh["x_edges"][0],
            mesh["x_edges"][-1],
            mesh["z_edges"][-1],
            mesh["z_edges"][0],
        ),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    axes[0].set_xlabel("East (m)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_title(
        f"East-West section (North = {mesh['y_centers'][iy]:g} m)"
    )

    image2 = axes[1].imshow(
        north_south,
        origin="upper",
        extent=(
            mesh["y_edges"][0],
            mesh["y_edges"][-1],
            mesh["z_edges"][-1],
            mesh["z_edges"][0],
        ),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    axes[1].set_xlabel("North (m)")
    axes[1].set_ylabel("Depth (m)")
    axes[1].set_title(
        f"North-South section (East = {mesh['x_centers'][ix]:g} m)"
    )

    fig.colorbar(
        image1,
        ax=axes[0],
        label="Susceptibility (SI)",
    )
    fig.colorbar(
        image2,
        ax=axes[1],
        label="Susceptibility (SI)",
    )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(
    mesh: dict,
    true_model: np.ndarray,
    inverted_model: np.ndarray,
    depth: float,
    output_file: Path,
) -> None:
    """Compare true, inverted, and difference horizontal sections."""
    iz = int(np.argmin(np.abs(mesh["z_centers"] - depth)))
    actual_depth = mesh["z_centers"][iz]

    true_section = true_model[:, :, iz].T
    inverted_section = inverted_model[:, :, iz].T
    difference = inverted_section - true_section

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    extent = (
        mesh["x_edges"][0],
        mesh["x_edges"][-1],
        mesh["y_edges"][0],
        mesh["y_edges"][-1],
    )

    vmax = max(
        float(np.max(true_section)),
        float(np.max(inverted_section)),
    )

    for ax, values, title in (
        (axes[0], true_section, "True model"),
        (axes[1], inverted_section, "Inverted model"),
    ):
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax if vmax > 0 else None,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        fig.colorbar(image, ax=ax, label="Susceptibility (SI)")

    diff_limit = max(float(np.max(np.abs(difference))), 1e-15)

    image = axes[2].imshow(
        difference,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="RdBu_r",
        vmin=-diff_limit,
        vmax=diff_limit,
        interpolation="nearest",
    )

    axes[2].set_title("Inverted - True")
    axes[2].set_xlabel("East (m)")
    axes[2].set_ylabel("North (m)")
    fig.colorbar(image, ax=axes[2], label="Susceptibility difference (SI)")

    fig.suptitle(f"Model comparison at depth = {actual_depth:g} m")
    fig.tight_layout()
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Magnetic data fit
# ---------------------------------------------------------------------

def plot_data_fit(
    data: np.ndarray | None,
    output_file: Path,
) -> None:
    """Plot observed versus predicted magnetic data."""
    if data is None:
        return

    observed = data[:, 0]
    predicted = data[:, 1]

    lower = min(float(np.min(observed)), float(np.min(predicted)))
    upper = max(float(np.max(observed)), float(np.max(predicted)))

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        observed,
        predicted,
        s=22,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.3,
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        "k--",
        linewidth=1.5,
        label="1:1 line",
    )

    ax.set_xlabel("Observed magnetic data (nT)")
    ax.set_ylabel("Predicted magnetic data (nT)")
    ax.set_title("Observed versus predicted magnetic data")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    residual = predicted - observed

    print_section("DATA FIT")

    print(f"Number of observations : {observed.size}")
    print(f"Observed RMS           : {np.sqrt(np.mean(observed**2)):.6f} nT")
    print(f"Predicted RMS          : {np.sqrt(np.mean(predicted**2)):.6f} nT")
    print(f"Residual RMS           : {np.sqrt(np.mean(residual**2)):.6f} nT")
    print(f"Residual mean          : {np.mean(residual):.6f} nT")
    print(f"Residual standard dev. : {np.std(residual):.6f} nT")


# ---------------------------------------------------------------------
# Inversion report
# ---------------------------------------------------------------------

def write_report(
    output_file: Path,
    noise_level: int,
    mesh: dict,
    true_model: np.ndarray,
    inverted_model: np.ndarray,
    data_fit: np.ndarray | None,
    threshold: float,
) -> None:
    """Write a plain-text inversion visualization report."""
    lines = [
        "GeoInvLab3D - POSITIVE-CONSTRAINED MAGNETIC INVERSION",
        "=" * 65,
        "",
        f"Noise level: {noise_level}%",
        "",
        "Mesh",
        "----",
        f"Dimensions: {mesh['nx']} x {mesh['ny']} x {mesh['nz']}",
        f"Number of cells: {mesh['nx'] * mesh['ny'] * mesh['nz']}",
        f"Cell size: {mesh['dx']} x {mesh['dy']} x {mesh['dz']} m",
        "",
        "True model",
        "----------",
        f"Minimum susceptibility: {np.min(true_model):.8e} SI",
        f"Maximum susceptibility: {np.max(true_model):.8e} SI",
        f"Mean susceptibility: {np.mean(true_model):.8e} SI",
        "",
        "Inverted model",
        "--------------",
        f"Minimum susceptibility: {np.min(inverted_model):.8e} SI",
        f"Maximum susceptibility: {np.max(inverted_model):.8e} SI",
        f"Mean susceptibility: {np.mean(inverted_model):.8e} SI",
        f"Negative cells: {np.count_nonzero(inverted_model < 0)}",
        "",
        "3-D anomaly visualization",
        "--------------------------",
        f"Display threshold: {threshold:.8e} SI",
        "",
    ]

    if data_fit is not None:
        observed = data_fit[:, 0]
        predicted = data_fit[:, 1]
        residual = predicted - observed

        lines.extend(
            [
                "Data fit",
                "--------",
                f"Number of observations: {observed.size}",
                f"Observed RMS: {np.sqrt(np.mean(observed**2)):.8e} nT",
                f"Predicted RMS: {np.sqrt(np.mean(predicted**2)):.8e} nT",
                f"Residual RMS: {np.sqrt(np.mean(residual**2)):.8e} nT",
                f"Residual mean: {np.mean(residual):.8e} nT",
                "",
            ]
        )

    output_file.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize positive-constrained noisy 3-D magnetic "
            "inversion results."
        )
    )

    parser.add_argument(
        "--noise-level",
        type=int,
        choices=(1, 3, 5),
        default=1,
        help="Noise level in percent.",
    )

    parser.add_argument(
        "--threshold-factor",
        type=float,
        default=DEFAULT_THRESHOLD_FACTOR,
        help=(
            "Relative 3-D display threshold as a fraction of the "
            "maximum susceptibility. Default: 0.10."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Absolute susceptibility threshold in SI. "
            "If specified, it overrides --threshold-factor."
        ),
    )

    parser.add_argument(
        "--section-depth",
        type=float,
        default=None,
        help="Depth in metres for horizontal section plots.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display Matplotlib figures.",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open interactive PyVista 3-D windows.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the visualization workflow."""
    args = parse_arguments()

    print_header(
        "GeoInvLab3D - POSITIVE-CONSTRAINED MAGNETIC INVERSION VISUALIZATION"
    )

    noise_label = f"noise_{args.noise_level}pct"

    result_dir = RESULTS_ROOT / noise_label
    figure_dir = result_dir / FIGURE_DIR_NAME
    figure_dir.mkdir(parents=True, exist_ok=True)

    mesh_file = DATA_DIR / "mesh.dat"
    true_model_file = DATA_DIR / "model_true.dat"

    inverted_model_file = result_dir / "model_inverted_positive.dat"
    data_fit_file = result_dir / "data_fit_positive.dat"
    statistics_file = result_dir / "inversion_statistics_positive.dat"

    print_section("INPUT FILES")

    print(f"Mesh file       : {mesh_file}")
    print(f"True model      : {true_model_file}")
    print(f"Inverted model  : {inverted_model_file}")
    print(f"Data fit        : {data_fit_file}")
    print(f"Statistics      : {statistics_file}")
    print(f"Figure directory: {figure_dir}")

    mesh = load_mesh(mesh_file)

    true_model = load_model(
        true_model_file,
        mesh,
        "True",
    )

    inverted_model = load_model(
        inverted_model_file,
        mesh,
        "Inverted",
    )

    data_fit = load_data_fit(data_fit_file)

    # Load the statistics file to check its availability.
    load_statistics(statistics_file)

    print_model_information(
        mesh,
        true_model,
        inverted_model,
    )

    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = args.threshold_factor * float(np.max(inverted_model))

    print_section("3-D DISPLAY SETTINGS")
    print(f"Noise level       : {args.noise_level}%")
    print(f"Threshold factor  : {args.threshold_factor:.3f}")
    print(f"Absolute threshold: {threshold:.6e} SI")
    print("Host medium       : transparent")
    print("Mesh outline      : visible")

    if args.section_depth is None:
        section_depth = float(mesh["z_centers"][mesh["nz"] // 2])
    else:
        section_depth = args.section_depth

    # 3-D true model.
    plot_model_3d(
        mesh=mesh,
        model=true_model,
        title="True magnetic susceptibility model",
        output_file=figure_dir / "true_model_3d.png",
        threshold_factor=args.threshold_factor,
        absolute_threshold=args.threshold,
        interactive=args.interactive,
    )

    # 3-D inverted model.
    plot_model_3d(
        mesh=mesh,
        model=inverted_model,
        title=f"Inverted model - {args.noise_level}% noise",
        output_file=figure_dir / "inverted_model_3d.png",
        threshold_factor=args.threshold_factor,
        absolute_threshold=args.threshold,
        interactive=args.interactive,
    )

    # Horizontal sections.
    plot_horizontal_section(
        mesh,
        true_model,
        section_depth,
        "True model",
        figure_dir / "true_model_horizontal.png",
    )

    plot_horizontal_section(
        mesh,
        inverted_model,
        section_depth,
        "Inverted model",
        figure_dir / "inverted_model_horizontal.png",
    )

    # Vertical sections.
    plot_vertical_sections(
        mesh,
        true_model,
        "True model - vertical sections",
        figure_dir / "true_model_vertical_sections.png",
    )

    plot_vertical_sections(
        mesh,
        inverted_model,
        "Inverted model - vertical sections",
        figure_dir / "inverted_model_vertical_sections.png",
    )

    # True versus inverted model.
    plot_model_comparison(
        mesh,
        true_model,
        inverted_model,
        section_depth,
        figure_dir / "model_comparison_2d.png",
    )

    # Observed versus predicted data.
    plot_data_fit(
        data_fit,
        figure_dir / "data_fit.png",
    )

    # Text report.
    write_report(
        output_file=figure_dir / "inversion_report.txt",
        noise_level=args.noise_level,
        mesh=mesh,
        true_model=true_model,
        inverted_model=inverted_model,
        data_fit=data_fit,
        threshold=threshold,
    )

    if not args.no_show:
        plt.show()

    print_header("VISUALIZATION COMPLETED")
    print(f"Figures and report saved in:\n{figure_dir}")


if __name__ == "__main__":
    main()
