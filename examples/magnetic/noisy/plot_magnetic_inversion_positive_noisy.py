#!/usr/bin/env python3
"""
Visualize positive-constrained 3-D magnetic inversion results.

Generated figures:
    1. model_3d.png
    2. model_sections_2d.png
    3. model_comparison_2d.png
    4. data_fit_maps.png
    5. inversion_parameters.png

Coordinate convention:
    x = East
    y = North
    z = positive downward
    surface = z = 0

Model ordering:
    z-index is the fastest varying index.

Run from the repository root:

    python examples/magnetic/noisy/visualize_magnetic_inversion_positive_noisy.py

Select the noise level:

    --noise-level 1
    --noise-level 3
    --noise-level 5

Optional arguments:

    --no-show
    --interactive
    
    Boukar Abdelghafour  GeoInvLab3D   MIT License Copyright (c) 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import pyvista as pv


# =============================================================================
# PATHS AND CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

SHARED_DATA_DIR = (
    PROJECT_ROOT / "examples" / "magnetic" / "synthetic" / "data"
)

RESULTS_ROOT = SCRIPT_DIR / "inversion_results"

NOISE_LEVELS = (1, 3, 5)

FIGURE_DPI = 300

# Threshold used to identify significant positive susceptibility.
ANOMALY_THRESHOLD_FACTOR = 0.10

# Matplotlib configuration.
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.titlesize": 15,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
    }
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mesh(path: Path) -> dict:
    """Load structured mesh geometry."""

    if not path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    values = np.asarray(
        np.loadtxt(path, comments="#"),
        dtype=float,
    ).ravel()

    if values.size != 9:
        raise ValueError(
            f"Expected 9 values in mesh.dat, got {values.size}."
        )

    nx, ny, nz = values[:3].astype(int)
    dx, dy, dz = values[3:6]
    ox, oy, oz = values[6:9]

    if min(nx, ny, nz) < 1:
        raise ValueError("Invalid mesh dimensions.")

    if min(dx, dy, dz) <= 0:
        raise ValueError("Invalid cell dimensions.")

    return {
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "ox": float(ox),
        "oy": float(oy),
        "oz": float(oz),
        "x": ox + (np.arange(nx) + 0.5) * dx,
        "y": oy + (np.arange(ny) + 0.5) * dy,
        "z": oz + (np.arange(nz) + 0.5) * dz,
        "xe": ox + np.arange(nx + 1) * dx,
        "ye": oy + np.arange(ny + 1) * dy,
        "ze": oz + np.arange(nz + 1) * dz,
        "n_cells": nx * ny * nz,
    }


def load_model(path: Path, mesh: dict, name: str) -> np.ndarray:
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

    if table.shape[0] != mesh["n_cells"]:
        raise ValueError(
            f"{path.name}: {table.shape[0]} cells found, "
            f"{mesh['n_cells']} expected."
        )

    if not np.all(np.isfinite(table)):
        raise ValueError(f"{path.name} contains non-finite values.")

    # Check that model coordinates match the mesh cell centres.
    X, Y, Z = np.meshgrid(
        mesh["x"],
        mesh["y"],
        mesh["z"],
        indexing="ij",
    )

    expected_centers = np.column_stack(
        (
            X.ravel(order="C"),
            Y.ravel(order="C"),
            Z.ravel(order="C"),
        )
    )

    if not np.allclose(
        table[:, :3],
        expected_centers,
        rtol=1e-8,
        atol=1e-8,
    ):
        raise ValueError(
            f"{path.name}: cell coordinates do not match "
            "the expected mesh ordering."
        )

    # The z-index is the fastest varying index.
    return table[:, 3].reshape(
        (mesh["nx"], mesh["ny"], mesh["nz"]),
        order="C",
    )


def load_data_fit(path: Path) -> dict:
    """Load observed, predicted and residual magnetic data."""

    if not path.is_file():
        raise FileNotFoundError(f"Data-fit file not found: {path}")

    table = np.loadtxt(
        path,
        comments="#",
        ndmin=2,
    )

    if table.shape[1] != 6:
        raise ValueError(
            "data_fit_positive.dat must contain six columns."
        )

    if not np.all(np.isfinite(table)):
        raise ValueError("Data-fit file contains non-finite values.")

    return {
        "x": table[:, 0],
        "y": table[:, 1],
        "z": table[:, 2],
        "observed": table[:, 3],
        "predicted": table[:, 4],
        "residual": table[:, 5],
    }


def load_statistics(path: Path) -> dict[str, float]:
    """Load inversion statistics saved by the inversion script."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Inversion statistics not found: {path}"
        )

    values = np.asarray(
        np.loadtxt(path, comments="#"),
        dtype=float,
    ).ravel()

    names = (
        "noise_level",
        "data_rms",
        "mean_absolute_error",
        "maximum_absolute_error",
        "relative_l2",
        "susceptibility_minimum",
        "susceptibility_maximum",
        "susceptibility_mean",
        "susceptibility_rms",
        "model_rmse",
        "negative_cells",
        "positive_cells",
        "zero_cells",
        "regularization_lambda",
    )

    if values.size != len(names):
        raise ValueError(
            f"Expected {len(names)} statistics, "
            f"got {values.size}."
        )

    return dict(zip(names, values))


# =============================================================================
# MODEL STATISTICS
# =============================================================================

def calculate_model_statistics(
    true_model: np.ndarray,
    inverted_model: np.ndarray,
) -> dict[str, float]:
    """Calculate model reconstruction statistics."""

    difference = inverted_model - true_model

    return {
        "true_min": float(true_model.min()),
        "true_max": float(true_model.max()),
        "true_mean": float(true_model.mean()),
        "inverted_min": float(inverted_model.min()),
        "inverted_max": float(inverted_model.max()),
        "inverted_mean": float(inverted_model.mean()),
        "model_rmse": float(np.sqrt(np.mean(difference**2))),
    }


# =============================================================================
# 3D MODEL VISUALIZATION
# =============================================================================

def create_pyvista_grid(
    mesh: dict,
    model: np.ndarray,
) -> pv.StructuredGrid:
    """Create a PyVista structured grid from cell-centred values."""

    X, Y, Z = np.meshgrid(
        mesh["xe"],
        mesh["ye"],
        mesh["ze"],
        indexing="ij",
    )

    grid = pv.StructuredGrid(X, Y, Z)

    # VTK structured-grid cell ordering uses x as the fastest index.
    # The input model uses z as the fastest index.
    vtk_values = model.ravel(order="F")

    grid.cell_data["Susceptibility"] = vtk_values

    return grid


def plot_model_3d(
    mesh: dict,
    true_model: np.ndarray,
    inverted_model: np.ndarray,
    output_path: Path,
    interactive: bool = False,
) -> None:
    """Generate a 3D visualization of the inverted model."""

    print("\nGenerating 3D model visualization...")

    grid = create_pyvista_grid(
        mesh,
        inverted_model,
    )

    maximum = float(np.max(inverted_model))

    if maximum <= 0.0:
        raise ValueError(
            "The inverted model contains no positive susceptibility."
        )

    threshold = (
        ANOMALY_THRESHOLD_FACTOR * maximum
    )

    anomalous_grid = grid.threshold(
        value=threshold,
        scalars="Susceptibility",
        preference="cell",
    )

    plotter = pv.Plotter(
        off_screen=not interactive,
        window_size=(1600, 1100),
    )

    plotter.set_background("white")

    if anomalous_grid.n_cells > 0:

        plotter.add_mesh(
            anomalous_grid,
            scalars="Susceptibility",
            cmap="viridis",
            show_edges=True,
            edge_color="black",
            line_width=0.35,
            opacity=1.0,
            scalar_bar_args={
                "title": "Susceptibility (SI)",
                "vertical": True,
                "title_font_size": 16,
                "label_font_size": 12,
            },
        )

    else:
        plotter.add_text(
            "No cells exceed the anomaly threshold.",
            font_size=12,
            color="black",
        )

    # Draw only the external mesh boundary.
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
        xtitle="East (m)",
        ytitle="North (m)",
        ztitle="Depth (m)",
        color="black",
    )

    plotter.add_text(
        "Positive-constrained magnetic inversion\n"
        f"Anomaly threshold: {threshold:.3e} SI",
        position="upper_left",
        font_size=12,
        color="black",
    )

    plotter.camera_position = "iso"

    # Preserve positive-downward depth convention.
    plotter.camera.up = (0.0, 1.0, 0.0)

    plotter.reset_camera()

    if interactive:
        plotter.show(
            auto_close=True,
        )
    else:
        plotter.show(
            screenshot=str(output_path),
            auto_close=True,
        )

    print(f"Saved: {output_path}")


# =============================================================================
# 2D MODEL SECTIONS
# =============================================================================

def plot_model_sections(
    mesh: dict,
    inverted_model: np.ndarray,
    output_path: Path,
) -> None:
    """Plot horizontal and vertical susceptibility sections."""

    print("\nGenerating 2D susceptibility sections...")

    nx = mesh["nx"]
    ny = mesh["ny"]
    nz = mesh["nz"]

    ix = nx // 2
    iy = ny // 2
    iz = nz // 2

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5.5),
    )

    fig.suptitle(
        "3D Magnetic Inversion - Susceptibility Sections",
        fontweight="bold",
    )

    maximum = max(
        float(np.max(inverted_model)),
        1e-15,
    )

    levels = np.linspace(
        0.0,
        maximum,
        31,
    )

    # ---------------------------------------------------------
    # Horizontal section
    # ---------------------------------------------------------

    ax = axes[0]

    section_xy = inverted_model[:, :, iz]

    image = ax.contourf(
        mesh["x"],
        mesh["y"],
        section_xy.T,
        levels=levels,
        cmap="viridis",
        extend="max",
    )

    ax.set_title(
        f"Horizontal section - z = {mesh['z'][iz]:.1f} m"
    )

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal")

    fig.colorbar(
        image,
        ax=ax,
        label="Susceptibility (SI)",
    )

    # ---------------------------------------------------------
    # Vertical section X-Z
    # ---------------------------------------------------------

    ax = axes[1]

    section_xz = inverted_model[:, iy, :]

    image = ax.contourf(
        mesh["x"],
        mesh["z"],
        section_xz.T,
        levels=levels,
        cmap="viridis",
        extend="max",
    )

    ax.set_title(
        f"Vertical section X-Z - y = {mesh['y'][iy]:.1f} m"
    )

    ax.set_xlabel("East (m)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_aspect("equal")

    fig.colorbar(
        image,
        ax=ax,
        label="Susceptibility (SI)",
    )

    # ---------------------------------------------------------
    # Vertical section Y-Z
    # ---------------------------------------------------------

    ax = axes[2]

    section_yz = inverted_model[ix, :, :]

    image = ax.contourf(
        mesh["y"],
        mesh["z"],
        section_yz.T,
        levels=levels,
        cmap="viridis",
        extend="max",
    )

    ax.set_title(
        f"Vertical section Y-Z - x = {mesh['x'][ix]:.1f} m"
    )

    ax.set_xlabel("North (m)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_aspect("equal")

    fig.colorbar(
        image,
        ax=ax,
        label="Susceptibility (SI)",
    )

    fig.tight_layout()

    fig.savefig(output_path)

    plt.close(fig)

    print(f"Saved: {output_path}")


# =============================================================================
# TRUE VS INVERTED MODEL
# =============================================================================

def plot_model_comparison(
    mesh: dict,
    true_model: np.ndarray,
    inverted_model: np.ndarray,
    output_path: Path,
) -> None:
    """Compare true and inverted susceptibility sections."""

    print("\nGenerating true versus inverted model comparison...")

    iz = mesh["nz"] // 2
    iy = mesh["ny"] // 2

    true_horizontal = true_model[:, :, iz].T
    inverted_horizontal = inverted_model[:, :, iz].T

    true_vertical = true_model[:, iy, :].T
    inverted_vertical = inverted_model[:, iy, :].T

    maximum = max(
        float(np.max(true_model)),
        float(np.max(inverted_model)),
        1e-15,
    )

    levels = np.linspace(
        0.0,
        maximum,
        31,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
    )

    fig.suptitle(
        "True and Inverted Magnetic Susceptibility Models",
        fontweight="bold",
    )

    # True horizontal.
    image = axes[0, 0].contourf(
        mesh["x"],
        mesh["y"],
        true_horizontal,
        levels=levels,
        cmap="viridis",
        extend="max",
    )

    axes[0, 0].set_title(
        f"True model - horizontal, z = {mesh['z'][iz]:.1f} m"
    )
    axes[0, 0].set_xlabel("East (m)")
    axes[0, 0].set_ylabel("North (m)")
    axes[0, 0].set_aspect("equal")

    fig.colorbar(
        image,
        ax=axes[0, 0],
        label="Susceptibility (SI)",
    )

    # Inverted horizontal.
    image = axes[0, 1].contourf(
        mesh["x"],
        mesh["y"],
        inverted_horizontal,
        levels=levels,
        cmap="viridis",
        extend="max",
    )

    axes[0, 1].set_title(
        f"Inverted model - horizontal, z = {mesh['z'][iz]:.1f} m"
    )
    axes[0, 1].set_xlabel("East (m)")
    axes[0, 1].set_ylabel("North (m)")
    axes[0, 1].set_aspect("equal")

    fig.colorbar(
        image,
        ax=axes[0, 1],
        label="Susceptibility (SI)",
    )

    # True vertical.
    image = axes[1, 0].contourf(
        mesh["x"],
        mesh["z"],
        true_vertical,
        levels=levels,
        cmap="viridis",
        extend="max",
    )

    axes[1, 0].set_title(
        f"True model - vertical X-Z, y = {mesh['y'][iy]:.1f} m"
    )
    axes[1, 0].set_xlabel("East (m)")
    axes[1, 0].set_ylabel("Depth (m)")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_aspect("equal")

    fig.colorbar(
        image,
        ax=axes[1, 0],
        label="Susceptibility (SI)",
    )

    # Inverted vertical.
    image = axes[1, 1].contourf(
        mesh["x"],
        mesh["z"],
        inverted_vertical,
        levels=levels,
        cmap="viridis",
        extend="max",
    )

    axes[1, 1].set_title(
        f"Inverted model - vertical X-Z, y = {mesh['y'][iy]:.1f} m"
    )
    axes[1, 1].set_xlabel("East (m)")
    axes[1, 1].set_ylabel("Depth (m)")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_aspect("equal")

    fig.colorbar(
        image,
        ax=axes[1, 1],
        label="Susceptibility (SI)",
    )

    fig.tight_layout()

    fig.savefig(output_path)

    plt.close(fig)

    print(f"Saved: {output_path}")


# =============================================================================
# MAGNETIC DATA FIT
# =============================================================================

def plot_data_fit(
    data: dict,
    output_path: Path,
) -> None:
    """Plot observed, predicted and residual magnetic anomalies."""

    print("\nGenerating magnetic data-fit maps...")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
    )

    fig.suptitle(
        "Magnetic Data Fit - Observed, Predicted and Residual",
        fontweight="bold",
    )

    fields = (
        ("observed", "Observed TMI"),
        ("predicted", "Predicted TMI"),
        ("residual", "Residual"),
    )

    for ax, (key, title) in zip(axes, fields):

        values = data[key]

        if key == "residual":
            limit = max(
                float(np.max(np.abs(values))),
                1e-15,
            )

            image = ax.scatter(
                data["x"],
                data["y"],
                c=values,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                s=35,
                edgecolors="black",
                linewidths=0.2,
            )

        else:
            image = ax.scatter(
                data["x"],
                data["y"],
                c=values,
                cmap="coolwarm",
                s=35,
                edgecolors="black",
                linewidths=0.2,
            )

        ax.set_title(title)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_aspect("equal")

        fig.colorbar(
            image,
            ax=ax,
            label="Magnetic anomaly (nT)",
        )

    fig.tight_layout()

    fig.savefig(output_path)

    plt.close(fig)

    print(f"Saved: {output_path}")


# =============================================================================
# INVERSION PARAMETERS AND STATISTICS
# =============================================================================

def plot_inversion_parameters(
    statistics: dict[str, float],
    model_statistics: dict[str, float],
    noise_level: int,
    output_path: Path,
) -> None:
    """Plot inversion parameters and quantitative statistics."""

    print("\nGenerating inversion parameters figure...")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    fig.suptitle(
        f"Positive-Constrained Magnetic Inversion - "
        f"{noise_level}% Noise",
        fontweight="bold",
    )

    # ---------------------------------------------------------
    # Data-fit metrics
    # ---------------------------------------------------------

    ax = axes[0, 0]

    names = [
        "Data RMS",
        "MAE",
        "Maximum error",
    ]

    values = [
        statistics["data_rms"],
        statistics["mean_absolute_error"],
        statistics["maximum_absolute_error"],
    ]

    bars = ax.bar(
        names,
        values,
    )

    ax.set_title("Data-fit statistics")
    ax.set_ylabel("Error (nT)")
    ax.tick_params(axis="x", rotation=15)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3g}",
            ha="center",
            va="bottom",
        )

    # ---------------------------------------------------------
    # Susceptibility statistics
    # ---------------------------------------------------------

    ax = axes[0, 1]

    names = [
        "Minimum",
        "Mean",
        "Maximum",
        "RMS",
    ]

    values = [
        statistics["susceptibility_minimum"],
        statistics["susceptibility_mean"],
        statistics["susceptibility_maximum"],
        statistics["susceptibility_rms"],
    ]

    bars = ax.bar(
        names,
        values,
    )

    ax.set_title("Inverted susceptibility statistics")
    ax.set_ylabel("Susceptibility (SI)")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # ---------------------------------------------------------
    # Model reconstruction
    # ---------------------------------------------------------

    ax = axes[1, 0]

    names = [
        "True model",
        "Inverted model",
    ]

    means = [
        model_statistics["true_mean"],
        model_statistics["inverted_mean"],
    ]

    bars = ax.bar(
        names,
        means,
    )

    ax.set_title("Mean susceptibility comparison")
    ax.set_ylabel("Susceptibility (SI)")

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2e}",
            ha="center",
            va="bottom",
        )

    # ---------------------------------------------------------
    # Inversion configuration
    # ---------------------------------------------------------

    ax = axes[1, 1]

    ax.axis("off")

    information = (
        "INVERSION CONFIGURATION\n"
        "\n"
        f"Noise level: {noise_level}%\n"
        f"Regularization lambda: "
        f"{statistics['regularization_lambda']:.4g}\n"
        "\n"
        "PHYSICAL CONSTRAINT\n"
        "\n"
        "Susceptibility >= 0 SI\n"
        "\n"
        "DATA-FIT QUALITY\n"
        "\n"
        f"Relative L2: {statistics['relative_l2']:.6e}\n"
        f"Model RMSE: {model_statistics['model_rmse']:.6e} SI\n"
        "\n"
        "MODEL CONSTRAINT CHECK\n"
        "\n"
        f"Negative cells: "
        f"{int(statistics['negative_cells'])}\n"
        f"Positive cells: "
        f"{int(statistics['positive_cells'])}\n"
        f"Zero cells: "
        f"{int(statistics['zero_cells'])}"
    )

    ax.text(
        0.05,
        0.95,
        information,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "white",
            "edgecolor": "black",
        },
    )

    fig.tight_layout()

    fig.savefig(output_path)

    plt.close(fig)

    print(f"Saved: {output_path}")


# =============================================================================
# SAVE TEXT REPORT
# =============================================================================

def save_report(
    path: Path,
    noise_level: int,
    statistics: dict[str, float],
    model_statistics: dict[str, float],
) -> None:
    """Save a readable ASCII inversion report."""

    lines = [
        "GEOINVLAB3D - POSITIVE-CONSTRAINED MAGNETIC INVERSION",
        "=" * 65,
        "",
        f"Noise level: {noise_level}%",
        "",
        "INVERSION CONFIGURATION",
        "-" * 65,
        f"Regularization lambda : {statistics['regularization_lambda']:.8e}",
        "Physical constraint   : susceptibility >= 0 SI",
        "",
        "DATA-FIT STATISTICS",
        "-" * 65,
        f"Data RMS              : {statistics['data_rms']:.8e} nT",
        f"Mean absolute error   : {statistics['mean_absolute_error']:.8e} nT",
        f"Maximum absolute error: {statistics['maximum_absolute_error']:.8e} nT",
        f"Relative L2            : {statistics['relative_l2']:.8e}",
        "",
        "INVERTED MODEL STATISTICS",
        "-" * 65,
        f"Minimum susceptibility: {statistics['susceptibility_minimum']:.8e} SI",
        f"Maximum susceptibility: {statistics['susceptibility_maximum']:.8e} SI",
        f"Mean susceptibility   : {statistics['susceptibility_mean']:.8e} SI",
        f"RMS susceptibility    : {statistics['susceptibility_rms']:.8e} SI",
        f"Model RMSE            : {statistics['model_rmse']:.8e} SI",
        "",
        "CONSTRAINT VERIFICATION",
        "-" * 65,
        f"Negative cells: {int(statistics['negative_cells'])}",
        f"Positive cells: {int(statistics['positive_cells'])}",
        f"Zero cells    : {int(statistics['zero_cells'])}",
        "",
        "TRUE VS INVERTED MODEL",
        "-" * 65,
        f"True model minimum    : {model_statistics['true_min']:.8e} SI",
        f"True model maximum    : {model_statistics['true_max']:.8e} SI",
        f"True model mean       : {model_statistics['true_mean']:.8e} SI",
        f"Inverted model minimum: {model_statistics['inverted_min']:.8e} SI",
        f"Inverted model maximum: {model_statistics['inverted_max']:.8e} SI",
        f"Inverted model mean   : {model_statistics['inverted_mean']:.8e} SI",
        f"Model reconstruction RMSE: {model_statistics['model_rmse']:.8e} SI",
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="ascii",
    )

    print(f"Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Visualize positive-constrained magnetic inversion results."
        )
    )

    parser.add_argument(
        "--noise-level",
        type=int,
        choices=NOISE_LEVELS,
        default=3,
        help="Noise level: 1, 3 or 5 percent.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without displaying interactive windows.",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Display the interactive 3D PyVista visualization.",
    )

    args = parser.parse_args()

    noise_level = args.noise_level

    results_dir = (
        RESULTS_ROOT / f"noise_{noise_level}pct"
    )

    figures_dir = results_dir / "visualization"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("GeoInvLab3D - MAGNETIC INVERSION VISUALIZATION")
    print("=" * 76)

    print(f"Noise level       : {noise_level}%")
    print(f"Shared data       : {SHARED_DATA_DIR}")
    print(f"Results directory : {results_dir}")
    print(f"Figures directory : {figures_dir}")

    # -------------------------------------------------------------------------
    # 1. Load mesh
    # -------------------------------------------------------------------------

    print("\n[1/7] Loading mesh...")

    mesh = load_mesh(
        SHARED_DATA_DIR / "mesh.dat"
    )

    print(
        f"Mesh dimensions: "
        f"{mesh['nx']} x {mesh['ny']} x {mesh['nz']}"
    )

    print(f"Number of cells: {mesh['n_cells']}")

    print(
        f"Cell dimensions: "
        f"{mesh['dx']} x {mesh['dy']} x {mesh['dz']} m"
    )

    print(
        f"Depth extent: "
        f"{mesh['ze'][0]} to {mesh['ze'][-1]} m"
    )

    # -------------------------------------------------------------------------
    # 2. Load models
    # -------------------------------------------------------------------------

    print("\n[2/7] Loading true and inverted models...")

    true_model = load_model(
        SHARED_DATA_DIR / "model_true.dat",
        mesh,
        "true_model",
    )

    inverted_model = load_model(
        results_dir / "model_inverted_positive.dat",
        mesh,
        "inverted_model",
    )

    print(
        f"True model range: "
        f"{true_model.min():.6e} to {true_model.max():.6e} SI"
    )

    print(
        f"Inverted model range: "
        f"{inverted_model.min():.6e} to "
        f"{inverted_model.max():.6e} SI"
    )

    # -------------------------------------------------------------------------
    # 3. Load data fit
    # -------------------------------------------------------------------------

    print("\n[3/7] Loading magnetic data fit...")

    data = load_data_fit(
        results_dir / "data_fit_positive.dat"
    )

    print(f"Number of observations: {data['observed'].size}")

    # -------------------------------------------------------------------------
    # 4. Load inversion statistics
    # -------------------------------------------------------------------------

    print("\n[4/7] Loading inversion statistics...")

    statistics = load_statistics(
        results_dir / "inversion_statistics_positive.dat"
    )

    model_statistics = calculate_model_statistics(
        true_model,
        inverted_model,
    )

    print(
        f"Data RMS: "
        f"{statistics['data_rms']:.6f} nT"
    )

    print(
        f"Relative L2: "
        f"{statistics['relative_l2']:.6e}"
    )

    print(
        f"Model RMSE: "
        f"{model_statistics['model_rmse']:.6e} SI"
    )

    # -------------------------------------------------------------------------
    # 5. 3D visualization
    # -------------------------------------------------------------------------

    print("\n[5/7] Generating 3D visualization...")

    plot_model_3d(
        mesh,
        true_model,
        inverted_model,
        figures_dir / "model_3d.png",
        interactive=args.interactive and not args.no_show,
    )

    # -------------------------------------------------------------------------
    # 6. 2D visualizations
    # -------------------------------------------------------------------------

    print("\n[6/7] Generating 2D visualizations...")

    plot_model_sections(
        mesh,
        inverted_model,
        figures_dir / "model_sections_2d.png",
    )

    plot_model_comparison(
        mesh,
        true_model,
        inverted_model,
        figures_dir / "model_comparison_2d.png",
    )

    plot_data_fit(
        data,
        figures_dir / "data_fit_maps.png",
    )

    # -------------------------------------------------------------------------
    # 7. Inversion parameters and report
    # -------------------------------------------------------------------------

    print("\n[7/7] Generating inversion statistics...")

    plot_inversion_parameters(
        statistics,
        model_statistics,
        noise_level,
        figures_dir / "inversion_parameters.png",
    )

    save_report(
        figures_dir / "inversion_report.txt",
        noise_level,
        statistics,
        model_statistics,
    )

    print("\n" + "=" * 76)
    print("MAGNETIC INVERSION VISUALIZATION COMPLETED")
    print("=" * 76)

    print(f"\nOutput directory: {figures_dir}")

    print("\nGenerated files:")

    for filename in (
        "model_3d.png",
        "model_sections_2d.png",
        "model_comparison_2d.png",
        "data_fit_maps.png",
        "inversion_parameters.png",
        "inversion_report.txt",
    ):
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
