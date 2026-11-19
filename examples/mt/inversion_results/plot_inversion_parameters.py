#!/usr/bin/env python3
"""
Plot Gauss-Newton inversion parameters from a GeoInvLab3D history file.

Expected whitespace-separated columns:
iteration data_rms normalized_rms phi_d phi_m phi model_min model_max
elapsed_s alpha relative_update

Lines beginning with '#' are treated as comments. The column names may be
provided in a commented header, as in the GeoInvLab3D history output.

Usage:
    python plot_inversion_parameters.py path/to/inversion_history.dat

Optional:
    python plot_inversion_parameters.py path/to/inversion_history.dat \
        --output figures/inversion_parameters.png

Both PNG and PDF versions are saved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ALIASES = {
    "iteration": ("iteration", "iter", "iteration_number"),
    "data_rms": ("data_rms", "rms", "data_misfit_rms"),
    "normalized_rms": ("normalized_rms", "norm_rms", "normalized_data_rms"),
    "phi_d": ("phi_d", "data_misfit", "data_objective"),
    "phi_m": ("phi_m", "regularization", "regularization_value"),
    "phi": ("phi", "objective", "objective_value", "total_objective"),
    "model_min": ("model_min", "min_model", "model_minimum"),
    "model_max": ("model_max", "max_model", "model_maximum"),
    "elapsed_s": ("elapsed_s", "time_s", "elapsed", "time"),
    "alpha": ("alpha", "step_length", "line_search_alpha"),
    "relative_update": ("relative_update", "relative_model_update", "update"),
}


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def read_history(path: Path) -> dict[str, np.ndarray]:
    """Read a GeoInvLab3D whitespace-delimited inversion history."""
    if not path.is_file():
        raise FileNotFoundError(f"History file not found: {path}")

    header_names = None
    numeric_rows: list[list[float]] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            candidate = line.lstrip("#").strip()
            tokens = candidate.split()
            normalized = [normalize_name(token) for token in tokens]
            known = {alias for values in ALIASES.values() for alias in values}

            if sum(token in known for token in normalized) >= 3:
                header_names = normalized
            continue

        try:
            numeric_rows.append([float(value) for value in line.split()])
        except ValueError:
            continue

    if not numeric_rows:
        raise ValueError(f"No numeric inversion records found in: {path}")

    width = len(numeric_rows[0])

    if any(len(row) != width for row in numeric_rows):
        raise ValueError("Inconsistent number of columns in inversion history.")

    data = np.asarray(numeric_rows, dtype=float)

    if header_names is None:
        default = list(ALIASES)

        if width != len(default):
            raise ValueError(
                "Could not identify the history columns from a commented "
                "header. Add a header beginning with '#', or use the "
                "11-column GeoInvLab3D history format."
            )

        header_names = default

    if len(header_names) != width:
        raise ValueError(
            f"Header has {len(header_names)} columns but data has {width}."
        )

    columns: dict[str, np.ndarray] = {}

    for canonical, aliases in ALIASES.items():
        for index, name in enumerate(header_names):
            if name in aliases:
                columns[canonical] = data[:, index]
                break

    if "iteration" not in columns:
        columns["iteration"] = np.arange(data.shape[0], dtype=float)

    return columns


def get_column(
    data: dict[str, np.ndarray],
    name: str,
) -> np.ndarray | None:
    values = data.get(name)

    if values is None or not np.any(np.isfinite(values)):
        return None

    return values


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """
    Add a publication-style panel label to the upper-left corner.
    """
    ax.text(
        0.025,
        0.96,
        f"({label})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "pad": 2.5,
        },
        zorder=10,
    )


def plot_history(
    data: dict[str, np.ndarray],
    output: Path,
    title: str,
) -> None:

    iteration = data["iteration"]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13, 12),
        constrained_layout=True,
    )

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
    )

    # ---------------------------------------------------------
    # (a) Data misfit
    # ---------------------------------------------------------
    ax = axes[0, 0]
    plotted = False

    for key, label in (
        ("data_rms", "Data RMS"),
        ("normalized_rms", "Normalized RMS"),
    ):
        values = get_column(data, key)

        if values is not None:
            ax.plot(
                iteration,
                values,
                marker="o",
                linewidth=1.8,
                label=label,
            )
            plotted = True

    ax.set_title("Data misfit")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMS")

    if plotted:
        ax.legend()

    ax.grid(True, alpha=0.3)
    add_panel_label(ax, "a")

    # ---------------------------------------------------------
    # (b) Objective function
    # ---------------------------------------------------------
    ax = axes[0, 1]
    plotted = False

    for key, label in (
        ("phi_d", r"$\Phi_d$ (data misfit)"),
        ("phi_m", r"$\Phi_m$ (regularization)"),
        ("phi", r"$\Phi$ (total objective)"),
    ):
        values = get_column(data, key)

        if values is not None:
            ax.plot(
                iteration,
                values,
                marker="o",
                linewidth=1.8,
                label=label,
            )
            plotted = True

    ax.set_title("Objective function")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")

    if plotted:
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.legend()

    ax.grid(True, alpha=0.3)
    add_panel_label(ax, "b")

    # ---------------------------------------------------------
    # (c) Model parameter range
    # ---------------------------------------------------------
    ax = axes[1, 0]
    plotted = False

    for key, label in (
        ("model_min", "Minimum model value"),
        ("model_max", "Maximum model value"),
    ):
        values = get_column(data, key)

        if values is not None:
            ax.plot(
                iteration,
                values,
                marker="o",
                linewidth=1.8,
                label=label,
            )
            plotted = True

    ax.set_title("Model parameter range")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Model value")

    if plotted:
        ax.legend()

    ax.grid(True, alpha=0.3)
    add_panel_label(ax, "c")

    # ---------------------------------------------------------
    # (d) Step length
    # ---------------------------------------------------------
    ax = axes[1, 1]

    values = get_column(data, "alpha")

    if values is not None:
        ax.plot(
            iteration,
            values,
            marker="o",
            linewidth=1.8,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No alpha values in history",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title("Step length")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\alpha$")
    ax.grid(True, alpha=0.3)
    add_panel_label(ax, "d")

    # ---------------------------------------------------------
    # (e) Relative model update
    # ---------------------------------------------------------
    ax = axes[2, 0]

    values = get_column(data, "relative_update")

    if values is not None:
        ax.plot(
            iteration,
            values,
            marker="o",
            linewidth=1.8,
        )
        ax.set_yscale("symlog", linthresh=1e-12)
    else:
        ax.text(
            0.5,
            0.5,
            "No relative-update values in history",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title("Relative model update")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative update")
    ax.grid(True, alpha=0.3)
    add_panel_label(ax, "e")

    # ---------------------------------------------------------
    # (f) Elapsed time
    # ---------------------------------------------------------
    ax = axes[2, 1]

    values = get_column(data, "elapsed_s")

    if values is not None:
        ax.plot(
            iteration,
            values,
            marker="o",
            linewidth=1.8,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No elapsed-time values in history",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title("Elapsed time")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time (s)")
    ax.grid(True, alpha=0.3)
    add_panel_label(ax, "f")

    # ---------------------------------------------------------
    # Final formatting
    # ---------------------------------------------------------
    for ax in axes.flat:
        ax.tick_params(direction="out")

    output.parent.mkdir(parents=True, exist_ok=True)

    # Save both raster (PNG) and vector (PDF) versions.
    # The PDF uses the same base filename as the requested output.
    png_output = output.with_suffix(".png")
    pdf_output = output.with_suffix(".pdf")

    fig.savefig(
        png_output,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_output,
        format="pdf",
        bbox_inches="tight",
    )

    print(f"PNG figure saved: {png_output.resolve()}")
    print(f"PDF figure saved: {pdf_output.resolve()}")

    plt.show()


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Plot GeoInvLab3D Gauss-Newton inversion parameters."
    )

    parser.add_argument(
        "history",
        type=Path,
        help="Path to inversion history file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure base path; both PNG and PDF are saved (default: beside history file)",
    )

    args = parser.parse_args()

    try:
        data = read_history(args.history)

        output = args.output or args.history.with_name(
            "inversion_parameters.png"
        )

        plot_history(
            data,
            output,
            title=f"GeoInvLab3D — Inversion history ({args.history.stem})",
        )

    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
