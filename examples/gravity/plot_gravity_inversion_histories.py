#!/usr/bin/env python3
"""
Compare the convergence histories of three 3-D gravity inversions
performed using noisy synthetic data.

Experiments:
    - 1% Gaussian noise
    - 3% Gaussian noise
    - 5% Gaussian noise

The figure contains six panels:
    (a) Data RMS
    (b) Normalized RMS
    (c) Data misfit: Phi_d
    (d) Model regularization: Phi_m
    (e) Total objective function: Phi
    (f) Relative model update

Input files:
    examples/gravity/inversion_results/noise_1pct/inversion_history.dat
    examples/gravity/inversion_results/noise_3pct/inversion_history.dat
    examples/gravity/inversion_results/noise_5pct/inversion_history.dat

Output files:
    examples/gravity/inversion_results/figures/
        gravity_inversion_histories.png
        gravity_inversion_histories.pdf

Run from the repository root:

    python examples/gravity/plot_gravity_inversion_histories.py

Coordinates:
    x = East
    y = North
    z = positive downward
    
    Boukar Abdelghafour  GeoInvLab3D   MIT License Copyright (c) 2026
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Paths and configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = SCRIPT_DIR / "inversion_results"
FIGURE_DIR = RESULTS_DIR / "figures"

EXPERIMENTS = {
    "1% noise": RESULTS_DIR / "noise_1pct" / "inversion_history.dat",
    "3% noise": RESULTS_DIR / "noise_3pct" / "inversion_history.dat",
    "5% noise": RESULTS_DIR / "noise_5pct" / "inversion_history.dat",
}

EXPECTED_COLUMNS = [
    "iteration",
    "data_rms",
    "normalized_rms",
    "phi_d",
    "phi_m",
    "phi",
    "model_min",
    "model_max",
    "elapsed_s",
    "alpha",
    "relative_update",
]

METRICS = [
    {
        "column": "data_rms",
        "title": "Data RMS",
        "ylabel": "Data RMS",
        "panel": "(a)",
    },
    {
        "column": "normalized_rms",
        "title": "Normalized RMS",
        "ylabel": "Normalized RMS",
        "panel": "(b)",
    },
    {
        "column": "phi_d",
        "title": r"Data misfit ($\Phi_d$)",
        "ylabel": r"$\Phi_d$",
        "panel": "(c)",
    },
    {
        "column": "phi_m",
        "title": r"Model regularization ($\Phi_m$)",
        "ylabel": r"$\Phi_m$",
        "panel": "(d)",
    },
    {
        "column": "phi",
        "title": r"Total objective function ($\Phi$)",
        "ylabel": r"$\Phi$",
        "panel": "(e)",
    },
    {
        "column": "relative_update",
        "title": "Relative model update",
        "ylabel": "Relative model update",
        "panel": "(f)",
    },
]


# =============================================================================
# History loading
# =============================================================================

def load_history(history_file):
    """
    Load an inversion history file.

    The expected format is a whitespace-separated ASCII file with a
    commented header containing the column names.
    """

    if not history_file.is_file():
        raise FileNotFoundError(
            f"Inversion history file not found:\n{history_file}"
        )

    header = None

    with history_file.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("#"):
                candidate = stripped.lstrip("#").strip().split()

                if candidate and candidate[0].lower() == "iteration":
                    header = candidate
                    break

            else:
                break

    data = np.loadtxt(history_file, comments="#", ndmin=2)

    if data.size == 0:
        raise ValueError(f"History file is empty: {history_file}")

    if header is None:
        if data.shape[1] != len(EXPECTED_COLUMNS):
            raise ValueError(
                f"Cannot determine the columns in {history_file}.\n"
                f"Expected {len(EXPECTED_COLUMNS)} columns, "
                f"but found {data.shape[1]}."
            )

        header = EXPECTED_COLUMNS

    if len(header) != data.shape[1]:
        raise ValueError(
            f"Header/data mismatch in {history_file}:\n"
            f"{len(header)} column names and {data.shape[1]} data columns."
        )

    header = [name.lower() for name in header]

    history = {
        name: data[:, index]
        for index, name in enumerate(header)
    }

    # Support an alternative name for the relative model update.
    if (
        "relative_update" not in history
        and "relative_model_update" in history
    ):
        history["relative_update"] = history["relative_model_update"]

    required_columns = [
        "iteration",
        "data_rms",
        "normalized_rms",
        "phi_d",
        "phi_m",
        "phi",
        "relative_update",
    ]

    missing = [
        column
        for column in required_columns
        if column not in history
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {history_file}:\n"
            f"{', '.join(missing)}"
        )

    return history


# =============================================================================
# Plotting
# =============================================================================

def plot_histories():
    """Generate the six-panel comparison of inversion histories."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    histories = {}

    for label, history_file in EXPERIMENTS.items():
        print(f"Loading history: {label}")

        histories[label] = load_history(history_file)

        iterations = histories[label]["iteration"]

        print(
            f"  Iterations: {len(iterations)} "
            f"(from {iterations[0]:.0f} to {iterations[-1]:.0f})"
        )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(13, 13),
        sharex=True,
    )

    axes = axes.ravel()

    for ax, metric in zip(axes, METRICS):

        column = metric["column"]

        for label, history in histories.items():

            iterations = history["iteration"]
            values = history[column]

            # Logarithmic axes cannot display zero or negative values.
            # Such values are masked without modifying the original data.
            plot_values = np.where(values > 0.0, values, np.nan)

            ax.plot(
                iterations,
                plot_values,
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=label,
            )

        ax.set_yscale("log")

        ax.set_title(
            metric["title"],
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        ax.set_ylabel(
            metric["ylabel"],
            fontsize=11,
        )

        ax.text(
            0.025,
            0.94,
            metric["panel"],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )

        ax.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.6,
            alpha=0.7,
        )

        ax.grid(
            True,
            which="minor",
            linestyle=":",
            linewidth=0.4,
            alpha=0.4,
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=9,
        )

    # X-axis labels.
    for ax in axes[-2:]:
        ax.set_xlabel(
            "Iteration",
            fontsize=11,
        )

    # Common legend.
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=True,
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        bottom=0.07,
        top=0.93,
        hspace=0.32,
        wspace=0.25,
    )

    # Save figures.
    png_file = FIGURE_DIR / "gravity_inversion_histories.png"
    pdf_file = FIGURE_DIR / "gravity_inversion_histories.pdf"

    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight",
    )

    print("\nFigures saved:")
    print(f"  PNG: {png_file}")
    print(f"  PDF: {pdf_file}")

    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    plot_histories()
