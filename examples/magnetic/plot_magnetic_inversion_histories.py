#!/usr/bin/env python3
"""
Compare magnetic inversion histories obtained using different
noise levels (1%, 3%, and 5%).

The figure contains six panels:

(a) Data RMS
(b) Normalized RMS
(c) Objective function components: Phi_d, Phi_m, and Phi
(d) Minimum and maximum magnetic susceptibility
(e) Cumulative elapsed time
(f) Relative objective-function reduction

Supported history formats
-------------------------
Standard format (9 columns):
iteration data_rms normalized_rms phi_d phi_m phi
model_min model_max elapsed_s

Extended format (11 columns):
iteration data_rms normalized_rms phi_d phi_m phi
model_min model_max elapsed_s alpha relative_update

Legacy format (7 columns):
iteration objective data_rms model_rms lambda
step_norm gradient_norm

Input files are whitespace-delimited ASCII files.
Comment lines beginning with '#' are supported.

GeoInvLab3D
Boukar Abdelghafour  GeoInvLab3D   MIT License Copyright (c) 2026
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# 1. PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = (
    SCRIPT_DIR / "noisy" / "inversion_results"
)

FIGURE_DIR = SCRIPT_DIR / "figures"

OUTPUT_PNG = (
    FIGURE_DIR / "magnetic_inversion_histories_comparison.png"
)

OUTPUT_PDF = (
    FIGURE_DIR / "magnetic_inversion_histories_comparison.pdf"
)


# ============================================================================
# 2. CONFIGURATION
# ============================================================================

NOISE_LEVELS = ("1", "3", "5")

NOISE_LABELS = {
    "1": "1% noise",
    "3": "3% noise",
    "5": "5% noise",
}

# Candidate scalar history filenames.
HISTORY_FILENAMES = (
    "inversion_history_positive.dat",
    "inversion_history.dat",
)

# Figure settings.
FIGURE_SIZE = (17, 12)
FIGURE_DPI = 300

FONT_SIZE = 10
TITLE_SIZE = 12
LABEL_SIZE = 10

LINE_WIDTH = 2.0
MARKER_SIZE = 4

GRID_ALPHA = 0.35

# Distinct colors for the three noise levels.
COLORS = {
    "1": "#2166AC",
    "3": "#D6604D",
    "5": "#1B7837",
}

# Objective-function line styles.
OBJECTIVE_STYLES = {
    "phi_d": "-",
    "phi_m": "--",
    "phi": "-.",
}


# ============================================================================
# 3. HISTORY FILE DISCOVERY
# ============================================================================

def find_history_file(noise_level: str) -> Path:
    """
    Find the scalar inversion history for a noise level.

    The per-cell model history file is deliberately excluded.
    """

    directory = (
        RESULTS_DIR / f"noise_{noise_level}pct"
    )

    for filename in HISTORY_FILENAMES:

        candidate = directory / filename

        if candidate.is_file():
            return candidate

    searched = [
        str(directory / filename)
        for filename in HISTORY_FILENAMES
    ]

    raise FileNotFoundError(
        f"No scalar inversion history found for "
        f"{noise_level}% noise.\n"
        "Files searched:\n  - "
        + "\n  - ".join(searched)
    )


# ============================================================================
# 4. HISTORY LOADING
# ============================================================================

def load_history(filename: Path) -> dict:
    """
    Load an inversion history.

    Supported formats:
        7 columns: legacy format
        9 columns: standard format
        11 columns: extended format
    """

    if not filename.is_file():
        raise FileNotFoundError(filename)

    data = np.loadtxt(
        filename,
        comments="#",
        dtype=float,
        ndmin=2,
    )

    if data.size == 0:
        raise ValueError(
            f"Empty inversion history: {filename}"
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            f"Non-finite values detected in {filename}."
        )

    ncolumns = data.shape[1]

    history = {
        "filename": filename,
        "format": None,
    }

    # ------------------------------------------------------------------------
    # Legacy format: 7 columns.
    # ------------------------------------------------------------------------

    if ncolumns == 7:

        history["format"] = "legacy"

        history["iteration"] = data[:, 0]
        history["objective"] = data[:, 1]
        history["data_rms"] = data[:, 2]
        history["model_rms"] = data[:, 3]
        history["lambda"] = data[:, 4]
        history["step_norm"] = data[:, 5]
        history["gradient_norm"] = data[:, 6]

    # ------------------------------------------------------------------------
    # Standard format: 9 columns.
    # Extended format: 11 columns.
    # ------------------------------------------------------------------------

    elif ncolumns in (9, 11):

        history["format"] = (
            "extended" if ncolumns == 11 else "standard"
        )

        history["iteration"] = data[:, 0]
        history["data_rms"] = data[:, 1]
        history["normalized_rms"] = data[:, 2]
        history["phi_d"] = data[:, 3]
        history["phi_m"] = data[:, 4]
        history["phi"] = data[:, 5]
        history["model_min"] = data[:, 6]
        history["model_max"] = data[:, 7]
        history["elapsed_s"] = data[:, 8]

        if ncolumns == 11:
            history["alpha"] = data[:, 9]
            history["relative_update"] = data[:, 10]

    else:

        raise ValueError(
            f"Unsupported history format in {filename}.\n"
            f"Found {ncolumns} columns; expected 7, 9, or 11."
        )

    # Ensure iterations are sorted.
    order = np.argsort(history["iteration"])

    for key, value in list(history.items()):

        if isinstance(value, np.ndarray):
            history[key] = value[order]

    return history


# ============================================================================
# 5. LOAD ALL HISTORIES
# ============================================================================

def load_all_histories() -> dict:
    """Load the three inversion histories."""

    histories = {}

    print("\nLoading inversion histories")
    print("-" * 70)

    for noise in NOISE_LEVELS:

        filename = find_history_file(noise)

        history = load_history(filename)

        histories[noise] = history

        print(f"\n{NOISE_LABELS[noise]}")
        print(f"  File: {filename}")
        print(f"  Format: {history['format']}")
        print(
            f"  Iterations: "
            f"{len(history['iteration'])}"
        )

        print(
            f"  First iteration: "
            f"{history['iteration'][0]:.0f}"
        )

        print(
            f"  Last iteration: "
            f"{history['iteration'][-1]:.0f}"
        )

    formats = {
        history["format"]
        for history in histories.values()
    }

    if len(formats) != 1:
        raise ValueError(
            "The three history files do not use the same format.\n"
            f"Detected formats: {sorted(formats)}"
        )

    return histories


# ============================================================================
# 6. AXIS FORMATTING
# ============================================================================

def format_axis(
    ax,
    title: str,
    xlabel: str = "Iteration",
    ylabel: str = "",
    logarithmic: bool = False,
):
    """Apply consistent formatting to a subplot."""

    ax.set_title(
        title,
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=10,
    )

    ax.set_xlabel(
        xlabel,
        fontsize=LABEL_SIZE,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_SIZE,
    )

    if logarithmic:
        ax.set_yscale("log")

    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE - 1,
        direction="out",
        length=4,
        width=0.8,
    )

    ax.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.7,
        alpha=GRID_ALPHA,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# ============================================================================
# 7. SAFE LOGARITHMIC SCALE
# ============================================================================

def can_use_log_scale(histories, key: str) -> bool:
    """Check whether all plotted values are strictly positive."""

    values = []

    for history in histories.values():

        if key not in history:
            return False

        values.extend(
            history[key][
                np.isfinite(history[key])
            ]
        )

    values = np.asarray(values)

    return (
        values.size > 0
        and np.all(values > 0.0)
    )


# ============================================================================
# 8. PLOT STANDARD HISTORIES
# ============================================================================

def plot_standard_histories(
    axes,
    histories: dict,
):
    """Plot standard and extended inversion histories."""

    # ------------------------------------------------------------------------
    # (a) Data RMS.
    # ------------------------------------------------------------------------

    ax = axes[0, 0]

    for noise, history in histories.items():

        ax.plot(
            history["iteration"],
            history["data_rms"],
            color=COLORS[noise],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label=NOISE_LABELS[noise],
        )

    format_axis(
        ax,
        "(a) Data RMS",
        ylabel="Data RMS",
        logarithmic=can_use_log_scale(
            histories,
            "data_rms",
        ),
    )

    ax.legend(
        fontsize=9,
        frameon=True,
        loc="best",
    )

    # ------------------------------------------------------------------------
    # (b) Normalized RMS.
    # ------------------------------------------------------------------------

    ax = axes[0, 1]

    for noise, history in histories.items():

        ax.plot(
            history["iteration"],
            history["normalized_rms"],
            color=COLORS[noise],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label=NOISE_LABELS[noise],
        )

    format_axis(
        ax,
        "(b) Normalized RMS",
        ylabel="Normalized RMS",
        logarithmic=can_use_log_scale(
            histories,
            "normalized_rms",
        ),
    )

    ax.legend(
        fontsize=9,
        frameon=True,
        loc="best",
    )

    # ------------------------------------------------------------------------
    # (c) Objective function components.
    # ------------------------------------------------------------------------

    ax = axes[1, 0]

    for noise, history in histories.items():

        for key, component_label in (
            ("phi_d", r"$\Phi_d$"),
            ("phi_m", r"$\Phi_m$"),
            ("phi", r"$\Phi$"),
        ):

            ax.plot(
                history["iteration"],
                history[key],
                color=COLORS[noise],
                linestyle=OBJECTIVE_STYLES[key],
                linewidth=LINE_WIDTH,
                label=f"{NOISE_LABELS[noise]} - {component_label}",
            )

    format_axis(
        ax,
        r"(c) Objective function: $\Phi_d$, $\Phi_m$, $\Phi$",
        ylabel="Objective function",
        logarithmic=False,
    )

    ax.legend(
        fontsize=7.5,
        frameon=True,
        loc="best",
        ncol=2,
    )

    # ------------------------------------------------------------------------
    # (d) Model minimum and maximum.
    # ------------------------------------------------------------------------

    ax = axes[1, 1]

    for noise, history in histories.items():

        ax.plot(
            history["iteration"],
            history["model_min"],
            color=COLORS[noise],
            linestyle="--",
            linewidth=LINE_WIDTH,
            label=f"{NOISE_LABELS[noise]} - minimum",
        )

        ax.plot(
            history["iteration"],
            history["model_max"],
            color=COLORS[noise],
            linestyle="-",
            linewidth=LINE_WIDTH,
            label=f"{NOISE_LABELS[noise]} - maximum",
        )

    format_axis(
        ax,
        "(d) Susceptibility range",
        ylabel="Susceptibility (SI)",
    )

    ax.legend(
        fontsize=8,
        frameon=True,
        loc="best",
        ncol=2,
    )

    # ------------------------------------------------------------------------
    # (e) Elapsed time.
    # ------------------------------------------------------------------------

    ax = axes[2, 0]

    for noise, history in histories.items():

        ax.plot(
            history["iteration"],
            history["elapsed_s"],
            color=COLORS[noise],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label=NOISE_LABELS[noise],
        )

    format_axis(
        ax,
        "(e) Cumulative elapsed time",
        ylabel="Elapsed time (s)",
    )

    ax.legend(
        fontsize=9,
        frameon=True,
        loc="best",
    )

    # ------------------------------------------------------------------------
    # (f) Relative objective-function reduction.
    # ------------------------------------------------------------------------

    ax = axes[2, 1]

    for noise, history in histories.items():

        phi = history["phi"]

        initial_phi = abs(float(phi[0]))

        if initial_phi <= 0.0:
            continue

        reduction = (
            100.0
            * (phi[0] - phi)
            / initial_phi
        )

        ax.plot(
            history["iteration"],
            reduction,
            color=COLORS[noise],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label=NOISE_LABELS[noise],
        )

    format_axis(
        ax,
        "(f) Relative objective reduction",
        ylabel="Reduction (%)",
    )

    ax.legend(
        fontsize=9,
        frameon=True,
        loc="best",
    )


# ============================================================================
# 9. PLOT LEGACY HISTORIES
# ============================================================================

def plot_legacy_histories(
    axes,
    histories: dict,
):
    """Plot the older seven-column history format."""

    panels = [
        (
            "objective",
            "(a) Objective function",
            "Objective",
        ),
        (
            "data_rms",
            "(b) Data RMS",
            "Data RMS",
        ),
        (
            "model_rms",
            "(c) Model RMS",
            "Model RMS",
        ),
        (
            "lambda",
            "(d) Regularization parameter",
            "Lambda",
        ),
        (
            "step_norm",
            "(e) Step norm",
            "Step norm",
        ),
        (
            "gradient_norm",
            "(f) Gradient norm",
            "Gradient norm",
        ),
    ]

    for index, (key, title, ylabel) in enumerate(panels):

        ax = axes.flat[index]

        for noise, history in histories.items():

            ax.plot(
                history["iteration"],
                history[key],
                color=COLORS[noise],
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=MARKER_SIZE,
                label=NOISE_LABELS[noise],
            )

        format_axis(
            ax,
            title,
            ylabel=ylabel,
            logarithmic=can_use_log_scale(
                histories,
                key,
            ),
        )

        ax.legend(
            fontsize=9,
            frameon=True,
            loc="best",
        )


# ============================================================================
# 10. FINAL STATISTICS
# ============================================================================

def print_final_statistics(histories: dict):
    """Print the final inversion metrics for each noise level."""

    print("\nFINAL INVERSION STATISTICS")
    print("=" * 90)

    for noise, history in histories.items():

        print(f"\n{NOISE_LABELS[noise]}")

        if history["format"] in ("standard", "extended"):

            print(
                f"  Data RMS          : "
                f"{history['data_rms'][-1]:.6g}"
            )

            print(
                f"  Normalized RMS    : "
                f"{history['normalized_rms'][-1]:.6g}"
            )

            print(
                f"  Phi_d             : "
                f"{history['phi_d'][-1]:.6g}"
            )

            print(
                f"  Phi_m             : "
                f"{history['phi_m'][-1]:.6g}"
            )

            print(
                f"  Phi               : "
                f"{history['phi'][-1]:.6g}"
            )

            print(
                f"  Model minimum     : "
                f"{history['model_min'][-1]:.6g} SI"
            )

            print(
                f"  Model maximum     : "
                f"{history['model_max'][-1]:.6g} SI"
            )

            print(
                f"  Elapsed time      : "
                f"{history['elapsed_s'][-1]:.3f} s"
            )

            if "alpha" in history:

                print(
                    f"  Final alpha       : "
                    f"{history['alpha'][-1]:.6g}"
                )

                print(
                    f"  Relative update   : "
                    f"{history['relative_update'][-1]:.6g}"
                )

        else:

            for key in (
                "objective",
                "data_rms",
                "model_rms",
                "lambda",
                "step_norm",
                "gradient_norm",
            ):

                print(
                    f"  {key:18s}: "
                    f"{history[key][-1]:.6g}"
                )

        print(
            f"  Number of iterations: "
            f"{len(history['iteration'])}"
        )

    print("\n" + "=" * 90)


# ============================================================================
# 11. MAIN
# ============================================================================

def main() -> int:
    """Generate the magnetic inversion history comparison."""

    print("=" * 90)
    print("GeoInvLab3D - COMPARISON OF MAGNETIC INVERSION HISTORIES")
    print("=" * 90)

    # Load histories.
    histories = load_all_histories()

    history_format = histories["1"]["format"]

    # Create output directory.
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create figure.
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=FIGURE_SIZE,
    )

    fig.patch.set_facecolor("white")

    # Plot according to the detected format.
    if history_format in ("standard", "extended"):

        plot_standard_histories(
            axes,
            histories,
        )

    elif history_format == "legacy":

        plot_legacy_histories(
            axes,
            histories,
        )

    # Main title.
    fig.suptitle(
        "Comparison of Magnetic Inversion Histories "
        "for Different Noise Levels",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    fig.tight_layout(
        rect=[0.02, 0.02, 0.98, 0.96],
        h_pad=2.5,
        w_pad=2.0,
    )

    # Save PNG.
    fig.savefig(
        OUTPUT_PNG,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    # Save PDF.
    fig.savefig(
        OUTPUT_PDF,
        format="pdf",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    # Print final metrics.
    print_final_statistics(histories)

    print("\nFIGURES GENERATED SUCCESSFULLY")
    print("-" * 70)
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")
    print(f"Resolution: {FIGURE_DPI} dpi")

    print("\n" + "=" * 90)

    return 0


# ============================================================================
# 12. ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:
        raise SystemExit(main())

    except Exception as error:

        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )

        raise
