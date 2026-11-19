#!/usr/bin/env python3
"""
Compare the true magnetic susceptibility model with three inverted models
obtained using 1%, 3%, and 5% noisy magnetic data.

Figure organization
-------------------
Row 1: Horizontal X-Y sections at the shallowest anomalous depth
       detected in the true model.
Row 2: Vertical X-Z sections at Y = 1250 m.
Row 3: Vertical Y-Z sections at X = 700 m.

Visualization rules
-------------------
- Automatically detect the shallowest anomalous depth in the true model.
- Use this exact depth for all four horizontal sections.
- Apply threshold filtering ONLY to the true model.
- Retain all cells in the inverted models.
- Reduce the displayed horizontal width of the true model.
- Preserve the original dimensions of inverted models.
- Use a common color scale.
- Display depth positive downward.
- Display numerical coordinates on all axes.

Boukar Abdelghafour  GeoInvLab3D   MIT License Copyright (c) 2026
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# ============================================================================
# 1. PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

TRUE_MODEL_FILE = (
    SCRIPT_DIR / "synthetic" / "data" / "model_true.dat"
)

NOISY_RESULTS_DIR = (
    SCRIPT_DIR / "noisy" / "inversion_results"
)

LEGACY_RESULTS_DIR = (
    SCRIPT_DIR / "inversion_results"
)

FIGURE_DIR = SCRIPT_DIR / "figures"

OUTPUT_PNG = (
    FIGURE_DIR / "magnetic_noise_comparison_2d.png"
)

OUTPUT_PDF = (
    FIGURE_DIR / "magnetic_noise_comparison_2d.pdf"
)


# ============================================================================
# 2. VISUALIZATION SETTINGS
# ============================================================================

BACKGROUND_SUSCEPTIBILITY = 0.010

# Thresholds used to identify anomalies in the true model.
TRUE_LOW_THRESHOLD = 0.005
TRUE_HIGH_THRESHOLD = 0.020

# Width reduction for the true horizontal section only.
TRUE_MODEL_WIDTH_SCALE = 0.75

# Vertical section positions.
TARGET_Y_SECTION = 1250.0
TARGET_X_SECTION = 700.0

# Color scale.
COLOR_PERCENTILE = 92.0
MIN_COLOR_HALF_RANGE = 0.002

LOW_COLOR = "#2166AC"
BACKGROUND_COLOR = "#F7F7F7"
HIGH_COLOR = "#B2182B"

GRID_COLOR = "#D9D9D9"
AXIS_COLOR = "#333333"

FIGURE_SIZE = (20, 14)
FIGURE_DPI = 300

FONT_SIZE = 10
TITLE_SIZE = 12
LABEL_SIZE = 10


# ============================================================================
# 3. INPUT FILES
# ============================================================================

def find_inverted_model(noise_level: str) -> Path:
    """Locate the inverted model for a given noise level."""

    directories = (
        NOISY_RESULTS_DIR / f"noise_{noise_level}",
        LEGACY_RESULTS_DIR / f"noise_{noise_level}",
    )

    filenames = (
        "model_inverted_positive.dat",
        "model_inverted.dat",
        "inverted_model.dat",
    )

    for directory in directories:
        for filename in filenames:

            candidate = directory / filename

            if candidate.is_file():
                return candidate

    searched = [
        str(directory / filename)
        for directory in directories
        for filename in filenames
    ]

    raise FileNotFoundError(
        f"Cannot find inverted model for {noise_level} noise.\n"
        "Files searched:\n  - "
        + "\n  - ".join(searched)
    )


# ============================================================================
# 4. MODEL LOADING
# ============================================================================

def load_model(filename: Path):
    """
    Load a magnetic model with four columns:

    x_center_m y_center_m z_center_m susceptibility_SI

    Return the coordinates and a model array with shape (nx, ny, nz).
    """

    if not filename.is_file():
        raise FileNotFoundError(
            f"Model file not found: {filename}"
        )

    data = np.loadtxt(
        filename,
        comments="#",
        dtype=float,
        ndmin=2,
    )

    if data.shape[1] < 4:
        raise ValueError(
            f"{filename} must contain at least four columns."
        )

    data = data[:, :4]

    if not np.all(np.isfinite(data)):
        raise ValueError(
            f"Non-finite values detected in {filename}."
        )

    x = np.unique(data[:, 0])
    y = np.unique(data[:, 1])
    z = np.unique(data[:, 2])

    nx, ny, nz = len(x), len(y), len(z)

    expected = nx * ny * nz

    if data.shape[0] != expected:
        raise ValueError(
            f"Incomplete or duplicated grid in {filename}: "
            f"{data.shape[0]} rows instead of {expected}."
        )

    values = np.empty(
        (nx, ny, nz),
        dtype=float,
    )

    occupied = np.zeros(
        (nx, ny, nz),
        dtype=bool,
    )

    ix = np.searchsorted(x, data[:, 0])
    iy = np.searchsorted(y, data[:, 1])
    iz = np.searchsorted(z, data[:, 2])

    for row, i, j, k in zip(data, ix, iy, iz):

        if occupied[i, j, k]:
            raise ValueError(
                f"Duplicate cell in {filename}."
            )

        values[i, j, k] = row[3]
        occupied[i, j, k] = True

    if not np.all(occupied):
        raise ValueError(
            f"Missing cells detected in {filename}."
        )

    return x, y, z, values


# ============================================================================
# 5. MESH VALIDATION
# ============================================================================

def check_same_mesh(reference, candidate, label):
    """Check that all models use identical cell-center coordinates."""

    for axis_name, ref, current in zip(
        ("X", "Y", "Z"),
        reference,
        candidate,
    ):

        if (
            ref.shape != current.shape
            or not np.allclose(
                ref,
                current,
                rtol=0.0,
                atol=1.0e-8,
            )
        ):

            raise ValueError(
                f"The {label} model does not use the same "
                f"{axis_name} coordinates as the true model."
            )


def centers_to_edges(centers, axis_name):
    """Calculate cell boundaries from regularly spaced centers."""

    centers = np.asarray(centers, dtype=float)

    if centers.size < 2:
        raise ValueError(
            f"At least two centers are required along {axis_name}."
        )

    differences = np.diff(centers)
    spacing = float(np.mean(differences))

    if not np.allclose(
        differences,
        spacing,
        rtol=1.0e-6,
        atol=1.0e-8,
    ):
        raise ValueError(
            f"Non-uniform cell spacing along {axis_name}."
        )

    edges = np.empty(
        centers.size + 1,
        dtype=float,
    )

    edges[1:-1] = (
        0.5 * (centers[:-1] + centers[1:])
    )

    edges[0] = centers[0] - 0.5 * spacing
    edges[-1] = centers[-1] + 0.5 * spacing

    return edges


# ============================================================================
# 6. AUTOMATIC DETECTION OF THE SHALLOWEST ANOMALOUS DEPTH
# ============================================================================

def find_shallowest_anomalous_depth(
    z: np.ndarray,
    true_values: np.ndarray,
):
    """
    Find the shallowest depth containing a detectable contrast
    in the true magnetic model.

    A cell is considered anomalous if:

        susceptibility < TRUE_LOW_THRESHOLD

    or:

        susceptibility > TRUE_HIGH_THRESHOLD

    The search starts at the shallowest layer and proceeds
    toward increasing depth.

    Returns
    -------
    iz : int
        Index of the first anomalous depth layer.

    actual_depth : float
        Depth of the selected cell-center layer.

    number_of_cells : int
        Number of anomalous cells in that layer.
    """

    print("\nSearching for the shallowest anomalous layer...")

    for iz in range(len(z)):

        layer = true_values[:, :, iz]

        anomaly_mask = (
            (layer < TRUE_LOW_THRESHOLD)
            | (layer > TRUE_HIGH_THRESHOLD)
        )

        number_of_cells = int(
            np.count_nonzero(anomaly_mask)
        )

        if number_of_cells > 0:

            actual_depth = float(z[iz])

            print(
                f"  First anomalous layer detected at "
                f"Z = {actual_depth:.2f} m"
            )

            print(
                f"  Number of anomalous cells: "
                f"{number_of_cells}"
            )

            print(
                f"  Susceptibility thresholds: "
                f"< {TRUE_LOW_THRESHOLD:.4f} or "
                f"> {TRUE_HIGH_THRESHOLD:.4f} SI"
            )

            return iz, actual_depth, number_of_cells

    raise ValueError(
        "No anomalous layer was detected in the true model.\n"
        "Check TRUE_LOW_THRESHOLD and TRUE_HIGH_THRESHOLD."
    )


# ============================================================================
# 7. COLOR SCALE
# ============================================================================

def determine_color_limits(inverted_models):
    """Determine robust common color limits from inverted models."""

    deviations = np.concatenate(
        [
            np.abs(
                values.ravel()
                - BACKGROUND_SUSCEPTIBILITY
            )
            for values in inverted_models.values()
        ]
    )

    half_range = float(
        np.percentile(
            deviations,
            COLOR_PERCENTILE,
        )
    )

    half_range = max(
        half_range,
        MIN_COLOR_HALF_RANGE,
    )

    vmin = max(
        0.0,
        BACKGROUND_SUSCEPTIBILITY - half_range,
    )

    vmax = (
        BACKGROUND_SUSCEPTIBILITY + half_range
    )

    if vmax <= vmin:
        vmax = vmin + MIN_COLOR_HALF_RANGE

    return vmin, vmax


def make_colormap(vmin, vmax):
    """Create a blue-white-red colormap centered on the background."""

    background_position = (
        BACKGROUND_SUSCEPTIBILITY - vmin
    ) / (vmax - vmin)

    background_position = float(
        np.clip(
            background_position,
            0.001,
            0.999,
        )
    )

    cmap = LinearSegmentedColormap.from_list(
        "magnetic_susceptibility",
        [
            (0.0, LOW_COLOR),
            (background_position, BACKGROUND_COLOR),
            (1.0, HIGH_COLOR),
        ],
        N=256,
    )

    cmap.set_bad("white")

    return cmap


# ============================================================================
# 8. TRUE MODEL FILTERING
# ============================================================================

def prepare_display_values(values, model_type):
    """
    Prepare display values.

    The true model is threshold-filtered.
    Inverted models are displayed without filtering.
    """

    if model_type == "true":

        mask = (
            (values >= TRUE_LOW_THRESHOLD)
            & (values <= TRUE_HIGH_THRESHOLD)
        )

        return np.ma.masked_where(
            mask,
            values,
        )

    return np.ma.asarray(values)


# ============================================================================
# 9. SECTION EXTRACTION
# ============================================================================

def nearest_index(coordinates, target):
    """Find the cell center nearest to a requested coordinate."""

    return int(
        np.argmin(
            np.abs(coordinates - target)
        )
    )


def extract_sections(
    x,
    y,
    z,
    values,
    horizontal_depth_index,
):
    """
    Extract the three sections.

    IMPORTANT:
    horizontal_depth_index is determined from the true model
    and is reused for every inverted model.
    """

    iy = nearest_index(
        y,
        TARGET_Y_SECTION,
    )

    ix = nearest_index(
        x,
        TARGET_X_SECTION,
    )

    return {
        "horizontal": values[
            :,
            :,
            horizontal_depth_index
        ],

        "vertical_xz": values[
            :,
            iy,
            :
        ],

        "vertical_yz": values[
            ix,
            :,
            :
        ],

        "indices": {
            "iz": horizontal_depth_index,
            "iy": iy,
            "ix": ix,
        },
    }


# ============================================================================
# 10. AXIS FORMATTING
# ============================================================================

def format_axis(ax, xlabel, ylabel):
    """Apply consistent formatting to a 2D panel."""

    ax.set_xlabel(
        xlabel,
        fontsize=LABEL_SIZE,
        fontweight="medium",
        labelpad=5,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_SIZE,
        fontweight="medium",
        labelpad=5,
    )

    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE - 1,
        direction="out",
        length=4,
        width=0.8,
        pad=3,
    )

    ax.grid(
        True,
        color=GRID_COLOR,
        linewidth=0.45,
        alpha=0.5,
    )

    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)


# ============================================================================
# 11. HORIZONTAL X-Y PLOT
# ============================================================================

def plot_horizontal(
    ax,
    xedges,
    yedges,
    values,
    cmap,
    clim,
    model_type,
):
    """Plot a horizontal X-Y susceptibility map."""

    if model_type == "true":

        x_center = 0.5 * (
            xedges[0] + xedges[-1]
        )

        plot_xedges = (
            x_center
            + TRUE_MODEL_WIDTH_SCALE
            * (xedges - x_center)
        )

    else:

        plot_xedges = xedges

    image = ax.pcolormesh(
        plot_xedges,
        yedges,
        values.T,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        shading="flat",
        rasterized=True,
    )

    ax.set_xlim(
        xedges[0],
        xedges[-1],
    )

    ax.set_ylim(
        yedges[0],
        yedges[-1],
    )

    ax.set_aspect("equal")

    format_axis(
        ax,
        "Easting X (m)",
        "Northing Y (m)",
    )

    return image


# ============================================================================
# 12. VERTICAL X-Z PLOT
# ============================================================================

def plot_vertical_xz(
    ax,
    xedges,
    zedges,
    values,
    cmap,
    clim,
):
    """Plot a vertical X-Z section at a fixed Y coordinate."""

    image = ax.pcolormesh(
        xedges,
        zedges,
        values.T,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        shading="flat",
        rasterized=True,
    )

    ax.set_xlim(
        xedges[0],
        xedges[-1],
    )

    ax.set_ylim(
        zedges[-1],
        zedges[0],
    )

    format_axis(
        ax,
        "Easting X (m)",
        "Depth Z (m)",
    )

    return image


# ============================================================================
# 13. VERTICAL Y-Z PLOT
# ============================================================================

def plot_vertical_yz(
    ax,
    yedges,
    zedges,
    values,
    cmap,
    clim,
):
    """Plot a vertical Y-Z section at a fixed X coordinate."""

    image = ax.pcolormesh(
        yedges,
        zedges,
        values.T,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        shading="flat",
        rasterized=True,
    )

    ax.set_xlim(
        yedges[0],
        yedges[-1],
    )

    ax.set_ylim(
        zedges[-1],
        zedges[0],
    )

    format_axis(
        ax,
        "Northing Y (m)",
        "Depth Z (m)",
    )

    return image


# ============================================================================
# 14. ROW TITLES
# ============================================================================

def add_row_titles(
    fig,
    axes,
    actual_y,
    actual_x,
    actual_z,
):
    """Add row titles separately from axis labels."""

    row_titles = [
        (
            "Horizontal sections (X-Y) "
            f"| Z = {actual_z:.2f} m"
        ),
        (
            "Vertical sections (X-Z) "
            f"| Y = {actual_y:.2f} m"
        ),
        (
            "Vertical sections (Y-Z) "
            f"| X = {actual_x:.2f} m"
        ),
    ]

    for row, title in enumerate(row_titles):

        position = axes[row, 0].get_position()

        fig.text(
            0.5,
            position.y1 + 0.018,
            title,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )


# ============================================================================
# 15. MAIN
# ============================================================================

def main():
    """Generate the 2D magnetic noise comparison figure."""

    print("=" * 78)
    print("GeoInvLab3D - 2D MAGNETIC NOISE COMPARISON")
    print("=" * 78)

    # ------------------------------------------------------------------------
    # Load model files.
    # ------------------------------------------------------------------------

    print("\n[1/7] Loading models")

    model_files = {
        "True model": TRUE_MODEL_FILE,
        "1% noise": find_inverted_model("1pct"),
        "3% noise": find_inverted_model("3pct"),
        "5% noise": find_inverted_model("5pct"),
    }

    loaded = {}
    reference_coordinates = None

    for label, filename in model_files.items():

        x, y, z, values = load_model(filename)

        coordinates = (x, y, z)

        if reference_coordinates is None:

            reference_coordinates = coordinates

        else:

            check_same_mesh(
                reference_coordinates,
                coordinates,
                label,
            )

        loaded[label] = {
            "x": x,
            "y": y,
            "z": z,
            "values": values,
            "filename": filename,
        }

        print(f"\n{label}")
        print(f"  File: {filename}")
        print(
            f"  Mesh: {len(x)} x {len(y)} x {len(z)}"
        )
        print(
            f"  Susceptibility range: "
            f"{values.min():.6g} to "
            f"{values.max():.6g} SI"
        )

    # ------------------------------------------------------------------------
    # Detect the shallowest anomalous depth in the true model.
    # ------------------------------------------------------------------------

    print("\n[2/7] Detecting the shallowest anomalous depth")

    x, y, z = reference_coordinates

    true_values = loaded["True model"]["values"]

    horizontal_depth_index, actual_z, anomaly_count = (
        find_shallowest_anomalous_depth(
            z,
            true_values,
        )
    )

    # ------------------------------------------------------------------------
    # Common color scale.
    # ------------------------------------------------------------------------

    print("\n[3/7] Computing common color scale")

    inverted_models = {
        label: loaded[label]["values"]
        for label in (
            "1% noise",
            "3% noise",
            "5% noise",
        )
    }

    clim = determine_color_limits(
        inverted_models
    )

    cmap = make_colormap(*clim)

    print(
        f"  Color limits: {clim[0]:.6g} to "
        f"{clim[1]:.6g} SI"
    )

    # ------------------------------------------------------------------------
    # Prepare mesh geometry.
    # ------------------------------------------------------------------------

    print("\n[4/7] Preparing sections")

    xedges = centers_to_edges(x, "X")
    yedges = centers_to_edges(y, "Y")
    zedges = centers_to_edges(z, "Z")

    sections = {}

    for label, model in loaded.items():

        model_type = (
            "true"
            if label == "True model"
            else "inverted"
        )

        display_values = prepare_display_values(
            model["values"],
            model_type,
        )

        sections[label] = extract_sections(
            x,
            y,
            z,
            display_values,
            horizontal_depth_index,
        )

    indices = sections["True model"]["indices"]

    actual_y = float(
        y[indices["iy"]]
    )

    actual_x = float(
        x[indices["ix"]]
    )

    print(
        f"\n  Common horizontal depth: Z = {actual_z:.2f} m"
    )

    print(
        f"  Vertical X-Z section: Y = {actual_y:.2f} m"
    )

    print(
        f"  Vertical Y-Z section: X = {actual_x:.2f} m"
    )

    # ------------------------------------------------------------------------
    # Create figure.
    # ------------------------------------------------------------------------

    print("\n[5/7] Creating figure")

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = [
        ("True model", "(a) True model"),
        ("1% noise", "(b) Inversion - 1% noise"),
        ("3% noise", "(c) Inversion - 3% noise"),
        ("5% noise", "(d) Inversion - 5% noise"),
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=FIGURE_SIZE,
        constrained_layout=False,
    )

    fig.patch.set_facecolor("white")

    for column, (_, title) in enumerate(labels):

        axes[0, column].set_title(
            title,
            fontsize=TITLE_SIZE,
            fontweight="bold",
            pad=12,
        )

    # ------------------------------------------------------------------------
    # Plot all models.
    # ------------------------------------------------------------------------

    for column, (key, _) in enumerate(labels):

        model_sections = sections[key]

        model_type = (
            "true"
            if key == "True model"
            else "inverted"
        )

        # Row 1: horizontal X-Y section.
        plot_horizontal(
            axes[0, column],
            xedges,
            yedges,
            model_sections["horizontal"],
            cmap,
            clim,
            model_type,
        )

        # Row 2: vertical X-Z section.
        plot_vertical_xz(
            axes[1, column],
            xedges,
            zedges,
            model_sections["vertical_xz"],
            cmap,
            clim,
        )

        # Row 3: vertical Y-Z section.
        plot_vertical_yz(
            axes[2, column],
            yedges,
            zedges,
            model_sections["vertical_yz"],
            cmap,
            clim,
        )

    # ------------------------------------------------------------------------
    # Figure layout.
    # ------------------------------------------------------------------------

    fig.subplots_adjust(
        left=0.105,
        right=0.89,
        bottom=0.085,
        top=0.91,
        wspace=0.30,
        hspace=0.58,
    )

    add_row_titles(
        fig,
        axes,
        actual_y,
        actual_x,
        actual_z,
    )

    # ------------------------------------------------------------------------
    # Shared colorbar.
    # ------------------------------------------------------------------------

    colorbar_axis = fig.add_axes(
        [0.925, 0.19, 0.018, 0.62]
    )

    scalar_map = plt.cm.ScalarMappable(
        norm=plt.Normalize(
            vmin=clim[0],
            vmax=clim[1],
        ),
        cmap=cmap,
    )

    scalar_map.set_array([])

    colorbar = fig.colorbar(
        scalar_map,
        cax=colorbar_axis,
    )

    colorbar.set_label(
        "Magnetic susceptibility (SI)",
        fontsize=11,
        fontweight="medium",
        labelpad=12,
    )

    colorbar.ax.tick_params(
        labelsize=9,
        direction="out",
    )

    colorbar.outline.set_linewidth(0.8)

    fig.suptitle(
        "Comparison of True and Inverted Magnetic Susceptibility Models",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )

    # ------------------------------------------------------------------------
    # Export figures.
    # ------------------------------------------------------------------------

    print("\n[6/7] Saving figures")

    fig.savefig(
        OUTPUT_PNG,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        OUTPUT_PDF,
        format="pdf",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    # ------------------------------------------------------------------------
    # Final report.
    # ------------------------------------------------------------------------

    print("\n[7/7] Completed")

    print("\n" + "=" * 78)
    print("2D MAGNETIC COMPARISON GENERATED SUCCESSFULLY")
    print("=" * 78)

    print("\nHORIZONTAL SECTION")
    print("-" * 50)

    print(
        f"Automatically detected depth: {actual_z:.2f} m"
    )

    print(
        f"Anomalous cells in true model: {anomaly_count}"
    )

    print(
        "The same depth is used for all four models."
    )

    print("\nVERTICAL SECTIONS")
    print("-" * 50)

    print(f"X-Z section: Y = {actual_y:.2f} m")
    print(f"Y-Z section: X = {actual_x:.2f} m")

    print("\nOUTPUT FILES")
    print("-" * 50)

    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")

    print("\n" + "=" * 78)

    return 0


# ============================================================================
# 16. ENTRY POINT
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

