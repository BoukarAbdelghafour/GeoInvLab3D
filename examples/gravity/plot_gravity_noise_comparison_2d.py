#!/usr/bin/env python3
"""
Compare the true gravity density-contrast model with three inverted models
obtained using 1%, 3%, and 5% noisy gravity data.

Figure organization
-------------------
Row 1: Horizontal X-Y sections at the shallowest depth where both positive
       and negative density contrasts are present in the true model.

Row 2: Vertical X-Z sections at Y = 1250 m.

Row 3: Vertical Y-Z sections at X = 700 m.

Visualization rules
-------------------
- Supports four-column and seven-column model files.
- Automatically converts absolute density values into density contrasts.
- Searches for a horizontal layer containing both dense and light bodies.
- Applies threshold filtering ONLY to the true model.
- Retains all cells in the inverted models.
- Preserves the original dimensions of all models.
- Supports models with different mesh coordinates.
- Uses a common color scale centered on zero density contrast.
- Displays depth positive downward.
- Displays numerical coordinates on all axes.

Supported model formats
-----------------------
Four columns:
    x_center_m y_center_m z_center_m density_contrast_kg_m3

Seven columns:
    ix iy iz x_center_m y_center_m z_center_m density_kg_m3

Output
------
figures/gravity_noise_comparison_2d.png
figures/gravity_noise_comparison_2d.pdf

GeoInvLab3D
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

TRUE_MODEL_CANDIDATES = (
    SCRIPT_DIR / "data" / "model_true.dat",
    SCRIPT_DIR / "synthetic" / "data" / "model_true.dat",
)

RESULT_DIRECTORY_CANDIDATES = (
    SCRIPT_DIR / "noisy" / "inversion_results",
    SCRIPT_DIR / "inversion_results",
    SCRIPT_DIR / "noisy",
)

FIGURE_DIR = SCRIPT_DIR / "figures"

OUTPUT_PNG = FIGURE_DIR / "gravity_noise_comparison_2d.png"
OUTPUT_PDF = FIGURE_DIR / "gravity_noise_comparison_2d.pdf"


# ============================================================================
# 2. VISUALIZATION SETTINGS
# ============================================================================

# Reference density contrast of the homogeneous background.
BACKGROUND_DENSITY = 0.0  # kg/m3

# Convert absolute density values in the true model into density contrasts.
CONVERT_TRUE_MODEL_TO_CONTRAST = True

# Threshold applied ONLY to the true model.
TRUE_ABS_THRESHOLD = 1.0  # kg/m3

# Requested vertical section positions.
TARGET_Y_SECTION = 1250.0  # m
TARGET_X_SECTION = 700.0   # m

# Common robust color scale.
COLOR_PERCENTILE = 92.0
MIN_COLOR_HALF_RANGE = 1.0  # kg/m3

# Diverging color palette:
# blue  = negative density contrast (light body)
# white = background
# red   = positive density contrast (dense body)
LOW_COLOR = "#2166AC"
BACKGROUND_COLOR = "#F7F7F7"
HIGH_COLOR = "#B2182B"

# Plot appearance.
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

def find_true_model() -> Path:
    """Locate the true gravity model."""

    for candidate in TRUE_MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate

    searched = "\n".join(
        f"  - {path}" for path in TRUE_MODEL_CANDIDATES
    )

    raise FileNotFoundError(
        "Cannot find the true gravity model.\n"
        "Files searched:\n" + searched
    )


def find_inverted_model(noise_level: str) -> Path:
    """Locate the inverted model for a given noise level."""

    folder_names = (
        f"noise_{noise_level}",
        noise_level,
    )

    filenames = (
        "model_inverted.dat",
        "inverted_model.dat",
        "model_inverted_positive.dat",
    )

    searched = []

    for root in RESULT_DIRECTORY_CANDIDATES:
        for folder in folder_names:
            directory = root / folder

            for filename in filenames:
                candidate = directory / filename
                searched.append(candidate)

                if candidate.is_file():
                    return candidate

            alternative = directory / "inversion_results"

            for filename in filenames:
                candidate = alternative / filename
                searched.append(candidate)

                if candidate.is_file():
                    return candidate

    raise FileNotFoundError(
        f"Cannot find the inverted model for {noise_level} noise.\n"
        "Files searched:\n  - "
        + "\n  - ".join(str(path) for path in searched)
    )


# ============================================================================
# 4. MODEL LOADING
# ============================================================================

def load_model(filename: Path):
    """
    Load a gravity model from a four-column or seven-column ASCII file.

    Return:
        x, y, z:
            Sorted cell-center coordinates.

        values:
            Model array with shape (nx, ny, nz).
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

    if data.shape[1] == 4:
        model_data = data[:, :4]

    elif data.shape[1] >= 7:
        model_data = data[:, 3:7]

    else:
        raise ValueError(
            f"Unsupported model format in {filename}.\n"
            f"Found {data.shape[1]} columns; expected 4 or at least 7."
        )

    if not np.all(np.isfinite(model_data)):
        raise ValueError(
            f"Non-finite values detected in {filename}."
        )

    x = np.unique(model_data[:, 0])
    y = np.unique(model_data[:, 1])
    z = np.unique(model_data[:, 2])

    nx, ny, nz = len(x), len(y), len(z)
    expected = nx * ny * nz

    if model_data.shape[0] != expected:
        raise ValueError(
            f"Incomplete or duplicated grid in {filename}: "
            f"{model_data.shape[0]} rows instead of {expected} "
            f"({nx} x {ny} x {nz})."
        )

    values = np.empty((nx, ny, nz), dtype=float)
    occupied = np.zeros((nx, ny, nz), dtype=bool)

    ix = np.searchsorted(x, model_data[:, 0])
    iy = np.searchsorted(y, model_data[:, 1])
    iz = np.searchsorted(z, model_data[:, 2])

    for row, i, j, k in zip(model_data, ix, iy, iz):
        if occupied[i, j, k]:
            raise ValueError(
                f"Duplicate cell in {filename}: "
                f"({row[0]}, {row[1]}, {row[2]})."
            )

        values[i, j, k] = row[3]
        occupied[i, j, k] = True

    if not np.all(occupied):
        raise ValueError(
            f"Missing cells detected in {filename}."
        )

    return x, y, z, values


# ============================================================================
# 5. TRUE MODEL DENSITY CONTRAST
# ============================================================================

def estimate_background_density(values: np.ndarray) -> float:
    """Estimate the homogeneous background using the most frequent value."""

    rounded = np.round(
        np.asarray(values, dtype=float),
        decimals=8,
    )

    unique_values, counts = np.unique(
        rounded,
        return_counts=True,
    )

    return float(unique_values[np.argmax(counts)])


def convert_true_model_to_contrast(values: np.ndarray):
    """Convert absolute density values into density contrasts."""

    background = estimate_background_density(values)

    contrast = (
        values
        - background
        + BACKGROUND_DENSITY
    )

    return contrast, background


# ============================================================================
# 6. MESH GEOMETRY
# ============================================================================

def centers_to_edges(centers, axis_name):
    """Calculate cell boundaries from regularly spaced cell centers."""

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

    edges = np.empty(centers.size + 1, dtype=float)

    edges[1:-1] = 0.5 * (
        centers[:-1] + centers[1:]
    )

    edges[0] = centers[0] - 0.5 * spacing
    edges[-1] = centers[-1] + 0.5 * spacing

    return edges


def nearest_index(coordinates, target):
    """Find the cell center nearest to a requested coordinate."""

    return int(
        np.argmin(
            np.abs(coordinates - target)
        )
    )


# ============================================================================
# 7. DETECT THE HORIZONTAL LAYER
# ============================================================================

def find_horizontal_depth(z, true_values):
    """
    Find the shallowest layer containing both positive and negative
    density contrasts.

    Positive contrast:
        Dense body.

    Negative contrast:
        Light body.

    If no layer contains both signs, select the layer with the largest
    number of cells belonging to the less represented anomaly type.
    """

    print("\nSearching for a layer containing both anomaly types...")

    best_index = None
    best_min_count = -1
    best_positive_count = 0
    best_negative_count = 0

    for iz in range(len(z)):
        layer = true_values[:, :, iz]

        positive_mask = (
            layer - BACKGROUND_DENSITY
            > TRUE_ABS_THRESHOLD
        )

        negative_mask = (
            layer - BACKGROUND_DENSITY
            < -TRUE_ABS_THRESHOLD
        )

        positive_count = int(np.count_nonzero(positive_mask))
        negative_count = int(np.count_nonzero(negative_mask))

        if positive_count > 0 and negative_count > 0:
            print(
                f"  Both anomaly types detected at "
                f"Z = {z[iz]:.2f} m."
            )
            print(
                f"  Dense-body cells: {positive_count}"
            )
            print(
                f"  Light-body cells: {negative_count}"
            )

            return (
                iz,
                float(z[iz]),
                positive_count,
                negative_count,
            )

        # Keep the layer with the largest minority anomaly count.
        minority_count = min(
            positive_count,
            negative_count,
        )

        if minority_count > best_min_count:
            best_min_count = minority_count
            best_index = iz
            best_positive_count = positive_count
            best_negative_count = negative_count

    if best_index is not None and best_min_count > 0:
        return (
            best_index,
            float(z[best_index]),
            best_positive_count,
            best_negative_count,
        )

    # If the bodies occupy separate depth ranges, no single horizontal
    # layer can display both. Use the shallowest anomalous layer.
    for iz in range(len(z)):
        layer = true_values[:, :, iz]

        anomaly_count = int(
            np.count_nonzero(
                np.abs(layer - BACKGROUND_DENSITY)
                > TRUE_ABS_THRESHOLD
            )
        )

        if anomaly_count > 0:
            print(
                "\nWARNING: The dense and light bodies do not occur "
                "in the same horizontal layer."
            )
            print(
                "The shallowest anomalous layer will be displayed."
            )

            positive_count = int(
                np.count_nonzero(
                    layer - BACKGROUND_DENSITY
                    > TRUE_ABS_THRESHOLD
                )
            )

            negative_count = int(
                np.count_nonzero(
                    layer - BACKGROUND_DENSITY
                    < -TRUE_ABS_THRESHOLD
                )
            )

            return (
                iz,
                float(z[iz]),
                positive_count,
                negative_count,
            )

    raise ValueError(
        "No anomalous cells were detected in the true model. "
        "Check TRUE_ABS_THRESHOLD and the model background."
    )


# ============================================================================
# 8. COLOR SCALE
# ============================================================================

def determine_color_limits(inverted_models):
    """Determine symmetric robust color limits from inverted models."""

    deviations = np.concatenate(
        [
            np.abs(
                values.ravel() - BACKGROUND_DENSITY
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

    return (
        BACKGROUND_DENSITY - half_range,
        BACKGROUND_DENSITY + half_range,
    )


def make_colormap(vmin, vmax):
    """Create a blue-white-red colormap centered on zero."""

    background_position = (
        BACKGROUND_DENSITY - vmin
    ) / (vmax - vmin)

    background_position = float(
        np.clip(
            background_position,
            0.001,
            0.999,
        )
    )

    cmap = LinearSegmentedColormap.from_list(
        "gravity_density_contrast",
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
# 9. TRUE MODEL FILTERING
# ============================================================================

def prepare_display_values(values, model_type):
    """
    Filter only the true model.

    Both positive and negative anomalies are retained.
    Inverted models are displayed without filtering.
    """

    if model_type == "true":
        background = BACKGROUND_DENSITY

        mask = (
            np.abs(values - background)
            <= TRUE_ABS_THRESHOLD
        )

        return np.ma.masked_where(
            mask,
            values,
        )

    return np.ma.asarray(values)


# ============================================================================
# 10. SECTION EXTRACTION
# ============================================================================

def extract_sections(
    x,
    y,
    z,
    values,
    reference_depth,
):
    """
    Extract horizontal and vertical sections.

    The target depth and vertical-section positions are defined using
    the true model. Each model independently selects its nearest
    available cell center.
    """

    iz = nearest_index(z, reference_depth)
    iy = nearest_index(y, TARGET_Y_SECTION)
    ix = nearest_index(x, TARGET_X_SECTION)

    return {
        "horizontal": values[:, :, iz],
        "vertical_xz": values[:, iy, :],
        "vertical_yz": values[ix, :, :],
        "actual_z": float(z[iz]),
        "actual_y": float(y[iy]),
        "actual_x": float(x[ix]),
    }


# ============================================================================
# 11. AXIS FORMATTING
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
# 12. HORIZONTAL X-Y PLOT
# ============================================================================

def plot_horizontal(
    ax,
    xedges,
    yedges,
    values,
    cmap,
    clim,
):
    """Plot a horizontal X-Y density-contrast map."""

    image = ax.pcolormesh(
        xedges,
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
# 13. VERTICAL X-Z PLOT
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
# 14. VERTICAL Y-Z PLOT
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
# 15. ROW TITLES
# ============================================================================

def add_row_titles(fig, axes, actual_y, actual_x, actual_z):
    """Add descriptive titles above each section row."""

    row_titles = [
        (
            "Horizontal sections (X-Y) "
            f"| Reference Z = {actual_z:.2f} m"
        ),
        (
            "Vertical sections (X-Z) "
            f"| Reference Y = {actual_y:.2f} m"
        ),
        (
            "Vertical sections (Y-Z) "
            f"| Reference X = {actual_x:.2f} m"
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
# 16. MAIN
# ============================================================================

def main():
    """Generate the 2D gravity noise comparison figure."""

    print("=" * 78)
    print("GeoInvLab3D - 2D GRAVITY NOISE COMPARISON")
    print("=" * 78)

    # ------------------------------------------------------------------------
    # Load model files.
    # ------------------------------------------------------------------------

    print("\n[1/7] Loading models")

    model_files = {
        "True model": find_true_model(),
        "1% noise": find_inverted_model("1pct"),
        "3% noise": find_inverted_model("3pct"),
        "5% noise": find_inverted_model("5pct"),
    }

    loaded = {}

    for label, filename in model_files.items():
        x, y, z, values = load_model(filename)

        if (
            label == "True model"
            and CONVERT_TRUE_MODEL_TO_CONTRAST
        ):
            original_min = float(values.min())
            original_max = float(values.max())

            values, estimated_background = (
                convert_true_model_to_contrast(values)
            )

            print(
                f"\n  Estimated true-model background: "
                f"{estimated_background:.6g} kg/m3"
            )

            print(
                f"  Original density range: "
                f"{original_min:.6g} to "
                f"{original_max:.6g} kg/m3"
            )

            print(
                f"  Converted contrast range: "
                f"{values.min():.6g} to "
                f"{values.max():.6g} kg/m3"
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
            f"  X range: {x.min():.2f} to {x.max():.2f} m"
        )
        print(
            f"  Y range: {y.min():.2f} to {y.max():.2f} m"
        )
        print(
            f"  Z range: {z.min():.2f} to {z.max():.2f} m"
        )
        print(
            f"  Density contrast range: "
            f"{values.min():.6g} to "
            f"{values.max():.6g} kg/m3"
        )

    # ------------------------------------------------------------------------
    # Detect the horizontal reference depth.
    # ------------------------------------------------------------------------

    print("\n[2/7] Detecting the horizontal reference depth")

    true_model = loaded["True model"]

    (
        horizontal_depth_index,
        reference_depth,
        dense_count,
        light_count,
    ) = find_horizontal_depth(
        true_model["z"],
        true_model["values"],
    )

    print(
        f"\n  Selected reference depth: "
        f"Z = {reference_depth:.2f} m"
    )
    print(f"  Dense-body cells: {dense_count}")
    print(f"  Light-body cells: {light_count}")

    # ------------------------------------------------------------------------
    # Determine the common color scale.
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

    clim = determine_color_limits(inverted_models)
    cmap = make_colormap(*clim)

    print(
        f"  Color limits: "
        f"{clim[0]:.6g} to {clim[1]:.6g} kg/m3"
    )

    # ------------------------------------------------------------------------
    # Prepare sections and individual mesh geometries.
    # ------------------------------------------------------------------------

    print("\n[4/7] Preparing sections")

    sections = {}
    geometries = {}

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
            model["x"],
            model["y"],
            model["z"],
            display_values,
            reference_depth,
        )

        geometries[label] = {
            "xedges": centers_to_edges(model["x"], "X"),
            "yedges": centers_to_edges(model["y"], "Y"),
            "zedges": centers_to_edges(model["z"], "Z"),
        }

        section = sections[label]

        print(f"\n  {label}:")
        print(
            f"    Horizontal depth: Z = {section['actual_z']:.2f} m"
        )
        print(
            f"    X-Z section: Y = {section['actual_y']:.2f} m"
        )
        print(
            f"    Y-Z section: X = {section['actual_x']:.2f} m"
        )

    # Use the true model's actual section coordinates as reference labels.
    actual_y = sections["True model"]["actual_y"]
    actual_x = sections["True model"]["actual_x"]

    # ------------------------------------------------------------------------
    # Create the figure.
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
    # Plot all four models.
    # ------------------------------------------------------------------------

    for column, (key, _) in enumerate(labels):
        model_sections = sections[key]
        geometry = geometries[key]

        # Row 1: horizontal X-Y section.
        plot_horizontal(
            axes[0, column],
            geometry["xedges"],
            geometry["yedges"],
            model_sections["horizontal"],
            cmap,
            clim,
        )

        # Row 2: vertical X-Z section.
        plot_vertical_xz(
            axes[1, column],
            geometry["xedges"],
            geometry["zedges"],
            model_sections["vertical_xz"],
            cmap,
            clim,
        )

        # Row 3: vertical Y-Z section.
        plot_vertical_yz(
            axes[2, column],
            geometry["yedges"],
            geometry["zedges"],
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
        reference_depth,
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
        "Density contrast (kg/m³)",
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
        "Comparison of True and Inverted Gravity Density-Contrast Models",
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
    print("2D GRAVITY COMPARISON GENERATED SUCCESSFULLY")
    print("=" * 78)

    print("\nHORIZONTAL SECTION")
    print("-" * 50)
    print(f"Reference depth: Z = {reference_depth:.2f} m")
    print(f"Dense-body cells: {dense_count}")
    print(f"Light-body cells: {light_count}")

    print("\nVERTICAL SECTIONS")
    print("-" * 50)
    print(f"X-Z reference section: Y = {actual_y:.2f} m")
    print(f"Y-Z reference section: X = {actual_x:.2f} m")

    print("\nVISUALIZATION RULES")
    print("-" * 50)
    print("True model: positive and negative anomalies retained.")
    print("True model: background cells filtered.")
    print("Inverted models: all cells retained.")
    print("True model width: original dimensions preserved.")
    print("Inverted model widths: original dimensions preserved.")
    print("Depth: positive downward.")
    print("Color scale: common and centered on zero.")

    print("\nOUTPUT FILES")
    print("-" * 50)
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")

    print("\n" + "=" * 78)

    return 0


# ============================================================================
# 17. ENTRY POINT
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

