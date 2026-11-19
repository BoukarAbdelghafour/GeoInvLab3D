#!/usr/bin/env python3
"""
Compare the true magnetic susceptibility model with positive-constrained
inversion models obtained using 1%, 3%, and 5% noisy data.

Visualization rules
-------------------
- The true model is threshold-filtered to display its two anomalies.
- ALL cells of the three inverted models are displayed; no inversion cells
  are removed or modified.
- A robust color range is estimated from the inverted models to prevent
  a few extreme values from hiding the recovered anomalies.
- The common colormap is centered on the homogeneous background
  susceptibility (0.01 SI).
- All four panels use the same geometry and camera orientation.

Input files
-----------
synthetic/data/model_true.dat
noisy/inversion_results/noise_{1,3,5}pct/model_inverted_positive.dat

Output
------
figures/magnetic_noise_comparison.png
figures/magnetic_noise_comparison.pdf

Boukar Abdelghafour  GeoInvLab3D   MIT License Copyright (c) 2026
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap


# ============================================================================
# 1. PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

TRUE_MODEL_FILE = SCRIPT_DIR / "synthetic" / "data" / "model_true.dat"

NOISY_RESULTS_DIR = SCRIPT_DIR / "noisy" / "inversion_results"
LEGACY_RESULTS_DIR = SCRIPT_DIR / "inversion_results"

FIGURE_DIR = SCRIPT_DIR / "figures"
OUTPUT_PNG = FIGURE_DIR / "magnetic_noise_comparison.png"
OUTPUT_PDF = FIGURE_DIR / "magnetic_noise_comparison.pdf"


# ============================================================================
# 2. VISUALIZATION SETTINGS
# ============================================================================

BACKGROUND_SUSCEPTIBILITY = 0.010  # SI

# Apply these thresholds ONLY to the true model.
TRUE_LOW_THRESHOLD = 0.005  # SI
TRUE_HIGH_THRESHOLD = 0.020  # SI

# Robust color scaling for the inverted models.
# A percentile below 100 reduces the influence of isolated extreme values.
COLOR_PERCENTILE = 92.0

# Minimum half-range around the background, in SI.
MIN_COLOR_HALF_RANGE = 0.002

# Camera and rendering.
CAMERA_ZOOM = 1.20
WINDOW_SIZE = (2200, 1600)
BACKGROUND_COLOR = "white"
SCALAR_NAME = "Susceptibility (SI)"

# Blue = lower susceptibility; white = background; red = higher susceptibility.
LOW_COLOR = "#2166AC"
BACKGROUND_COLOR_MAP = "#F7F7F7"
HIGH_COLOR = "#B2182B"


# ============================================================================
# 3. INPUT FILES
# ============================================================================

def find_inverted_model(noise_level: str) -> Path:
    """Locate the inverted model file for a given noise level."""
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
        f"Cannot find the inverted model for {noise_level} noise.\n"
        "Files searched:\n  - " + "\n  - ".join(searched)
    )


def load_model(
    filename: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a model stored as four ASCII columns:
    x_center_m, y_center_m, z_center_m, susceptibility_SI.
    """
    if not filename.is_file():
        raise FileNotFoundError(f"Model file not found: {filename}")

    data = np.loadtxt(filename, comments="#", dtype=float, ndmin=2)

    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(
            f"{filename} must contain at least four columns: "
            "x_center_m y_center_m z_center_m susceptibility_SI."
        )

    data = data[:, :4]

    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite values found in {filename}.")

    x = np.unique(data[:, 0])
    y = np.unique(data[:, 1])
    z = np.unique(data[:, 2])

    shape = (x.size, y.size, z.size)
    expected = int(np.prod(shape))

    if data.shape[0] != expected:
        raise ValueError(
            f"Incomplete or duplicated grid in {filename}: "
            f"{data.shape[0]} rows found; expected {expected} "
            f"({x.size} x {y.size} x {z.size})."
        )

    values = np.empty(shape, dtype=float)
    occupied = np.zeros(shape, dtype=bool)

    ix = np.searchsorted(x, data[:, 0])
    iy = np.searchsorted(y, data[:, 1])
    iz = np.searchsorted(z, data[:, 2])

    for row, i, j, k in zip(data, ix, iy, iz):
        if occupied[i, j, k]:
            raise ValueError(
                f"Duplicate cell center in {filename}: "
                f"({row[0]}, {row[1]}, {row[2]})."
            )
        values[i, j, k] = row[3]
        occupied[i, j, k] = True

    if not np.all(occupied):
        raise ValueError(f"Missing cells detected in {filename}.")

    return x, y, z, values


def check_same_mesh(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray],
    label: str,
) -> None:
    """Ensure that all models use exactly the same cell-center coordinates."""
    for axis_name, ref, current in zip(("X", "Y", "Z"), reference, candidate):
        if ref.shape != current.shape or not np.allclose(
            ref, current, rtol=0.0, atol=1.0e-8
        ):
            raise ValueError(
                f"The {label} model does not use the same {axis_name} "
                "coordinates as the true model."
            )


# ============================================================================
# 4. GRID CONSTRUCTION
# ============================================================================

def centers_to_edges(centers: np.ndarray, axis_name: str) -> np.ndarray:
    """Convert regularly spaced cell centers to cell-edge coordinates."""
    centers = np.asarray(centers, dtype=float)

    if centers.size < 2:
        raise ValueError(
            f"At least two cell centers are required along {axis_name}."
        )

    differences = np.diff(centers)
    spacing = float(np.mean(differences))

    if not np.allclose(differences, spacing, rtol=1.0e-6, atol=1.0e-8):
        raise ValueError(f"Non-uniform cell spacing along {axis_name}.")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * spacing
    edges[-1] = centers[-1] + 0.5 * spacing
    return edges


def build_structured_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
) -> pv.StructuredGrid:
    """Build a structured grid with depth displayed positive downward."""
    xe = centers_to_edges(x, "X")
    ye = centers_to_edges(y, "Y")
    ze = centers_to_edges(z, "Z")

    # Reverse the vertical display coordinate so depth increases downward.
    display_ze = -ze[::-1]

    xx, yy, zz = np.meshgrid(
        xe, ye, display_ze, indexing="ij"
    )
    grid = pv.StructuredGrid(xx, yy, zz)

    # Reverse cell values consistently with the reversed vertical axis.
    grid.cell_data[SCALAR_NAME] = values[:, :, ::-1].ravel(order="F")
    return grid


# ============================================================================
# 5. COLOR SCALE
# ============================================================================

def make_background_centered_colormap(
    vmin: float,
    vmax: float,
) -> LinearSegmentedColormap:
    """
    Create a blue-white-red colormap whose white point is exactly the
    background susceptibility, even when the color limits are asymmetric.
    """
    span = vmax - vmin
    if span <= 0.0:
        raise ValueError("Invalid color limits.")

    background_position = (
        BACKGROUND_SUSCEPTIBILITY - vmin
    ) / span
    background_position = float(np.clip(background_position, 0.001, 0.999))

    return LinearSegmentedColormap.from_list(
        "magnetic_background_centered",
        [
            (0.0, LOW_COLOR),
            (background_position, BACKGROUND_COLOR_MAP),
            (1.0, HIGH_COLOR),
        ],
        N=256,
    )


def determine_color_limits(
    inverted_models: dict[str, np.ndarray],
) -> tuple[float, float]:
    """
    Determine robust common limits from all inverted models.

    The percentile is applied to the absolute deviation from the
    homogeneous background. No model values are changed; values outside
    the display range are simply shown with the endpoint colors.
    """
    deviations = np.concatenate(
        [
            np.abs(values.ravel() - BACKGROUND_SUSCEPTIBILITY)
            for values in inverted_models.values()
        ]
    )

    half_range = float(np.percentile(deviations, COLOR_PERCENTILE))
    half_range = max(half_range, MIN_COLOR_HALF_RANGE)

    # Susceptibility is non-negative, so do not use a negative lower limit.
    vmin = max(0.0, BACKGROUND_SUSCEPTIBILITY - half_range)
    vmax = BACKGROUND_SUSCEPTIBILITY + half_range

    if vmax <= vmin:
        vmax = vmin + MIN_COLOR_HALF_RANGE

    return vmin, vmax


# ============================================================================
# 6. ANOMALY EXTRACTION
# ============================================================================

def extract_display_grid(
    grid: pv.StructuredGrid,
    model_type: str,
) -> pv.UnstructuredGrid:
    """
    Extract cells for display.

    The true model is threshold-filtered to isolate its anomalies.
    Inverted models are returned with ALL cells intact.
    """
    if model_type == "true":
        values = np.asarray(grid.cell_data[SCALAR_NAME])
        mask = (
            (values < TRUE_LOW_THRESHOLD)
            | (values > TRUE_HIGH_THRESHOLD)
        )

        if not np.any(mask):
            raise ValueError(
                "No anomalous cells remain in the true model. "
                "Check TRUE_LOW_THRESHOLD and TRUE_HIGH_THRESHOLD."
            )

        return grid.extract_cells(mask)

    # Important: no threshold or masking is applied to inverted models.
    return grid


# ============================================================================
# 7. PLOTTING
# ============================================================================

def add_model_panel(
    plotter: pv.Plotter,
    position: tuple[int, int],
    title: str,
    model_file: Path,
    model_type: str,
    clim: tuple[float, float],
    cmap: LinearSegmentedColormap,
    camera_position,
) -> None:
    """Add one model to the requested subplot."""
    x, y, z, values = load_model(model_file)

    full_grid = build_structured_grid(x, y, z, values)
    display_grid = extract_display_grid(full_grid, model_type)

    plotter.subplot(*position)
    plotter.set_background(BACKGROUND_COLOR)

    plotter.add_mesh(
        display_grid,
        scalars=SCALAR_NAME,
        cmap=cmap,
        clim=clim,
        show_edges=False,
        show_scalar_bar=(position == (1, 1)),
        scalar_bar_args={
            "title": "Magnetic susceptibility (SI)",
            "vertical": True,
            "title_font_size": 13,
            "label_font_size": 11,
            "n_labels": 6,
            "color": "black",
            "fmt": "%.3f",
            "position_x": 0.84,
            "position_y": 0.12,
            "height": 0.72,
            "width": 0.10,
        } if position == (1, 1) else None,
        lighting=True,
        smooth_shading=False,
        ambient=0.25,
        diffuse=0.75,
        specular=0.05,
    )

    # Show only the external computational-domain outline.
    plotter.add_mesh(
        full_grid.outline(),
        color="black",
        line_width=1.0,
        opacity=0.70,
    )

    plotter.add_text(
        title,
        position="upper_edge",
        font_size=17,
        color="black",
        font="arial",
    )

    plotter.show_axes()
    plotter.camera_position = camera_position
    plotter.camera.zoom(CAMERA_ZOOM)


# ============================================================================
# 8. MAIN
# ============================================================================

def main() -> int:
    """Generate the four-panel magnetic noise comparison."""
    print("=" * 78)
    print("GeoInvLab3D - MAGNETIC NOISE COMPARISON")
    print("=" * 78)

    if not TRUE_MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"True magnetic model not found: {TRUE_MODEL_FILE}"
        )

    model_files = {
        "True model": TRUE_MODEL_FILE,
        "1% noise": find_inverted_model("1pct"),
        "3% noise": find_inverted_model("3pct"),
        "5% noise": find_inverted_model("5pct"),
    }

    loaded = {}
    reference_coordinates = None

    print("\n[1/4] Loading and validating models")

    for label, filename in model_files.items():
        x, y, z, values = load_model(filename)

        coordinates = (x, y, z)
        if reference_coordinates is None:
            reference_coordinates = coordinates
        else:
            check_same_mesh(reference_coordinates, coordinates, label)

        loaded[label] = (x, y, z, values, filename)

        print(
            f"{label:12s}: {filename}\n"
            f"             mesh = {x.size} x {y.size} x {z.size}\n"
            f"             susceptibility = "
            f"{values.min():.6g} to {values.max():.6g} SI"
        )

    inverted_models = {
        label: loaded[label][3]
        for label in ("1% noise", "3% noise", "5% noise")
    }

    print("\n[2/4] Computing robust common color limits")
    clim = determine_color_limits(inverted_models)
    cmap = make_background_centered_colormap(*clim)

    print(f"  Color limits : {clim[0]:.6g} to {clim[1]:.6g} SI")
    print(f"  Background   : {BACKGROUND_SUSCEPTIBILITY:.6g} SI")
    print(f"  Percentile   : {COLOR_PERCENTILE:.1f}")
    print("  Inverted models: all cells retained")
    print("  Color bar     : linked to the actual susceptibility mapper")

    # Build one common camera from the full computational domain.
    x, y, z = reference_coordinates
    xe = centers_to_edges(x, "X")
    ye = centers_to_edges(y, "Y")
    ze = centers_to_edges(z, "Z")
    xmin, xmax = float(xe[0]), float(xe[-1])
    ymin, ymax = float(ye[0]), float(ye[-1])
    zmin, zmax = float(-ze[-1]), float(-ze[0])
    center = (
        0.5 * (xmin + xmax),
        0.5 * (ymin + ymax),
        0.5 * (zmin + zmax),
    )
    length = max(xmax - xmin, ymax - ymin, zmax - zmin)
    camera_position = (
        (center[0] + 2.8 * length,
         center[1] + 2.8 * length,
         center[2] + 2.2 * length),
        center,
        (0.0, 0.0, 1.0),
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[3/4] Rendering comparison figure")

    plotter = pv.Plotter(
        shape=(2, 2),
        off_screen=True,
        window_size=WINDOW_SIZE,
        border=False,
    )

    panels = [
        ((0, 0), "(a) True magnetic model", "True model", "true"),
        ((0, 1), "(b) Inversion - 1% noise", "1% noise", "inverted"),
        ((1, 0), "(c) Inversion - 3% noise", "3% noise", "inverted"),
        ((1, 1), "(d) Inversion - 5% noise", "5% noise", "inverted"),
    ]

    try:
        for position, title, key, model_type in panels:
            add_model_panel(
                plotter=plotter,
                position=position,
                title=title,
                model_file=loaded[key][4],
                model_type=model_type,
                clim=clim,
                cmap=cmap,
                camera_position=camera_position,
            )

        # Link the four subplot cameras to guarantee a consistent view.
        plotter.link_views()

        plotter.screenshot(str(OUTPUT_PNG))

    finally:
        plotter.close()

    print("\n[4/4] Exporting PDF")
    image = plt.imread(OUTPUT_PNG)
    figure = plt.figure(figsize=(14, 10))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(image)
    axis.axis("off")
    figure.savefig(
        OUTPUT_PDF,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(figure)

    print("\n" + "=" * 78)
    print("COMPARISON FIGURE GENERATED")
    print("=" * 78)
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
