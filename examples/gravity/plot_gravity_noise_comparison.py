#!/usr/bin/env python3
"""
Compare the true gravity density-contrast model with inverted models
obtained using 1%, 3%, and 5% noisy data.

Visualization rules
-------------------
- Supports four-column and seven-column model files.
- The true model is automatically converted to density contrast
  when it contains absolute density values.
- The true model is threshold-filtered to display its anomalies.
- ALL cells of the three inverted models are displayed.
- No filtering or modification is applied to inverted models.
- A robust common color range is estimated from the inverted models.
- The colormap is centered on the homogeneous background (0 kg/m3).
- All four panels use the same geometry and camera orientation.
- Depth is displayed positive downward.
- Only the external computational-domain outline is displayed.

Supported model formats
-----------------------
Four columns:
    x_center_m y_center_m z_center_m density_kg_m3

Seven columns:
    ix iy iz x_center_m y_center_m z_center_m density_kg_m3

Output
------
figures/gravity_noise_comparison.png
figures/gravity_noise_comparison.pdf

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

OUTPUT_PNG = FIGURE_DIR / "gravity_noise_comparison.png"
OUTPUT_PDF = FIGURE_DIR / "gravity_noise_comparison.pdf"


# ============================================================================
# 2. VISUALIZATION SETTINGS
# ============================================================================

# Density contrast represented by the homogeneous background.
BACKGROUND_DENSITY = 0.0  # kg/m3

# Automatically convert the true model to density contrast by subtracting
# its dominant density value.
#
# This is useful when model_true.dat stores absolute densities, for example
# a background of 1000 kg/m3 with anomalous values of 50 and 1950 kg/m3.
#
# The inverted models are assumed to already contain density contrasts.
CONVERT_TRUE_MODEL_TO_CONTRAST = True

# Threshold applied ONLY to the true model, after conversion to contrast.
TRUE_ABS_THRESHOLD = 1.0  # kg/m3

# Robust color scaling for inverted models.
COLOR_PERCENTILE = 92.0

# Minimum color half-range.
MIN_COLOR_HALF_RANGE = 1.0  # kg/m3

# Rendering.
CAMERA_ZOOM = 1.20
WINDOW_SIZE = (2200, 1600)
BACKGROUND_COLOR = "white"

SCALAR_NAME = "Density contrast (kg/m3)"

# Diverging color palette.
LOW_COLOR = "#2166AC"
BACKGROUND_COLOR_MAP = "#F7F7F7"
HIGH_COLOR = "#B2182B"

# Camera orientation.
CAMERA_X_FACTOR = 2.8
CAMERA_Y_FACTOR = 2.8
CAMERA_Z_FACTOR = 2.2


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

            # Alternative directory layout.
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
# 4. DENSITY BACKGROUND
# ============================================================================

def estimate_background_density(values: np.ndarray) -> float:
    """
    Estimate the homogeneous background density.

    The most frequently occurring rounded value is used. This is suitable
    for synthetic block models containing a homogeneous background and
    a limited number of anomalous cells.
    """

    values = np.asarray(values, dtype=float)

    rounded = np.round(values, decimals=8)

    unique_values, counts = np.unique(
        rounded,
        return_counts=True,
    )

    index = int(np.argmax(counts))

    return float(unique_values[index])


# ============================================================================
# 5. MODEL LOADING
# ============================================================================

def load_model(
    filename: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a model stored in four-column or seven-column ASCII format.

    Four columns:
        x_center_m, y_center_m, z_center_m, density_kg_m3

    Seven columns:
        ix, iy, iz, x_center_m, y_center_m, z_center_m, density_kg_m3
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

        # Four-column model.
        model_data = data[:, :4]

    elif data.shape[1] >= 7:

        # Seven-column model.
        # Ignore the integer cell indices.
        model_data = data[:, 3:7]

    else:

        raise ValueError(
            f"Unsupported model format in {filename}.\n"
            f"Found {data.shape[1]} columns.\n"
            "Expected four or at least seven columns."
        )

    if not np.all(np.isfinite(model_data)):
        raise ValueError(
            f"Non-finite values found in {filename}."
        )

    x = np.unique(model_data[:, 0])
    y = np.unique(model_data[:, 1])
    z = np.unique(model_data[:, 2])

    shape = (
        x.size,
        y.size,
        z.size,
    )

    expected = int(np.prod(shape))

    if model_data.shape[0] != expected:
        raise ValueError(
            f"Incomplete or duplicated grid in {filename}.\n"
            f"Found {model_data.shape[0]} rows; expected {expected} "
            f"({x.size} x {y.size} x {z.size})."
        )

    values = np.empty(shape, dtype=float)
    occupied = np.zeros(shape, dtype=bool)

    ix = np.searchsorted(
        x,
        model_data[:, 0],
    )

    iy = np.searchsorted(
        y,
        model_data[:, 1],
    )

    iz = np.searchsorted(
        z,
        model_data[:, 2],
    )

    for row, i, j, k in zip(
        model_data,
        ix,
        iy,
        iz,
    ):

        if occupied[i, j, k]:
            raise ValueError(
                f"Duplicate cell center in {filename}: "
                f"({row[0]}, {row[1]}, {row[2]})."
            )

        values[i, j, k] = row[3]
        occupied[i, j, k] = True

    if not np.all(occupied):
        raise ValueError(
            f"Missing cells detected in {filename}."
        )

    return x, y, z, values


def convert_true_model_to_contrast(
    values: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Convert absolute density values to density contrasts.

    The dominant value is interpreted as the homogeneous background.
    """

    background = estimate_background_density(values)

    contrast = values - background + BACKGROUND_DENSITY

    return contrast, background


# ============================================================================
# 6. MESH VALIDATION
# ============================================================================

def check_same_mesh(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray],
    label: str,
) -> None:
    """Verify that models use identical cell-center coordinates."""

    for axis_name, ref, current in zip(
        ("X", "Y", "Z"),
        reference,
        candidate,
    ):

        if ref.shape != current.shape or not np.allclose(
            ref,
            current,
            rtol=0.0,
            atol=1.0e-8,
        ):

            raise ValueError(
                f"The {label} model does not use the same "
                f"{axis_name} coordinates as the true model.\n"
                f"True model range: "
                f"{ref.min():.6f} to {ref.max():.6f}\n"
                f"Current model range: "
                f"{current.min():.6f} to {current.max():.6f}"
            )


# ============================================================================
# 7. GRID CONSTRUCTION
# ============================================================================

def centers_to_edges(
    centers: np.ndarray,
    axis_name: str,
) -> np.ndarray:
    """Convert regularly spaced cell centers into cell edges."""

    centers = np.asarray(
        centers,
        dtype=float,
    )

    if centers.size < 2:
        raise ValueError(
            f"At least two cell centers are required along {axis_name}."
        )

    differences = np.diff(centers)

    spacing = float(
        np.mean(differences)
    )

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

    edges[1:-1] = 0.5 * (
        centers[:-1] + centers[1:]
    )

    edges[0] = centers[0] - 0.5 * spacing
    edges[-1] = centers[-1] + 0.5 * spacing

    return edges


def build_structured_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
) -> pv.StructuredGrid:
    """
    Build a PyVista structured grid.

    Depth is displayed positive downward.
    """

    xe = centers_to_edges(x, "X")
    ye = centers_to_edges(y, "Y")
    ze = centers_to_edges(z, "Z")

    # Reverse vertical display coordinates.
    display_ze = -ze[::-1]

    xx, yy, zz = np.meshgrid(
        xe,
        ye,
        display_ze,
        indexing="ij",
    )

    grid = pv.StructuredGrid(
        xx,
        yy,
        zz,
    )

    # Reverse values consistently with the vertical display coordinates.
    grid.cell_data[SCALAR_NAME] = (
        values[:, :, ::-1].ravel(order="F")
    )

    return grid


# ============================================================================
# 8. COLOR SCALE
# ============================================================================

def make_background_centered_colormap(
    vmin: float,
    vmax: float,
) -> LinearSegmentedColormap:
    """
    Create a blue-white-red colormap centered on zero.
    """

    span = vmax - vmin

    if span <= 0.0:
        raise ValueError(
            "Invalid color limits."
        )

    background_position = (
        BACKGROUND_DENSITY - vmin
    ) / span

    background_position = float(
        np.clip(
            background_position,
            0.001,
            0.999,
        )
    )

    return LinearSegmentedColormap.from_list(
        "gravity_background_centered",
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
    Determine robust common color limits from inverted models.

    Model values are not modified. Values outside the display range
    are represented using the endpoint colors.
    """

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

    vmin = BACKGROUND_DENSITY - half_range
    vmax = BACKGROUND_DENSITY + half_range

    return vmin, vmax


# ============================================================================
# 9. ANOMALY EXTRACTION
# ============================================================================

def extract_display_grid(
    grid: pv.StructuredGrid,
    model_type: str,
) -> pv.UnstructuredGrid:
    """
    Extract cells for display.

    True model:
        Display only significant density anomalies.

    Inverted models:
        Display all cells without filtering.
    """

    if model_type == "true":

        values = np.asarray(
            grid.cell_data[SCALAR_NAME]
        )

        mask = (
            np.abs(
                values - BACKGROUND_DENSITY
            ) > TRUE_ABS_THRESHOLD
        )

        if not np.any(mask):
            raise ValueError(
                "No anomalous cells remain in the true model.\n"
                "Check the background density and threshold."
            )

        return grid.extract_cells(mask)

    # No filtering for inverted models.
    return grid


# ============================================================================
# 10. COMMON CAMERA
# ============================================================================

def create_common_camera(
    coordinates: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
):
    """Create a common camera for all four panels."""

    x, y, z = coordinates

    xe = centers_to_edges(x, "X")
    ye = centers_to_edges(y, "Y")
    ze = centers_to_edges(z, "Z")

    xmin, xmax = float(xe[0]), float(xe[-1])
    ymin, ymax = float(ye[0]), float(ye[-1])

    # Negative display coordinates correspond to positive depth.
    zmin = float(-ze[-1])
    zmax = float(-ze[0])

    center = (
        0.5 * (xmin + xmax),
        0.5 * (ymin + ymax),
        0.5 * (zmin + zmax),
    )

    length = max(
        xmax - xmin,
        ymax - ymin,
        zmax - zmin,
    )

    camera_position = (
        (
            center[0] + CAMERA_X_FACTOR * length,
            center[1] + CAMERA_Y_FACTOR * length,
            center[2] + CAMERA_Z_FACTOR * length,
        ),
        center,
        (0.0, 0.0, 1.0),
    )

    return camera_position


# ============================================================================
# 11. PLOTTING
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
    model_data=None,
) -> None:
    """Add one gravity model to a subplot."""

    if model_data is None:

        x, y, z, values = load_model(
            model_file
        )

    else:

        x, y, z, values = model_data

    full_grid = build_structured_grid(
        x,
        y,
        z,
        values,
    )

    display_grid = extract_display_grid(
        full_grid,
        model_type,
    )

    plotter.subplot(*position)

    plotter.set_background(
        BACKGROUND_COLOR
    )

    show_scalar_bar = (
        position == (1, 1)
    )

    scalar_bar_args = None

    if show_scalar_bar:

        scalar_bar_args = {
            "title": "Density contrast (kg/m3)",
            "vertical": True,
            "title_font_size": 13,
            "label_font_size": 11,
            "n_labels": 7,
            "color": "black",
            "fmt": "%.1f",
            "position_x": 0.84,
            "position_y": 0.12,
            "height": 0.72,
            "width": 0.10,
        }

    plotter.add_mesh(
        display_grid,
        scalars=SCALAR_NAME,
        cmap=cmap,
        clim=clim,
        show_edges=False,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args=scalar_bar_args,
        lighting=True,
        smooth_shading=False,
        ambient=0.25,
        diffuse=0.75,
        specular=0.05,
    )

    # External domain outline only.
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
# 12. MAIN
# ============================================================================

def main() -> int:
    """Generate the four-panel gravity noise comparison."""

    print("=" * 78)
    print("GeoInvLab3D - GRAVITY NOISE COMPARISON")
    print("=" * 78)

    # ----------------------------------------------------------------------
    # Locate input files.
    # ----------------------------------------------------------------------

    true_model_file = find_true_model()

    model_files = {
        "True model": true_model_file,
        "1% noise": find_inverted_model("1pct"),
        "3% noise": find_inverted_model("3pct"),
        "5% noise": find_inverted_model("5pct"),
    }

    # ----------------------------------------------------------------------
    # Load the true model.
    # ----------------------------------------------------------------------

    print("\n[1/5] Loading the true model")

    true_x, true_y, true_z, true_values = load_model(
        true_model_file
    )

    original_min = float(true_values.min())
    original_max = float(true_values.max())

    print(f"  File          : {true_model_file}")
    print(
        f"  Mesh          : "
        f"{true_x.size} x {true_y.size} x {true_z.size}"
    )
    print(f"  Original min  : {original_min:.6g} kg/m3")
    print(f"  Original max  : {original_max:.6g} kg/m3")

    if CONVERT_TRUE_MODEL_TO_CONTRAST:

        true_values, estimated_background = (
            convert_true_model_to_contrast(
                true_values
            )
        )

        print(
            f"  Estimated background density: "
            f"{estimated_background:.6g} kg/m3"
        )

        print(
            f"  Converted contrast range: "
            f"{true_values.min():.6g} to "
            f"{true_values.max():.6g} kg/m3"
        )

    true_model_data = (
        true_x,
        true_y,
        true_z,
        true_values,
    )

    reference_coordinates = (
        true_x,
        true_y,
        true_z,
    )

    loaded = {
        "True model": (
            true_x,
            true_y,
            true_z,
            true_values,
            true_model_file,
        )
    }

    # ----------------------------------------------------------------------
    # Load and validate inverted models.
    # ----------------------------------------------------------------------

    print("\n[2/5] Loading and validating inverted models")

    for label in (
        "1% noise",
        "3% noise",
        "5% noise",
    ):

        filename = model_files[label]

        x, y, z, values = load_model(
            filename
        )

        check_same_mesh(
            reference_coordinates,
            (x, y, z),
            label,
        )

        loaded[label] = (
            x,
            y,
            z,
            values,
            filename,
        )

        print(
            f"\n{label}:"
            f"\n  File    : {filename}"
            f"\n  Mesh    : {x.size} x {y.size} x {z.size}"
            f"\n  Cells   : {values.size}"
            f"\n  Minimum : {values.min():.6g} kg/m3"
            f"\n  Maximum : {values.max():.6g} kg/m3"
        )

    # ----------------------------------------------------------------------
    # Determine common color limits.
    # ----------------------------------------------------------------------

    print("\n[3/5] Computing robust common color limits")

    inverted_models = {
        label: loaded[label][3]
        for label in (
            "1% noise",
            "3% noise",
            "5% noise",
        )
    }

    clim = determine_color_limits(
        inverted_models
    )

    cmap = make_background_centered_colormap(
        *clim
    )

    print(
        f"  Color limits: "
        f"{clim[0]:.4f} to {clim[1]:.4f} kg/m3"
    )

    print(
        f"  Background: {BACKGROUND_DENSITY:.4f} kg/m3"
    )

    print(
        f"  Percentile: {COLOR_PERCENTILE:.1f}"
    )

    print(
        f"  True-model threshold: "
        f"|contrast| > {TRUE_ABS_THRESHOLD:.4f} kg/m3"
    )

    print("  Inverted models: all cells retained")

    # ----------------------------------------------------------------------
    # Common camera.
    # ----------------------------------------------------------------------

    camera_position = create_common_camera(
        reference_coordinates
    )

    # ----------------------------------------------------------------------
    # Create output directory.
    # ----------------------------------------------------------------------

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Render the four-panel figure.
    # ----------------------------------------------------------------------

    print("\n[4/5] Rendering comparison figure")

    plotter = pv.Plotter(
        shape=(2, 2),
        off_screen=True,
        window_size=WINDOW_SIZE,
        border=False,
    )

    panels = [
        (
            (0, 0),
            "(a) True gravity model",
            "True model",
            "true",
        ),
        (
            (0, 1),
            "(b) Inversion - 1% noise",
            "1% noise",
            "inverted",
        ),
        (
            (1, 0),
            "(c) Inversion - 3% noise",
            "3% noise",
            "inverted",
        ),
        (
            (1, 1),
            "(d) Inversion - 5% noise",
            "5% noise",
            "inverted",
        ),
    ]

    try:

        for position, title, key, model_type in panels:

            model_data = None

            if key == "True model":
                model_data = true_model_data

            add_model_panel(
                plotter=plotter,
                position=position,
                title=title,
                model_file=loaded[key][4],
                model_type=model_type,
                clim=clim,
                cmap=cmap,
                camera_position=camera_position,
                model_data=model_data,
            )

        # Link the four cameras.
        plotter.link_views()

        plotter.screenshot(
            str(OUTPUT_PNG)
        )

    finally:

        plotter.close()

    # ----------------------------------------------------------------------
    # Export PDF.
    # ----------------------------------------------------------------------

    print("\n[5/5] Exporting PDF")

    image = plt.imread(
        OUTPUT_PNG
    )

    figure = plt.figure(
        figsize=(14, 10)
    )

    axis = figure.add_axes(
        [0, 0, 1, 1]
    )

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

    # ----------------------------------------------------------------------
    # Final summary.
    # ----------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("GRAVITY COMPARISON FIGURE GENERATED")
    print("=" * 78)

    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")

    print("=" * 78)

    return 0


# ============================================================================
# 13. EXECUTION
# ============================================================================

if __name__ == "__main__":

    try:
        raise SystemExit(
            main()
        )

    except Exception as error:

        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )

        raise

