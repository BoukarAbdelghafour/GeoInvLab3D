#!/usr/bin/env python3
"""
Plot the true 3D gravity density-contrast model and measurement stations.

The visualization displays:
    - Positive density contrast: dense body.
    - Negative density contrast: low-density body.
    - Surface gravity stations as black triangles.
    - External model-domain outline without internal mesh lines.

Coordinates:
    X: East (m)
    Y: North (m)
    Z: Depth, positive downward (m)

Output:
    - gravity_true_model_3d.png (300 DPI)
    - gravity_true_model_3d.pdf (vector PDF)

Boukar Abdelghafour
GeoInvLab3D
MIT License
Copyright (c) 2026
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "data" / "model_true.dat"
STATIONS_FILE = BASE_DIR / "data" / "stations.dat"

# Output files
OUTPUT_PNG = BASE_DIR / "gravity_true_model_3d.png"
OUTPUT_PDF = BASE_DIR / "gravity_true_model_3d.pdf"

# Model dimensions
NX, NY, NZ = 20, 20, 15

DX = DY = DZ = 100.0

# Density-contrast thresholds
DENSE_THRESHOLD = 1.0
LIGHT_THRESHOLD = -1.0

# Body colors
DENSE_COLOR = "firebrick"
LIGHT_COLOR = "royalblue"

# Station display
STATION_COLOR = "black"
STATION_MARKER = "^"
STATION_SIZE = 25

# Slight graphical elevation above the surface
STATION_DISPLAY_Z = -15.0

# Figure settings
FIGURE_SIZE = (11, 9)
FIGURE_DPI = 300


# =============================================================================
# Data loading
# =============================================================================

def load_true_model(filename):
    """Load the true density-contrast model."""

    if not filename.exists():
        raise FileNotFoundError(
            f"Model file not found: {filename}"
        )

    data = np.loadtxt(filename, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 7:
        raise ValueError(
            "Expected at least seven columns: "
            "ix iy iz x_center y_center z_center density_contrast."
        )

    x = data[:, 3]
    y = data[:, 4]
    z = data[:, 5]
    density = data[:, 6]

    return x, y, z, density


def load_stations(filename):
    """Load gravity station coordinates."""

    if not filename.exists():
        raise FileNotFoundError(
            f"Station file not found: {filename}"
        )

    stations = np.loadtxt(filename, comments="#")

    if stations.ndim == 1:
        stations = stations.reshape(1, -1)

    if stations.shape[1] < 3:
        raise ValueError(
            "The station file must contain x, y and z coordinates."
        )

    return stations[:, :3]


# =============================================================================
# Geometry
# =============================================================================

def create_cuboid_faces(xmin, xmax, ymin, ymax, zmin, zmax):
    """Return the six faces of a rectangular prism."""

    vertices = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ])

    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [3, 0, 4, 7]],
    ]

    return faces


def add_body(ax, bounds, color):
    """Add a semi-transparent 3D rectangular body."""

    faces = create_cuboid_faces(*bounds)

    collection = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors="none",
        alpha=0.85,
    )

    ax.add_collection3d(collection)


def add_domain_outline(ax):
    """Draw only the external outline of the complete model domain."""

    faces = create_cuboid_faces(
        0.0, NX * DX,
        0.0, NY * DY,
        0.0, NZ * DZ,
    )

    for face in faces:
        closed = face + [face[0]]

        ax.plot(
            [point[0] for point in closed],
            [point[1] for point in closed],
            [point[2] for point in closed],
            color="black",
            linewidth=0.8,
        )


# =============================================================================
# Plotting
# =============================================================================

def plot_model(x, y, z, density, stations):
    """Create and save the 3D visualization of the true gravity model."""

    dense_mask = density >= DENSE_THRESHOLD
    light_mask = density <= LIGHT_THRESHOLD

    if not np.any(dense_mask):
        raise ValueError("No positive density anomaly found.")

    if not np.any(light_mask):
        raise ValueError("No negative density anomaly found.")

    fig = plt.figure(figsize=FIGURE_SIZE)

    ax = fig.add_subplot(111, projection="3d")

    # -------------------------------------------------------------------------
    # Dense body
    # -------------------------------------------------------------------------

    add_body(
        ax,
        (
            500.0, 900.0,
            500.0, 900.0,
            300.0, 700.0,
        ),
        DENSE_COLOR,
    )

    # -------------------------------------------------------------------------
    # Low-density body
    # -------------------------------------------------------------------------

    add_body(
        ax,
        (
            1100.0, 1500.0,
            1100.0, 1500.0,
            500.0, 900.0,
        ),
        LIGHT_COLOR,
    )

    # -------------------------------------------------------------------------
    # External model-domain outline
    # -------------------------------------------------------------------------

    add_domain_outline(ax)

    # -------------------------------------------------------------------------
    # Gravity stations at the surface
    # -------------------------------------------------------------------------

    ax.scatter(
        stations[:, 0],
        stations[:, 1],
        np.full(len(stations), STATION_DISPLAY_Z),
        marker=STATION_MARKER,
        color=STATION_COLOR,
        s=STATION_SIZE,
        depthshade=False,
        edgecolors="white",
        linewidths=0.3,
    )

    # -------------------------------------------------------------------------
    # Axes limits
    # -------------------------------------------------------------------------

    ax.set_xlim(0.0, NX * DX)
    ax.set_ylim(0.0, NY * DY)
    ax.set_zlim(NZ * DZ, STATION_DISPLAY_Z)

    # -------------------------------------------------------------------------
    # Axis labels
    # -------------------------------------------------------------------------

    ax.set_xlabel(
        "X (m)",
        fontsize=11,
        labelpad=10,
    )

    ax.set_ylabel(
        "Y (m)",
        fontsize=11,
        labelpad=10,
    )

    ax.set_zlabel(
        "Depth Z (m)",
        fontsize=11,
        labelpad=10,
    )

    ax.set_xticks(np.arange(0, NX * DX + 1, 500))
    ax.set_yticks(np.arange(0, NY * DY + 1, 500))
    ax.set_zticks(np.arange(0, NZ * DZ + 1, 300))

    ax.tick_params(axis="both", labelsize=9)
    ax.tick_params(axis="z", labelsize=9)

    # -------------------------------------------------------------------------
    # Equal visual scaling
    # -------------------------------------------------------------------------

    ax.set_box_aspect(
        (
            NX * DX,
            NY * DY,
            NZ * DZ,
        )
    )

    # -------------------------------------------------------------------------
    # View angle
    # -------------------------------------------------------------------------

    ax.view_init(
        elev=25,
        azim=-55,
    )

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    legend_elements = [
        Patch(
            facecolor=DENSE_COLOR,
            label="Dense body (+500 kg/m³)",
        ),
        Patch(
            facecolor=LIGHT_COLOR,
            label="Low-density body (-300 kg/m³)",
        ),
        Line2D(
            [0], [0],
            marker=STATION_MARKER,
            color="black",
            linestyle="None",
            markersize=7,
            markerfacecolor=STATION_COLOR,
            label=f"Gravity stations ({len(stations)})",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=9,
        frameon=True,
    )

    # -------------------------------------------------------------------------
    # Figure title
    # -------------------------------------------------------------------------

    ax.set_title(
        "True 3D Gravity Density-Contrast Model",
        fontsize=14,
        pad=20,
        fontweight="bold",
    )

    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Save output files
    # -------------------------------------------------------------------------

    OUTPUT_PNG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save high-resolution PNG
    fig.savefig(
        OUTPUT_PNG,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        format="png",
    )

    # Save vector PDF
    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        format="pdf",
    )

    print("\nFigures successfully saved:")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")

    # -------------------------------------------------------------------------
    # Display figure
    # -------------------------------------------------------------------------

    plt.show()

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():

    print("Loading true gravity model...")

    x, y, z, density = load_true_model(MODEL_FILE)

    print(f"Number of cells: {len(density)}")

    print(
        f"Density contrast range: "
        f"{density.min():.1f} to {density.max():.1f} kg/m3"
    )

    print(
        f"Positive anomaly cells: "
        f"{np.count_nonzero(density > 0)}"
    )

    print(
        f"Negative anomaly cells: "
        f"{np.count_nonzero(density < 0)}"
    )

    print("\nLoading gravity stations...")

    stations = load_stations(STATIONS_FILE)

    print(f"Number of stations: {len(stations)}")

    # Generate and save figure
    plot_model(
        x,
        y,
        z,
        density,
        stations,
    )


if __name__ == "__main__":
    main()

