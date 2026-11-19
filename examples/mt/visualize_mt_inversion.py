"""
Visualize and compare true, initial, and inverted MT resistivity models.

GeoInvLab3D conventions:
- X: East
- Y: North
- Z: positive downward
- Surface: z = 0
- Model values use z as the fastest-varying index.
  MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR
RESULTS_DIR = SCRIPT_DIR / "inversion_results"
FIGURE_DIR = RESULTS_DIR / "figures"

MESH_FILE = DATA_DIR / "mesh.dat"

MODEL_FILES = {
    "True model": DATA_DIR / "model_true.dat",
    "Initial model": DATA_DIR / "model_initial.dat",
}

INVERTED_CANDIDATES = [
    RESULTS_DIR / "model_inverted.dat",
    RESULTS_DIR / "inverted_model.dat",
    RESULTS_DIR / "model_final.dat",
]

STATION_CANDIDATES = [
    DATA_DIR / "stations.dat",
    SCRIPT_DIR / "stations.dat",
]


# ============================================================
# PHYSICAL MODEL CONFIGURATION
# ============================================================

# Values used by the synthetic true model.
BACKGROUND_RESISTIVITY = 100.0
TRUE_CONDUCTIVE_RESISTIVITY = 10.0
TRUE_RESISTIVE_RESISTIVITY = 1000.0

# General thresholds used for displaying the true and
# initial models.
#
# These are visualization thresholds only.
CONDUCTIVE_THRESHOLD = 10.0
RESISTIVE_THRESHOLD = 1000.0

# Thresholds used ONLY for the inverted model.
#
# Conductive body:
#       0 <= rho <= 10 ohm.m
#
# Resistive body:
#       rho > 1000 ohm.m
INVERTED_CONDUCTIVE_MIN = 0.0
INVERTED_CONDUCTIVE_MAX = 20.0
INVERTED_RESISTIVE_MIN = 500.0
INVERTED_CONDUCTIVE_COLOR = "orange"
INVERTED_RESISTIVE_COLOR = "green"

# Target horizontal section depth.
TARGET_DEPTH = 200.0
TARGET_X = -150.0

# Plot settings.
CMAP = "viridis"

CONDUCTIVE_COLOR = "red"
RESISTIVE_COLOR = "blue"
BACKGROUND_COLOR = "lightgray"


# ============================================================
# FILE LOADING
# ============================================================

def load_numeric_table(path):
    """Load a numerical ASCII table."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    data = np.loadtxt(
        path,
        comments="#",
        dtype=float,
        ndmin=2,
    )

    if data.size == 0:
        raise ValueError(
            f"File is empty: {path}"
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            f"Non-finite values detected in: {path}"
        )

    return data


def load_mesh(path):
    """
    Load the structured mesh.

    Expected columns:

        nx ny nz dx dy dz origin_x origin_y origin_z
    """

    data = load_numeric_table(path)

    if data.shape[0] != 1 or data.shape[1] < 9:
        raise ValueError(
            "The mesh file must contain one row with at least "
            "nine columns:\n"
            "nx ny nz dx dy dz origin_x origin_y origin_z"
        )

    row = data[0]

    nx = int(round(row[0]))
    ny = int(round(row[1]))
    nz = int(round(row[2]))

    dx = float(row[3])
    dy = float(row[4])
    dz = float(row[5])

    x0 = float(row[6])
    y0 = float(row[7])
    z0 = float(row[8])

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(
            "Mesh dimensions must be positive."
        )

    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError(
            "Cell dimensions must be positive."
        )

    # Cell boundaries.
    x_edges = x0 + np.arange(nx + 1) * dx
    y_edges = y0 + np.arange(ny + 1) * dy
    z_edges = z0 + np.arange(nz + 1) * dz

    # Cell centers.
    x_centers = 0.5 * (
        x_edges[:-1] + x_edges[1:]
    )

    y_centers = 0.5 * (
        y_edges[:-1] + y_edges[1:]
    )

    z_centers = 0.5 * (
        z_edges[:-1] + z_edges[1:]
    )

    shape = (nx, ny, nz)

    print("\nMesh information")
    print("-" * 55)
    print(
        f"Dimensions       : "
        f"{nx} x {ny} x {nz}"
    )
    print(
        f"Number of cells  : "
        f"{nx * ny * nz}"
    )
    print(
        f"Cell dimensions  : "
        f"{dx:g} x {dy:g} x {dz:g} m"
    )
    print(
        f"X extent         : "
        f"{x_edges[0]:g} to {x_edges[-1]:g} m"
    )
    print(
        f"Y extent         : "
        f"{y_edges[0]:g} to {y_edges[-1]:g} m"
    )
    print(
        f"Z extent         : "
        f"{z_edges[0]:g} to {z_edges[-1]:g} m"
    )
    print("-" * 55)

    return {
        "shape": shape,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "z_edges": z_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "z_centers": z_centers,
    }


def load_model(path, shape):
    """
    Load a resistivity model.

    The last column contains resistivity values.

    GeoInvLab3D ordering:
        index = i * (ny * nz) + j * nz + k

    Therefore z is the fastest-varying index.
    """

    data = load_numeric_table(path)

    expected_cells = int(np.prod(shape))

    if data.shape[0] != expected_cells:
        raise ValueError(
            f"Invalid model size in {path}.\n"
            f"Expected {expected_cells} rows, "
            f"found {data.shape[0]}."
        )

    values = data[:, -1]

    if np.any(values <= 0.0):
        raise ValueError(
            f"Resistivity values must be positive in {path}."
        )

    return values.reshape(
        shape,
        order="C",
    )


def find_inverted_model():
    """Find the available inverted model file."""

    for path in INVERTED_CANDIDATES:
        if path.is_file():
            return path

    candidates = "\n".join(
        str(path)
        for path in INVERTED_CANDIDATES
    )

    raise FileNotFoundError(
        "Could not find the inverted model.\n"
        "Checked:\n"
        + candidates
    )


def find_stations_file():
    """Find the MT station file."""

    for path in STATION_CANDIDATES:
        if path.is_file():
            return path

    return None


def load_stations(path):
    """Load MT station coordinates."""

    if path is None:
        print(
            "Station file not found. "
            "Stations will not be displayed."
        )
        return None

    data = load_numeric_table(path)

    if data.shape[1] < 3:
        raise ValueError(
            "The station file must contain at least "
            "three columns: x, y, z."
        )

    stations = data[:, :3]

    print(
        f"Stations loaded  : {len(stations)}"
    )
    print(
        f"Station file     : {path}"
    )

    if not np.allclose(
        stations[:, 2],
        0.0,
        atol=1e-8,
    ):
        print(
            "Warning: some stations are not located "
            "at z = 0 m."
        )

    return stations


# ============================================================
# COLOR NORMALIZATION
# ============================================================

def create_shared_norm(models):
    """
    Create a common logarithmic normalization for all models.
    """

    positive_values = []

    for model in models:
        values = model[
            np.isfinite(model)
            & (model > 0.0)
        ]

        if values.size > 0:
            positive_values.append(values)

    if not positive_values:
        raise ValueError(
            "No positive resistivity values found."
        )

    all_values = np.concatenate(positive_values)

    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))

    if np.isclose(vmin, vmax):
        vmax = vmin * 1.01

    return LogNorm(
        vmin=vmin,
        vmax=vmax,
    )


# ============================================================
# 3D VISUALIZATION
# ============================================================

def plot_3d_comparison(
    models,
    mesh,
    stations,
):
    """
    Plot true, initial, and inverted models in 3D.

    True and initial models:
        Conductive: rho <= 10 ohm.m
        Resistive:  rho >= 1000 ohm.m

    Inverted model:
        Conductive: rho <= 20 ohm.m
        Resistive:  rho >= 500 ohm.m

    Background cells are hidden in the inverted model.

    GeoInvLab3D convention:
        Z is positive downward.
    """

    titles = [
        "True model",
        "Initial model",
        "Inverted model",
    ]

    x_edges = mesh["x_edges"]
    y_edges = mesh["y_edges"]
    z_edges = mesh["z_edges"]

    # Create 3D coordinate arrays for voxel visualization.
    X, Y, Z = np.meshgrid(
        x_edges,
        y_edges,
        z_edges,
        indexing="ij",
    )

    fig = plt.figure(figsize=(20, 8))

    for panel_index, (title, model) in enumerate(
        zip(titles, models),
        start=1,
    ):

        ax = fig.add_subplot(
            1,
            3,
            panel_index,
            projection="3d",
        )

        # ----------------------------------------------------
        # Select thresholds and colors for each model.
        # ----------------------------------------------------

        if title == "Inverted model":

            conductive_color = INVERTED_CONDUCTIVE_COLOR
            resistive_color = INVERTED_RESISTIVE_COLOR

            conductive_mask = (
                (model >= INVERTED_CONDUCTIVE_MIN)
                & (model <= INVERTED_CONDUCTIVE_MAX)
            )

            resistive_mask = (
                model >= INVERTED_RESISTIVE_MIN
            )

            # Hide the background in the inverted model.
            background_mask = np.zeros(
                model.shape,
                dtype=bool,
            )

        else:

            conductive_color = CONDUCTIVE_COLOR
            resistive_color = RESISTIVE_COLOR

            conductive_mask = (
                model <= CONDUCTIVE_THRESHOLD
            )

            resistive_mask = (
                model >= RESISTIVE_THRESHOLD
            )

            background_mask = ~(
                conductive_mask | resistive_mask
            )

        # ----------------------------------------------------
        # Print selection information.
        # ----------------------------------------------------

        conductive_count = int(
            np.count_nonzero(conductive_mask)
        )

        resistive_count = int(
            np.count_nonzero(resistive_mask)
        )

        background_count = int(
            np.count_nonzero(background_mask)
        )

        print(f"\n{title}")
        print("-" * 55)
        print(f"Conductive cells : {conductive_count}")
        print(f"Resistive cells  : {resistive_count}")
        print(f"Background cells : {background_count}")

        # ----------------------------------------------------
        # Background.
        # ----------------------------------------------------

        if np.any(background_mask):

            ax.voxels(
                X,
                Y,
                Z,
                background_mask,
                facecolors=BACKGROUND_COLOR,
                edgecolor="gray",
                linewidth=0.15,
                alpha=0.08,
            )

        # ----------------------------------------------------
        # Conductive anomaly.
        # ----------------------------------------------------

        if np.any(conductive_mask):

            ax.voxels(
                X,
                Y,
                Z,
                conductive_mask,
                facecolors=conductive_color,
                edgecolor="black",
                linewidth=0.25,
                alpha=1.0,
            )

        # ----------------------------------------------------
        # Resistive anomaly.
        # ----------------------------------------------------

        if np.any(resistive_mask):

            ax.voxels(
                X,
                Y,
                Z,
                resistive_mask,
                facecolors=resistive_color,
                edgecolor="black",
                linewidth=0.25,
                alpha=1.0,
            )

        # ----------------------------------------------------
        # MT stations.
        # ----------------------------------------------------

        if stations is not None:

            ax.scatter(
                stations[:, 0],
                stations[:, 1],
                stations[:, 2],
                color="black",
                marker="^",
                s=22,
                depthshade=False,
            )

        # ----------------------------------------------------
        # Axes.
        # ----------------------------------------------------

        ax.set_title(
            title,
            fontsize=19,
            fontweight="bold",
            color="black",
            pad=24,
            backgroundcolor="white",
        )

        ax.set_xlabel("X East (m)")
        ax.set_ylabel("Y North (m)")
        ax.set_zlabel("Z Depth (m)")

        ax.set_xlim(
            x_edges[0],
            x_edges[-1],
        )

        ax.set_ylim(
            y_edges[0],
            y_edges[-1],
        )

        # Z is positive downward.
        ax.set_zlim(
            z_edges[-1],
            z_edges[0],
        )

        ax.set_box_aspect(
            (
                x_edges[-1] - x_edges[0],
                y_edges[-1] - y_edges[0],
                z_edges[-1] - z_edges[0],
            )
        )

        ax.view_init(
            elev=25,
            azim=-55,
        )

    # --------------------------------------------------------
    # Legend.
    # --------------------------------------------------------

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=CONDUCTIVE_COLOR,
            markeredgecolor="black",
            markersize=9,
            label=r"True conductor ($\rho \leq 10\ \Omega\cdot m$)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=RESISTIVE_COLOR,
            markeredgecolor="black",
            markersize=9,
            label=r"True resistor ($\rho \geq 1000\ \Omega\cdot m$)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=INVERTED_CONDUCTIVE_COLOR,
            markeredgecolor="black",
            markersize=9,
            label=r"Inverted conductor ($\rho \leq 20\ \Omega\cdot m$)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=INVERTED_RESISTIVE_COLOR,
            markeredgecolor="black",
            markersize=9,
            label=r"Inverted resistor ($\rho \geq 500\ \Omega\cdot m$)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="black",
            linestyle="None",
            markersize=7,
            label="MT stations",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        frameon=True,
    )

    fig.suptitle(
        "MT resistivity models: 3D comparison",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(
        rect=(0, 0.09, 1, 0.93)
    )

    output_path = (
        FIGURE_DIR / "model_comparison_3d.png"
    )
    pdf_path = (
        FIGURE_DIR / "model_comparison_3d.pdf"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
    )

    print("\n3D figure saved:")
    print(f"  PNG: {output_path}")
    print(f"  PDF: {pdf_path}")

    plt.show()



# ============================================================
# DEPTH INDEX
# ============================================================

def find_depth_index(
    depths,
    target_depth,
):
    """Find the cell-center index closest to target depth."""

    return int(
        np.argmin(
            np.abs(
                depths
                - target_depth
            )
        )
    )


# ============================================================
# 2D SECTIONS
# ============================================================

def plot_sections(
    models,
    mesh,
    norm,
):
    """
    Plot XY, XZ, and YZ resistivity sections.

    All resistivity values are displayed continuously.
    No threshold is applied here.
    """

    titles = [
        "True model",
        "Initial model",
        "Inverted model",
    ]

    nx = mesh["nx"]
    ny = mesh["ny"]

    x_centers = mesh["x_centers"]
    y_centers = mesh["y_centers"]
    z_centers = mesh["z_centers"]

    depth_index = find_depth_index(
        z_centers,
        TARGET_DEPTH,
    )

    center_y = ny // 2

    # Find the X cell closest to the requested coordinate.
    x_index = int(
    np.argmin(
        np.abs(x_centers - TARGET_X)
    )
    )
    actual_depth = z_centers[
        depth_index
    ]

    actual_x = x_centers[x_index]

    actual_y = y_centers[
        center_y
    ]

    print("\n2D section positions")
    print("-" * 55)
    print(
        f"Requested depth : "
        f"{TARGET_DEPTH:g} m"
    )
    print(
        f"Actual depth    : "
        f"{actual_depth:g} m"
    )
    print(
        f"XZ section Y    : "
        f"{actual_y:g} m"
    )
    print(
        f"YZ section X    : "
        f"{actual_x:g} m"
    )
    print("-" * 55)

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(17, 13),
        constrained_layout=True,
    )

    image = None

    for row, (
        title,
        model,
    ) in enumerate(
        zip(titles, models)
    ):

        # ----------------------------------------------------
        # XY
        # ----------------------------------------------------

        horizontal = model[
            :,
            :,
            depth_index,
        ].T

        image = axes[row, 0].pcolormesh(
            x_centers,
            y_centers,
            horizontal,
            shading="nearest",
            cmap=CMAP,
            norm=norm,
        )

        axes[row, 0].set_title(
            f"{title} - XY section "
            f"at z = {actual_depth:g} m"
        )

        axes[row, 0].set_xlabel(
            "X East (m)"
        )

        axes[row, 0].set_ylabel(
            "Y North (m)"
        )

        axes[row, 0].set_aspect(
            "equal"
        )

        # ----------------------------------------------------
        # XZ
        # ----------------------------------------------------

        vertical_xz = model[
            :,
            center_y,
            :,
        ].T

        axes[row, 1].pcolormesh(
            x_centers,
            z_centers,
            vertical_xz,
            shading="nearest",
            cmap=CMAP,
            norm=norm,
        )

        axes[row, 1].set_title(
            f"{title} - XZ section "
            f"at Y = {actual_y:g} m"
        )

        axes[row, 1].set_xlabel(
            "X East (m)"
        )

        axes[row, 1].set_ylabel(
            "Z Depth (m)"
        )

        # Z increases downward.
        axes[row, 1].invert_yaxis()

        # ----------------------------------------------------
        # YZ
        # ----------------------------------------------------

        vertical_yz = model[
        x_index,
        :,
        :,
        ].T

        axes[row, 2].pcolormesh(
            y_centers,
            z_centers,
            vertical_yz,
            shading="nearest",
            cmap=CMAP,
            norm=norm,
        )

        axes[row, 2].set_title(
            f"{title} - YZ section "
            f"at X = {actual_x:g} m"
        )

        axes[row, 2].set_xlabel(
            "Y North (m)"
        )

        axes[row, 2].set_ylabel(
            "Z Depth (m)"
        )

        # Z increases downward.
        axes[row, 2].invert_yaxis()

    # ========================================================
    # COMMON COLORBAR
    # ========================================================

    colorbar = fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.85,
        pad=0.02,
    )

    colorbar.set_label(
        "Resistivity (ohm.m)"
    )

    fig.suptitle(
        "MT resistivity models: 2D sections",
        fontsize=16,
        fontweight="bold",
    )

    output_path = (
        FIGURE_DIR
        / "model_comparison_sections.png"
    )
    pdf_path = (FIGURE_DIR/ "model_comparison_sections.pdf")
    fig.savefig(output_path,dpi=300,bbox_inches="tight",)
     

    fig.savefig(
    pdf_path,
    format="pdf",
    bbox_inches="tight",)
    print(f"\n2D sections saved: "f"{output_path}")
    print(f"  PDF: {pdf_path}")
    plt.show()


# ============================================================
# MODEL STATISTICS
# ============================================================

def print_model_statistics(
    models,
):
    """Print statistics for all models."""

    titles = [
        "True model",
        "Initial model",
        "Inverted model",
    ]

    print("\nMODEL STATISTICS")
    print("=" * 65)

    for title, model in zip(
        titles,
        models,
    ):

        print(f"\n{title}")
        print("-" * 45)

        print(
            f"Minimum resistivity : "
            f"{np.min(model):.6g} ohm.m"
        )

        print(
            f"Maximum resistivity : "
            f"{np.max(model):.6g} ohm.m"
        )

        print(
            f"Mean resistivity    : "
            f"{np.mean(model):.6g} ohm.m"
        )

        print(
            f"Median resistivity  : "
            f"{np.median(model):.6g} ohm.m"
        )

        # ----------------------------------------------------
        # True model exact-value information
        # ----------------------------------------------------

        if title == "True model":

            conductive_count = np.count_nonzero(
                np.isclose(
                    model,
                    TRUE_CONDUCTIVE_RESISTIVITY,
                    rtol=0.0,
                    atol=1e-10,
                )
            )

            background_count = np.count_nonzero(
                np.isclose(
                    model,
                    BACKGROUND_RESISTIVITY,
                    rtol=0.0,
                    atol=1e-10,
                )
            )

            resistive_count = np.count_nonzero(
                np.isclose(
                    model,
                    TRUE_RESISTIVE_RESISTIVITY,
                    rtol=0.0,
                    atol=1e-10,
                )
            )

            print(
                f"10 ohm.m cells   : "
                f"{conductive_count}"
            )

            print(
                f"100 ohm.m cells  : "
                f"{background_count}"
            )

            print(
                f"1000 ohm.m cells : "
                f"{resistive_count}"
            )

        # ----------------------------------------------------
        # Inverted model threshold information
        # ----------------------------------------------------

        if title == "Inverted model":

            conductive_count = np.count_nonzero(
                model <= INVERTED_CONDUCTIVE_MAX
            )

            resistive_count = np.count_nonzero(
                model >= INVERTED_RESISTIVE_MIN
            )

            print(
                "Conductive cells (<= 20 ohm.m): "
                f"{conductive_count}"
            )

            print(
                "Resistive cells (>= 500 ohm.m): "
                f"{resistive_count}"
            )

    print("=" * 65)


# ============================================================
# TRUE / INVERTED COMPARISON
# ============================================================

def compare_true_and_inverted(
    true_model,
    inverted_model,
):
    """Calculate global model-comparison metrics."""

    true_values = true_model.ravel()
    inverted_values = inverted_model.ravel()

    difference = (
        inverted_values
        - true_values
    )

    denominator = max(
        np.linalg.norm(true_values),
        np.finfo(float).eps,
    )

    relative_l2 = (
        np.linalg.norm(difference)
        / denominator
    )

    mae = np.mean(
        np.abs(difference)
    )

    rmse = np.sqrt(
        np.mean(
            difference ** 2
        )
    )

    max_error = np.max(
        np.abs(difference)
    )

    if (
        np.std(true_values) > 0.0
        and np.std(inverted_values) > 0.0
    ):
        correlation = np.corrcoef(
            true_values,
            inverted_values,
        )[0, 1]
    else:
        correlation = np.nan

    print(
        "\nTRUE / INVERTED MODEL COMPARISON"
    )
    print("=" * 55)

    print(
        f"Relative L2 error : "
        f"{relative_l2:.6e}"
    )

    print(
        f"MAE               : "
        f"{mae:.6g} ohm.m"
    )

    print(
        f"RMSE              : "
        f"{rmse:.6g} ohm.m"
    )

    print(
        f"Maximum error     : "
        f"{max_error:.6g} ohm.m"
    )

    print(
        f"Correlation       : "
        f"{correlation:.6f}"
    )

    print("=" * 55)


# ============================================================
# INVERSION REPORT
# ============================================================

def print_inversion_report():
    """Print the inversion report when available."""

    report_candidates = [
        RESULTS_DIR / "inversion_report.txt",
        RESULTS_DIR / "report.txt",
    ]

    for report_path in report_candidates:

        if report_path.is_file():

            print(
                "\nINVERSION REPORT"
            )

            print("=" * 65)

            print(
                report_path.read_text(
                    encoding="utf-8"
                )
            )

            print("=" * 65)

            return

    print(
        "\nNo inversion report found."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "GeoInvLab3D - MT INVERSION VISUALIZATION"
    )
    print("=" * 70)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Mesh
    # --------------------------------------------------------

    print(
        "\n[1/6] Loading mesh..."
    )

    mesh = load_mesh(
        MESH_FILE
    )

    # --------------------------------------------------------
    # 2. Stations
    # --------------------------------------------------------

    print(
        "\n[2/6] Loading MT stations..."
    )

    stations_file = (
        find_stations_file()
    )

    stations = load_stations(
        stations_file
    )

    # --------------------------------------------------------
    # 3. True and initial models
    # --------------------------------------------------------

    print(
        "\n[3/6] Loading true and initial models..."
    )

    true_model = load_model(
        MODEL_FILES["True model"],
        mesh["shape"],
    )

    initial_model = load_model(
        MODEL_FILES["Initial model"],
        mesh["shape"],
    )

    # --------------------------------------------------------
    # 4. Inverted model
    # --------------------------------------------------------

    print(
        "\n[4/6] Loading inverted model..."
    )

    inverted_path = (
        find_inverted_model()
    )

    inverted_model = load_model(
        inverted_path,
        mesh["shape"],
    )

    print(
        f"Inverted model file: "
        f"{inverted_path}"
    )

    models = [
        true_model,
        initial_model,
        inverted_model,
    ]

    # --------------------------------------------------------
    # 5. Statistics
    # --------------------------------------------------------

    print(
        "\n[5/6] Computing model statistics..."
    )

    print_model_statistics(
        models
    )

    compare_true_and_inverted(
        true_model,
        inverted_model,
    )

    print_inversion_report()

    # --------------------------------------------------------
    # 6. Figures
    # --------------------------------------------------------

    print(
        "\n[6/6] Generating figures..."
    )

    norm = create_shared_norm(
        models
    )

    plot_3d_comparison(
        models,
        mesh,
        stations,
    )

    plot_sections(
        models,
        mesh,
        norm,
    )

    print(
        "\nVisualization completed."
    )

    print(
        f"Figures directory: "
        f"{FIGURE_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
