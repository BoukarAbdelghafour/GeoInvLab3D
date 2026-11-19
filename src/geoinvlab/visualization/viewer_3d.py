"""
3D visualization for GeoInvLab3D.

This module provides a PyVista-based 3D viewer for:

- observation data
- structured 3D meshes
- physical models

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from geoinvlab.core.data import DataSet
from geoinvlab.core.mesh import StructuredMesh3D
from geoinvlab.core.model import Model3D


class Viewer3D(QWidget):
    """
    Interactive 3D viewer for GeoInvLab3D.

    Displays observation data, structured 3D meshes,
    and physical models in the same scene.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._plotter = QtInteractor(self)

        self._create_layout()
        self._configure_plotter()

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def _create_layout(self) -> None:
        """Create the viewer layout."""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plotter)

    def _configure_plotter(self) -> None:
        """Configure the PyVista scene."""

        self._plotter.set_background("#20252b")
        self._add_axes_and_grid()

    def _add_axes_and_grid(self) -> None:
        """Add coordinate axes and the reference grid."""

        self._plotter.add_axes()

        self._plotter.show_grid(
            color="gray",
            xtitle="X",
            ytitle="Y",
            ztitle="Z",
        )

    # ==================================================================
    # GENERAL OPERATIONS
    # ==================================================================

    def clear(self) -> None:
        """Remove all objects from the scene."""

        self._plotter.clear()
        self._add_axes_and_grid()
        self._plotter.render()

    def reset_camera(self) -> None:
        """Reset the camera to show the complete scene."""

        self._plotter.reset_camera()
        self._plotter.render()

    def reset_view(self) -> None:
        """Alias for reset_camera()."""

        self.reset_camera()

    # ==================================================================
    # DATA
    # ==================================================================

    def show_data(
        self,
        dataset: DataSet,
        *,
        point_size: float = 9.0,
        clear: bool = True,
    ) -> None:
        """Display observation locations."""

        if clear:
            self.clear()

        self._add_data(
            dataset,
            point_size=point_size,
        )

        self._plotter.reset_camera()
        self._plotter.render()

    def _add_data(
        self,
        dataset: DataSet,
        *,
        point_size: float = 9.0,
    ) -> None:
        """Add observation data without clearing the scene."""

        locations = np.asarray(
            dataset.locations,
            dtype=float,
        )

        values = np.asarray(
            dataset.values,
            dtype=float,
        )

        if locations.size == 0:
            return

        if locations.ndim != 2 or locations.shape[1] != 3:
            raise ValueError(
                "Dataset locations must have shape (n_points, 3)."
            )

        if values.ndim != 1 or values.size != locations.shape[0]:
            raise ValueError(
                "Dataset values must contain exactly one scalar "
                "per observation location."
            )

        cloud = pv.PolyData(locations)
        cloud["values"] = values

        self._plotter.add_points(
            cloud,
            scalars="values",
            cmap="plasma",
            point_size=point_size,
            render_points_as_spheres=True,
            name="data",
            scalar_bar_args={
                "title": dataset.name or "Data",
            },
        )

    # ==================================================================
    # MESH
    # ==================================================================

    def show_mesh(
        self,
        mesh: StructuredMesh3D,
        *,
        clear: bool = True,
        show_edges: bool = True,
    ) -> None:
        """Display a structured 3D mesh."""

        if clear:
            self.clear()

        self._add_mesh(
            mesh,
            show_edges=show_edges,
        )

        self._plotter.reset_camera()
        self._plotter.render()

    def _add_mesh(
        self,
        mesh: StructuredMesh3D,
        *,
        show_edges: bool = True,
    ) -> None:
        """Add the structured mesh without clearing the scene."""

        ox, oy, oz = mesh.origin

        grid = pv.ImageData()

        grid.dimensions = (
            mesh.nx + 1,
            mesh.ny + 1,
            mesh.nz + 1,
        )

        grid.origin = (ox, oy, oz)

        grid.spacing = (
            mesh.dx,
            mesh.dy,
            mesh.dz,
        )

        self._plotter.add_mesh(
            grid,
            style="wireframe",
            color="white",
            line_width=1.0,
            opacity=0.6,
            show_edges=show_edges,
            name="mesh",
        )

    # ==================================================================
    # MODEL
    # ==================================================================

    def show_model(
        self,
        model: Model3D,
        *,
        clear: bool = True,
        cmap: str = "coolwarm",
        opacity: float = 0.15,
        show_edges: bool = False,
    ) -> None:
        """Display a physical model on a structured mesh."""

        if clear:
            self.clear()

        self._add_model(
            model,
            cmap=cmap,
            opacity=opacity,
            show_edges=show_edges,
        )

        self._plotter.reset_camera()
        self._plotter.render()

    def _add_model(
        self,
        model: Model3D,
        *,
        cmap: str = "coolwarm",
        opacity: float = 0.15,
        show_edges: bool = False,
    ) -> None:
        """Add a physical model and highlight internal anomalies."""

        mesh = model.mesh

        ox, oy, oz = mesh.origin

        grid = pv.ImageData()

        grid.dimensions = (
            mesh.nx + 1,
            mesh.ny + 1,
            mesh.nz + 1,
        )

        grid.origin = (ox, oy, oz)

        grid.spacing = (
            mesh.dx,
            mesh.dy,
            mesh.dz,
        )

        # --------------------------------------------------------------
        # Read model values
        # --------------------------------------------------------------

        values = np.asarray(
            model.values,
            dtype=float,
        ).ravel()

        expected_size = mesh.nx * mesh.ny * mesh.nz

        if values.size != expected_size:
            raise ValueError(
                f"Model values size mismatch. "
                f"Expected {expected_size}, received {values.size}."
            )

        # GeoInvLab3D ordering:
        # q = i * (ny * nz) + j * nz + k
        # The z index varies fastest.
        model_3d = values.reshape(
            (mesh.nx, mesh.ny, mesh.nz),
            order="C",
        )

        # VTK ImageData ordering:
        # The x index varies fastest.
        vtk_values = model_3d.transpose(
            2, 1, 0
        ).ravel(order="C")

        grid.cell_data["model"] = vtk_values

        # --------------------------------------------------------------
        # Identify anomalous cells
        # --------------------------------------------------------------

        unique_values, counts = np.unique(
            vtk_values,
            return_counts=True,
        )

        background_value = unique_values[
            np.argmax(counts)
        ]

        anomaly_mask = ~np.isclose(
            vtk_values,
            background_value,
            rtol=1e-10,
            atol=1e-12,
        )

        anomaly_indices = np.flatnonzero(
            anomaly_mask
        )

        print("\n========== 3D MODEL DISPLAY ==========")
        print(f"Model name:       {model.name}")
        print(f"Mesh dimensions:  {mesh.nx} x {mesh.ny} x {mesh.nz}")
        print(f"Total cells:      {expected_size}")
        print(f"Background:       {background_value}")
        print(f"Anomaly cells:    {anomaly_indices.size}")

        for value, count in zip(unique_values, counts):
            print(
                f"Resistivity: {value:10.3f} ohm.m | "
                f"Cells: {count}"
            )

        # --------------------------------------------------------------
        # Display the complete model with transparency
        # --------------------------------------------------------------

        self._plotter.add_mesh(
            grid,
            scalars="model",
            cmap=cmap,
            opacity=opacity,
            show_edges=show_edges,
            name="model_background",
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Resistivity (ohm.m)",
            },
        )

        # --------------------------------------------------------------
        # Extract and display internal anomalies
        # --------------------------------------------------------------

        if anomaly_indices.size > 0:

            anomaly_grid = grid.extract_cells(
                anomaly_indices
            )

            print("\n========== EXTRACTED ANOMALIES ==========")
            print(f"Number of cells: {anomaly_grid.n_cells}")
            print(f"Values: {anomaly_grid.cell_data['model']}")
            print(f"Bounds: {anomaly_grid.bounds}")
            print("==========================================")

            self._plotter.add_mesh(
                anomaly_grid,
                scalars="model",
                cmap=cmap,
                clim=(
                    float(np.min(vtk_values)),
                    float(np.max(vtk_values)),
                ),
                opacity=1.0,
                show_edges=True,
                line_width=2.0,
                name="model_anomalies",
                show_scalar_bar=False,
            )

        else:
            print("No anomalous cells detected.")

        print("======================================\n")

    # ==================================================================
    # COMPLETE PROJECT
    # ==================================================================

    def show_project(
        self,
        *,
        dataset: DataSet | None = None,
        mesh: StructuredMesh3D | None = None,
        model: Model3D | None = None,
    ) -> None:
        """
        Display the complete project.

        Objects are displayed in this order:

            1. mesh
            2. model
            3. observation data
        """

        self.clear()

        if mesh is not None:
            self._add_mesh(mesh)

        if model is not None:
            self._add_model(model)

        if dataset is not None:
            self._add_data(dataset)

        self._plotter.reset_camera()
        self._plotter.render()

    # ==================================================================
    # CLEANUP
    # ==================================================================

    def closeEvent(self, event) -> None:
        """Release the PyVista interactor."""

        try:
            self._plotter.close()
        except Exception:
            pass

        super().closeEvent(event)

