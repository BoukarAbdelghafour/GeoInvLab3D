"""
Main application window for GeoInvLab3D.

Supported inversion methods: Gravity, Magnetotellurics (MT), and Magnetic.

MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QScrollArea, QStatusBar, QTableWidget, QTableWidgetItem,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from geoinvlab.project import Project
from geoinvlab.core.mesh import StructuredMesh3D, load_mesh
from geoinvlab.core.model import Model3D, load_model
from geoinvlab.core.data.mt_dataset import MTDataSet
from geoinvlab.io.data_importer import load_data_file
from geoinvlab.gui.help.help_window import HelpWindow


class MainWindow(QMainWindow):
    """Main GeoInvLab3D application window."""

    INVERSION_METHODS = {
        "Gravity": {
            "id": "gravity", "parameter": "density",
            "unit": "kg/m³",
            "description": "Gravity inversion using density contrast.",
        },
        "MT": {
            "id": "mt", "parameter": "resistivity",
            "unit": "ohm·m",
            "description": "Magnetotelluric inversion using electrical resistivity.",
        },
        "Magnetic": {
            "id": "magnetic", "parameter": "susceptibility",
            "unit": "SI",
            "description": "Magnetic inversion using magnetic susceptibility.",
        },
    }

    CLASSICAL_HEADERS = ["X", "Y", "Z", "Value", "Uncertainty"]
    MT_COMPONENTS = (
        "rho_xx", "rho_xy", "rho_yx", "rho_yy",
        "phase_xx", "phase_xy", "phase_yx", "phase_yy",
    )

    def __init__(self) -> None:
        super().__init__()
        self.project = Project(name="Untitled Project")
        self.viewer_3d = None
        self._help_window = None
        self.model_metadata = {}
        self._viewer_available = False
        self.inversion_config = {
            "method": "gravity", "lambda": 1e-8,
            "max_iterations": 100, "tolerance": 1e-6,
            "regularization": "tikhonov", "reference_model": True,
        }
        self.setWindowTitle("GeoInvLab3D")
        self.resize(1500, 950)

        # Load the application logo from the GUI package assets directory.
        self.logo_path = (
            Path(__file__).resolve().parent / "assets" / "geoinvlab3d_symbol.png"
        )
        if self.logo_path.is_file():
            self.setWindowIcon(QIcon(str(self.logo_path)))
        self._create_menu()
        self._create_central_widget()
        self._create_status_bar()
        self._apply_professional_theme()
        self._update_project_tree()
        self._update_project_information()
        self._display_inversion_info()
        self.status_bar.showMessage("Ready - Untitled Project")

    # ------------------------------------------------------------------
    # General helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_number(value) -> str:
        """Format a scalar numeric value safely."""
        array = np.asarray(value)
        if array.size != 1:
            return str(array.tolist())
        return f"{float(array.reshape(-1)[0]):.8g}"

    def _method_definition(self) -> dict:
        method = self.inversion_config.get("method", "gravity")
        for definition in self.INVERSION_METHODS.values():
            if definition["id"] == method:
                return definition
        return self.INVERSION_METHODS["Gravity"]

    def _method_label(self) -> str:
        method_id = self._method_definition()["id"]
        return next(
            (label for label, item in self.INVERSION_METHODS.items()
             if item["id"] == method_id),
            "Gravity",
        )

    def _method_parameter(self) -> str:
        return self._method_definition()["parameter"]

    def _default_model_value(self) -> float:
        return {"gravity": 0.0, "mt": 100.0, "magnetic": 0.01}.get(
            self.inversion_config.get("method", "gravity"), 0.0
        )

    def _default_lambda(self) -> float:
        return {"gravity": 1e-8, "mt": 1e-2, "magnetic": 1e-6}.get(
            self.inversion_config.get("method", "gravity"), 1e-8
        )

    def _inversion_is_configured(self) -> bool:
        return (
            isinstance(self.inversion_config, dict)
            and self.inversion_config.get("method")
            in {"gravity", "mt", "magnetic"}
        )

    # ------------------------------------------------------------------
    # Menus and main layout
    # ------------------------------------------------------------------

    def _create_menu(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        for label, slot in (
            ("New Project", self._new_project),
            ("Open Project...", self._open_project),
            ("Save Project...", self._save_project),
        ):
            action = QAction(label, self)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        for label, slot in (
            ("Import Data...", self._import_data),
            ("Export Data...", self._export_data),
        ):
            action = QAction(label, self)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        action = QAction("Exit", self)
        action.triggered.connect(self.close)
        file_menu.addAction(action)

        project_menu = menu_bar.addMenu("&Project")
        action = QAction("Project Information", self)
        action.triggered.connect(self._show_project_information)
        project_menu.addAction(action)

        data_menu = menu_bar.addMenu("&Data")
        for label, slot in (
            ("Import Data...", self._import_data),
            ("Export Data...", self._export_data),
        ):
            action = QAction(label, self)
            action.triggered.connect(slot)
            data_menu.addAction(action)

        mesh_menu = menu_bar.addMenu("&Mesh")
        for label, slot in (
            ("Create Mesh...", self._create_mesh),
            ("Load Mesh...", self._load_mesh),
            ("Remove Mesh", self._remove_mesh),
        ):
            action = QAction(label, self)
            action.triggered.connect(slot)
            mesh_menu.addAction(action)

        model_menu = menu_bar.addMenu("&Model")
        for label, slot in (
            ("Create Model...", self._create_model),
            ("Load Model...", self._load_model),
            ("Create Reference Model...", self._create_reference_model),
            ("Remove Model", self._remove_model),
            ("Remove Reference Model", self._remove_reference_model),
        ):
            action = QAction(label, self)
            action.triggered.connect(slot)
            model_menu.addAction(action)

        inversion_menu = menu_bar.addMenu("&Inversion")
        for label, slot in (
            ("Configure Inversion", self._configure_inversion),
            ("Run Inversion", self._run_inversion),
        ):
            action = QAction(label, self)
            action.triggered.connect(slot)
            inversion_menu.addAction(action)

        view_menu = menu_bar.addMenu("&View")
        for label, slot in (
            ("Show Project in 3D", self._show_project_3d),
            ("Reset 3D View", self._reset_view),
            ("Clear 3D View", self._clear_3d_view),
        ):
            action = QAction(label, self)
            action.triggered.connect(slot)
            view_menu.addAction(action)

        help_menu = menu_bar.addMenu("&Help")

        manual_action = QAction("User Manual", self)
        manual_action.setShortcut("F1")
        manual_action.setStatusTip("Open the GeoInvLab3D user manual.")
        manual_action.triggered.connect(
            lambda: self._open_help("index.html")
        )
        help_menu.addAction(manual_action)

        quick_action = QAction("Quick Start", self)
        quick_action.setStatusTip("Open the GeoInvLab3D quick-start guide.")
        quick_action.triggered.connect(
            lambda: self._open_help("quick_start.html")
        )
        help_menu.addAction(quick_action)

        tutorials_menu = help_menu.addMenu("Tutorials")
        for label, page in (
            ("Gravity", "gravity.html"),
            ("Magnetic", "magnetic.html"),
            ("Magnetotelluric (MT)", "magnetotelluric.html"),
        ):
            tutorial_action = QAction(label, self)
            tutorial_action.triggered.connect(
                lambda checked=False, target=page: self._open_help(target)
            )
            tutorials_menu.addAction(tutorial_action)

        formats_action = QAction("Data File Formats", self)
        formats_action.triggered.connect(
            lambda: self._open_help("data_formats.html")
        )
        help_menu.addAction(formats_action)

        troubleshooting_action = QAction("Troubleshooting", self)
        troubleshooting_action.triggered.connect(
            lambda: self._open_help("troubleshooting.html")
        )
        help_menu.addAction(troubleshooting_action)

        help_menu.addSeparator()
        about_action = QAction("About GeoInvLab3D", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_central_widget(self) -> None:
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(6)

        # Application branding header.
        brand_header = QWidget()
        brand_header.setObjectName("brandHeader")
        brand_layout = QHBoxLayout(brand_header)
        brand_layout.setContentsMargins(10, 2, 12, 2)
        brand_layout.setSpacing(8)

        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setFixedSize(126, 104)
        self.logo_label.setStyleSheet("""
        QLabel {
        background-color: transparent;
        border: none;
        padding: 0px;
        margin: 0px;
        }
        """)
        if self.logo_path.is_file():
            logo_pixmap = QPixmap(str(self.logo_path))
            if not logo_pixmap.isNull():
                self.logo_label.setPixmap(
                    logo_pixmap.scaled(
                        122, 100,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        else:
            self.logo_label.setText("GeoInvLab3D")
            self.logo_label.setStyleSheet(
                "font-size:18px;font-weight:bold;color:#123f7a;"
            )

        brand_text = QWidget()
        brand_text_layout = QVBoxLayout(brand_text)
        brand_text_layout.setContentsMargins(0, 0, 0, 0)
        brand_text_layout.setSpacing(3)

        brand_title = QLabel("   GeoInvLab3D")
        brand_title.setStyleSheet(
            "font-size:25px;font-weight:700;color:#174D35;"
        )
        brand_subtitle = QLabel("   3D GEOPHYSICAL MODELING & INVERSION")
        brand_subtitle.setStyleSheet(
            "font-size:12px;font-weight:700;letter-spacing:1px;color:#D94848;"
        )
        brand_description = QLabel(
            "   Integrated tools for Magnetotelluric, Magnetic and Gravity modelling and inversion"
        )
        brand_description.setWordWrap(True)
        brand_description.setStyleSheet("font-size:12px;color:#526458;")

        brand_text_layout.addWidget(brand_title)
        brand_text_layout.addWidget(brand_subtitle)
        brand_text_layout.addWidget(brand_description)
        brand_text_layout.addStretch(1)

        brand_layout.addWidget(self.logo_label, 0, Qt.AlignmentFlag.AlignVCenter)
        brand_layout.addWidget(brand_text, 1)
        brand_header.setMaximumHeight(116)
        brand_header.setStyleSheet(
            "#brandHeader { background:#ffffff; "
            "border:1px solid #d9e3ef; border-radius:7px; }"
        )
        main_layout.addWidget(brand_header)

        left = QWidget()
        left.setObjectName("projectPanel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(7)
        title = QLabel("Project")
        title.setObjectName("panelTitle")
        title.setStyleSheet(
            "font-size:18px;font-weight:700;color:#174D35;"
            "padding:7px 8px;border-bottom:2px solid #D94848;"
        )
        left_layout.addWidget(title)
        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderLabel("GeoInvLab3D")
        self.project_tree.setMinimumHeight(205)
        self.project_tree.setRootIsDecorated(True)
        self.project_tree.setItemsExpandable(True)
        self.project_tree.setIndentation(18)
        self.project_tree.setUniformRowHeights(True)
        self.project_tree.itemClicked.connect(self._project_item_clicked)
        left_layout.addWidget(self.project_tree, 1)
        for label, slot in (
            ("Import Data", self._import_data),
            ("Create Mesh", self._create_mesh),
            ("Create Model", self._create_model),
            ("Reference Model", self._create_reference_model),
            ("Configure Inversion", self._configure_inversion),
            ("Run Inversion", self._run_inversion),
            ("Show Project in 3D", self._show_project_3d),
        ):
            button = QPushButton(label)
            button.setObjectName(
                "runInversionButton" if label == "Run Inversion" else "sidebarButton"
            )
            button.setMinimumHeight(30)
            button.setMaximumHeight(34)
            button.clicked.connect(slot)
            left_layout.addWidget(button)
        left.setMinimumWidth(220)
        left.setMaximumWidth(320)
        left.setMinimumHeight(570)

        # A scrollable sidebar prevents the project tree from being compressed
        # on displays with limited vertical space.
        left_scroll = QScrollArea()
        left_scroll.setObjectName("projectSidebarScroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        left_scroll.setWidget(left)

        self.tabs = QTabWidget()
        self.data_table = QTableWidget()
        self._set_classical_table_headers()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSortingEnabled(True)
        self.tabs.addTab(self.data_table, "Data Table")

        self.viewer_3d = self._create_3d_viewer()
        self.tabs.addTab(self.viewer_3d, "3D View")

        self.mesh_info = QLabel("No mesh created.")
        self.mesh_info.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.mesh_info.setWordWrap(True)
        self.tabs.addTab(self.mesh_info, "Mesh")

        self.model_info = QLabel("No model created.")
        self.model_info.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.model_info.setWordWrap(True)
        self.tabs.addTab(self.model_info, "Model")

        self.inversion_widget = QWidget()
        inversion_layout = QVBoxLayout(self.inversion_widget)
        heading = QLabel("Inversion Configuration")
        heading.setStyleSheet("font-size:18px;font-weight:bold;padding:5px;")
        inversion_layout.addWidget(heading)
        method_form = QFormLayout()
        self.method_combo = QComboBox()
        self.method_combo.setMinimumWidth(350)
        for label, definition in self.INVERSION_METHODS.items():
            self.method_combo.addItem(label, definition["id"])
        index = self.method_combo.findData(self.inversion_config["method"])
        self.method_combo.setCurrentIndex(index)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_form.addRow("Method:", self.method_combo)
        inversion_layout.addLayout(method_form)
        self.method_info_label = QLabel()
        self.method_info_label.setWordWrap(True)
        self.method_info_label.setMinimumHeight(110)
        self.method_info_label.setStyleSheet(
            "padding:12px;background:#eef3f8;border:1px solid #ccd6e0;"
            "border-radius:5px;"
        )
        inversion_layout.addWidget(self.method_info_label)
        self.inversion_info = QLabel("Inversion not configured.")
        self.inversion_info.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.inversion_info.setWordWrap(True)
        inversion_layout.addWidget(self.inversion_info, 1)
        for label, slot in (
            ("Configure Inversion...", self._configure_inversion),
            ("Run Inversion", self._run_inversion),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            inversion_layout.addWidget(button)
        inversion_scroll = QScrollArea()
        inversion_scroll.setWidgetResizable(True)
        inversion_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inversion_scroll.setWidget(self.inversion_widget)
        self.tabs.addTab(inversion_scroll, "Inversion")
        self._update_method_widget()

        right = QWidget()
        right.setObjectName("propertiesPanel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(7)
        title = QLabel("Properties")
        title.setObjectName("panelTitle")
        title.setStyleSheet("font-size:17px;font-weight:bold;padding:5px;")
        right_layout.addWidget(title)
        self.properties_label = QLabel("No selection.")
        self.properties_label.setObjectName("propertiesCard")
        self.properties_label.setWordWrap(True)
        self.properties_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_layout.addWidget(self.properties_label, 1)
        right.setMinimumWidth(260)
        right.setMaximumWidth(380)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _apply_professional_theme(self) -> None:
        """Apply the GeoInvLab3D green, red and white visual identity."""
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #F5F8F5;
                color: #26352C;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 10pt;
            }
            QMenuBar {
                background-color: #174D35;
                color: #FFFFFF;
                padding: 5px 8px;
                border-bottom: 2px solid #D94848;
            }
            QMenuBar::item { background: transparent; padding: 7px 11px; border-radius: 4px; }
            QMenuBar::item:selected { background-color: #267A50; }
            QMenu { background-color: #FFFFFF; color: #26352C; border: 1px solid #C9D9CC; padding: 5px; }
            QMenu::item { padding: 7px 28px 7px 22px; border-radius: 3px; }
            QMenu::item:selected { background-color: #E5F3E9; color: #174D35; }
            QWidget#brandHeader {background-color: #FFFFFF;border: 1px solid #D7E4D9;border-radius: 7px;}
            QWidget#projectPanel { background-color: #174D35; border: 1px solid #123F2B; border-radius: 7px; }
            QWidget#projectPanel QLabel#panelTitle { color: #174D35; }
            QTreeWidget {
                background-color: #FFFFFF;
                alternate-background-color: #F0F6F1;
                color: #263F30;
                border: 2px solid #D94848;
                border-radius: 6px;
                padding: 5px;
                selection-background-color: #D7EEDD;
                selection-color: #174D35;
                outline: 0;
            }
            QTreeWidget QHeaderView::section {
                background-color: #DCEBDF;
                color: #174D35;
                border: none;
                border-bottom: 1px solid #C5D8C8;
                padding: 8px 7px;
                min-height: 30px;
                font-size: 11pt;
                font-weight: bold;
            }
            QTreeWidget::item { padding: 4px 3px; min-height: 25px; }
            QTreeWidget::item:hover { background-color: #E8F4EA; }
            QPushButton#sidebarButton { background-color: #26734B; color: #FFFFFF; border: 1px solid #39865D; border-radius: 5px; padding: 7px 10px; text-align: left; }
            QPushButton#sidebarButton:hover { background-color: #328B5C; border-color: #75B58B; }
            QPushButton#sidebarButton:pressed { background-color: #123F2B; }
            QPushButton#runInversionButton { background-color: #D94848; color: #FFFFFF; border: 1px solid #BE3535; border-radius: 5px; padding: 7px 12px; min-height: 22px; font-weight: bold; }
            QPushButton#runInversionButton:hover { background-color: #E45B5B; }
            QPushButton#runInversionButton:pressed { background-color: #AD3030; }
            QWidget#propertiesPanel { background-color: #EAF2EB; border: 1px solid #D1E0D3; border-radius: 7px; }
            QWidget#propertiesPanel QLabel#panelTitle { color: #174D35; }
            QLabel#propertiesCard { background-color: #FFFFFF; color: #263F30; border: 1px solid #D4E2D6; border-radius: 5px; padding: 12px; }
            QTabWidget::pane { background-color: #FFFFFF; border: 1px solid #C7D9CA; border-radius: 5px; top: -1px; }
            QTabBar::tab { background-color: #E8EFE9; color: #31513B; border: 1px solid #D0DDD2; padding: 8px 16px; margin-right: 2px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
            QTabBar::tab:hover { background-color: #D9EBDD; }
            QTabBar::tab:selected { background-color: #278653; color: #FFFFFF; border-color: #278653; font-weight: bold; }
            QTableWidget { background-color: #FFFFFF; alternate-background-color: #F3F7F3; gridline-color: #DFE8E0; border: 1px solid #D5E1D6; selection-background-color: #D7EEDD; selection-color: #174D35; }
            QHeaderView::section { background-color: #E2EEE4; color: #244B32; border: none; border-right: 1px solid #D0DED2; border-bottom: 1px solid #C5D8C8; padding: 7px; font-weight: bold; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #FFFFFF; color: #26352C; border: 1px solid #BFD1C2; border-radius: 4px; padding: 6px 8px; min-height: 20px; }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #278653; }
            QPushButton { background-color: #278653; color: #FFFFFF; border: 1px solid #207448; border-radius: 5px; padding: 7px 12px; min-height: 22px; }
            QPushButton:hover { background-color: #329761; }
            QPushButton:pressed { background-color: #174D35; }
            QDialog, QMessageBox { background-color: #F5F8F5; }
            QScrollArea { background-color: transparent; border: none; }
            QScrollArea#projectSidebarScroll { background-color: #174D35; border: none; }
            QStatusBar { background-color: #174D35; color: #FFFFFF; border-top: 2px solid #D94848; padding: 4px; }
            QSplitter::handle { background-color: #D7E3D9; width: 4px; }
            QSplitter::handle:hover { background-color: #70A982; }
            QToolTip { background-color: #174D35; color: #FFFFFF; border: 1px solid #D94848; padding: 5px; }
            """
        )

    def _create_status_bar(self) -> None:
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _set_classical_table_headers(self) -> None:
        self.data_table.setColumnCount(5)
        self.data_table.setHorizontalHeaderLabels(self.CLASSICAL_HEADERS)

    # ------------------------------------------------------------------
    # Method selection
    # ------------------------------------------------------------------

    def _on_method_changed(self, index: int) -> None:
        del index
        method_id = self.method_combo.currentData()
        if method_id not in {"gravity", "mt", "magnetic"}:
            return
        old = self.inversion_config.get("method", "gravity")
        self.inversion_config["method"] = method_id
        if method_id != old:
            self.inversion_config["lambda"] = self._default_lambda()
        self.project.inversion_config = dict(self.inversion_config)
        self._update_method_widget()
        self._display_inversion_info()
        self._update_project_tree()
        self._update_project_information()
        self.status_bar.showMessage(f"{self._method_label()} method selected.")

    def _update_method_widget(self) -> None:
        method_id = self.inversion_config.get("method", "gravity")
        definition = self._method_definition()
        if hasattr(self, "method_combo"):
            index = self.method_combo.findData(method_id)
            if index >= 0:
                self.method_combo.blockSignals(True)
                self.method_combo.setCurrentIndex(index)
                self.method_combo.blockSignals(False)
        if hasattr(self, "method_info_label"):
            self.method_info_label.setText(
                f"<b>Selected method:</b> {self._method_label()}<br><br>"
                f"<b>Physical parameter:</b> {definition['parameter']}<br>"
                f"<b>Unit:</b> {definition['unit']}<br><br>"
                f"{definition['description']}"
            )

    # ------------------------------------------------------------------
    # 3-D viewer
    # ------------------------------------------------------------------

    def _create_3d_viewer(self) -> QWidget:
        try:
            from geoinvlab.visualization.viewer_3d import Viewer3D
            viewer = Viewer3D()
            self._viewer_available = True
            return viewer
        except Exception as exc:
            self._viewer_available = False
            widget = QWidget()
            layout = QVBoxLayout(widget)
            title = QLabel("3D Visualization")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            message = QLabel(f"3D viewer unavailable.\n\n{exc}")
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            message.setWordWrap(True)
            layout.addStretch()
            layout.addWidget(title)
            layout.addWidget(message)
            layout.addStretch()
            return widget

    def _show_project_3d(self) -> None:
        if not self._viewer_available:
            QMessageBox.warning(self, "3D View", "The 3D viewer is not available.")
            return
        try:
            self.viewer_3d.show_project(
                dataset=self.project.dataset, mesh=self.project.mesh,
                model=self.project.model,
            )
            self.tabs.setCurrentWidget(self.viewer_3d)
            self.status_bar.showMessage("Project displayed in 3D.")
        except Exception as exc:
            QMessageBox.critical(self, "3D View Error", f"Could not display project:\n\n{exc}")

    def _update_3d_project(self) -> None:
        if self._viewer_available:
            try:
                self.viewer_3d.show_project(
                    dataset=self.project.dataset, mesh=self.project.mesh,
                    model=self.project.model,
                )
            except Exception as exc:
                self.status_bar.showMessage(f"3D update failed: {exc}")

    def _update_3d_data(self) -> None:
        if self._viewer_available and self.project.dataset is not None:
            try:
                self.viewer_3d.show_data(self.project.dataset)
            except Exception as exc:
                self.status_bar.showMessage(f"3D data update failed: {exc}")

    def _reset_view(self) -> None:
        if not self._viewer_available:
            return
        try:
            self.viewer_3d.reset_view()
            self.tabs.setCurrentWidget(self.viewer_3d)
        except Exception as exc:
            QMessageBox.warning(self, "3D View", f"Could not reset view:\n\n{exc}")

    def _clear_3d_view(self) -> None:
        if self._viewer_available:
            try:
                self.viewer_3d.clear()
            except Exception as exc:
                self.status_bar.showMessage(f"Could not clear 3D view: {exc}")

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def _new_project(self) -> None:
        name, ok = QInputDialog.getText(
            self, "New Project", "Project name:", text="Untitled Project"
        )
        if not ok:
            return
        name = name.strip() or "Untitled Project"
        self.project = Project(name=name)
        self.model_metadata = {}
        self.inversion_config = {
            "method": "gravity", "lambda": 1e-8,
            "max_iterations": 100, "tolerance": 1e-6,
            "regularization": "tikhonov", "reference_model": True,
        }
        self._clear_interface()
        self._refresh_all()
        self.setWindowTitle(f"GeoInvLab3D - {name}")
        self.status_bar.showMessage(f"New project created: {name}")

    def _clear_interface(self) -> None:
        self._clear_data_table()
        self.mesh_info.setText("No mesh created.")
        self.model_info.setText("No model created.")
        self.inversion_info.setText("Inversion not configured.")
        self.properties_label.setText("No selection.")
        self._clear_3d_view()
        self._update_method_widget()

    def _open_project(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open GeoInvLab3D Project", "",
            "GeoInvLab3D Project (*.json);;All Files (*)",
        )
        if not filename:
            return
        try:
            self.project = Project.load(filename)
            self.model_metadata = {}
            config = getattr(self.project, "inversion_config", None)
            if isinstance(config, dict):
                self.inversion_config = dict(config)
            self._refresh_all()
            self.setWindowTitle(f"GeoInvLab3D - {self.project.name}")
            self.status_bar.showMessage(f"Project loaded: {self.project.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Open Project", f"Could not open project:\n\n{exc}")

    def _save_project(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "Save Project", "No project is available.")
            return
        self.project.inversion_config = dict(self.inversion_config)
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save GeoInvLab3D Project", self.project.name,
            "GeoInvLab3D Project (*.json);;All Files (*)",
        )
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        try:
            saved_path = self.project.save(str(path))
            self.status_bar.showMessage(f"Project saved: {saved_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Project", f"Could not save project:\n\n{exc}")

    def _refresh_all(self) -> None:
        dataset = self.project.dataset
        if dataset is None:
            self._clear_data_table()
        else:
            self._display_dataset(dataset)
        if self.project.mesh is None:
            self.mesh_info.setText("No mesh created.")
        else:
            self._display_mesh_info(self.project.mesh)
        if self.project.model is None:
            self.model_info.setText("No model created.")
            self.model_metadata = {}
        else:
            self._display_model_info(self.project.model)
        self._update_method_widget()
        self._display_inversion_info()
        self._update_project_tree()
        self._update_project_information()
        self._update_3d_project()

    # ------------------------------------------------------------------
    # Data import, table and export
    # ------------------------------------------------------------------

    def _import_data(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Geophysical Data", "",
            "Data files (*.csv *.txt *.dat *.xyz);;All Files (*)",
        )
        if not filename:
            return
        try:
            dataset = load_data_file(filename)
            self.project.set_data(dataset)
            self._display_dataset(dataset)
            self._update_project_tree()
            self._show_dataset_properties()
            self._update_3d_data()
            if isinstance(dataset, MTDataSet):
                status = (
                    f"Imported {dataset.n_observations} MT station-frequency "
                    f"observations ({dataset.n_data} scalar data values)."
                )
            else:
                status = f"Imported {dataset.n_data} data observations."
            self.status_bar.showMessage(status)
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", f"Could not import data:\n\n{exc}")

    def _display_dataset(self, dataset) -> None:
        """Display classical data or one scalar row per MT component."""
        self.data_table.setSortingEnabled(False)
        self.data_table.clearContents()

        if isinstance(dataset, MTDataSet):
            has_uncertainties = dataset.uncertainties is not None
            headers = ["X", "Y", "Z", "Frequency (Hz)", "Component", "Value"]
            if has_uncertainties:
                headers.append("Uncertainty")
            self.data_table.setColumnCount(len(headers))
            self.data_table.setHorizontalHeaderLabels(headers)
            n_observations = dataset.n_observations
            self.data_table.setRowCount(dataset.n_data)

            # Canonical uncertainties are (n_observations, 8). The
            # MTDataSet class also accepts a legacy (n_data, 8) array;
            # its weights() method uses the first n_observations rows.
            uncertainty_table = None
            if has_uncertainties:
                uncertainty_table = np.asarray(dataset.uncertainties, dtype=float)
                if uncertainty_table.shape[0] < n_observations:
                    raise ValueError(
                        "MT uncertainty table has fewer rows than observations."
                    )

            values = dataset.values.reshape(n_observations, 8)
            for observation in range(n_observations):
                x, y, z = dataset.locations[observation]
                frequency = dataset.frequencies[observation]
                for component_index, component in enumerate(self.MT_COMPONENTS):
                    row = observation * 8 + component_index
                    row_values = [
                        x, y, z, frequency, component,
                        values[observation, component_index],
                    ]
                    for column, value in enumerate(row_values):
                        item = QTableWidgetItem(
                            str(value) if isinstance(value, str)
                            else self._format_number(value)
                        )
                        if not isinstance(value, str):
                            item.setTextAlignment(
                                Qt.AlignmentFlag.AlignRight
                                | Qt.AlignmentFlag.AlignVCenter
                            )
                        self.data_table.setItem(row, column, item)
                    if has_uncertainties:
                        uncertainty = uncertainty_table[
                            observation, component_index
                        ]
                        item = QTableWidgetItem(self._format_number(uncertainty))
                        item.setTextAlignment(
                            Qt.AlignmentFlag.AlignRight
                            | Qt.AlignmentFlag.AlignVCenter
                        )
                        self.data_table.setItem(row, 6, item)
        else:
            self._set_classical_table_headers()
            n = int(dataset.n_data)
            self.data_table.setRowCount(n)
            has_uncertainties = dataset.uncertainties is not None
            for row in range(n):
                x, y, z = dataset.locations[row]
                numbers = [x, y, z, dataset.values[row]]
                for column, number in enumerate(numbers):
                    item = QTableWidgetItem(self._format_number(number))
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                    self.data_table.setItem(row, column, item)
                uncertainty = (
                    self._format_number(dataset.uncertainties[row])
                    if has_uncertainties else "—"
                )
                item = QTableWidgetItem(uncertainty)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight
                    | Qt.AlignmentFlag.AlignVCenter
                )
                self.data_table.setItem(row, 4, item)

        self.data_table.resizeColumnsToContents()
        self.data_table.setSortingEnabled(True)

    def _clear_data_table(self) -> None:
        self.data_table.setSortingEnabled(False)
        self.data_table.clearContents()
        self.data_table.setRowCount(0)
        self._set_classical_table_headers()
        self.data_table.setSortingEnabled(True)

    def _export_data(self) -> None:
        """Export data using the importer's native ASCII table layouts."""
        dataset = self.project.dataset
        if dataset is None:
            QMessageBox.warning(self, "Export Data", "No data available.")
            return
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Data", "",
            "CSV files (*.csv);;Text files (*.txt);;DAT files (*.dat);;XYZ files (*.xyz)",
        )
        if not filename:
            return
        path = Path(filename)
        if not path.suffix:
            suffix = ".csv" if "CSV" in selected_filter else ".dat"
            path = path.with_suffix(suffix)
        delimiter = "," if path.suffix.lower() == ".csv" else " "

        if isinstance(dataset, MTDataSet):
            base = np.column_stack((
                dataset.locations,
                dataset.frequencies,
                dataset.rho_xx, dataset.rho_xy, dataset.rho_yx, dataset.rho_yy,
                dataset.phase_xx, dataset.phase_xy, dataset.phase_yx, dataset.phase_yy,
            ))
            if dataset.uncertainties is None:
                output = base
                header = (
                    "x,y,z,frequency,rho_xx,rho_xy,rho_yx,rho_yy,"
                    "phase_xx,phase_xy,phase_yx,phase_yy"
                )
            else:
                uncertainty_table = np.asarray(dataset.uncertainties, dtype=float)
                # Export the N physical observation rows. This also handles
                # the legacy (8*N, 8) storage form accepted by MTDataSet.
                uncertainty_table = uncertainty_table[:dataset.n_observations, :]
                if uncertainty_table.shape != (dataset.n_observations, 8):
                    QMessageBox.critical(
                        self, "Export Error",
                        "MT uncertainty array cannot be represented as one row "
                        "per station-frequency observation.",
                    )
                    return
                output = np.column_stack((base, uncertainty_table))
                header = (
                    "x,y,z,frequency,rho_xx,rho_xy,rho_yx,rho_yy,"
                    "phase_xx,phase_xy,phase_yx,phase_yy,"
                    "unc_rho_xx,unc_rho_xy,unc_rho_yx,unc_rho_yy,"
                    "unc_phase_xx,unc_phase_xy,unc_phase_yx,unc_phase_yy"
                )
        else:
            locations = np.asarray(dataset.locations, dtype=float)
            values = np.asarray(dataset.values, dtype=float).reshape(-1, 1)
            output = np.column_stack((locations, values))
            if dataset.uncertainties is not None:
                output = np.column_stack((
                    output, np.asarray(dataset.uncertainties, dtype=float).reshape(-1, 1)
                ))
                header = "x,y,z,value,uncertainty"
            else:
                header = "x,y,z,value"

        try:
            np.savetxt(
                path, output, delimiter=delimiter, header=header,
                comments="", fmt="%.12g",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            return
        self.status_bar.showMessage(f"Data exported: {path.name}")

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------

    def _create_mesh(self) -> None:
        if self.project.dataset is None:
            QMessageBox.warning(self, "Create Mesh", "Import data before creating a mesh.")
            return
        parameters = self._mesh_parameters_dialog()
        if parameters is None:
            return
        nx, ny, nz, dx, dy, dz = parameters
        try:
            mesh = StructuredMesh3D(
                nx=int(nx), ny=int(ny), nz=int(nz),
                dx=float(dx), dy=float(dy), dz=float(dz),
            )
            self.project.set_mesh(mesh)
            if hasattr(self.project, "clear_reference_model"):
                self.project.clear_reference_model()
            elif hasattr(self.project, "reference_model"):
                self.project.reference_model = None
            self.model_metadata = {}
            self.inversion_config["reference_model"] = False
            self.project.inversion_config = dict(self.inversion_config)
            self._display_mesh_info(mesh)
            self._update_project_tree()
            self._show_mesh_properties()
            self._update_3d_project()
            self.status_bar.showMessage(f"Mesh created: {mesh.n_cells} cells.")
        except Exception as exc:
            QMessageBox.critical(self, "Mesh Error", f"Could not create mesh:\n\n{exc}")

    def _load_mesh(self) -> None:
        """Load a structured mesh from a GeoInvLab3D ASCII mesh file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Structured Mesh", "",
            "GeoInvLab3D mesh files (*.mesh *.dat *.txt);;All Files (*)",
        )
        if not filename:
            return
        reference = getattr(self.project, "reference_model", None)
        if self.project.mesh is not None and (
            self.project.model is not None or reference is not None
        ):
            answer = QMessageBox.question(
                self, "Replace Mesh",
                "Replacing the mesh will remove the initial model, reference model, "
                "and inversion result. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            mesh = load_mesh(filename)
            self.project.set_mesh(mesh)
            # Older Project implementations do not clear the reference model.
            if hasattr(self.project, "clear_reference_model"):
                self.project.clear_reference_model()
            elif hasattr(self.project, "reference_model"):
                self.project.reference_model = None
            self.model_metadata = {}
            self.inversion_config["reference_model"] = False
            self.project.inversion_config = dict(self.inversion_config)
            self._refresh_all()
            self._show_mesh_properties()
            self.status_bar.showMessage(
                f"Mesh loaded: {mesh.n_cells} cells from {Path(filename).name}."
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Mesh Error", f"Could not load mesh:\n\n{exc}"
            )

    def _mesh_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Mesh")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        counts = []
        sizes = []
        for label in ("NX:", "NY:", "NZ:"):
            spin = QSpinBox()
            spin.setRange(1, 1000)
            spin.setValue(10)
            form.addRow(label, spin)
            counts.append(spin)
        for label in ("DX:", "DY:", "DZ:"):
            spin = QDoubleSpinBox()
            spin.setRange(1e-12, 1e9)
            spin.setDecimals(6)
            spin.setValue(1.0)
            form.addRow(label, spin)
            sizes.append(spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return tuple(spin.value() for spin in counts + sizes)

    def _display_mesh_info(self, mesh: StructuredMesh3D) -> None:
        dimensions = mesh.dimensions
        self.mesh_info.setText(
            f"<b>Structured 3D Mesh</b><br><br>Cells: {mesh.n_cells}<br>"
            f"Shape: {mesh.nx} × {mesh.ny} × {mesh.nz}<br><br>"
            f"Cell size:<br>dx = {mesh.dx}<br>dy = {mesh.dy}<br>dz = {mesh.dz}<br><br>"
            f"Physical dimensions:<br>X = {dimensions[0]}<br>"
            f"Y = {dimensions[1]}<br>Z = {dimensions[2]}<br><br>"
            f"Padding cells (x/y/z): {mesh.padding_x} / {mesh.padding_y} / {mesh.padding_z}<br>"
            f"Padding factor: {mesh.padding_factor}<br><br>"
            f"Cell volume: {mesh.cell_volume}"
        )

    def _remove_mesh(self) -> None:
        if self.project.mesh is None:
            return
        answer = QMessageBox.question(
            self, "Remove Mesh",
            "Remove the mesh?\n\nThe model and reference model will also be removed.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.project.clear_mesh()
        if hasattr(self.project, "clear_reference_model"):
            self.project.clear_reference_model()
        elif hasattr(self.project, "reference_model"):
            self.project.reference_model = None
        self.model_metadata = {}
        self.inversion_config["reference_model"] = False
        self.project.inversion_config = dict(self.inversion_config)
        self.mesh_info.setText("No mesh created.")
        self.model_info.setText("No model created.")
        self._update_project_tree()
        self._update_3d_project()

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def _create_model(self) -> None:
        mesh = self.project.mesh
        if mesh is None:
            QMessageBox.warning(self, "Create Model", "Create a mesh first.")
            return
        parameter = self._method_parameter()
        unit = self._method_definition()["unit"]
        value, ok = QInputDialog.getDouble(
            self, "Create Model", f"Initial {parameter} ({unit}):",
            self._default_model_value(), -1e12, 1e12, 8,
        )
        if not ok:
            return
        name, ok = QInputDialog.getText(
            self, "Model Name", "Physical parameter name:", text=parameter
        )
        if not ok:
            return
        try:
            model = Model3D(
                mesh=mesh, values=np.full(mesh.n_cells, value, dtype=float),
                name=name.strip() or parameter,
            )
            self.project.set_model(model)
            self.model_metadata = {
                "name": model.name,
                "unit": self._method_definition()["unit"],
                "value_count": model.n_parameters,
                "ordering": "x-y-z (z fastest)",
            }
            self._display_model_info(model)
            self._update_project_tree()
            self._show_model_properties()
            self._update_3d_project()
            self.status_bar.showMessage(f"{self._method_label()} initial model created.")
        except Exception as exc:
            QMessageBox.critical(self, "Model Error", f"Could not create model:\n\n{exc}")

    def _load_model(self) -> None:
        """Load a cell-wise physical-property model for the current mesh."""
        mesh = self.project.mesh
        if mesh is None:
            QMessageBox.warning(self, "Load Model", "Load or create a mesh first.")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Physical Model", "",
            "GeoInvLab3D model files (*.model *.dat *.txt);;All Files (*)",
        )
        if not filename:
            return
        try:
            model, metadata = load_model(
                filename, mesh, return_metadata=True
            )
            self.project.set_model(model)
            self.model_metadata = dict(metadata)
            self._display_model_info(model)
            self._update_project_tree()
            self._show_model_properties()
            self._update_3d_project()
            unit = metadata.get("unit", "")
            unit_text = f" {unit}" if unit else ""
            self.status_bar.showMessage(
                f"Model loaded: {model.n_parameters} cell values{unit_text}."
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Model Error", f"Could not load model:\n\n{exc}"
            )

    def _create_reference_model(self) -> None:
        mesh = self.project.mesh
        if mesh is None:
            QMessageBox.warning(self, "Reference Model", "Create a mesh first.")
            return
        parameter = self._method_parameter()
        unit = self._method_definition()["unit"]
        value, ok = QInputDialog.getDouble(
            self, "Reference Model", f"Reference {parameter} ({unit}):",
            self._default_model_value(), -1e12, 1e12, 8,
        )
        if not ok:
            return
        try:
            reference = Model3D(
                mesh=mesh, values=np.full(mesh.n_cells, value, dtype=float),
                name="reference",
            )
            if hasattr(self.project, "set_reference_model"):
                self.project.set_reference_model(reference)
            else:
                self.project.reference_model = reference
            self.inversion_config["reference_model"] = True
            self.project.inversion_config = dict(self.inversion_config)
            self._update_project_tree()
            self._display_inversion_info()
            self.status_bar.showMessage(f"{self._method_label()} reference model created.")
        except Exception as exc:
            QMessageBox.critical(
                self, "Reference Model Error",
                f"Could not create reference model:\n\n{exc}",
            )

    def _remove_model(self) -> None:
        if self.project.model is not None:
            self.project.clear_model()
            self.model_info.setText("No model created.")
            self._update_project_tree()
            self._update_3d_project()

    def _remove_reference_model(self) -> None:
        if hasattr(self.project, "clear_reference_model"):
            self.project.clear_reference_model()
        elif hasattr(self.project, "reference_model"):
            self.project.reference_model = None
        self.inversion_config["reference_model"] = False
        self.project.inversion_config = dict(self.inversion_config)
        self._update_project_tree()
        self._display_inversion_info()
        self.status_bar.showMessage("Reference model removed.")

    def _display_model_info(self, model: Model3D) -> None:
        parameter = self.model_metadata.get("name", self._method_parameter())
        unit = self.model_metadata.get("unit", self._method_definition()["unit"])
        self.model_info.setText(
            f"<b>{model.name}</b><br><br>Method: {self._method_label()}<br>"
            f"Parameter: {parameter}<br>Unit: {unit}<br><br>"
            f"Parameters: {model.n_parameters}<br>"
            f"Shape: {model.mesh.nx} × {model.mesh.ny} × {model.mesh.nz}<br><br>"
            f"Minimum: {model.min():.8g}<br>Maximum: {model.max():.8g}<br>"
            f"Mean: {model.mean():.8g}<br>Norm: {model.norm():.8g}"
        )

    # ------------------------------------------------------------------
    # Project tree and properties
    # ------------------------------------------------------------------

    def _update_project_tree(self) -> None:
        self.project_tree.clear()
        root = QTreeWidgetItem(self.project_tree, [self.project.name])
        root.setExpanded(True)
        dataset = self.project.dataset
        if dataset is None:
            data_label = "Data"
        elif isinstance(dataset, MTDataSet):
            data_label = f"Data ({dataset.n_observations} MT observations; {dataset.n_data} values)"
        else:
            data_label = f"Data ({dataset.n_data})"
        self.data_item = QTreeWidgetItem(root, [data_label])
        mesh = self.project.mesh
        self.mesh_item = QTreeWidgetItem(
            root, ["Mesh" if mesh is None else f"Mesh ({mesh.n_cells} cells)"]
        )
        model = self.project.model
        self.model_item = QTreeWidgetItem(
            root, ["Model" if model is None else f"Model ({model.name})"]
        )
        reference = getattr(self.project, "reference_model", None)
        self.reference_item = QTreeWidgetItem(
            root, ["Reference Model" if reference is None else "Reference Model (available)"]
        )
        self.inversion_item = QTreeWidgetItem(
            root, [f"Inversion ({self._method_label()})"]
        )
        if self._inversion_is_configured():
            QTreeWidgetItem(self.inversion_item, ["Configured"])
        if getattr(self.project, "result", None) is not None:
            QTreeWidgetItem(self.inversion_item, ["Result"])
        self.project_root = root

    def _project_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        del column
        label = item.text(0)
        if "Data" in label:
            self._show_dataset_properties()
        elif "Reference Model" in label:
            self._show_reference_model_properties()
        elif "Mesh" in label:
            self._show_mesh_properties()
        elif "Model" in label:
            self._show_model_properties()
        elif "Result" in label:
            self._show_result_properties()
        elif "Inversion" in label:
            self._show_inversion_properties()
        else:
            self._update_project_information()

    def _show_dataset_properties(self) -> None:
        dataset = self.project.dataset
        if dataset is None:
            self.properties_label.setText("No data imported.")
            return
        if isinstance(dataset, MTDataSet):
            values = dataset.values
            self.properties_label.setText(
                f"<b>Magnetotelluric Data</b><br><br>"
                f"Name: {dataset.name}<br>"
                f"Station-frequency observations: {dataset.n_observations}<br>"
                f"Frequencies: {dataset.n_frequencies}<br>"
                f"Scalar data values: {dataset.n_data}<br>"
                f"Uncertainties: {'Yes' if dataset.uncertainties is not None else 'No'}<br><br>"
                f"Frequency range: {np.min(dataset.frequencies):.8g} – "
                f"{np.max(dataset.frequencies):.8g} Hz<br>"
                f"Minimum response: {np.min(values):.8g}<br>"
                f"Maximum response: {np.max(values):.8g}<br>"
                f"Mean response: {np.mean(values):.8g}"
            )
            return
        values = np.asarray(dataset.values)
        dimension = getattr(dataset, "dimension", "—")
        units = getattr(dataset, "units", None)
        self.properties_label.setText(
            f"<b>Data</b><br><br>Name: {dataset.name}<br>"
            f"Observations: {dataset.n_data}<br>Dimension: {dimension}D<br>"
            f"Units: {units or '—'}<br>"
            f"Uncertainties: {'Yes' if dataset.uncertainties is not None else 'No'}<br><br>"
            f"Minimum: {np.min(values):.8g}<br>Maximum: {np.max(values):.8g}<br>"
            f"Mean: {np.mean(values):.8g}"
        )

    def _show_mesh_properties(self) -> None:
        mesh = self.project.mesh
        if mesh is None:
            self.properties_label.setText("No mesh created.")
            return
        self.properties_label.setText(
            f"<b>Mesh</b><br><br>Shape: {mesh.nx} × {mesh.ny} × {mesh.nz}<br>"
            f"Cells: {mesh.n_cells}<br><br>dx: {mesh.dx}<br>dy: {mesh.dy}<br>"
            f"dz: {mesh.dz}<br><br>Padding X: {mesh.padding_x}<br>"
            f"Padding Y: {mesh.padding_y}<br>Padding Z: {mesh.padding_z}<br>"
            f"Padding factor: {mesh.padding_factor}<br><br>"
            f"Cell volume: {mesh.cell_volume}"
        )

    def _show_model_properties(self) -> None:
        model = self.project.model
        if model is None:
            self.properties_label.setText("No model created.")
            return
        self.properties_label.setText(
            f"<b>Initial Model</b><br><br>Method: {self._method_label()}<br>"
            f"Parameter: {self.model_metadata.get('name', self._method_parameter())}<br>"
            f"Unit: {self.model_metadata.get('unit', self._method_definition()['unit']) or '—'}<br><br>"
            f"Name: {model.name}<br>Parameters: {model.n_parameters}<br>"
            f"Ordering: {self.model_metadata.get('ordering', 'x-y-z (z fastest)')}<br>"
            f"Shape: {model.mesh.nx} × {model.mesh.ny} × {model.mesh.nz}<br><br>"
            f"Minimum: {model.min():.8g}<br>Maximum: {model.max():.8g}<br>"
            f"Mean: {model.mean():.8g}<br>Norm: {model.norm():.8g}"
        )

    def _show_reference_model_properties(self) -> None:
        model = getattr(self.project, "reference_model", None)
        if model is None:
            self.properties_label.setText("No reference model.")
            return
        self.properties_label.setText(
            f"<b>Reference Model</b><br><br>Method: {self._method_label()}<br>"
            f"Parameter: {self._method_parameter()}<br>"
            f"Unit: {self._method_definition()['unit']}<br><br>"
            f"Parameters: {model.n_parameters}<br>Minimum: {model.min():.8g}<br>"
            f"Maximum: {model.max():.8g}<br>Mean: {model.mean():.8g}"
        )

    def _update_project_information(self) -> None:
        self.properties_label.setText(
            f"<b>Project</b><br><br>Name: {self.project.name}<br><br>"
            f"Data: {self.project.n_data}<br>Mesh cells: {self.project.n_cells}<br>"
            f"Model parameters: {self.project.n_model_parameters}<br><br>"
            f"Inversion method: {self._method_label()}<br>"
            f"Inversion configured: {'Yes' if self._inversion_is_configured() else 'No'}"
        )

    def _show_project_information(self) -> None:
        ready = getattr(self.project, "is_ready_for_inversion", False)
        if callable(ready):
            ready = ready()
        QMessageBox.information(
            self, "Project Information",
            f"<b>Project:</b> {self.project.name}<br><br>"
            f"<b>Data:</b> {self.project.n_data} values<br>"
            f"<b>Mesh:</b> {self.project.n_cells} cells<br>"
            f"<b>Model:</b> {self.project.n_model_parameters} parameters<br><br>"
            f"<b>Inversion method:</b> {self._method_label()}<br>"
            f"<b>Parameter:</b> {self._method_parameter()}<br>"
            f"<b>Inversion configured:</b> {'Yes' if self._inversion_is_configured() else 'No'}<br>"
            f"<b>Ready:</b> {'Yes' if ready else 'No'}"
        )

    # ------------------------------------------------------------------
    # Inversion configuration and execution
    # ------------------------------------------------------------------

    def _configure_inversion(self) -> None:
        if self.project.dataset is None:
            QMessageBox.warning(self, "Configure Inversion", "Import data first.")
            return
        if self.project.mesh is None:
            QMessageBox.warning(self, "Configure Inversion", "Create a mesh first.")
            return
        if self.project.model is None:
            QMessageBox.warning(self, "Configure Inversion", "Create an initial model first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Inversion")
        dialog.resize(650, 500)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        method_combo = QComboBox()
        method_combo.setMinimumWidth(350)
        for label, definition in self.INVERSION_METHODS.items():
            method_combo.addItem(label, definition["id"])
        method_combo.setCurrentIndex(
            max(0, method_combo.findData(self.inversion_config.get("method", "gravity")))
        )

        lambda_spin = QDoubleSpinBox()
        lambda_spin.setRange(0.0, 1e12)
        lambda_spin.setDecimals(12)
        lambda_spin.setSingleStep(1e-6)
        lambda_spin.setValue(float(self.inversion_config.get("lambda", self._default_lambda())))

        iterations_spin = QSpinBox()
        iterations_spin.setRange(1, 100000)
        iterations_spin.setValue(int(self.inversion_config.get("max_iterations", 100)))

        tolerance_spin = QDoubleSpinBox()
        tolerance_spin.setRange(1e-15, 1e3)
        tolerance_spin.setDecimals(12)
        tolerance_spin.setValue(float(self.inversion_config.get("tolerance", 1e-6)))

        reference_combo = QComboBox()
        reference_combo.addItem("Enabled", True)
        reference_combo.addItem("Disabled", False)
        reference_combo.setCurrentIndex(
            reference_combo.findData(bool(self.inversion_config.get("reference_model", True)))
        )

        description = QLabel()
        description.setWordWrap(True)
        description.setMinimumHeight(120)
        description.setStyleSheet(
            "padding:12px;background:#eef3f8;border:1px solid #ccd6e0;border-radius:5px;"
        )

        def update_description() -> None:
            method_id = method_combo.currentData()
            definition = next(
                (item for item in self.INVERSION_METHODS.values()
                 if item["id"] == method_id), None
            )
            if definition is None:
                return
            description.setText(
                f"<b>{method_combo.currentText()}</b><br><br>"
                f"<b>Parameter:</b> {definition['parameter']}<br>"
                f"<b>Unit:</b> {definition['unit']}<br><br>{definition['description']}"
            )

        method_combo.currentIndexChanged.connect(update_description)
        update_description()
        form.addRow("Method:", method_combo)
        form.addRow("Tikhonov λ:", lambda_spin)
        form.addRow("Maximum iterations:", iterations_spin)
        form.addRow("Tolerance:", tolerance_spin)
        form.addRow("Reference model:", reference_combo)
        layout.addLayout(form)
        layout.addWidget(description)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.inversion_config = {
            "method": method_combo.currentData(),
            "lambda": float(lambda_spin.value()),
            "max_iterations": int(iterations_spin.value()),
            "tolerance": float(tolerance_spin.value()),
            "regularization": "tikhonov",
            "reference_model": bool(reference_combo.currentData()),
        }
        self.project.inversion_config = dict(self.inversion_config)
        self._update_method_widget()
        self._display_inversion_info()
        self._update_project_tree()
        self._show_inversion_properties()
        self.status_bar.showMessage(f"{self._method_label()} inversion configured.")

    def _display_inversion_info(self) -> None:
        config = self.inversion_config
        reference = getattr(self.project, "reference_model", None)
        definition = self._method_definition()
        self.inversion_info.setText(
            f"<b>Inversion Configuration</b><br><br>"
            f"Method: {self._method_label()}<br>"
            f"Parameter: {definition['parameter']}<br>Unit: {definition['unit']}<br>"
            f"Regularization: {config.get('regularization', 'tikhonov')}<br>"
            f"Lambda: {config.get('lambda', 0.0):.8g}<br>"
            f"Maximum iterations: {config.get('max_iterations', 100)}<br>"
            f"Tolerance: {config.get('tolerance', 1e-6):.8g}<br>"
            f"Reference model: {'Available' if reference is not None else 'Not available'}"
        )
        self._update_method_widget()

    def _show_inversion_properties(self) -> None:
        config = self.inversion_config
        definition = self._method_definition()
        self.properties_label.setText(
            f"<b>Inversion</b><br><br>Method: {self._method_label()}<br>"
            f"Parameter: {definition['parameter']}<br>Unit: {definition['unit']}<br>"
            f"Regularization: {config.get('regularization', 'tikhonov')}<br>"
            f"Lambda: {config.get('lambda', 0.0):.8g}<br>"
            f"Iterations: {config.get('max_iterations', 100)}<br>"
            f"Tolerance: {config.get('tolerance', 1e-6):.8g}<br>"
            f"Reference model: {'Enabled' if config.get('reference_model', False) else 'Disabled'}"
        )

    def _run_inversion(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "Run Inversion", "No project is available.")
            return
        if self.project.dataset is None:
            QMessageBox.warning(self, "Run Inversion", "No data has been imported.")
            return
        if self.project.mesh is None:
            QMessageBox.warning(self, "Run Inversion", "No mesh has been created.")
            return
        if self.project.model is None:
            QMessageBox.warning(self, "Run Inversion", "No initial model has been created.")
            return
        if not self._inversion_is_configured():
            QMessageBox.warning(self, "Run Inversion", "Configure the inversion first.")
            return

        method = self.inversion_config["method"]
        if method not in {"gravity", "mt", "magnetic"}:
            QMessageBox.warning(self, "Run Inversion", f"Unsupported inversion method: {method}")
            return

        if self.inversion_config.get("reference_model", False):
            if getattr(self.project, "reference_model", None) is None:
                QMessageBox.warning(
                    self, "Run Inversion",
                    "A reference model is required by the current configuration. "
                    "Create a reference model first.",
                )
                return

        ready = getattr(self.project, "is_ready_for_inversion", False)
        if callable(ready):
            ready = ready()
        if not ready:
            QMessageBox.warning(
                self, "Run Inversion",
                "The project is not ready for inversion. Check data, mesh and model.",
            )
            return

        label = self._method_label()
        self.status_bar.showMessage(f"Starting {label} inversion...")
        QApplication.processEvents()
        try:
            if not hasattr(self.project, "run_inversion"):
                raise RuntimeError("Project.run_inversion() is not available.")
            self.project.inversion_config = dict(self.inversion_config)
            result = self.project.run_inversion(dict(self.inversion_config))
            self.project.result = result
            self._update_project_tree()
            self._update_project_information()
            self._display_inversion_info()
            self._update_3d_project()
            self.status_bar.showMessage(f"{label} inversion completed successfully.")
            self.tabs.setCurrentWidget(self.inversion_widget)
            QMessageBox.information(
                self, "Inversion Complete",
                f"{label} inversion completed successfully.",
            )
        except Exception as exc:
            self.status_bar.showMessage(f"{label} inversion failed.")
            QMessageBox.critical(
                self, "Inversion Error",
                f"The {label} inversion failed.\n\n{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------
    # Result and About
    # ------------------------------------------------------------------

    def _show_result_properties(self) -> None:
        result = getattr(self.project, "result", None)
        if result is None:
            self.properties_label.setText("No inversion result.")
            return
        text = (
            "<b>Inversion Result</b><br><br>"
            f"Method: {self._method_label()}<br>"
            f"Parameter: {self._method_parameter()}<br><br>"
        )
        for name in ("objective", "data_misfit", "residual_rms", "iterations"):
            if hasattr(result, name):
                text += f"{name}: {getattr(result, name)}<br>"
        if hasattr(result, "model"):
            model = result.model
            text += (
                f"<br>Model minimum: {model.min():.8g}<br>"
                f"Model maximum: {model.max():.8g}<br>"
                f"Model mean: {model.mean():.8g}<br>"
            )
        self.properties_label.setText(text)

    def _open_help(self, page: str = "index.html") -> None:
        """Open the integrated, offline GeoInvLab3D help viewer."""
        if self._help_window is None:
            self._help_window = HelpWindow(self)
        self._help_window.show_page(page)
        self._help_window.show()
        self._help_window.raise_()
        self._help_window.activateWindow()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About GeoInvLab3D",
            "<div style='min-width:360px;'>"
            "<h2 style='color:#123f7a;'>GeoInvLab3D</h2>"
            "<p><b>3-D Geophysical Modeling &amp; Inversion</b></p>"
            "<p>An integrated environment for geophysical modelling and inversion.</p>"
            "<p><b>Supported methods:</b></p>"
            "<ul>"
            "<li>Gravity</li>"
            "<li>Magnetotelluric (MT)</li>"
            "<li>Magnetic</li>"
            "</ul>"
            "<hr>"
            "<p><b>Developed by:</b><br>"
            "Dr. Boukar Abdelghafour </p>"
            "<p><b>Version:</b> 0.1.0   ( c ) MIT License 2026</p>"
            "</div>",
        )
