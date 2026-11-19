"""
Application entry point for GeoInvLab3D.
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from geoinvlab.gui.main_window import MainWindow


def main() -> int:
    """Start GeoInvLab3D."""

    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("GeoInvLab3D")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("GeoInvLab")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

