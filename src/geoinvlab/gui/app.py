"""
GeoInvLab3D graphical application.
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

import sys

from PySide6.QtWidgets import QApplication

from geoinvlab.gui.main_window import MainWindow


def main() -> int:
    """Start the GeoInvLab3D GUI."""

    app = QApplication(sys.argv)

    app.setApplicationName(
        "GeoInvLab3D"
    )

    app.setOrganizationName(
        "GeoInvLab"
    )

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

