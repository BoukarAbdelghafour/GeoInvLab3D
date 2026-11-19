"""Integrated offline help window for GeoInvLab3D."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
)


class HelpWindow(QDialog):
    """Searchable help browser backed by HTML files shipped with the package."""

    PAGES = (
        ("Overview", "index.html"),
        ("Quick Start", "quick_start.html"),
        ("General Workflow", "workflow.html"),
        ("Gravity", "gravity.html"),
        ("Magnetic", "magnetic.html"),
        ("Magnetotelluric (MT)", "magnetotelluric.html"),
        ("Data File Formats", "data_formats.html"),
        ("Troubleshooting", "troubleshooting.html"),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("GeoInvLab3D Help")
        self.resize(1100, 720)
        self._content_dir = Path(__file__).resolve().parent / "content"

        layout = QVBoxLayout(self)
        heading = QLabel("GeoInvLab3D Help")
        heading.setObjectName("helpHeading")
        heading.setStyleSheet(
            "font-size:20px;font-weight:700;color:#174D35;padding:5px;"
        )
        layout.addWidget(heading)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search the user manual...")
        self.search_edit.textChanged.connect(self._filter_pages)
        layout.addWidget(self.search_edit)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.page_list = QListWidget()
        self.page_list.setMinimumWidth(210)
        for title, filename in self.PAGES:
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, filename)
            self.page_list.addItem(item)
        self.page_list.currentItemChanged.connect(self._on_page_selected)
        splitter.addWidget(self.page_list)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setOpenLinks(True)
        self.browser.setSearchPaths([str(self._content_dir)])
        self.browser.setStyleSheet(
            "QTextBrowser{background:#ffffff;color:#26352c;"
            "border:1px solid #d7e4d9;padding:12px;}"
        )
        splitter.addWidget(self.browser)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        self.page_list.setCurrentRow(0)

    def _filter_pages(self, text: str) -> None:
        query = text.strip().casefold()
        for index in range(self.page_list.count()):
            item = self.page_list.item(index)
            item.setHidden(query not in item.text().casefold())

    def _on_page_selected(self, current, previous=None) -> None:
        del previous
        if current is None:
            return
        filename = current.data(Qt.ItemDataRole.UserRole)
        self._load_page(filename)

    def _load_page(self, filename: str) -> None:
        path = (self._content_dir / filename).resolve()
        if path.parent != self._content_dir.resolve() or not path.is_file():
            self.browser.setHtml(
                "<h2>Help page unavailable</h2>"
                "<p>The requested help page could not be found.</p>"
            )
            return
        self.browser.setSource(QUrl.fromLocalFile(str(path)))

    def show_page(self, filename: str = "index.html") -> None:
        """Display a named help page and select it in the navigation list."""
        self.search_edit.clear()
        for index in range(self.page_list.count()):
            item = self.page_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == filename:
                self.page_list.setCurrentRow(index)
                self._load_page(filename)
                return
        self._load_page("index.html")
