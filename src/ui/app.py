"""Application entry helpers for launching the GUI."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from .main_window import MainWindow


def run() -> int:
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_teal.xml")
    window = MainWindow()
    window.show()
    return app.exec()


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
