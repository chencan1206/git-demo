"""Entry point: ``python -m plant_hsi`` or the ``phs`` console script."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from plant_hsi import __version__
from plant_hsi.core.config import load_config
from plant_hsi.core.logging_setup import setup_logging


def main() -> int:
    cfg = load_config()
    log = setup_logging(Path.home() / ".config" / "plant_hsi" / "logs", level=cfg.log_level)
    log.info("PHS Studio starting v%s", __version__)

    # Headless check: if Qt is not installed, exit gracefully.
    try:
        from PySide6.QtWidgets import QApplication  # noqa: WPS433 — optional dep
    except ImportError:
        log.warning("PySide6 is not installed; running headless.")
        print(f"PHS v{__version__} (headless) — install PySide6 to launch the GUI.")
        return 0

    # If DISPLAY is missing but Qt is present, switch to offscreen platform so
    # the app loads without crashing (useful for CI and smoke tests).
    _headless_platform = not os.environ.get("DISPLAY")
    if _headless_platform and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QApplication(sys.argv)
    app.setApplicationName("PHS Studio")
    app.setOrganizationName("PHS")

    from plant_hsi.ui.main_window import MainWindow  # noqa: WPS433

    window = MainWindow()
    window.show()

    # Offscreen mode (no real display) → quit immediately after construction
    # so the process can be used as a smoke test.
    if _headless_platform:
        log.info("offscreen mode — MainWindow instantiated, quitting cleanly")
        window.close()
        app.quit()
        return 0

    rc: int = int(app.exec())
    log.info("PHS Studio exiting with rc=%s", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
