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

    # End-to-end pipeline check: pure computation, no Qt involved, so it can
    # run even when the GUI cannot start (headless servers, broken Qt plugins).
    if "--e2e" in sys.argv:
        import json

        from plant_hsi import selftest  # noqa: WPS433

        payload: dict[str, object] = {"ok": False}
        try:
            payload.update(selftest.run())
            payload["ok"] = True
            print("e2e OK samples={samples} report_bytes={report_bytes}".format(**payload))
        except Exception as exc:  # noqa: BLE001 — surface the real cause
            import traceback

            traceback.print_exc()
            payload["error"] = f"{type(exc).__name__}: {exc}"
            print(f"e2e FAILED: {payload['error']}")
        out = os.environ.get("PHS_E2E_OUT")
        if out:
            Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0 if payload["ok"] else 1

    # Headless check: if Qt is not installed, exit gracefully.
    try:
        from PySide6.QtWidgets import QApplication  # noqa: WPS433 — optional dep
    except ImportError:
        log.warning("PySide6 is not installed; running headless.")
        print(f"PHS v{__version__} (headless) — install PySide6 to launch the GUI.")
        return 0

    # Headless fallback: only Linux has DISPLAY/WAYLAND_DISPLAY. Windows and
    # macOS never set them, so testing them here would make the frozen app
    # construct the window and quit immediately (looks like a silent crash).
    _linux = sys.platform.startswith("linux")
    _self_test = "--self-test" in sys.argv
    _headless_platform = _self_test or (
        _linux
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    )
    if _headless_platform and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QApplication(sys.argv)
    app.setApplicationName("PHS Studio")
    app.setOrganizationName("PHS")

    from plant_hsi.ui.main_window import MainWindow  # noqa: WPS433

    try:
        window = MainWindow()
        window.show()
    except Exception as exc:  # noqa: BLE001 — surface startup failures, never die silently
        import traceback

        log.exception("startup failed")
        crash_log = Path.home() / ".config" / "plant_hsi" / "crash.log"
        crash_log.parent.mkdir(parents=True, exist_ok=True)
        crash_log.write_text(traceback.format_exc(), encoding="utf-8")
        try:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(
                None,
                "PHS Studio 启动失败",
                f"{type(exc).__name__}: {exc}\n\n详细堆栈已写入:\n{crash_log}",
            )
        except Exception:  # noqa: BLE001 — dialog is best effort
            pass
        return 1

    # Offscreen mode (no real display) → quit immediately after construction
    # so the process can be used as a smoke test.
    if _headless_platform:
        log.info("offscreen mode — MainWindow instantiated, quitting cleanly")
        print(f"PHS Studio v{__version__} self-test OK (offscreen)")
        window.close()
        app.quit()
        return 0

    rc: int = int(app.exec())
    log.info("PHS Studio exiting with rc=%s", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
