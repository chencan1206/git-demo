"""HTML report generation and run manifests."""

from __future__ import annotations

from plant_hsi.report.builder import build_report, confusion_matrix_svg, scatter_svg

__all__ = ["build_report", "confusion_matrix_svg", "scatter_svg"]
