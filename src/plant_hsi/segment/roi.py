"""ROI data structures and CSV persistence."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from plant_hsi.segment.core import ComponentResult


@dataclass
class ROI:
    """One labelled region of interest."""

    id: int  # 1-based stable ID
    area: int  # pixel count
    col_centroid: float  # x (column) centroid
    row_centroid: float  # y (row) centroid
    bbox: tuple[int, int, int, int]  # (left, top, width, height)
    sort_key: float | int | None = None  # user-defined override for ordering
    note: str = ""


@dataclass
class ROISet:
    """Collection of ROIs + supporting data."""

    rois: list[ROI] = field(default_factory=list)
    labels: NDArray[np.int32] | None = None  # full label map (H, W)
    source_shape: tuple[int, int] | None = None  # original (H, W)
    method: str = ""  # e.g. "ra_otsu"

    # ----- construct from core outputs -------------------------------------
    @classmethod
    def from_components(
        cls,
        comp: ComponentResult,
        *,
        sort_by: str = "area_desc",
        labels: NDArray[np.int32] | None = None,
        method: str = "",
    ) -> ROISet:
        rois: list[ROI] = []
        for i in range(comp.n_components):
            col, row = comp.centroids[i]
            bb_vals = tuple(int(v) for v in comp.bboxes[i])
            assert len(bb_vals) == 4
            bb: tuple[int, int, int, int] = (bb_vals[0], bb_vals[1], bb_vals[2], bb_vals[3])
            rois.append(
                ROI(
                    id=i + 1,
                    area=int(comp.areas[i]),
                    col_centroid=float(col),
                    row_centroid=float(row),
                    bbox=bb,
                )
            )
        rois = _sort_rois(rois, sort_by)
        # Renumber IDs to match sorted order but also keep sort_key
        for new_idx, r in enumerate(rois):
            r.sort_key = new_idx + 1
        h = labels.shape[0] if labels is not None else 0
        w = labels.shape[1] if labels is not None else 0
        return cls(rois=rois, labels=labels, source_shape=(h, w), method=method)

    # ----- access helpers ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.rois)

    def __iter__(self) -> Iterator[ROI]:
        return iter(self.rois)

    def get(self, roi_id: int) -> ROI | None:
        for r in self.rois:
            if r.id == roi_id:
                return r
        return None

    # ----- persistence ------------------------------------------------------
    def to_dataframe(
        self,
        *,
        mean_spectra: NDArray[np.floating] | None = None,
        wavelength_nm: list[float] | None = None,
    ) -> pd.DataFrame:
        rows = []
        for r in self.rois:
            rows.append(
                {
                    "label": r.id,
                    "sort_order": r.sort_key or r.id,
                    "area_px": r.area,
                    "row_centroid": r.row_centroid,
                    "col_centroid": r.col_centroid,
                    "bbox_left": r.bbox[0],
                    "bbox_top": r.bbox[1],
                    "bbox_w": r.bbox[2],
                    "bbox_h": r.bbox[3],
                    "note": r.note,
                }
            )
        df = pd.DataFrame(rows)

        # Optionally append per-band mean spectra (pd.concat avoids the
        # "highly fragmented DataFrame" warning from looped df[col] = ...).
        if mean_spectra is not None:
            ms = np.asarray(mean_spectra, dtype=np.float64)
            if ms.shape[0] != len(self.rois):
                raise ValueError(f"mean_spectra rows ({ms.shape[0]}) != n_rois ({len(self.rois)})")
            bands = ms.shape[1]
            if wavelength_nm is not None and len(wavelength_nm) == bands:
                band_labels = [f"mean_{wl:.1f}nm" for wl in wavelength_nm]
            else:
                band_labels = [f"mean_band_{i}" for i in range(bands)]
            spec_df = pd.DataFrame(ms, columns=band_labels)
            df = pd.concat([df, spec_df.reset_index(drop=True)], axis=1)
        return df

    def write_csv(
        self,
        path: Path | str,
        *,
        mean_spectra: NDArray[np.floating] | None = None,
        wavelength_nm: list[float] | None = None,
    ) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe(mean_spectra=mean_spectra, wavelength_nm=wavelength_nm)
        df.to_csv(out, index=False)
        return out


# -----------------------------------------------------------------------------
# Sort helpers
# -----------------------------------------------------------------------------


def _sort_rois(rois: list[ROI], by: str) -> list[ROI]:
    if by == "area_desc":
        return sorted(rois, key=lambda r: -r.area)
    if by == "area_asc":
        return sorted(rois, key=lambda r: r.area)
    if by == "row_col":  # raster order: top to bottom, left to right
        return sorted(rois, key=lambda r: (r.row_centroid, r.col_centroid))
    if by == "col_row":
        return sorted(rois, key=lambda r: (r.col_centroid, r.row_centroid))
    if by == "id":
        return sorted(rois, key=lambda r: r.id)
    raise ValueError(f"unknown sort {by!r}; expected area_desc|area_asc|row_col|col_row|id")


def compute_mean_spectra(
    cube: NDArray[np.floating],
    labels: NDArray[np.integer],
    roi_ids: Iterable[int],
) -> NDArray[np.float32]:
    """Return ``(n_rois, bands)`` mean spectrum matrix."""
    ids = list(roi_ids)
    h, w, bands = cube.shape
    out = np.zeros((len(ids), bands), dtype=np.float32)
    for i, rid in enumerate(ids):
        mask = labels == rid
        if mask.any():
            out[i] = cube[mask].mean(axis=0).astype(np.float32)
    return out
