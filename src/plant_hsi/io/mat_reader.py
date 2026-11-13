"""MATLAB (.mat) file reader.

Supports MAT v5 (via scipy.io.loadmat) and v7.3 (via h5py). Returns the
real 3-D numeric cube variable — Zenodo Javidan files expose it as
``Day_one_1`` etc. alongside a few tiny scalar variables (none carry
wavelength metadata in the Zenodo dump).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _is_v73(path: Path) -> bool:
    hdr = path.read_bytes()[:128]
    return b"MATLAB" in hdr and b"HDF5" in hdr


def _inspect_v5(data: dict[str, Any]) -> list[tuple[str, tuple[int, ...], str]]:
    out = []
    for k, v in data.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v)
        out.append((k, tuple(arr.shape), str(arr.dtype)))
    return out


def _list_h5(root: Any, prefix: str = "") -> list[tuple[str, tuple[int, ...], str]]:
    import contextlib

    import h5py

    items: list[tuple[str, tuple[int, ...], str]] = []
    for name, obj in root.items():
        full = f"{prefix}/{name}" if prefix else name
        if isinstance(obj, h5py.Group):
            items.extend(_list_h5(obj, full))
        elif isinstance(obj, h5py.Dataset):
            with contextlib.suppress(Exception):
                items.append((full, tuple(obj.shape), str(obj.dtype)))
    return items


def inspect_mat(path: str | Path) -> list[tuple[str, tuple[int, ...], str]]:
    """Return ``(var_name, shape, dtype)`` for every non-meta variable."""
    p = Path(path)
    if _is_v73(p):
        import h5py

        with h5py.File(p, "r") as f:
            return _list_h5(f)
    from scipy.io import loadmat

    data = loadmat(str(p), squeeze_me=True, simplify_cells=True)
    return _inspect_v5(data)


def load_mat_cube(path: str | Path) -> tuple[np.ndarray, str, dict[str, np.ndarray]]:
    """Load the real 3-D numeric cube variable from ``path``.

    Returns ``(cube, var_name, other_vars)`` where ``other_vars`` contains
    every remaining variable (scalars, wavelength arrays, etc.) so the
    caller can fish out metadata.
    """
    p = Path(path)
    other: dict[str, np.ndarray] = {}

    if _is_v73(p):
        import h5py

        with h5py.File(p, "r") as f:
            all_vars = _list_h5(f)
            cube_name = _pick_cube_name(all_vars)
            if cube_name is None:
                raise ValueError(f"no 3-D numeric variable in {p}")
            # h5py stores transposed for MATLAB's column-major convention
            raw = f[cube_name][()]
            cube = _squeeze_and_transpose(raw)
            # Load everything else into other
            for name, _, _ in all_vars:
                if name == cube_name:
                    continue
                try:
                    val = np.asarray(f[name][()])
                    if val.ndim <= 2:
                        other[name.replace("/", ".")] = val
                except Exception:  # noqa: BLE001
                    pass
            varname = cube_name
    else:
        from scipy.io import loadmat

        data = loadmat(str(p), squeeze_me=True, simplify_cells=True)
        real_vars = {k: v for k, v in data.items() if not k.startswith("__")}
        all_info = [
            (k, tuple(np.asarray(v).shape), str(np.asarray(v).dtype)) for k, v in real_vars.items()
        ]
        cube_name = _pick_cube_name(all_info)
        if cube_name is None:
            raise ValueError(f"no 3-D numeric variable in {p}")
        raw = np.asarray(real_vars[cube_name])
        cube = _squeeze_and_transpose(raw)
        varname = cube_name
        for k, v in real_vars.items():
            if k == cube_name:
                continue
            arr = np.asarray(v)
            if arr.ndim <= 2:
                other[k] = arr

    return cube.astype(np.float32), varname, other


def _pick_cube_name(items: list[tuple[str, tuple[int, ...], str]]) -> str | None:
    """Among all variables in a .mat file, pick the one that looks like
    a hyperspectral cube — the first 3-D numeric variable whose total
    size dominates the file."""
    candidates = [
        (n, s, d) for (n, s, d) in items if len(s) == 3 and "u" not in d and "bool" not in d
    ]
    if not candidates:
        return None
    # Prefer the biggest — the cube dominates any MAT file.
    candidates.sort(key=lambda c: c[1][0] * c[1][1] * c[1][2], reverse=True)
    return candidates[0][0]


def _squeeze_and_transpose(arr: np.ndarray) -> np.ndarray:
    """MAT stores cubes in (band, y, x) or (y, x, band) depending on writer.

    We normalise everything to ``(y, x, band)`` by picking whichever order
    gives a plausible spatial-first layout after comparing the largest
    two axes."""
    # Handle 1-element dims
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        # Pad with a band dim
        while arr.ndim < 3:
            arr = arr[..., None]
    # Largest axis = band if (last, first, middle) arrangement. Detect via
    # the shape; we want spatial (y, x) first, band last.
    sh = arr.shape
    if sh[0] > sh[1] and sh[0] > sh[2]:
        # band is first — transpose to (y, x, band)
        arr = np.moveaxis(arr, 0, -1)
    # Else assume already (y, x, band)
    return arr
