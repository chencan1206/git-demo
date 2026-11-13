"""Stage-3 preprocessing tests (acceptance criteria 1-5)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.signal import savgol_filter

from plant_hsi.preprocess import (
    PreprocessChain,
    PreprocessOp,
    fractional_derivative,
    integer_derivative,
    mean_center,
    min_max,
    moving_average,
    msc,
    savitzky_golay,
    snv,
    standardize,
    vector_normalize,
)


def _spectra(n: int = 12, bands: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.linspace(0.2, 0.8, n * bands, dtype=np.float64).reshape(n, bands) + rng.normal(
        0, 0.01, (n, bands)
    )


# ---------------------------------------------------------------- FOD (ac1)
def test_fod_order_one_matches_first_difference() -> None:
    x = _spectra(5, 30, seed=1)
    fod = fractional_derivative(x, order=1.0, memory=8)
    np.testing.assert_allclose(fod[:, 1:], np.diff(x, axis=1), atol=1e-6)


def test_fod_order_zero_is_identity() -> None:
    x = _spectra(4, 25, seed=2)
    np.testing.assert_allclose(fractional_derivative(x, order=0.0), x, atol=1e-6)


def test_fod_integer_derivative_agrees_and_scales_with_delta() -> None:
    x = _spectra(3, 20, seed=3)
    np.testing.assert_allclose(
        integer_derivative(x, order=1), fractional_derivative(x, order=1.0), atol=1e-6
    )
    np.testing.assert_allclose(
        fractional_derivative(x, order=1.0, delta=2.0)[:, 1:],
        np.diff(x, axis=1) / 2.0,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="order must be >= 0"):
        fractional_derivative(x, order=-0.5)


# ------------------------------------------------------- SNV/MSC/SG (ac2)
def test_snv_matches_reference_formula() -> None:
    x = _spectra(6, 15, seed=4)
    expected = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    np.testing.assert_allclose(snv(x), expected, atol=2e-6)


def test_msc_matches_reference_regression() -> None:
    x = _spectra(5, 18, seed=5)
    ref = x.mean(axis=0)
    ref = ref - ref.mean()
    design = np.column_stack([ref, np.ones_like(ref)])
    expected = np.empty_like(x)
    for i, row in enumerate(x):
        a, b = np.linalg.lstsq(design, row, rcond=None)[0]
        expected[i] = (row - b) / a
    np.testing.assert_allclose(msc(x), expected, atol=2e-5)


def test_sg_reproduces_quadratic_and_matches_scipy() -> None:
    bands = 31
    t = np.linspace(-1, 1, bands)
    quad = np.tile(1.0 + 2.0 * t + 3.0 * t**2, (4, 1))
    np.testing.assert_allclose(savitzky_golay(quad, window=7, polyorder=2), quad, atol=1e-5)

    x = _spectra(4, bands, seed=6)
    expected = savgol_filter(x, 7, 2, axis=1)
    np.testing.assert_allclose(savitzky_golay(x, window=7, polyorder=2), expected, atol=1e-6)


def test_moving_average_preserves_shape_and_constant() -> None:
    x = np.full((3, 11), 0.42)
    np.testing.assert_allclose(moving_average(x, window=5), x, atol=1e-6)
    assert moving_average(_spectra(2, 9), window=4).shape == (2, 9)


def test_normalisation_transforms() -> None:
    x = _spectra(6, 12, seed=7)
    centered = mean_center(x)
    np.testing.assert_allclose(centered.mean(axis=0), 0, atol=1e-5)

    z = standardize(x)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-4)

    mm = min_max(x)
    assert float(mm.min()) >= -1e-6
    assert float(mm.max()) <= 1.0 + 1e-6

    vn = vector_normalize(x)
    np.testing.assert_allclose(np.linalg.norm(vn, axis=1), 1, atol=1e-5)


# --------------------------------------------------------- chain (ac3, ac5)
def test_chain_order_changes_result_and_round_trips() -> None:
    x = _spectra(8, 25, seed=8)
    a = PreprocessChain([PreprocessOp("sg", {"window": 7}), PreprocessOp("standardize")])
    b = PreprocessChain([PreprocessOp("standardize"), PreprocessOp("sg", {"window": 7})])

    assert not np.allclose(a.apply(x), b.apply(x))

    snapshot = a.to_dict()
    restored = PreprocessChain.from_dict(snapshot)
    assert restored.describe() == a.describe()
    np.testing.assert_allclose(restored.apply(x), a.apply(x), atol=1e-6)


def test_chain_editing_and_param_validation() -> None:
    chain = PreprocessChain()
    chain.add("snv")
    chain.add("sg", window=7, polyorder=2)
    chain.add("vector_normalize")
    assert len(chain) == 3

    new_idx = chain.move(2, -1)
    assert new_idx == 1
    assert [op.kind for op in chain] == ["snv", "vector_normalize", "sg"]

    removed = chain.remove(0)
    assert removed.kind == "snv"
    chain.clear()
    assert len(chain) == 0

    with pytest.raises(ValueError, match="unknown preprocess kind"):
        PreprocessChain().add("nope")
    with pytest.raises(ValueError, match="unknown parameter"):
        PreprocessChain().add("snv", bogus=1)


def test_chain_records_out_of_range_without_clipping() -> None:
    x = _spectra(5, 10, seed=9)
    chain = PreprocessChain([PreprocessOp("mean_center")])
    out, report = chain.apply_with_report(x)

    assert report.ok
    assert report.nan_after == 0 and report.inf_after == 0
    assert report.out_of_range_before == 0
    assert report.out_of_range_after > 0
    assert float(out.min()) < 0  # values are recorded, not silently clipped

    assert chain.apply(x).dtype == np.float32
    with pytest.raises(ValueError, match="expected 2-D"):
        chain.apply(np.zeros(10))


def test_chain_performance_on_1000x300() -> None:
    rng = np.random.default_rng(10)
    x = rng.random((1000, 300)).astype(np.float32)
    chain = PreprocessChain(
        [
            PreprocessOp("snv"),
            PreprocessOp("msc"),
            PreprocessOp("sg", {"window": 11, "polyorder": 2}),
            PreprocessOp("fractional_derivative", {"order": 0.8, "memory": 32}),
            PreprocessOp("vector_normalize"),
        ]
    )

    start = time.perf_counter()
    out = chain.apply(x)
    elapsed = time.perf_counter() - start

    assert out.shape == x.shape
    assert np.isfinite(out).all()
    assert elapsed < 2.0, f"chain took {elapsed:.2f}s (> 2 s)"
