"""Stage-3 feature tests: PCA and SPA band selection."""

from __future__ import annotations

import numpy as np
import pytest

from plant_hsi.features import pca, spa


def test_pca_reconstructs_rank_two_data() -> None:
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(60, 2))
    basis = rng.normal(size=(2, 30))
    spectra = latent @ basis + 0.3

    result = pca(spectra, n_components=2)

    assert result.scores.shape == (60, 2)
    assert result.components.shape == (2, 30)
    assert result.explained_variance_ratio.sum() <= 1.0 + 1e-6
    assert result.explained_variance_ratio.sum() > 0.999

    np.testing.assert_allclose(result.components @ result.components.T, np.eye(2), atol=1e-5)
    recon = result.scores @ result.components + result.mean
    np.testing.assert_allclose(recon, spectra, atol=2e-4)
    np.testing.assert_allclose(result.transform(spectra), result.scores, atol=1e-4)


def test_pca_component_count_is_bounded() -> None:
    spectra = np.random.default_rng(1).normal(size=(8, 12))
    default = pca(spectra)
    assert 1 <= len(default.explained_variance_ratio) <= 8

    with pytest.raises(ValueError, match="exceeds maximum"):
        pca(spectra, n_components=50)

    with pytest.raises(ValueError, match="expected 2-D"):
        pca(np.zeros(12))


def test_spa_selects_informative_bands() -> None:
    rng = np.random.default_rng(2)
    n, bands = 200, 40
    spectra = 0.05 * rng.normal(size=(n, bands))
    spectra[:, 5] = rng.normal(size=n)
    spectra[:, 17] = rng.normal(size=n)
    y = 3.0 * spectra[:, 5] - 2.0 * spectra[:, 17]

    selected = spa(spectra, y, n_select=2)

    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert all(0 <= i < bands for i in selected)
    assert set(selected) == {5, 17}


def test_spa_unsupervised_and_validation() -> None:
    rng = np.random.default_rng(3)
    spectra = rng.normal(size=(30, 25))
    spectra[:, 9] *= 5.0

    sel = spa(spectra, n_select=4)
    assert sel[0] == 9
    assert len(set(sel)) == 4

    assert len(spa(spectra, n_select=100)) == 25
    with pytest.raises(ValueError, match="n_select must be >= 1"):
        spa(spectra, n_select=0)
    with pytest.raises(ValueError, match="y length"):
        spa(spectra, np.zeros(3), n_select=2)
