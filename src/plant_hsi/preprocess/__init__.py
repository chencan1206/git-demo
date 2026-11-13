"""Spectral preprocessing: scatter correction, smoothing, derivatives and
normalisation, plus a serialisable chain.

All functions are pure ndarray -> ndarray transforms so they can be reused by
the GUI, the Step engine and tests alike.
"""

from __future__ import annotations

from plant_hsi.preprocess.chain import (
    OPS,
    ChainReport,
    PreprocessChain,
    PreprocessOp,
)
from plant_hsi.preprocess.derivative import fractional_derivative, integer_derivative
from plant_hsi.preprocess.normalize import (
    mean_center,
    min_max,
    standardize,
    vector_normalize,
)
from plant_hsi.preprocess.scatter import msc, snv
from plant_hsi.preprocess.smoothing import moving_average, savitzky_golay

__all__ = [
    "OPS",
    "ChainReport",
    "PreprocessChain",
    "PreprocessOp",
    "fractional_derivative",
    "integer_derivative",
    "mean_center",
    "min_max",
    "moving_average",
    "msc",
    "savitzky_golay",
    "snv",
    "standardize",
    "vector_normalize",
]
