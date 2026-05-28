from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.fftpack import dct

from .common import create_dct_masks, energy_ratio, safe_ratio


def dct2(image: np.ndarray) -> np.ndarray:
    """2D DCT with orthogonal normalization."""
    return dct(dct(image.T, norm="ortho").T, norm="ortho")


def extract_dct_features(gray: np.ndarray) -> Dict[str, float]:
    coeff = dct2(gray)

    coeff_no_dc = coeff.copy()
    coeff_no_dc[0, 0] = 0.0

    total_energy = float(np.sum(coeff_no_dc**2)) + 1e-8
    masks = create_dct_masks(gray.shape[0])

    low = energy_ratio(coeff_no_dc, masks["low"], total_energy)
    mid = energy_ratio(coeff_no_dc, masks["mid"], total_energy)
    high = energy_ratio(coeff_no_dc, masks["high"], total_energy)

    return {
        "dct_low_energy_ratio": low,
        "dct_mid_energy_ratio": mid,
        "dct_high_energy_ratio": high,
        "dct_high_to_low_ratio": safe_ratio(high, low),
    }