from __future__ import annotations

from typing import Dict, List

import numpy as np

from .common import create_dct_masks, energy_ratio
from .dct_features import dct2


def extract_block_inconsistency_features(
    gray: np.ndarray,
    block_size: int = 32,
) -> Dict[str, float]:
    """Measure local DCT high-frequency inconsistency across blocks.

    Face-swapping or compositing can create local regions with different
    frequency statistics. The standard deviation of block high-frequency
    ratios is useful for detecting this inconsistency.
    """
    h, w = gray.shape
    ratios: List[float] = []
    masks = create_dct_masks(block_size)

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = gray[y : y + block_size, x : x + block_size]
            coeff = dct2(block)
            coeff[0, 0] = 0.0

            total = float(np.sum(coeff**2)) + 1e-8
            high_ratio = energy_ratio(coeff, masks["high"], total)
            ratios.append(high_ratio)

    if not ratios:
        ratios = [0.0]

    arr = np.array(ratios, dtype=np.float32)

    return {
        "block_dct_high_ratio_mean": float(np.mean(arr)),
        "block_dct_high_ratio_std": float(np.std(arr)),
        "block_dct_high_ratio_max": float(np.max(arr)),
        "block_dct_high_ratio_min": float(np.min(arr)),
    }