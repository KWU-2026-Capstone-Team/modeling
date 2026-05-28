from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def resize_gray(frame_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    """Convert BGR image to normalized grayscale square image."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def safe_ratio(a: float, b: float, eps: float = 1e-8) -> float:
    return float(a / (b + eps))


def create_dct_masks(size: int) -> Dict[str, np.ndarray]:
    """Create low/mid/high masks for DCT coefficient matrix.

    In DCT, low-frequency coefficients are near the top-left corner.
    """
    y, x = np.ogrid[:size, :size]
    r = np.sqrt(x**2 + y**2)
    r_norm = r / r.max()

    return {
        "low": r_norm <= 0.15,
        "mid": (r_norm > 0.15) & (r_norm <= 0.45),
        "high": r_norm > 0.45,
    }


def create_frequency_masks(size: int) -> Dict[str, np.ndarray]:
    """Create low/mid/high radial masks for centered FFT spectrum."""
    y, x = np.ogrid[:size, :size]
    center = (size - 1) / 2.0
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    r_norm = r / r.max()

    return {
        "low": r_norm <= 0.15,
        "mid": (r_norm > 0.15) & (r_norm <= 0.45),
        "high": r_norm > 0.45,
    }


def energy_ratio(coeff: np.ndarray, mask: np.ndarray, total_energy: float) -> float:
    return safe_ratio(float(np.sum(coeff[mask] ** 2)), total_energy)