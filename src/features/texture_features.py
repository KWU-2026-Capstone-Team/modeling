from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def extract_noise_features(gray: np.ndarray) -> Dict[str, float]:
    """Extract high-frequency texture/noise features."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    laplacian_var = float(np.var(lap))

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    highpass = gray - blurred

    return {
        "laplacian_var": laplacian_var,
        "highpass_mean_abs": float(np.mean(np.abs(highpass))),
        "highpass_std": float(np.std(highpass)),
    }