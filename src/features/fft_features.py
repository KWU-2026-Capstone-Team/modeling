from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .common import create_frequency_masks, energy_ratio


def radial_profile(power_spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial average of a centered 2D power spectrum."""
    h, w = power_spectrum.shape
    y, x = np.indices((h, w))
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])

    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(np.int32)

    radial_sum = np.bincount(r.ravel(), weights=power_spectrum.ravel())
    radial_count = np.bincount(r.ravel())
    radial_mean = radial_sum / np.maximum(radial_count, 1)
    radii = np.arange(len(radial_mean))

    return radii, radial_mean


def extract_fft_features(gray: np.ndarray) -> Dict[str, float]:
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2
    total_energy = float(np.sum(power)) + 1e-8

    masks = create_frequency_masks(gray.shape[0])

    low = energy_ratio(power, masks["low"], total_energy)
    mid = energy_ratio(power, masks["mid"], total_energy)
    high = energy_ratio(power, masks["high"], total_energy)

    radii, radial_mean = radial_profile(power)
    radial_energy = radial_mean + 1e-8
    radius_norm = radii / max(float(radii.max()), 1.0)

    spectral_centroid = float(np.sum(radius_norm * radial_energy) / np.sum(radial_energy))
    spectral_spread = float(
        np.sqrt(
            np.sum(((radius_norm - spectral_centroid) ** 2) * radial_energy)
            / np.sum(radial_energy)
        )
    )

    valid = (radii > 2) & (radial_energy > 0)
    if np.sum(valid) >= 8:
        x = np.log(radii[valid].astype(np.float32))
        y = np.log(radial_energy[valid].astype(np.float32))
        radial_slope = float(np.polyfit(x, y, deg=1)[0])
    else:
        radial_slope = 0.0

    return {
        "fft_low_energy_ratio": low,
        "fft_mid_energy_ratio": mid,
        "fft_high_energy_ratio": high,
        "fft_spectral_centroid": spectral_centroid,
        "fft_spectral_spread": spectral_spread,
        "fft_radial_slope": radial_slope,
    }