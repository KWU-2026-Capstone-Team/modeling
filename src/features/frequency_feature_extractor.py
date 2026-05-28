from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.face_crop import FaceCropper
from features.block_frequency_features import extract_block_inconsistency_features
from features.common import resize_gray
from features.dct_features import extract_dct_features
from features.fft_features import extract_fft_features
from features.texture_features import extract_noise_features
from inference.scoring import heuristic_score


@dataclass
class FrameFrequencyFeatures:
    video_path: str
    frame_index: int
    timestamp_sec: float
    used_face_crop: bool

    dct_high_energy_ratio: float
    dct_mid_energy_ratio: float
    dct_low_energy_ratio: float
    dct_high_to_low_ratio: float

    fft_high_energy_ratio: float
    fft_mid_energy_ratio: float
    fft_low_energy_ratio: float
    fft_spectral_centroid: float
    fft_spectral_spread: float
    fft_radial_slope: float

    laplacian_var: float
    highpass_mean_abs: float
    highpass_std: float

    block_dct_high_ratio_mean: float
    block_dct_high_ratio_std: float
    block_dct_high_ratio_max: float
    block_dct_high_ratio_min: float

    heuristic_fake_score: float
    model_fake_score: Optional[float] = None


def extract_frame_frequency_features(
    frame_bgr: np.ndarray,
    video_path: str,
    frame_index: int,
    timestamp_sec: float,
    face_cropper: Optional[FaceCropper] = None,
    image_size: int = 256,
) -> FrameFrequencyFeatures:
    used_face_crop = False
    target = frame_bgr

    if face_cropper is not None:
        target, used_face_crop = face_cropper.crop_largest_face(frame_bgr)

    gray = resize_gray(target, size=image_size)

    feature_dict = {}
    feature_dict.update(extract_dct_features(gray))
    feature_dict.update(extract_fft_features(gray))
    feature_dict.update(extract_noise_features(gray))
    feature_dict.update(extract_block_inconsistency_features(gray, block_size=32))

    feature_dict["heuristic_fake_score"] = heuristic_score(feature_dict)

    return FrameFrequencyFeatures(
        video_path=video_path,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        used_face_crop=used_face_crop,
        dct_high_energy_ratio=feature_dict["dct_high_energy_ratio"],
        dct_mid_energy_ratio=feature_dict["dct_mid_energy_ratio"],
        dct_low_energy_ratio=feature_dict["dct_low_energy_ratio"],
        dct_high_to_low_ratio=feature_dict["dct_high_to_low_ratio"],
        fft_high_energy_ratio=feature_dict["fft_high_energy_ratio"],
        fft_mid_energy_ratio=feature_dict["fft_mid_energy_ratio"],
        fft_low_energy_ratio=feature_dict["fft_low_energy_ratio"],
        fft_spectral_centroid=feature_dict["fft_spectral_centroid"],
        fft_spectral_spread=feature_dict["fft_spectral_spread"],
        fft_radial_slope=feature_dict["fft_radial_slope"],
        laplacian_var=feature_dict["laplacian_var"],
        highpass_mean_abs=feature_dict["highpass_mean_abs"],
        highpass_std=feature_dict["highpass_std"],
        block_dct_high_ratio_mean=feature_dict["block_dct_high_ratio_mean"],
        block_dct_high_ratio_std=feature_dict["block_dct_high_ratio_std"],
        block_dct_high_ratio_max=feature_dict["block_dct_high_ratio_max"],
        block_dct_high_ratio_min=feature_dict["block_dct_high_ratio_min"],
        heuristic_fake_score=feature_dict["heuristic_fake_score"],
    )