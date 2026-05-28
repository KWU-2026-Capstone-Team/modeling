from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


def heuristic_score(features: Dict[str, float]) -> float:
    """Calculate heuristic fake score in [0, 1].

    This is not a final scientific detector.
    It is useful for prototype analysis before model training.
    """

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    dct_high = features["dct_high_energy_ratio"]
    fft_high = features["fft_high_energy_ratio"]
    centroid = features["fft_spectral_centroid"]
    block_std = features["block_dct_high_ratio_std"]
    highpass = features["highpass_std"]
    slope = features["fft_radial_slope"]

    s1 = sigmoid((dct_high - 0.20) * 10.0)
    s2 = sigmoid((fft_high - 0.25) * 10.0)
    s3 = sigmoid((centroid - 0.32) * 8.0)
    s4 = sigmoid((block_std - 0.035) * 40.0)
    s5 = sigmoid((highpass - 0.035) * 50.0)
    s6 = sigmoid((slope + 2.0) * 1.2)

    score = 0.20 * s1 + 0.20 * s2 + 0.15 * s3 + 0.20 * s4 + 0.15 * s5 + 0.10 * s6
    return float(np.clip(score, 0.0, 1.0))


def aggregate_scores(scores: np.ndarray, trim_ratio: float = 0.1) -> float:
    """Aggregate frame-level fake scores into a video-level fake score."""
    if len(scores) == 0:
        return 0.0

    scores = np.sort(scores.astype(np.float32))
    n = len(scores)
    k = int(n * trim_ratio)

    if n > 2 * k:
        scores = scores[k : n - k]

    return float(np.mean(scores))


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "video_path",
        "frame_index",
        "timestamp_sec",
        "used_face_crop",
        "heuristic_fake_score",
        "model_fake_score",
        "label",
    }

    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def apply_model_scores(df: pd.DataFrame, model) -> pd.DataFrame:
    """Add model_fake_score column using a trained classifier."""
    if model is None or df.empty:
        df["model_fake_score"] = None
        return df

    model_cols = get_model_feature_columns(df)
    x = df[model_cols]

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)[:, 1]
    else:
        raw = model.decision_function(x)
        scores = 1.0 / (1.0 + np.exp(-raw))

    df = df.copy()
    df["model_fake_score"] = scores.astype(float)
    return df