from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


NON_FEATURE_COLUMNS = {
    "video_path",
    "frame_index",
    "timestamp_sec",
    "used_face_crop",
    "label",
    "label_name",
    "heuristic_fake_score",
    "model_fake_score",
}


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric frame-level feature columns for video-level aggregation."""
    feature_cols: list[str] = []

    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def build_video_level_features(
    frame_csv: str | Path,
    output_csv: str | Path,
    video_col: str = "video_path",
    label_col: str = "label",
    label_name_col: str = "label_name",
    aggregation_methods: Iterable[str] = ("mean", "std", "min", "max"),
) -> pd.DataFrame:
    """Aggregate frame-level frequency features into video-level features.

    Input:
        One row = one sampled frame.

    Output:
        One row = one video.

    Example output columns:
        dct_high_energy_ratio_mean
        dct_high_energy_ratio_std
        dct_high_energy_ratio_min
        dct_high_energy_ratio_max
    """
    frame_csv = Path(frame_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(frame_csv)

    if video_col not in df.columns:
        raise ValueError(f"Missing video column: {video_col}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    feature_cols = get_numeric_feature_columns(df)

    if not feature_cols:
        raise ValueError("No numeric feature columns found for aggregation.")

    rows: list[dict] = []

    for video_path, group in df.groupby(video_col):
        row: dict = {
            video_col: video_path,
            label_col: int(group[label_col].iloc[0]),
            "frame_count": int(len(group)),
        }

        if label_name_col in group.columns:
            row[label_name_col] = str(group[label_name_col].iloc[0])

        for col in feature_cols:
            values = group[col].astype(float)

            if "mean" in aggregation_methods:
                row[f"{col}_mean"] = float(values.mean())
            if "std" in aggregation_methods:
                # std can be NaN if only one frame exists; fill with 0.
                row[f"{col}_std"] = float(values.std(ddof=0))
            if "min" in aggregation_methods:
                row[f"{col}_min"] = float(values.min())
            if "max" in aggregation_methods:
                row[f"{col}_max"] = float(values.max())
            if "median" in aggregation_methods:
                row[f"{col}_median"] = float(values.median())

        rows.append(row)

    video_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    video_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return video_df