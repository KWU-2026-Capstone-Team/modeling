from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_analysis_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    x = df["timestamp_sec"]

    plt.figure(figsize=(12, 5))
    plt.plot(x, df["heuristic_fake_score"], label="Heuristic fake score")

    if "model_fake_score" in df.columns and df["model_fake_score"].notna().any():
        plt.plot(x, df["model_fake_score"], label="Model fake score")

    plt.xlabel("Time (sec)")
    plt.ylabel("Fake score")
    plt.ylim(0, 1)
    plt.title("Frame-level fake score over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fake_score_timeline.png", dpi=160)
    plt.close()

    selected_cols = [
        "dct_high_energy_ratio",
        "fft_high_energy_ratio",
        "fft_spectral_centroid",
        "block_dct_high_ratio_std",
        "highpass_std",
    ]
    available = [c for c in selected_cols if c in df.columns]

    plt.figure(figsize=(12, 6))
    for col in available:
        values = df[col].to_numpy(dtype=np.float32)
        if values.max() > values.min():
            values = (values - values.min()) / (values.max() - values.min())
        plt.plot(x, values, label=col)

    plt.xlabel("Time (sec)")
    plt.ylabel("Min-max normalized feature value")
    plt.title("Key frequency features over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "frequency_feature_timeline.png", dpi=160)
    plt.close()

    corr_cols = [
        c
        for c in selected_cols + ["heuristic_fake_score", "model_fake_score"]
        if c in df.columns
    ]
    corr_df = df[corr_cols].dropna(axis=1, how="all")

    if corr_df.shape[1] >= 2:
        corr = corr_df.corr(numeric_only=True)

        plt.figure(figsize=(8, 6))
        plt.imshow(corr, interpolation="nearest")
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Feature correlation heatmap")
        plt.tight_layout()
        plt.savefig(out_dir / "feature_correlation.png", dpi=160)
        plt.close()