from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from inference.scoring import get_model_feature_columns


def aggregate_video_scores(
    df: pd.DataFrame,
    score_col: str = "model_fake_score",
    label_col: str = "label",
    video_col: str = "video_path",
) -> pd.DataFrame:
    rows = []

    for video_path, group in df.groupby(video_col):
        video_score = float(group[score_col].mean())
        video_label = int(group[label_col].iloc[0])

        rows.append(
            {
                "video_path": video_path,
                "label": video_label,
                "video_fake_score": video_score,
                "frame_count": int(len(group)),
            }
        )

    return pd.DataFrame(rows)


def evaluate_video_level(
    feature_csv: str,
    model_path: str,
    output_json: str,
    threshold: float = 0.5,
    label_col: str = "label",
) -> dict:
    df = pd.read_csv(feature_csv)
    model = joblib.load(model_path)

    feature_cols = get_model_feature_columns(df)
    x = df[feature_cols]

    df = df.copy()
    df["model_fake_score"] = model.predict_proba(x)[:, 1]

    video_df = aggregate_video_scores(
        df,
        score_col="model_fake_score",
        label_col=label_col,
        video_col="video_path",
    )

    y_true = video_df["label"].astype(int)
    y_score = video_df["video_fake_score"].astype(float)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "video_count": int(len(video_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0,
        ),
    }

    if len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== Video-level evaluation ===")
    print(metrics["classification_report"])

    if "roc_auc" in metrics:
        print(f"Video-level ROC-AUC: {metrics['roc_auc']:.4f}")

    print(f"Saved video-level metrics: {output_json}")

    return metrics