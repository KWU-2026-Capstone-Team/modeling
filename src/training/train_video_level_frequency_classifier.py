from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS = {
    "video_path",
    "label",
    "label_name",
}


def get_video_level_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols: list[str] = []

    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def downsample_majority_class(
    df: pd.DataFrame,
    label_col: str = "label",
    majority_label: int = 1,
    minority_label: int = 0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Downsample majority class in the training set only."""
    majority_df = df[df[label_col] == majority_label]
    minority_df = df[df[label_col] == minority_label]

    if len(majority_df) == 0 or len(minority_df) == 0:
        raise ValueError(
            f"Both classes must exist. majority={len(majority_df)}, minority={len(minority_df)}"
        )

    sampled_majority_df = majority_df.sample(
        n=len(minority_df),
        random_state=random_state,
        replace=False,
    )

    balanced_df = pd.concat([minority_df, sampled_majority_df], axis=0)
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return balanced_df


def evaluate_classifier(model, x, y, threshold: float = 0.5) -> dict:
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = model.predict(x)
        y_score = None

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(
            y,
            y_pred,
            digits=4,
            zero_division=0,
        ),
    }

    if y_score is not None and len(set(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, y_score))

    return metrics


def train_video_level_frequency_model(
    train_csv: str,
    val_csv: str,
    test_csv: str | None,
    label_col: str,
    model_out: str,
    metrics_out: str | None = None,
    random_state: int = 42,
    balance_train: bool = True,
    threshold: float = 0.5,
) -> None:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if label_col not in train_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in train CSV")

    original_train_distribution = train_df[label_col].value_counts().sort_index().to_dict()

    print("\n=== Original train video label distribution ===")
    print(train_df[label_col].value_counts().sort_index())

    if balance_train:
        train_df = downsample_majority_class(
            train_df,
            label_col=label_col,
            majority_label=1,
            minority_label=0,
            random_state=random_state,
        )
        print("\n=== Balanced train video label distribution ===")
        print(train_df[label_col].value_counts().sort_index())
    else:
        print("\n=== Train balancing disabled ===")

    used_train_distribution = train_df[label_col].value_counts().sort_index().to_dict()

    feature_cols = get_video_level_feature_columns(train_df)

    x_train = train_df[feature_cols]
    y_train = train_df[label_col].astype(int)

    x_val = val_df[feature_cols]
    y_val = val_df[label_col].astype(int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight=None,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    val_metrics = evaluate_classifier(model, x_val, y_val, threshold=threshold)

    result = {
        "feature_columns": feature_cols,
        "threshold": threshold,
        "train_balancing": {
            "enabled": balance_train,
            "method": "downsample_majority_class" if balance_train else None,
            "majority_label": 1,
            "minority_label": 0,
            "original_train_distribution": original_train_distribution,
            "used_train_distribution": used_train_distribution,
        },
        "validation": val_metrics,
    }

    print("\n=== Validation result ===")
    print(val_metrics["classification_report"])
    if "roc_auc" in val_metrics:
        print(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")

    if test_csv is not None:
        test_df = pd.read_csv(test_csv)
        x_test = test_df[feature_cols]
        y_test = test_df[label_col].astype(int)

        test_metrics = evaluate_classifier(model, x_test, y_test, threshold=threshold)
        result["test"] = test_metrics

        print("\n=== Test result ===")
        print(test_metrics["classification_report"])
        if "roc_auc" in test_metrics:
            print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    if metrics_out is not None:
        Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved video-level model: {model_out}")
    if metrics_out:
        print(f"Saved video-level metrics: {metrics_out}")