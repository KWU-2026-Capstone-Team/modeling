import argparse
import json
import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def load_csv_features(csv_path, label_col):
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

    y = df[label_col].astype(int)

    drop_cols = [label_col]

    for col in ["video", "video_path", "path", "filename", "file", "id"]:
        if col in df.columns:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols, errors="ignore")

    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        print(f"[WARN] Dropping non-numeric columns: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)

    X = X.fillna(0)

    return X, y


def build_model(model_name, random_state=42):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )

    if model_name == "extra":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )

    if model_name == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model: {model_name}")


def evaluate(model, X, y, threshold=0.5):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = model.decision_function(X)

    pred = (prob >= threshold).astype(int)

    print("\n=== Probability debug ===")
    print("threshold:", threshold)
    print("prob min :", prob.min())
    print("prob max :", prob.max())
    print("prob mean:", prob.mean())
    print("prob p01 :", pd.Series(prob).quantile(0.01))
    print("prob p05 :", pd.Series(prob).quantile(0.05))
    print("prob p50 :", pd.Series(prob).quantile(0.50))
    print("prob p95 :", pd.Series(prob).quantile(0.95))
    print("prob p99 :", pd.Series(prob).quantile(0.99))
    print("pred distribution:")
    print(pd.Series(pred).value_counts())

    acc = accuracy_score(y, pred)
    balanced_acc = balanced_accuracy_score(y, pred)
    auc = roc_auc_score(y, prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0
    )

    result = {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "roc_auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(
            y, pred, digits=4, zero_division=0
        ),
    }

    return result


def search_best_threshold(model, X_val, y_val):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_val)[:, 1]
    else:
        prob = model.decision_function(X_val)

    best = {
        "threshold": 0.5,
        "balanced_accuracy": 0.0,
        "f1_macro": -1.0,
        "precision_macro": 0.0,
        "recall_macro": 0.0,
    }

    for i in range(5, 96):
        threshold = i / 100
        pred = (prob >= threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, pred, average="macro", zero_division=0
        )

        balanced_acc = balanced_accuracy_score(y_val, pred)

        if i % 5 == 0:
            print(
                f"thr={threshold:.2f} "
                f"bal_acc={balanced_acc:.4f} "
                f"f1_macro={f1:.4f} "
                f"pred_real={(pred == 0).sum()} "
                f"pred_fake={(pred == 1).sum()}"
            )

        if f1 > best["f1_macro"]:
            best = {
                "threshold": float(threshold),
            "balanced_accuracy": float(balanced_acc),
            "f1_macro": float(f1),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            }

    return best


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--label-col", default="label")

    parser.add_argument(
        "--model",
        choices=["rf", "extra", "logreg"],
        default="extra",
    )

    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)

    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--search-threshold", action="store_true")

    args = parser.parse_args()

    X_train, y_train = load_csv_features(args.train_csv, args.label_col)
    X_val, y_val = load_csv_features(args.val_csv, args.label_col)
    X_test, y_test = load_csv_features(args.test_csv, args.label_col)

    common_cols = list(
        set(X_train.columns)
        & set(X_val.columns)
        & set(X_test.columns)
    )
    common_cols = sorted(common_cols)

    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    X_test = X_test[common_cols]

    print("=== Dataset shape ===")
    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    print("\n=== Original train label distribution ===")
    print(y_train.value_counts())

    # Balance train set by undersampling majority class
    train_df = X_train.copy()
    train_df["_label"] = y_train.values

    real_df = train_df[train_df["_label"] == 0]
    fake_df = train_df[train_df["_label"] == 1]

    n = min(len(real_df), len(fake_df))

    real_sampled = real_df.sample(n=n, random_state=42)
    fake_sampled = fake_df.sample(n=n, random_state=42)

    balanced_df = pd.concat([real_sampled, fake_sampled], axis=0)
    balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    y_train_balanced = balanced_df["_label"].astype(int)
    X_train_balanced = balanced_df.drop(columns=["_label"])

    print("\n=== Balanced train label distribution ===")
    print(y_train_balanced.value_counts())

    model = build_model(args.model)
    model.fit(X_train_balanced, y_train_balanced)

    if args.search_threshold:
        best = search_best_threshold(model, X_val, y_val)
        threshold = best["threshold"]
        print("\n=== Best validation threshold ===")
        print(best)
    else:
        threshold = args.threshold if args.threshold is not None else 0.5

    print("\n=== Validation result ===")
    val_result = evaluate(model, X_val, y_val, threshold)
    print(val_result["classification_report"])
    print("Confusion matrix:")
    print(val_result["confusion_matrix"])

    print("\n=== Test result ===")
    test_result = evaluate(model, X_test, y_test, threshold)
    print(test_result["classification_report"])
    print("Confusion matrix:")
    print(test_result["confusion_matrix"])

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": common_cols,
            "threshold": threshold,
            "label_col": args.label_col,
        },
        args.model_out,
    )

    metrics = {
        "model": args.model,
        "threshold": threshold,
        "val": val_result,
        "test": test_result,
        "feature_count": len(common_cols),
        "feature_columns": common_cols,
    }

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nSaved model to:", args.model_out)
    print("Saved metrics to:", args.metrics_out)


if __name__ == "__main__":
    main()