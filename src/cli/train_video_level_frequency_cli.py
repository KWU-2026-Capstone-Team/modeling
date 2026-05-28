from __future__ import annotations

import argparse

from training.train_video_level_frequency_classifier import train_video_level_frequency_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train video-level frequency-domain classifier")

    parser.add_argument("--train-csv", required=True, type=str)
    parser.add_argument("--val-csv", required=True, type=str)
    parser.add_argument("--test-csv", default=None, type=str)
    parser.add_argument("--label-col", default="label", type=str)
    parser.add_argument("--model-out", required=True, type=str)
    parser.add_argument("--metrics-out", required=True, type=str)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument(
        "--no-balance-train",
        action="store_true",
        help="Disable train-set downsampling. By default, fake videos are downsampled to match real videos.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_video_level_frequency_model(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        label_col=args.label_col,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        random_state=args.random_state,
        balance_train=not args.no_balance_train,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()