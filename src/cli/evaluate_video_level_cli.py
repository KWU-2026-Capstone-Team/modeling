from __future__ import annotations

import argparse

from training.evaluate_video_level import evaluate_video_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model at video level")

    parser.add_argument("--feature-csv", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--label-col", default="label", type=str)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluate_video_level(
        feature_csv=args.feature_csv,
        model_path=args.model,
        output_json=args.output_json,
        threshold=args.threshold,
        label_col=args.label_col,
    )


if __name__ == "__main__":
    main()