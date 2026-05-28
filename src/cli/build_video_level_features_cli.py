from __future__ import annotations

import argparse

from preprocessing.build_video_level_features import build_video_level_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build video-level frequency features from frame-level feature CSV"
    )

    parser.add_argument("--frame-csv", required=True, type=str)
    parser.add_argument("--output-csv", required=True, type=str)
    parser.add_argument("--video-col", default="video_path", type=str)
    parser.add_argument("--label-col", default="label", type=str)
    parser.add_argument("--label-name-col", default="label_name", type=str)
    parser.add_argument(
        "--aggs",
        nargs="+",
        default=["mean", "std", "min", "max"],
        choices=["mean", "std", "min", "max", "median"],
        help="Aggregation methods to apply per video",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_df = build_video_level_features(
        frame_csv=args.frame_csv,
        output_csv=args.output_csv,
        video_col=args.video_col,
        label_col=args.label_col,
        label_name_col=args.label_name_col,
        aggregation_methods=args.aggs,
    )

    print(f"Saved video-level features: {args.output_csv}")
    print(f"Rows/videos: {len(video_df)}")
    print(video_df[args.label_col].value_counts().sort_index())


if __name__ == "__main__":
    main()