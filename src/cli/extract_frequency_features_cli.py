from __future__ import annotations

import argparse

from preprocessing.extract_frequency_dataset import extract_frequency_features_from_video_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frequency features from video split CSV")

    parser.add_argument("--video-csv", required=True, type=str)
    parser.add_argument("--output-csv", required=True, type=str)
    parser.add_argument("--sample-fps", default=1.0, type=float)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--face-only", action="store_true")
    parser.add_argument("--max-frames-per-video", default=None, type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = extract_frequency_features_from_video_csv(
        video_csv=args.video_csv,
        output_csv=args.output_csv,
        sample_fps=args.sample_fps,
        image_size=args.image_size,
        face_only=args.face_only,
        max_frames_per_video=args.max_frames_per_video,
    )

    print(f"Saved: {args.output_csv}")
    print(f"Rows: {len(df)}")
    print(df["label_name"].value_counts())


if __name__ == "__main__":
    main()