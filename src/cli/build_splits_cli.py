from __future__ import annotations

import argparse

from data.split_builder import build_video_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val/test video splits")

    parser.add_argument("--index-csv", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--random-state", default=42, type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_video_splits(
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()