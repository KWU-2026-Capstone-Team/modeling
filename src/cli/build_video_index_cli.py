from __future__ import annotations

import argparse

from data.dataset_paths import get_dataset_root
from data.video_index_builder import build_video_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build video index CSV from dataset root")

    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="../../datasets/splits/ff_c23_video_index.csv",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = get_dataset_root("ff-c23", args.dataset_root)

    df = build_video_index(
        dataset_root=dataset_root,
        output_csv=args.output_csv,
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Saved index: {args.output_csv}")
    print(df["label_name"].value_counts())


if __name__ == "__main__":
    main()