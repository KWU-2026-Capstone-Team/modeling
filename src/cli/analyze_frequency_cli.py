from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from inference.analyze_video_frequency import analyze_video_frequency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze video using frequency-domain features")

    parser.add_argument("--video", required=True, type=str, help="Input video path")
    parser.add_argument("--out", default="outputs/frequency_analysis", type=str, help="Output directory")
    parser.add_argument("--sample-fps", default=2.0, type=float, help="Frame sampling FPS")
    parser.add_argument("--image-size", default=256, type=int, help="Input size for frequency analysis")
    parser.add_argument("--face-only", action="store_true", help="Analyze largest face crop only")
    parser.add_argument("--model", default=None, type=str, help="Optional trained model path")
    parser.add_argument("--threshold", default=0.5, type=float, help="Decision threshold")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot saving")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _, summary = analyze_video_frequency(
        video_path=args.video,
        out_dir=args.out,
        sample_fps=args.sample_fps,
        image_size=args.image_size,
        face_only=args.face_only,
        model_path=args.model,
        threshold=args.threshold,
        save_plots=not args.no_plots,
    )

    print("\n=== Frequency-domain Deepfake Analysis Summary ===")
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()