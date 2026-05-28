from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data.face_crop import FaceCropper
from data.video_reader import iter_sampled_frames
from features.frequency_feature_extractor import extract_frame_frequency_features


def extract_frequency_features_from_video_csv(
    video_csv: str | Path,
    output_csv: str | Path,
    sample_fps: float = 1.0,
    image_size: int = 256,
    face_only: bool = True,
    max_frames_per_video: int | None = None,
) -> pd.DataFrame:
    video_df = pd.read_csv(video_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    face_cropper = FaceCropper() if face_only else None

    rows = []

    for _, video_row in tqdm(video_df.iterrows(), total=len(video_df), desc="Videos"):
        video_path = video_row["video_path"]
        label = int(video_row["label"])
        label_name = video_row.get("label_name", "fake" if label == 1 else "real")

        frame_count = 0

        try:
            for frame_index, timestamp_sec, frame_bgr in iter_sampled_frames(
                video_path,
                sample_fps=sample_fps,
            ):
                feature = extract_frame_frequency_features(
                    frame_bgr=frame_bgr,
                    video_path=video_path,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    face_cropper=face_cropper,
                    image_size=image_size,
                )

                row = asdict(feature)
                row["label"] = label
                row["label_name"] = label_name
                rows.append(row)

                frame_count += 1

                if max_frames_per_video is not None and frame_count >= max_frames_per_video:
                    break

        except Exception as e:
            print(f"[WARN] Failed to process video: {video_path}")
            print(f"Reason: {e}")

    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return feature_df