from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from tqdm import tqdm

from data.face_crop import FaceCropper
from data.video_reader import get_video_metadata, iter_sampled_frames
from features.frequency_feature_extractor import extract_frame_frequency_features
from inference.scoring import aggregate_scores, apply_model_scores
from visualization.plot_frequency_report import save_analysis_plots


@dataclass
class VideoAnalysisSummary:
    video_path: str
    frame_count_analyzed: int
    fps: float
    duration_sec: float
    face_only: bool
    aggregate_heuristic_fake_score: float
    aggregate_model_fake_score: Optional[float]
    decision: str
    threshold: float


def analyze_video_frequency(
    video_path: str,
    out_dir: str,
    sample_fps: float = 2.0,
    image_size: int = 256,
    face_only: bool = False,
    model_path: Optional[str] = None,
    threshold: float = 0.5,
    save_plots: bool = True,
) -> tuple[pd.DataFrame, VideoAnalysisSummary]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata = get_video_metadata(video_path)
    face_cropper = FaceCropper() if face_only else None
    model = joblib.load(model_path) if model_path else None

    rows = []

    frame_iter = iter_sampled_frames(video_path, sample_fps=sample_fps)

    for frame_index, timestamp_sec, frame_bgr in tqdm(
        frame_iter,
        desc="Analyzing sampled frames",
    ):
        feature = extract_frame_frequency_features(
            frame_bgr=frame_bgr,
            video_path=video_path,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            face_cropper=face_cropper,
            image_size=image_size,
        )
        rows.append(asdict(feature))

    df = pd.DataFrame(rows)
    df = apply_model_scores(df, model)

    csv_path = out / "frame_frequency_features.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    heuristic_score = aggregate_scores(df["heuristic_fake_score"].to_numpy()) if len(df) else 0.0

    if "model_fake_score" in df.columns and df["model_fake_score"].notna().any():
        model_score = aggregate_scores(df["model_fake_score"].dropna().to_numpy())
        final_score = model_score
    else:
        model_score = None
        final_score = heuristic_score

    decision = "FAKE/SUSPICIOUS" if final_score >= threshold else "REAL/LOW-SUSPICION"

    summary = VideoAnalysisSummary(
        video_path=video_path,
        frame_count_analyzed=len(df),
        fps=metadata.fps,
        duration_sec=metadata.duration_sec,
        face_only=face_only,
        aggregate_heuristic_fake_score=heuristic_score,
        aggregate_model_fake_score=model_score,
        decision=decision,
        threshold=threshold,
    )

    summary_path = out / "video_frequency_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

    if save_plots and len(df):
        save_analysis_plots(df, out)

    return df, summary