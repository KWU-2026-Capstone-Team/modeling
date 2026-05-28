# detection_model.py
"""
Detection model pipeline for identifying deepfake videos based on frequency‑domain features.

This module implements a lightweight feature extractor and classifier training
workflow inspired by the analysis of StyleGAN and recent deepfake detection
literature.  The central idea is to derive statistical descriptors from the
two–dimensional discrete Fourier transform (FFT) of video frames and to
aggregate these descriptors across multiple sampled frames in a video.  The
resulting feature vector can be fed into a standard machine learning model
such as LightGBM, XGBoost or scikit‑learn's RandomForest.

Key design choices:

* **Radial frequency bands**: The FFT magnitude spectrum is partitioned into
  concentric rings that represent low, mid and high frequency content.  The
  boundaries are expressed as fractions of the Nyquist radius.
* **Band statistics**: For each band we compute summary statistics – mean,
  standard deviation, skewness, kurtosis, energy and Shannon entropy – that
  capture both average and distributional characteristics of the spectral
  magnitudes.
* **Frame sampling**: Only a subset of frames is processed to reduce
  computational overhead.  Frames are uniformly sampled across the video
  duration.
* **Video‑level aggregation**: Frame‑level features are aggregated with
  statistics such as the mean, standard deviation, maximum and percentiles
  across all processed frames.  This captures temporal variability without
  heavy sequence models.

Example usage:

```
from detection_model import extract_video_features, train_lightgbm
import glob

# Gather video feature vectors and labels
X, y = [], []
for video_path in glob.glob("dataset/real/*mp4"):
    X.append(extract_video_features(video_path))
    y.append(0)  # label 0 for real
for video_path in glob.glob("dataset/fake/*mp4"):
    X.append(extract_video_features(video_path))
    y.append(1)  # label 1 for deepfake

# Train a classifier
model = train_lightgbm(X, y)
```

Note: This module is self‑contained and does not require heavy deep learning
frameworks.  It uses standard scientific Python libraries available in most
environments.  For face detection or more sophisticated preprocessing, you
could integrate additional logic such as cropping detected faces before
computing the FFT.
"""

from __future__ import annotations

import os
import random
from typing import Iterable, List, Sequence, Tuple, Dict, Optional

import cv2  # type: ignore
import numpy as np
import pandas as pd
from scipy import fftpack, stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

try:
    import lightgbm as lgb  # type: ignore
except ImportError:
    lgb = None  # LightGBM is optional

try:
    import xgboost as xgb  # type: ignore
except ImportError:
    xgb = None  # XGBoost is optional


################################################################################
# Utility functions
################################################################################

def _radial_distance_indices(shape: Tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    center_r, center_c = rows // 2, cols // 2
    y_indices, x_indices = np.ogrid[:rows, :cols]
    dist = np.sqrt((y_indices - center_r) ** 2 + (x_indices - center_c) ** 2)
    max_dist = np.sqrt(center_r ** 2 + center_c ** 2)
    return dist / (max_dist if max_dist != 0 else 1.0)

def _compute_band_statistics(magnitude: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    region = magnitude[mask]
    if region.size == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'skew': 0.0,
            'kurt': 0.0,
            'energy': 0.0,
            'entropy': 0.0,
        }
    region = region.astype(np.float64)
    mean = float(np.mean(region))
    std = float(np.std(region))
    skew = float(stats.skew(region))
    kurt = float(stats.kurtosis(region))
    energy = float(np.sum(region ** 2) / region.size)
    total = np.sum(region)
    if total == 0:
        entropy = 0.0
    else:
        p = region / total
        epsilon = 1e-12
        entropy = float(-np.sum(p * np.log(p + epsilon)))
    return {
        'mean': mean,
        'std': std,
        'skew': skew,
        'kurt': kurt,
        'energy': energy,
        'entropy': entropy,
    }

def compute_fft_features(
    frame: np.ndarray,
    band_bounds: Sequence[Tuple[float, float]] = ((0.0, 0.1), (0.1, 0.3), (0.3, 1.0)),
) -> Dict[str, float]:
    if frame.ndim == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame.copy()
    h, w = frame_gray.shape
    new_size = 2 ** int(np.ceil(np.log2(max(h, w))))
    frame_resized = cv2.resize(frame_gray, (new_size, new_size), interpolation=cv2.INTER_AREA)
    fft_result = fftpack.fft2(frame_resized)
    fft_shifted = fftpack.fftshift(fft_result)
    magnitude = np.abs(fft_shifted)
    mag_norm = magnitude / (np.sum(magnitude) + 1e-12)
    rdist = _radial_distance_indices(mag_norm.shape)
    features: Dict[str, float] = {}
    for idx, (r_in, r_out) in enumerate(band_bounds):
        mask = (rdist >= r_in) & (rdist < r_out)
        stats_dict = _compute_band_statistics(mag_norm, mask)
        for key, value in stats_dict.items():
            features[f"band{idx}_{key}"] = value
    return features

def extract_video_features(
    video_path: str,
    max_frames: int = 30,
    sample_every: Optional[int] = None,
    band_bounds: Sequence[Tuple[float, float]] = ((0.0, 0.1), (0.1, 0.3), (0.3, 1.0)),
) -> Dict[str, float]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Unable to open video file: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process: List[int] = []
    if sample_every is not None and sample_every > 0:
        frames_to_process = list(range(0, frame_count, sample_every))
    else:
        if frame_count <= max_frames:
            frames_to_process = list(range(frame_count))
        else:
            frames_to_process = sorted(random.sample(range(frame_count), max_frames))
    per_frame_features: List[Dict[str, float]] = []
    for idx in frames_to_process:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        feats = compute_fft_features(frame, band_bounds=band_bounds)
        per_frame_features.append(feats)
    cap.release()
    if not per_frame_features:
        raise RuntimeError(f"No frames were processed for {video_path}")
    df = pd.DataFrame(per_frame_features)
    aggs: Dict[str, float] = {}
    percentiles = [0.90, 0.95, 0.99]
    for col in df.columns:
        values = df[col].values
        aggs[f"{col}_mean"] = float(np.mean(values))
        aggs[f"{col}_std"] = float(np.std(values))
        aggs[f"{col}_max"] = float(np.max(values))
        for p in percentiles:
            aggs[f"{col}_p{int(p*100)}"] = float(np.percentile(values, p*100))
    return aggs

################################################################################
# Classifier training functions
################################################################################

def train_random_forest(
    X: Sequence[Dict[str, float]] | pd.DataFrame,
    y: Sequence[int],
    n_estimators: int = 200,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
):
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(list(X))
    y_arr = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_df, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    metrics = {
        'accuracy': acc,
        'roc_auc': auc,
    }
    return clf, metrics

def train_lightgbm(
    X: Sequence[Dict[str, float]] | pd.DataFrame,
    y: Sequence[int],
    num_leaves: int = 31,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
):
    if lgb is None:
        return None, {}
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(list(X))
    y_arr = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_df, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )
    model = lgb.LGBMClassifier(
        num_leaves=num_leaves,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        objective='binary',
        class_weight='balanced',
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    metrics = {
        'accuracy': acc,
        'roc_auc': auc,
    }
    return model, metrics

def train_xgboost(
    X: Sequence[Dict[str, float]] | pd.DataFrame,
    y: Sequence[int],
    max_depth: int = 6,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
):
    if xgb is None:
        return None, {}
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(list(X))
    y_arr = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_df, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )
    model = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=(len(y_arr) - sum(y_arr)) / (sum(y_arr) + 1e-6),
        seed=random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    metrics = {
        'accuracy': acc,
        'roc_auc': auc,
    }
    return model, metrics

################################################################################
# Demonstration when run as a script
################################################################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Demo feature extraction and model training.")
    parser.add_argument(
        '--video-dir', type=str, default=None,
        help='Directory containing subfolders "real" and "fake" with video files.'
    )
    parser.add_argument(
        '--sample-every', type=int, default=None,
        help='Sample every Nth frame instead of limiting to max_frames.'
    )
    parser.add_argument('--max-frames', type=int, default=30, help='Max frames per video.')
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'lgbm', 'xgb'], help='Classifier to train.')
    args = parser.parse_args()
    if args.video_dir is None:
        print("Please provide --video-dir to run the demo. See docstring for usage.")
    else:
        X, y = [], []
        for class_label, sub in enumerate(['real', 'fake']):
            sub_dir = os.path.join(args.video_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            for fname in os.listdir(sub_dir):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    path = os.path.join(sub_dir, fname)
                    try:
                        feats = extract_video_features(
                            path,
                            max_frames=args.max_frames,
                            sample_every=args.sample_every
                        )
                        X.append(feats)
                        y.append(class_label)
                        print(f"Processed {path}")
                    except Exception as e:
                        print(f"Error processing {path}: {e}")
        if not X:
            print("No video features extracted. Check your video directory structure.")
        else:
            if args.model == 'rf':
                clf, metrics = train_random_forest(X, y)
            elif args.model == 'lgbm':
                clf, metrics = train_lightgbm(X, y)
            else:
                clf, metrics = train_xgboost(X, y)
            print(f"Validation metrics: {metrics}")
