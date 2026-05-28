# AGENTS.md

## Project

This repository is for a graduation project on visual deepfake detection.

The current research direction is lightweight video-level deepfake detection using frequency-domain features and classical machine learning classifiers.

The main goal is not simply high accuracy, but interpretable experimentation:
- compare classifier behavior
- diagnose class imbalance
- evaluate threshold effects
- improve feature separability
- document results for a graduation project report

## Language preference

Explain project decisions in Korean unless the user explicitly asks for English.

Keep technical terms such as `threshold`, `balanced accuracy`, `macro F1`, `feature extraction`, `RandomForest`, `ExtraTrees`, and `frequency-domain` in English when appropriate.

## Current project structure

Expected structure:

```text
research/
├─ src/
│  ├─ cli/
│  │  └─ train_detection_model_from_csv.py
│  └─ detection_model.py
├─ datasets/
│  └─ processed/
│     ├─ train_video_level_frequency_features.csv
│     ├─ val_video_level_frequency_features.csv
│     └─ test_video_level_frequency_features.csv
├─ checkpoints/
├─ AGENTS.md
└─ PROJECT_STATUS.md