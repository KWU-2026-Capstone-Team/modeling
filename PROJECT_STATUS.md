# Graduation Project Status

## Topic

Visual deepfake detection using video-level frequency-domain features.

## Current state

The project currently uses pre-extracted CSV features rather than raw mp4 files.

Main feature files:

- `datasets/processed/train_video_level_frequency_features.csv`
- `datasets/processed/val_video_level_frequency_features.csv`
- `datasets/processed/test_video_level_frequency_features.csv`

## Current issue

The current 69-dimensional frequency feature set does not strongly separate real and fake videos.

Initial models predicted all samples as fake due to class imbalance.

Balanced undersampling made the model predict both classes, but threshold sensitivity remains high.

## Best observed ExtraTrees result so far

With balanced undersampling and threshold 0.2:

- Test accuracy: 0.8381
- Test macro F1: 0.4674
- Real recall: 0.0133
- Fake recall: 0.9756
- Confusion matrix:

```text
[[2, 148],
 [22, 878]]