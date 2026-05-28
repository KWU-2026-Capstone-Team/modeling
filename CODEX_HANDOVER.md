# Deeptector Codex Handover Document

> Purpose: This document is intended to let Codex continue development of the Deeptector graduation project immediately.  
> Basis: This handover is based on the project conversation history, observed commands, file names, training logs, and recent Git workflow.  
> Important limitation: The repository itself was not directly inspected while writing this document. Codex must verify every file path, function name, label mapping, and command against the actual GitHub repository before editing.

---

## 0. First Instruction for Codex

Before editing anything, inspect the actual repository and verify whether the file names, commands, label mapping, metrics paths, and assumptions in this handover are correct. If anything differs from the repository state, update `PROJECT_STATUS.md` and proceed based on the actual repository.

Start by reading these files/directories:

```text
AGENTS.md
PROJECT_STATUS.md
.gitignore
src/
src/detection_model.py
src/cli/train_video_level_frequency_cli.py
experiments/
checkpoints/  # only if metric JSON files are intentionally available
```

Do not assume that all checkpoint/model files are committed. The project appears to intentionally avoid committing large model artifacts.

---

## 1. Project Goal

Deeptector is a graduation project for detecting deepfake videos. The current direction is to build a deepfake detection pipeline based on frequency-domain features rather than an end-to-end RGB-frame CNN.

The project aims to:

1. Extract frequency-domain features from video frames.
2. Aggregate frame-level frequency features into video-level features.
3. Train a binary classifier to distinguish real and fake videos.
4. Evaluate the model at video level, not merely frame level.
5. Save reproducible models and metrics.
6. Support threshold-based experimentation using validation data.
7. Provide interpretable results suitable for a graduation project report.

Current assumed label mapping:

```text
0 = real
1 = fake
```

This must be verified in the repository and dataset generation code before further work.

---

## 2. Core Features

### 2.1 Frequency-domain deepfake detection

The project uses frequency-domain features as the main signal for detecting deepfake artifacts.

The rationale is that deepfake generation, face synthesis, warping, blending, resizing, and compression may leave abnormal frequency patterns that are not always obvious in RGB pixel space.

Expected advantages:

- More explainable than a pure CNN black-box pipeline.
- Suitable for classical ML models such as Random Forest.
- Easier to analyze with feature importance.
- Relevant to artifacts caused by synthesis, interpolation, compression, and blending.

---

### 2.2 Frame-level model: previous direction

The earlier approach was frame-level classification.

In this approach:

- Each frame is treated as an independent sample.
- Each feature row corresponds to one frame.
- The model predicts whether each frame is real or fake.
- Video-level prediction requires post-processing such as averaging probabilities, majority vote, max probability, or thresholding.

Advantages:

- More training samples because each video yields many frames.
- Simpler feature extraction and training setup.
- Allows frame-by-frame inspection.

Problems:

- High risk of data leakage if frames from the same video are split across train/validation/test.
- Final task is video-level detection, but the model is trained on frame-level labels.
- Individual frame predictions can be unstable.
- Reported frame-level accuracy may overstate actual video-level performance.

Important precaution:

```text
Never allow frames from the same source video to appear in different dataset splits.
```

---

### 2.3 Video-level aggregation model: current direction

The current direction is video-level aggregation.

In this approach:

- Frequency features are extracted from multiple frames in a video.
- Features are aggregated per video.
- One video becomes one feature row.
- The classifier directly predicts whether the video is real or fake.

Observed processed CSV files:

```text
datasets/processed/train_video_level_frequency_features.csv
datasets/processed/val_video_level_frequency_features.csv
datasets/processed/test_video_level_frequency_features.csv
```

Likely aggregation statistics may include:

```text
mean
standard deviation
minimum
maximum
median
percentile-based statistics
```

The actual aggregation functions and feature names must be verified in the repository.

Advantages:

- Matches the final use case: video-level deepfake detection.
- Reduces frame leakage risk if split is performed by video.
- Produces metrics that are more meaningful for the final project.
- Allows one prediction per video, simplifying evaluation.

Potential disadvantages:

- Fewer training samples than frame-level training.
- Aggregation may discard temporal or local frame-level details.
- Class imbalance can have stronger effects.
- Poor aggregation design may reduce separability between real and fake videos.

---

## 3. Technology Stack

The following stack is inferred from commands, file names, and model artifacts.

### Language

```text
Python
```

### Likely Python libraries

```text
pandas
numpy
scikit-learn
joblib
```

### Model family

The project currently appears to use Random Forest models for frequency feature classification.

Observed file naming pattern:

```text
rf = Random Forest
balanced = class-balanced training experiment
t02/t03/t04/t05/t06/t07 = threshold experiments, likely 0.2 through 0.7
```

Codex should confirm the exact model class in the source code. It is likely one of:

```python
RandomForestClassifier
```

or another scikit-learn classifier saved with `joblib`.

---

## 4. How to Run

### 4.1 Known training command

The following command was observed in the project conversation.

Likely working directory:

```text
C:\Users\user\Desktop\deeptector\research\src
```

PowerShell command:

```powershell
python -m cli.train_video_level_frequency_cli `
  --train-csv ..\datasets\processed\train_video_level_frequency_features.csv `
  --val-csv ..\datasets\processed\val_video_level_frequency_features.csv `
  --test-csv ..\datasets\processed\test_video_level_frequency_features.csv `
  --label-col label `
  --model-out ..\checkpoints\video_level_frequency_rf_balanced_t04.joblib `
  --metrics-out ..\checkpoints\video_level_frequency_rf_balanced_metrics_t04.json `
  --threshold 0.4
```

Equivalent Unix-style command, if run from a compatible environment:

```bash
python -m cli.train_video_level_frequency_cli \
  --train-csv ../datasets/processed/train_video_level_frequency_features.csv \
  --val-csv ../datasets/processed/val_video_level_frequency_features.csv \
  --test-csv ../datasets/processed/test_video_level_frequency_features.csv \
  --label-col label \
  --model-out ../checkpoints/video_level_frequency_rf_balanced_t04.joblib \
  --metrics-out ../checkpoints/video_level_frequency_rf_balanced_metrics_t04.json \
  --threshold 0.4
```

### 4.2 Important path warning

The command uses relative paths. It may fail if executed from the repository root instead of `research/src` or the expected working directory.

Recommended improvement:

- Resolve paths with `Path(...).resolve()`.
- Document the correct working directory in `README.md` and `PROJECT_STATUS.md`.
- Consider making commands runnable from the `research/` root.

---

## 5. Implemented Features to Date

The following features appear to already exist or have recently been worked on.

### 5.1 Video-level frequency training CLI

Observed module:

```text
cli.train_video_level_frequency_cli
```

Observed CLI arguments:

```text
--train-csv
--val-csv
--test-csv
--label-col
--model-out
--metrics-out
--threshold
```

Likely responsibilities:

1. Parse command-line arguments.
2. Load train/validation/test CSV files.
3. Select feature columns and label column.
4. Balance the training set.
5. Train a Random Forest or similar classifier.
6. Predict class probabilities.
7. Apply a custom probability threshold.
8. Print validation/test classification reports.
9. Save the trained model as `.joblib`.
10. Save metrics as `.json`.

Potential function names to inspect:

```text
main()
parse_args()
load_dataset()
balance_train_data()
train_model()
evaluate_model()
save_model()
save_metrics()
```

These function names are inferred and must be verified in the actual source code.

---

### 5.2 Class balancing

A recent training log showed severe class imbalance in the original video-level training set.

Observed original training distribution:

```text
label
0     700
1    4200
```

Observed balanced training distribution:

```text
label
0    700
1    700
```

This suggests downsampling of the majority class.

Important concern:

- If label `1 = fake`, the fake class has far more samples than real.
- Downsampling fake from 4200 to 700 may discard substantial manipulation diversity.
- Consider using `class_weight="balanced"` or repeated balanced sampling rather than simple downsampling.

Codex must verify:

```text
- Which class is real and which is fake.
- Whether balancing is random and reproducible.
- Whether random_state is fixed.
- Whether validation/test sets remain untouched.
```

---

### 5.3 Threshold-based prediction

Recent experiments use thresholds such as:

```text
0.2
0.3
0.4
0.5
0.6
0.7
```

Observed command used:

```text
--threshold 0.4
```

Likely prediction logic:

```python
pred = (fake_probability >= threshold).astype(int)
```

Codex must verify:

- Which probability column corresponds to fake.
- Whether `model.classes_` is used safely.
- Whether label order is assumed incorrectly.

Correct implementation should avoid assuming that fake probability is always column index `1` unless `model.classes_` confirms it.

Recommended robust pattern:

```python
classes = list(model.classes_)
fake_index = classes.index(1)
fake_proba = model.predict_proba(X)[:, fake_index]
y_pred = (fake_proba >= threshold).astype(int)
```

This assumes label `1` is fake. If label mapping differs, update accordingly.

---

### 5.4 Model and metrics saving

Observed model artifact path:

```text
checkpoints/video_level_frequency_rf_balanced_t04.joblib
```

Observed metrics path:

```text
checkpoints/video_level_frequency_rf_balanced_metrics_t04.json
```

Observed checkpoint/metric file names from Git status-related command:

```text
checkpoints/frequency_rf_balanced_metrics.json
checkpoints/frequency_rf_balanced_video_metrics.json
checkpoints/frequency_rf_balanced_video_metrics_t03.json
checkpoints/frequency_rf_balanced_video_metrics_t04.json
checkpoints/frequency_rf_balanced_video_metrics_t06.json
checkpoints/frequency_rf_balanced_video_metrics_t07.json
checkpoints/frequency_rf_test_metrics.json
checkpoints/frequency_rf_unbalanced_video_metrics.json
checkpoints/new_frequency_extra_balacc_metrics.json
checkpoints/new_frequency_extra_balanced_metrics.json
checkpoints/new_frequency_extra_balanced_metrics_t02.json
checkpoints/new_frequency_extra_balanced_metrics_t03.json
checkpoints/new_frequency_extra_balanced_metrics_t05.json
checkpoints/new_frequency_extra_metrics.json
```

These files may be untracked and may not exist in GitHub.

Recommendation:

- Do not commit `.joblib` model files unless explicitly required.
- Consider committing selected small metric JSON files under `reports/metrics/` or `experiments/results/`.
- Standardize metric file naming.

---

## 6. Main File and Directory Structure

The following structure is inferred from project conversation and commands.

```text
deeptector/
└─ research/
   ├─ .gitignore
   ├─ AGENTS.md
   ├─ PROJECT_STATUS.md
   ├─ src/
   │  ├─ detection_model.py
   │  └─ cli/
   │     └─ train_video_level_frequency_cli.py
   ├─ experiments/
   ├─ checkpoints/
   │  ├─ frequency_rf_balanced_metrics.json
   │  ├─ frequency_rf_balanced_video_metrics.json
   │  ├─ frequency_rf_balanced_video_metrics_t03.json
   │  ├─ frequency_rf_balanced_video_metrics_t04.json
   │  ├─ frequency_rf_balanced_video_metrics_t06.json
   │  ├─ frequency_rf_balanced_video_metrics_t07.json
   │  ├─ frequency_rf_test_metrics.json
   │  ├─ frequency_rf_unbalanced_video_metrics.json
   │  ├─ new_frequency_extra_balacc_metrics.json
   │  ├─ new_frequency_extra_balanced_metrics.json
   │  ├─ new_frequency_extra_balanced_metrics_t02.json
   │  ├─ new_frequency_extra_balanced_metrics_t03.json
   │  ├─ new_frequency_extra_balanced_metrics_t05.json
   │  ├─ new_frequency_extra_metrics.json
   │  └─ video_level_frequency_rf_balanced_t04.joblib
   └─ datasets/
      └─ processed/
         ├─ train_video_level_frequency_features.csv
         ├─ val_video_level_frequency_features.csv
         └─ test_video_level_frequency_features.csv
```

Codex must inspect the repository to confirm the actual structure.

---

## 7. Important Files and Their Roles

### 7.1 `.gitignore`

Purpose:

- Prevent large or local-only files from being committed.
- Exclude datasets, checkpoints, Python cache files, virtual environments, and environment variable files.

Recommended entries:

```gitignore
__pycache__/
*.pyc
.venv/
venv/
.env
.env.*

# Datasets
datasets/raw/
datasets/processed/

# Model artifacts
checkpoints/
*.joblib
*.pkl

# Notebook/cache artifacts
.ipynb_checkpoints/
.DS_Store
```

Decision needed:

- Whether metric JSON files should be committed.
- If yes, store them under a separate lightweight directory such as `reports/metrics/` instead of `checkpoints/`.

---

### 7.2 `.gitattributes`

A recent Git warning occurred:

```text
warning: in the working copy of 'src/detection_model.py', LF will be replaced by CRLF the next time Git touches it
```

This is a Windows line-ending warning, not a code failure.

Recommended `.gitattributes`:

```gitattributes
*.py text eol=lf
*.md text eol=lf
*.json text eol=lf
*.yml text eol=lf
*.yaml text eol=lf
*.txt text eol=lf
```

Purpose:

- Avoid meaningless CRLF/LF diffs.
- Keep Codex/Linux/Windows edits consistent.

---

### 7.3 `AGENTS.md`

Purpose:

- Instructions for Codex or other AI coding agents.
- Should include project rules, safety constraints, data handling rules, and preferred workflow.

Recommended content:

```text
- Do not commit large datasets or model checkpoints.
- Preserve train/validation/test split by video ID.
- Avoid frame-level leakage.
- Prefer video-level evaluation metrics.
- Select thresholds using validation data only.
- Do not optimize using the test set.
- Update PROJECT_STATUS.md after meaningful changes.
```

---

### 7.4 `PROJECT_STATUS.md`

Purpose:

- Current status of the project.
- Completed features.
- Known issues.
- Latest experimental results.
- Next development priorities.

Codex should update this after each major change.

Recommended sections:

```text
Current pipeline
Current best command
Current best model
Current best metrics
Known bugs/issues
Next task priority
Notes for report writing
```

---

### 7.5 `src/detection_model.py`

Purpose:

- Appears to contain model-related code.
- It triggered the LF/CRLF Git warning.

Codex should inspect this file for:

```text
- Model class definitions
- Prediction logic
- Threshold logic
- Feature preprocessing
- Inference functions
- Any frame-level legacy code
```

Risk:

- If this file still assumes frame-level operation, it may need to be aligned with the video-level aggregation direction.

---

### 7.6 `src/cli/train_video_level_frequency_cli.py`

Purpose:

- Main training CLI for video-level frequency features.
- Current most important file for model experimentation.

Codex should inspect for:

```text
- Argument parser
- Dataset loading
- Feature/label separation
- Class balancing
- Model initialization
- Random seed
- Threshold handling
- Validation/test evaluation
- Metric JSON schema
- Model saving
```

Immediate improvement targets:

```text
- Add confusion matrix output.
- Add threshold sweep on validation set.
- Standardize metrics JSON.
- Confirm safe use of predict_proba columns.
- Add train/val/test label distribution to metrics.
```

---

### 7.7 `experiments/`

Purpose:

- Experimental scripts and analysis code.
- Should contain threshold sweep, metrics comparison, feature importance, and probability distribution analysis.

Recommended scripts to add if not present:

```text
experiments/sweep_video_level_thresholds.py
experiments/compare_metrics.py
experiments/analyze_probability_distribution.py
experiments/feature_importance.py
```

---

### 7.8 `checkpoints/`

Purpose:

- Stores model binaries and metric files.

Recommendation:

- Keep model binaries out of Git.
- Use a consistent naming convention.
- Move selected metric summaries to a Git-tracked reports directory if needed.

Example naming convention:

```text
video_level_frequency_rf_balanced_threshold_0_40_metrics.json
video_level_frequency_rf_balanced_threshold_0_40.joblib
```

---

## 8. Recent Work

### 8.1 Direction changed from frame-level to video-level

Recent project discussion clarified that the model direction changed from frame-level frequency features to video-level aggregation features.

The video-level approach is now the preferred direction because:

- The final output should be per video.
- It better matches the real use case.
- It reduces frame leakage risk.
- It makes evaluation more defensible for a graduation project.

---

### 8.2 Recent training experiment

Observed command used threshold `0.4` and balanced training.

Original training distribution:

```text
label
0     700
1    4200
```

Balanced training distribution:

```text
label
0    700
1    700
```

Observed validation result excerpt:

```text
class 0:
precision 0.0830
recall    0.1333
f1-score  0.1023
support   150

class 1:
precision 0.8393
recall    0.754...
```

Interpretation:

- Class `0`, assumed real, performs extremely poorly.
- The model is biased toward class `1`, assumed fake.
- Balancing alone did not solve the issue.
- Threshold `0.4` may not be optimal.
- There may be feature, split, label mapping, or probability interpretation issues.

---

### 8.3 Recent Git workflow

Observed Git command:

```powershell
git add .gitignore AGENTS.md PROJECT_STATUS.md src/ experiments/
```

Interpretation:

- Source files and project docs were staged.
- `checkpoints/` was not explicitly staged.
- This suggests model artifacts are likely intended to remain untracked.

Observed warning:

```text
warning: in the working copy of 'src/detection_model.py', LF will be replaced by CRLF the next time Git touches it
```

Action:

- Add `.gitattributes` if not present.
- Normalize line endings if necessary.

---

## 9. Confirmed Bugs and Technical Issues

### 9.1 Very poor class 0 performance

Observed metric:

```text
label 0 precision: 0.0830
label 0 recall:    0.1333
label 0 f1-score:  0.1023
```

This is the most important current issue.

Possible causes:

1. Label mapping is misunderstood.
2. `predict_proba` column is interpreted incorrectly.
3. Features do not separate real/fake videos well.
4. Train/validation/test distribution differs significantly.
5. Class balancing discards too much majority-class diversity.
6. Aggregation statistics are too weak.
7. Dataset split has source/manipulation bias.
8. Threshold is poorly chosen.
9. Validation set may have a different class ratio or feature distribution.
10. Data preprocessing differs between train and validation/test.

Immediate checks:

```text
- Verify label mapping.
- Print train/validation/test label distributions.
- Print confusion matrix.
- Check model.classes_.
- Plot or summarize predict_proba distribution by true class.
- Sweep thresholds on validation set.
- Confirm no feature columns accidentally include labels or IDs.
```

---

### 9.2 Potential unsafe probability column assumption

If code uses:

```python
proba[:, 1]
```

it may be wrong unless `model.classes_[1] == 1`.

Codex should replace unsafe assumptions with explicit class-index lookup.

---

### 9.3 Threshold experiments are not sufficiently organized

Observed metric files use several naming styles:

```text
frequency_rf_*
new_frequency_extra_*
video_level_frequency_rf_*
```

Problem:

- Hard to compare experiments.
- Hard to know which metrics correspond to which feature set, threshold, or balancing mode.

Recommended fix:

Create a summary CSV or Markdown report:

```text
experiments/results_summary.csv
```

Suggested columns:

```text
experiment_name
model_type
feature_type
aggregation_type
balanced
threshold
val_accuracy
val_balanced_accuracy
val_macro_f1
val_f1_real
val_f1_fake
test_accuracy
test_balanced_accuracy
test_macro_f1
test_f1_real
test_f1_fake
model_path
metrics_path
notes
```

---

### 9.4 Relative path fragility

Current CLI command depends on relative paths such as:

```text
..\datasets\processed\train_video_level_frequency_features.csv
```

Problem:

- Running from a different directory may fail.

Recommended fix:

- Use `pathlib.Path` and resolved paths.
- Add examples for both Windows PowerShell and Bash.
- Document the expected working directory.

---

### 9.5 Line ending warning

Issue:

```text
LF will be replaced by CRLF
```

Fix:

- Add `.gitattributes`.
- Avoid large line-ending-only diffs.

---

### 9.6 Checkpoint management policy is unclear

Problem:

- `checkpoints/` contains both models and metric JSON files.
- Some metric JSON files are useful for reproducibility.
- Model files can be too large for Git.

Recommended policy:

```text
- Do not commit model binaries.
- Do not commit datasets.
- Commit only curated metric summaries if needed.
- Store final model artifacts externally if required.
```

---

## 10. Database, API, and Environment Variables

No database, external API, or required environment variable has been confirmed from the project conversation.

The current pipeline appears to be local-file based.

Local artifacts:

```text
datasets/processed/*.csv
checkpoints/*.joblib
checkpoints/*.json
```

Possible optional environment variables, if the project later needs them:

```env
PROJECT_ROOT=C:\Users\user\Desktop\deeptector
DATASET_DIR=C:\Users\user\Desktop\deeptector\datasets
CHECKPOINT_DIR=C:\Users\user\Desktop\deeptector\research\checkpoints
```

However, the current CLI argument approach is acceptable and may be preferable for reproducibility.

---

## 11. Important Design Decisions and Precautions

### 11.1 Final evaluation must be video-level

The project should not claim final performance based only on frame-level metrics.

Recommended report language:

```text
Frame-level classification treats each extracted frame as an independent sample, while video-level aggregation summarizes multiple frame-level frequency features into one representation per video. Since the final detection target is a video, video-level evaluation is the primary metric for this project.
```

---

### 11.2 Avoid frame leakage

Never split frames from the same video across train/validation/test.

Bad example:

```text
video_A_frame_001 -> train
video_A_frame_120 -> validation
video_A_frame_240 -> test
```

Correct example:

```text
video_A -> train only
video_B -> validation only
video_C -> test only
```

---

### 11.3 Use validation set for threshold selection

Thresholds must be selected using validation data only.

Correct procedure:

```text
1. Train model on train set.
2. Predict probabilities on validation set.
3. Sweep thresholds on validation set.
4. Select threshold by validation balanced accuracy, macro F1, or another predefined criterion.
5. Evaluate once on test set using the selected threshold.
```

Do not choose threshold based on test performance.

---

### 11.4 Handle class imbalance carefully

Current observed imbalance:

```text
label 0 = 700
label 1 = 4200
```

If label `1` is fake, then fake data is much larger than real data.

Simple downsampling can reduce training diversity.

Possible alternatives:

```text
- class_weight="balanced"
- repeated downsampling with multiple seeds
- ensemble over balanced subsets
- stratified sampling by manipulation type or source if metadata exists
- threshold optimization by balanced accuracy or macro F1
```

---

### 11.5 Record label mapping in every metric output

Every metrics JSON should include:

```json
"label_mapping": {
  "0": "real",
  "1": "fake"
}
```

This prevents misinterpretation of classification reports.

---

### 11.6 Preserve interpretability

Because this is a graduation project, interpretability matters.

Recommended analysis:

```text
- Random Forest feature_importances_
- Top frequency features
- Confusion matrix
- Threshold curve
- Probability distribution by class
```

These are useful for both debugging and final presentation/report writing.

---

## 12. Not Yet Implemented or Needs Modification

### 12.1 Confusion matrix output

Add confusion matrix to metrics JSON.

Recommended JSON structure:

```json
"confusion_matrix": {
  "labels": [0, 1],
  "matrix": [[tn, fp], [fn, tp]]
}
```

Also print it in CLI output.

---

### 12.2 Threshold sweep automation

Current threshold experiments appear manual.

Add one of the following:

Option A: new experiment script

```text
experiments/sweep_video_level_thresholds.py
```

Option B: CLI arguments

```text
--threshold-sweep
--threshold-min
--threshold-max
--threshold-step
--threshold-metric balanced_accuracy
```

Suggested output:

```text
threshold
accuracy
balanced_accuracy
macro_f1
real_precision
real_recall
real_f1
fake_precision
fake_recall
fake_f1
```

---

### 12.3 Standard metrics JSON

Recommended metrics JSON schema:

```json
{
  "experiment_name": "video_level_frequency_rf_balanced_t04",
  "model_type": "RandomForestClassifier",
  "feature_type": "frequency",
  "aggregation": "video_level",
  "balanced": true,
  "threshold": 0.4,
  "label_mapping": {
    "0": "real",
    "1": "fake"
  },
  "train_distribution_original": {
    "0": 700,
    "1": 4200
  },
  "train_distribution_used": {
    "0": 700,
    "1": 700
  },
  "validation": {
    "accuracy": null,
    "balanced_accuracy": null,
    "macro_f1": null,
    "classification_report": {},
    "confusion_matrix": {}
  },
  "test": {
    "accuracy": null,
    "balanced_accuracy": null,
    "macro_f1": null,
    "classification_report": {},
    "confusion_matrix": {}
  }
}
```

---

### 12.4 Probability distribution analysis

Add a script or CLI option to save probability distribution summaries.

Recommended script:

```text
experiments/analyze_probability_distribution.py
```

Analyze:

```text
- fake probability distribution for true real videos
- fake probability distribution for true fake videos
- false positive videos
- false negative videos
- threshold impact on each class
```

---

### 12.5 Feature importance analysis

If Random Forest is used, add feature importance output.

Recommended script:

```text
experiments/feature_importance.py
```

Outputs:

```text
reports/feature_importance_top20.csv
reports/feature_importance_top20.png
```

Use in final report to explain which frequency features were most influential.

---

### 12.6 README update

The repository should contain clear setup and execution instructions.

Recommended README sections:

```text
Project overview
Directory structure
Dataset preparation
Feature extraction
Training command
Evaluation command
Current known issues
Experiment tracking
Artifact policy
```

---

## 13. Priority of Next Tasks

### Priority 1: Diagnose class 0 failure

Do this first.

Checklist:

```text
- Verify label mapping.
- Print train/validation/test label distributions.
- Confirm model.classes_ order.
- Add confusion matrix.
- Analyze predict_proba distribution by true label.
- Sweep thresholds on validation set.
- Confirm feature columns do not include leakage or non-feature identifiers.
```

Goal:

```text
Understand why label 0 performance is extremely poor.
```

---

### Priority 2: Add threshold sweep

Automate threshold selection using validation data.

Preferred criterion:

```text
balanced_accuracy
```

Alternative criteria:

```text
macro_f1
real_recall with acceptable fake precision
```

Do not use test set for threshold selection.

---

### Priority 3: Standardize metric output

Metrics should include:

```text
- experiment name
- model type
- feature type
- aggregation type
- threshold
- label mapping
- train/validation/test distributions
- classification reports
- confusion matrices
- selected threshold criterion
```

---

### Priority 4: Fix Git hygiene

Tasks:

```text
- Verify .gitignore.
- Add .gitattributes if missing.
- Ensure datasets are ignored.
- Ensure model binaries are ignored.
- Decide where curated metric summaries should live.
```

---

### Priority 5: Add interpretability outputs

Tasks:

```text
- Feature importance analysis.
- Probability distribution plots or summaries.
- Confusion matrix visualizations.
- Threshold sweep result table.
```

These will help both debugging and the graduation presentation.

---

## 14. Suggested Immediate Codex Task Prompt

Use this as the first task prompt to Codex:

```text
You are continuing development of the Deeptector graduation project.

First, inspect the actual repository and verify the assumptions in CODEX_HANDOVER.md. Pay special attention to:
- AGENTS.md
- PROJECT_STATUS.md
- .gitignore
- src/detection_model.py
- src/cli/train_video_level_frequency_cli.py
- experiments/

The project is a deepfake detection pipeline using frequency-domain features. The recent direction changed from frame-level classification to video-level aggregation features.

Known recent training command:

python -m cli.train_video_level_frequency_cli \
  --train-csv ../datasets/processed/train_video_level_frequency_features.csv \
  --val-csv ../datasets/processed/val_video_level_frequency_features.csv \
  --test-csv ../datasets/processed/test_video_level_frequency_features.csv \
  --label-col label \
  --model-out ../checkpoints/video_level_frequency_rf_balanced_t04.joblib \
  --metrics-out ../checkpoints/video_level_frequency_rf_balanced_metrics_t04.json \
  --threshold 0.4

Known issue:
Validation performance for label 0 is extremely poor. Observed values were approximately:
- precision: 0.0830
- recall: 0.1333
- f1-score: 0.1023

Observed original train distribution:
- label 0: 700
- label 1: 4200

Observed balanced train distribution:
- label 0: 700
- label 1: 700

Your first priority:
1. Verify label mapping.
2. Print train/validation/test label distributions.
3. Confirm model.classes_ and safe probability column handling.
4. Add confusion matrix output.
5. Add threshold sweep on validation set.
6. Save standardized metrics JSON.
7. Analyze predict_proba distribution for each class.
8. Do not optimize thresholds on the test set.
9. Avoid committing checkpoints or datasets unless explicitly configured.
10. Add .gitattributes to prevent LF/CRLF noise if missing.
11. Update PROJECT_STATUS.md after changes.

Be careful about frame leakage. Final evaluation should be video-level, not frame-level.
```

---

## 15. Final Current Status Summary

Current project status:

```text
- Frequency-domain deepfake detection project.
- Previous frame-level direction changed to video-level aggregation.
- Video-level frequency CSV training CLI appears to exist.
- Random Forest-style model appears to be used.
- Class balancing exists or was recently added.
- Threshold-based prediction exists.
- Several metric JSON files and threshold experiments exist locally.
- GitHub upload/staging work was recently performed.
```

Main unresolved issue:

```text
The model performs very poorly on label 0, assumed to be the real class.
```

Most important next step:

```text
Do not immediately add a new model. First diagnose the existing pipeline by verifying labels, probability columns, confusion matrix, threshold behavior, and probability distributions.
```

