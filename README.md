# NFL Pass Completion Probability

Predict NFL pass completion probability using play-level tracking data, contextual features, spatial geometry, and computer vision (OpenCV + YOLOv8). The primary model is an XGBoost classifier trained on features extracted at the moment the ball is thrown (`pass_forward`).

## Overview

This project combines tabular NFL tracking data with a computer vision pipeline:

1. **Contextual features** — down, distance, quarter, and situational flags
2. **Spatial features** — receiver separation, QB pressure, pass length from tracking coordinates
3. **Computer vision features** — OpenCV-rendered field snapshots and optional YOLOv8 detections from broadcast video
4. **XGBoost** — binary classifier outputting completion probability

The baseline uses NFL Big Data Bowl-style CSVs (games, plays, players, tracking). No broadcast video is required to run the default pipeline; OpenCV features are derived from tracking positions rendered as top-down field images.

## Project Structure

```
nfl-completion-prob/
├── data/
│   ├── raw/                  # games.csv, plays.csv, players.csv, tracking_week1.csv
│   ├── processed/frames/     # OpenCV-rendered field images (generated)
│   └── video/                # Optional broadcast clips for YOLOv8
├── models/                   # Saved XGBoost models and feature lists
├── results/                  # Metrics and feature importance CSVs
├── src/
│   ├── data/
│   │   ├── load_data.py      # CSV loaders
│   │   └── merge_sources.py  # Build modeling dataset
│   ├── features/
│   │   ├── contextual_features.py
│   │   └── spatial_features.py
│   ├── vision/
│   │   ├── field_renderer.py # OpenCV top-down field rendering
│   │   ├── detector.py       # YOLOv8 wrapper
│   │   └── visual_features.py
│   └── models/
│       ├── train_baseline.py # Training pipeline
│       └── predict.py        # Inference CLI
└── requirements.txt
```

## Setup

```bash
git clone <repo-url>
cd nfl-completion-prob

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Data

Place the following files in `data/raw/` (not included in the repo due to size):

| File | Description |
|------|-------------|
| `games.csv` | Game metadata |
| `plays.csv` | Play-by-play data |
| `players.csv` | Player info |
| `tracking_week1.csv` | Player tracking per frame |

The modeling pipeline filters tracking data to `pass_forward` events (ball release) and merges with play context. The current week 1 subset contains ~68 pass plays.

## Usage

### Train the model

```bash
python src/models/train_baseline.py
```

This will:

- Load and merge tracking + play data
- Engineer contextual, spatial, and CV features
- Save OpenCV field frames to `data/processed/frames/`
- Train logistic regression and XGBoost (80/20 stratified split)
- Write model artifacts to `models/` and metrics to `results/`

### Enable YOLOv8

In `src/models/train_baseline.py`, set:

```python
USE_YOLO = True
```

Optionally place broadcast clips in `data/video/` using the naming convention:

```
data/video/{gameId}_{playId}.mp4
```

YOLOv8 detects players and the ball on video frames (or rendered field images when no video is available) and adds detection-based features to the model.

### Run inference

Single prediction from situational inputs:

```bash
python src/models/predict.py \
  --model models/cv_xgboost_<timestamp>.pkl \
  --features models/cv_features_<timestamp>.txt \
  --down 3 --yards 7 --quarter 1
```

Predict on the full dataset:

```bash
python src/models/predict.py \
  --model models/cv_xgboost_<timestamp>.pkl \
  --features models/cv_features_<timestamp>.txt \
  --dataset
```

Add `--use-yolo` to run YOLOv8 during dataset inference.

## Features

### Contextual (11)

Down, yards to go, quarter, and binary flags for down type, yardage bucket, and third-and-long.

### Spatial (5)

Derived from tracking `x`, `y`, `s` at `pass_forward`:

- `spatial_receiver_defender_sep` — minimum distance from primary receiver to nearest defender
- `spatial_qb_pressure` — defenders within 5 yards of the QB
- `spatial_pass_length` — distance from QB to primary receiver
- `spatial_avg_defender_speed` — mean defender speed
- `spatial_receiver_depth` — receiver depth relative to QB

### Computer vision (10)

OpenCV geometry from rendered field images, plus optional YOLOv8 outputs:

- `cv_receiver_defender_sep`, `cv_qb_pressure_count`, `cv_throw_lane_defenders`, `cv_offense_spread`, `cv_max_receiver_speed`
- `yolo_player_count`, `yolo_ball_detected`, `yolo_min_player_sep`, `yolo_mean_confidence`, `yolo_offense_width`

## Pipeline

```
tracking CSV  ──► merge with plays  ──► contextual features
                    │                      spatial features
                    ▼                      visual features (OpenCV / YOLOv8)
              pass_forward frames              │
                    │                          │
                    └──────────► aggregate per play ──► XGBoost ──► completion probability
```

## Model Outputs

After training, artifacts are timestamped under:

- `models/cv_xgboost_<timestamp>.pkl` — trained XGBoost model
- `models/cv_features_<timestamp>.txt` — feature column order for inference
- `results/cv_metrics_<timestamp>.csv` — log loss, AUC, accuracy
- `results/cv_feature_importance_<timestamp>.csv` — XGBoost feature importances

## Dependencies

- **pandas**, **numpy** — data handling
- **xgboost**, **scikit-learn** — modeling
- **opencv-python** — field rendering and image I/O
- **ultralytics** — YOLOv8 object detection
- **matplotlib** — plotting (optional)
