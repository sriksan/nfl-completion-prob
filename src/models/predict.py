"""Predict pass completion probability using trained XGBoost model."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
project_root = src_dir.parent

sys.path.append(str(src_dir / "data"))
sys.path.append(str(src_dir / "features"))
sys.path.append(str(src_dir / "vision"))

from merge_sources import create_modeling_dataset
from contextual_features import add_contextual_features
from spatial_features import add_spatial_features, SPATIAL_FEATURE_COLS
from visual_features import add_visual_features, VISUAL_FEATURE_COLS


CONTEXTUAL_FEATURES = [
    "down",
    "yardsToGo",
    "quarter",
    "is_first_down",
    "is_second_down",
    "is_third_down",
    "is_fourth_down",
    "is_short_yardage",
    "is_medium_yardage",
    "is_long_yardage",
    "is_third_and_long",
]


def load_model(model_path: Path, features_path: Path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    return model, feature_cols


def build_feature_row(
    down: int,
    yards_to_go: int,
    quarter: int,
    spatial: dict | None = None,
    visual: dict | None = None,
) -> pd.DataFrame:
    """Build a single play-level feature row for inference."""
    row = {
        "down": down,
        "yardsToGo": yards_to_go,
        "quarter": quarter,
        "is_first_down": int(down == 1),
        "is_second_down": int(down == 2),
        "is_third_down": int(down == 3),
        "is_fourth_down": int(down == 4),
        "is_short_yardage": int(yards_to_go <= 3),
        "is_medium_yardage": int(3 < yards_to_go < 10),
        "is_long_yardage": int(yards_to_go >= 10),
        "is_third_and_long": int(down == 3 and yards_to_go >= 7),
    }
    if spatial:
        row.update(spatial)
    if visual:
        row.update(visual)
    return pd.DataFrame([row])


def predict_from_dataset(
    model_path: Path,
    features_path: Path,
    use_yolo: bool = False,
    video_dir: str | None = None,
) -> pd.DataFrame:
    """Run inference on the full modeling dataset."""
    model, feature_cols = load_model(model_path, features_path)

    df = create_modeling_dataset()
    df = add_contextual_features(df)
    df = add_spatial_features(df)
    df = add_visual_features(df, video_dir=video_dir, use_yolo=use_yolo, save_frames=True)

    play_df = df.groupby("playId").agg(
        {"gameId": "first", "complete": "first", **{f: "first" for f in feature_cols if f in df.columns}}
    ).reset_index()

    available = [f for f in feature_cols if f in play_df.columns]
    X = play_df[available].values
    probs = model.predict_proba(X)[:, 1]
    play_df["completion_probability"] = probs
    return play_df


def main():
    parser = argparse.ArgumentParser(description="Predict NFL pass completion probability")
    parser.add_argument("--model", type=Path, required=True, help="Path to .pkl model")
    parser.add_argument("--features", type=Path, required=True, help="Path to feature list .txt")
    parser.add_argument("--down", type=int, help="Down (1-4) for single prediction")
    parser.add_argument("--yards", type=int, help="Yards to go for single prediction")
    parser.add_argument("--quarter", type=int, default=1, help="Quarter for single prediction")
    parser.add_argument("--dataset", action="store_true", help="Predict on full dataset")
    parser.add_argument("--use-yolo", action="store_true", help="Enable YOLOv8 detection")
    parser.add_argument("--video-dir", type=Path, default=project_root / "data" / "video")
    args = parser.parse_args()

    if args.dataset:
        results = predict_from_dataset(args.model, args.features, args.use_yolo, args.video_dir)
        print(results[["playId", "complete", "completion_probability"]].head(10))
        return

    if args.down is None or args.yards is None:
        parser.error("--down and --yards are required unless --dataset is set")

    model, feature_cols = load_model(args.model, args.features)
    row = build_feature_row(args.down, args.yards, args.quarter)
    available = [f for f in feature_cols if f in row.columns]
    missing = [f for f in feature_cols if f not in row.columns]
    if missing:
        print(f"Warning: missing features defaulting to 0: {missing}")
        for col in missing:
            row[col] = 0.0
        available = feature_cols

    prob = model.predict_proba(row[available].values)[:, 1][0]
    print(f"Completion probability: {prob:.1%}")


if __name__ == "__main__":
    main()
