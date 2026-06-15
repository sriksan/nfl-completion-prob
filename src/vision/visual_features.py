"""Extract computer-vision features from OpenCV frames and YOLO detections."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from detector import COCO_PERSON_CLASS, COCO_SPORTS_BALL_CLASS, YOLODetector
from field_renderer import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    render_play_frame,
    tracking_to_bboxes,
)

VISUAL_FEATURE_COLS = [
    "cv_receiver_defender_sep",
    "cv_qb_pressure_count",
    "cv_throw_lane_defenders",
    "cv_offense_spread",
    "cv_max_receiver_speed",
    "yolo_player_count",
    "yolo_ball_detected",
    "yolo_min_player_sep",
    "yolo_mean_confidence",
    "yolo_offense_width",
]


def _bbox_center(bbox: dict) -> tuple[float, float]:
    return (bbox["x1"] + bbox["x2"]) / 2, (bbox["y1"] + bbox["y2"]) / 2


def _euclidean_field(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x1 - x2, y1 - y2))


def _identify_roles(play_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Identify QB (slowest offensive player) and primary receiver (fastest offensive)."""
    possession = play_df["possessionTeam"].iloc[0]
    offense = play_df[play_df["team"] == possession].copy()
    defense = play_df[play_df["team"] != possession].copy()

    if offense.empty or defense.empty:
        return offense.iloc[:0], defense.iloc[:0]

    qb = offense.loc[offense["s"].idxmin()]
    receivers = offense[offense.index != qb.name]
    primary_receiver = receivers.loc[receivers["s"].idxmax()] if not receivers.empty else qb
    return primary_receiver, qb


def _features_from_tracking_bboxes(
    bboxes: list[dict],
    play_df: pd.DataFrame,
) -> dict[str, float]:
    """OpenCV-aligned geometric features from tracking-derived bounding boxes."""
    offense_boxes = [b for b in bboxes if b["team"] == "offense"]
    defense_boxes = [b for b in bboxes if b["team"] == "defense"]

    primary_receiver, qb = _identify_roles(play_df)

    defaults = {
        "cv_receiver_defender_sep": 0.0,
        "cv_qb_pressure_count": 0.0,
        "cv_throw_lane_defenders": 0.0,
        "cv_offense_spread": 0.0,
        "cv_max_receiver_speed": 0.0,
    }
    if not offense_boxes or not defense_boxes or primary_receiver.empty:
        return defaults

    recv_x, recv_y = primary_receiver["x"], primary_receiver["y"]
    qb_x, qb_y = qb["x"], qb["y"]

    min_sep = min(
        _euclidean_field(recv_x, recv_y, d["field_x"], d["field_y"])
        for d in defense_boxes
    )

    pressure_radius = 5.0
    qb_pressure = sum(
        1
        for d in defense_boxes
        if _euclidean_field(qb_x, qb_y, d["field_x"], d["field_y"]) <= pressure_radius
    )

    lane_width = 4.0
    throw_lane = 0
    for d in defense_boxes:
        dx, dy = d["field_x"] - qb_x, d["field_y"] - qb_y
        throw_len = np.hypot(recv_x - qb_x, recv_y - qb_y)
        if throw_len < 1:
            continue
        proj = (dx * (recv_x - qb_x) + dy * (recv_y - qb_y)) / throw_len
        if 0 < proj < throw_len:
            perp = abs(dx * (recv_y - qb_y) - dy * (recv_x - qb_x)) / throw_len
            if perp <= lane_width:
                throw_lane += 1

    offense_ys = [b["field_y"] for b in offense_boxes]
    offense_spread = float(np.std(offense_ys)) if len(offense_ys) > 1 else 0.0
    max_speed = float(play_df[play_df["team"] == play_df["possessionTeam"].iloc[0]]["s"].max())

    return {
        "cv_receiver_defender_sep": min_sep,
        "cv_qb_pressure_count": float(qb_pressure),
        "cv_throw_lane_defenders": float(throw_lane),
        "cv_offense_spread": offense_spread,
        "cv_max_receiver_speed": max_speed,
    }


def _features_from_yolo(detections: list) -> dict[str, float]:
    """Aggregate YOLOv8 detection statistics for a single frame."""
    persons = [d for d in detections if d.class_id == COCO_PERSON_CLASS]
    balls = [d for d in detections if d.class_id == COCO_SPORTS_BALL_CLASS]

    if not persons:
        return {
            "yolo_player_count": 0.0,
            "yolo_ball_detected": 0.0,
            "yolo_min_player_sep": 0.0,
            "yolo_mean_confidence": 0.0,
            "yolo_offense_width": 0.0,
        }

    centers = [d.center for d in persons]
    min_sep = float("inf")
    for i, c1 in enumerate(centers):
        for c2 in centers[i + 1 :]:
            min_sep = min(min_sep, np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
    if min_sep == float("inf"):
        min_sep = 0.0

    xs = [d.center[0] for d in persons]
    offense_width = float(max(xs) - min(xs)) if xs else 0.0

    return {
        "yolo_player_count": float(len(persons)),
        "yolo_ball_detected": float(len(balls) > 0),
        "yolo_min_player_sep": min_sep,
        "yolo_mean_confidence": float(np.mean([d.confidence for d in persons])),
        "yolo_offense_width": offense_width,
    }


def _find_video_for_play(
    game_id: int,
    play_id: int,
    video_dir: Path,
) -> Path | None:
    """Look for broadcast video clip matching a play."""
    candidates = [
        video_dir / f"{game_id}_{play_id}.mp4",
        video_dir / f"{game_id}_{play_id}.avi",
        video_dir / f"game_{game_id}" / f"play_{play_id}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def compute_play_visual_features(
    play_df: pd.DataFrame,
    video_dir: Path | None = None,
    use_yolo: bool = False,
    detector: YOLODetector | None = None,
    save_frame_dir: Path | None = None,
) -> dict[str, float]:
    """
    Compute CV features for one play at pass_forward.

    Always derives OpenCV/tracking features from field coordinates.
    Optionally runs YOLOv8 on broadcast video when available.
    """
    frame = render_play_frame(play_df)
    bboxes = tracking_to_bboxes(play_df)
    features = _features_from_tracking_bboxes(bboxes, play_df)

    if save_frame_dir is not None:
        game_id = int(play_df["gameId"].iloc[0])
        play_id = int(play_df["playId"].iloc[0])
        save_frame_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_frame_dir / f"{game_id}_{play_id}.png"), frame)

    yolo_features = {
        "yolo_player_count": 0.0,
        "yolo_ball_detected": 0.0,
        "yolo_min_player_sep": 0.0,
        "yolo_mean_confidence": 0.0,
        "yolo_offense_width": 0.0,
    }

    video_path = None
    if video_dir is not None:
        video_path = _find_video_for_play(
            int(play_df["gameId"].iloc[0]),
            int(play_df["playId"].iloc[0]),
            Path(video_dir),
        )

    if use_yolo:
        if detector is None:
            detector = YOLODetector()
        if video_path is not None:
            _, detections = detector.detect_video_frame(video_path)
        else:
            detections = detector.detect(frame)
        yolo_features = _features_from_yolo(detections)

    features.update(yolo_features)
    return features


def add_visual_features(
    df: pd.DataFrame,
    video_dir: str | Path | None = None,
    use_yolo: bool = False,
    save_frames: bool = False,
    frames_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Add play-level CV features to a player-level modeling dataframe.

    Features are computed once per play and broadcast to all player rows.
    """
    video_dir = Path(video_dir) if video_dir else None
    frames_dir = Path(frames_dir) if frames_dir else None
    if save_frames and frames_dir is None:
        frames_dir = Path("data/processed/frames")

    detector = YOLODetector() if use_yolo else None
    play_features = []

    for (game_id, play_id), play_df in df.groupby(["gameId", "playId"]):
        feats = compute_play_visual_features(
            play_df,
            video_dir=video_dir,
            use_yolo=use_yolo,
            detector=detector,
            save_frame_dir=frames_dir if save_frames else None,
        )
        feats["gameId"] = game_id
        feats["playId"] = play_id
        play_features.append(feats)

    features_df = pd.DataFrame(play_features)
    merged = df.merge(features_df, on=["gameId", "playId"], how="left")

    added = [c for c in VISUAL_FEATURE_COLS if c in merged.columns]
    print(f"Added {len(added)} computer-vision features")
    if use_yolo:
        print("  (YOLOv8 detection enabled)")
    else:
        print("  (OpenCV tracking features; pass use_yolo=True for YOLOv8)")

    return merged
