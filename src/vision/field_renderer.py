"""Render NFL tracking coordinates as OpenCV field images."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FIELD_LENGTH_YARDS = 120.0
FIELD_WIDTH_YARDS = 53.3
DEFAULT_IMAGE_WIDTH = 1200
DEFAULT_IMAGE_HEIGHT = 533
PLAYER_RADIUS_PX = 12
BBOX_SIZE_PX = 40


def field_to_pixel(
    x: float,
    y: float,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
) -> tuple[int, int]:
    """Map NFL field coordinates (yards) to image pixel coordinates."""
    px = int(np.clip(x / FIELD_LENGTH_YARDS * width, 0, width - 1))
    py = int(np.clip((FIELD_WIDTH_YARDS - y) / FIELD_WIDTH_YARDS * height, 0, height - 1))
    return px, py


def pixel_to_field(
    px: int,
    py: int,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
) -> tuple[float, float]:
    """Map image pixel coordinates back to NFL field yards."""
    x = px / width * FIELD_LENGTH_YARDS
    y = FIELD_WIDTH_YARDS - (py / height * FIELD_WIDTH_YARDS)
    return x, y


def render_play_frame(
    play_df: pd.DataFrame,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
) -> np.ndarray:
    """
    Render a top-down field snapshot for one play using OpenCV.

    Offense (possession team) is drawn in blue, defense in red.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (34, 120, 34)  # field green (BGR)

    # Yard lines every 10 yards
    for yard in range(0, int(FIELD_LENGTH_YARDS) + 1, 10):
        px, _ = field_to_pixel(yard, 0, width, height)
        cv2.line(frame, (px, 0), (px, height - 1), (220, 220, 220), 1)

    possession = play_df["possessionTeam"].iloc[0] if "possessionTeam" in play_df.columns else None

    for _, player in play_df.iterrows():
        px, py = field_to_pixel(player["x"], player["y"], width, height)
        is_offense = possession is not None and player.get("team") == possession
        color = (220, 140, 40) if is_offense else (40, 60, 200)  # BGR
        cv2.circle(frame, (px, py), PLAYER_RADIUS_PX, color, -1)
        cv2.circle(frame, (px, py), PLAYER_RADIUS_PX, (255, 255, 255), 1)

    return frame


def tracking_to_bboxes(
    play_df: pd.DataFrame,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
    bbox_size: int = BBOX_SIZE_PX,
) -> list[dict]:
    """
    Convert tracking positions to pixel bounding boxes for CV feature extraction.

    Used when broadcast video is unavailable; mirrors YOLO detection format.
    """
    possession = play_df["possessionTeam"].iloc[0] if "possessionTeam" in play_df.columns else None
    bboxes = []
    half = bbox_size // 2

    for _, player in play_df.iterrows():
        px, py = field_to_pixel(player["x"], player["y"], width, height)
        is_offense = possession is not None and player.get("team") == possession
        bboxes.append(
            {
                "x1": max(0, px - half),
                "y1": max(0, py - half),
                "x2": min(width - 1, px + half),
                "y2": min(height - 1, py + half),
                "confidence": 1.0,
                "class_id": 0,
                "class_name": "person",
                "team": "offense" if is_offense else "defense",
                "field_x": player["x"],
                "field_y": player["y"],
                "speed": player.get("s", 0.0),
            }
        )
    return bboxes


def save_play_frame(
    play_df: pd.DataFrame,
    output_path: Path,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
) -> Path:
    """Render and save a play frame to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = render_play_frame(play_df, width, height)
    cv2.imwrite(str(output_path), frame)
    return output_path
