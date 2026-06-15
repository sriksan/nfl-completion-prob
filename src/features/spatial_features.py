"""Spatial features derived from NFL tracking coordinates at pass_forward."""

import numpy as np
import pandas as pd

SPATIAL_FEATURE_COLS = [
    "spatial_receiver_defender_sep",
    "spatial_qb_pressure",
    "spatial_pass_length",
    "spatial_avg_defender_speed",
    "spatial_receiver_depth",
]


def _euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x1 - x2, y1 - y2))


def _compute_play_spatial(play_df: pd.DataFrame) -> dict[str, float]:
    possession = play_df["possessionTeam"].iloc[0]
    offense = play_df[play_df["team"] == possession]
    defense = play_df[play_df["team"] != possession]

    defaults = {col: 0.0 for col in SPATIAL_FEATURE_COLS}
    if offense.empty or defense.empty:
        return defaults

    qb = offense.loc[offense["s"].idxmin()]
    receivers = offense[offense.index != qb.name]
    if receivers.empty:
        return defaults

    primary_receiver = receivers.loc[receivers["s"].idxmax()]

    recv_pos = (primary_receiver["x"], primary_receiver["y"])
    qb_pos = (qb["x"], qb["y"])

    min_sep = min(_euclidean(recv_pos[0], recv_pos[1], d["x"], d["y"]) for _, d in defense.iterrows())

    pressure = sum(1 for _, d in defense.iterrows() if _euclidean(qb_pos[0], qb_pos[1], d["x"], d["y"]) <= 5.0)

    pass_length = _euclidean(recv_pos[0], recv_pos[1], qb_pos[0], qb_pos[1])
    avg_def_speed = float(defense["s"].mean())
    receiver_depth = float(recv_pos[0] - qb_pos[0])

    pass_length_col = play_df["PassLength"].iloc[0] if "PassLength" in play_df.columns else np.nan
    if pd.notna(pass_length_col) and str(pass_length_col).strip():
        try:
            pass_length = float(pass_length_col)
        except (TypeError, ValueError):
            pass

    return {
        "spatial_receiver_defender_sep": min_sep,
        "spatial_qb_pressure": float(pressure),
        "spatial_pass_length": pass_length,
        "spatial_avg_defender_speed": avg_def_speed,
        "spatial_receiver_depth": receiver_depth,
    }


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add play-level spatial tracking features to player-level modeling data."""
    play_features = []
    for (game_id, play_id), play_df in df.groupby(["gameId", "playId"]):
        feats = _compute_play_spatial(play_df)
        feats["gameId"] = game_id
        feats["playId"] = play_id
        play_features.append(feats)

    features_df = pd.DataFrame(play_features)
    merged = df.merge(features_df, on=["gameId", "playId"], how="left")
    print(f"Added {len(SPATIAL_FEATURE_COLS)} spatial tracking features")
    return merged
