"""
features.py
Builds the feature matrix from raw race and qualifying CSVs.
Outputs data/processed/features.csv — one row per driver per race.
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    races = pd.read_csv(os.path.join(RAW_DIR, "race_results.csv"))
    qualis = pd.read_csv(os.path.join(RAW_DIR, "quali_results.csv"))

    races["finish_position"] = pd.to_numeric(races["finish_position"], errors="coerce")
    races["grid_position"] = pd.to_numeric(races["grid_position"], errors="coerce")
    races["points"] = pd.to_numeric(races["points"], errors="coerce").fillna(0)
    qualis["quali_position"] = pd.to_numeric(qualis["quali_position"], errors="coerce")

    races = races.sort_values(["season", "round"]).reset_index(drop=True)
    qualis = qualis.sort_values(["season", "round"]).reset_index(drop=True)

    return races, qualis


def add_dnf_flag(races: pd.DataFrame) -> pd.DataFrame:
    races["dnf"] = (~races["status"].str.lower().str.startswith("finish")).astype(int)
    return races


def rolling_driver_features(races: pd.DataFrame) -> pd.DataFrame:
    """
    Per driver: rolling stats over the previous 3 and 5 races.
    Shifted by 1 so the current race is not included in its own features.
    """
    df = races.copy()
    df = df.sort_values(["driver_code", "season", "round"])

    grp = df.groupby("driver_code")

    for window in (3, 5):
        df[f"avg_finish_{window}"] = (
            grp["finish_position"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_points_{window}"] = (
            grp["points"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"dnf_rate_{window}"] = (
            grp["dnf"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    df["grid_to_finish"] = df["grid_position"] - df["finish_position"]
    df["avg_grid_to_finish_5"] = (
        grp["grid_to_finish"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    return df


def add_quali_features(races: pd.DataFrame, qualis: pd.DataFrame) -> pd.DataFrame:
    q = qualis.copy()
    q["best_q_time_s"] = q[["q1_time_s", "q2_time_s", "q3_time_s"]].min(axis=1)

    pole_time = (
        q.groupby(["season", "round"])["best_q_time_s"]
        .min()
        .rename("pole_time_s")
        .reset_index()
    )
    q = q.merge(pole_time, on=["season", "round"])
    q["gap_to_pole_s"] = q["best_q_time_s"] - q["pole_time_s"]

    q_cols = q[["season", "round", "driver_code", "quali_position", "best_q_time_s", "gap_to_pole_s"]]
    return races.merge(q_cols, on=["season", "round", "driver_code"], how="left")


def add_teammate_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Gap to teammate in qualifying. Positive = driver is slower."""
    team_avg = (
        df.groupby(["season", "round", "team"])["best_q_time_s"]
        .mean()
        .rename("team_avg_q_time_s")
        .reset_index()
    )
    df = df.merge(team_avg, on=["season", "round", "team"], how="left")
    df["vs_teammate_q_gap_s"] = df["best_q_time_s"] - df["team_avg_q_time_s"]
    return df.drop(columns=["team_avg_q_time_s"])


def add_constructor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 5-race average combined points per constructor."""
    team_pts = (
        df.groupby(["season", "round", "team"])["points"]
        .sum()
        .reset_index()
        .rename(columns={"points": "team_pts_race"})
        .sort_values(["team", "season", "round"])
    )
    team_pts["team_avg_points_5"] = (
        team_pts.groupby("team")["team_pts_race"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    return df.merge(
        team_pts[["season", "round", "team", "team_avg_points_5"]],
        on=["season", "round", "team"],
        how="left",
    )


def add_season_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative driver points so far this season (excluding current race)."""
    df = df.sort_values(["driver_code", "season", "round"])
    df["cum_points"] = (
        df.groupby(["driver_code", "season"])["points"]
        .transform(lambda s: s.shift(1).cumsum().fillna(0))
    )
    max_round = df.groupby("season")["round"].transform("max")
    df["season_progress"] = (df["round"] - 1) / max_round
    return df


def build_features() -> pd.DataFrame:
    races, qualis = load_raw()
    races = add_dnf_flag(races)
    df = rolling_driver_features(races)
    df = add_quali_features(df, qualis)
    df = add_teammate_gap(df)
    df = add_constructor_features(df)
    df = add_season_progress(df)
    df = df.dropna(subset=["finish_position"])

    feature_cols = [
        "season", "round", "gp_name", "driver_code", "team",
        "finish_position",
        "avg_finish_3", "avg_finish_5",
        "avg_points_3", "avg_points_5",
        "dnf_rate_3", "dnf_rate_5",
        "avg_grid_to_finish_5",
        "quali_position", "gap_to_pole_s", "vs_teammate_q_gap_s",
        "team_avg_points_5",
        "cum_points", "season_progress",
    ]

    return df[feature_cols].reset_index(drop=True)


def main():
    print("Building features...")
    df = build_features()

    out_path = os.path.join(PROCESSED_DIR, "features.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows x {len(df.columns)} cols -> {out_path}")
    print(f"\nMissing values per feature:\n{df.isnull().sum().to_string()}")


if __name__ == "__main__":
    main()
