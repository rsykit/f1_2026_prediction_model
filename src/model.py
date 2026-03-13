"""
model.py
Trains an XGBoost model to predict finish position from engineered features.
Saves the trained model to models/xgb_model.json.
Prints cross-validation MAE and feature importances.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import mean_absolute_error
import json

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "avg_finish_3", "avg_finish_5",
    "avg_points_3", "avg_points_5",
    "dnf_rate_3", "dnf_rate_5",
    "avg_grid_to_finish_5",
    "quali_position", "gap_to_pole_s", "vs_teammate_q_gap_s",
    "team_avg_points_5",
    "cum_points", "season_progress",
]
TARGET = "finish_position"


def load_features() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "features.csv")
    df = pd.read_csv(path)
    return df


def train(df: pd.DataFrame) -> xgb.XGBRegressor:
    X = df[FEATURE_COLS]
    y = df[TARGET]

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
    )

    # Group k-fold by season so we never leak future races into training
    # (each fold holds out one full season)
    groups = df["season"]
    gkf = GroupKFold(n_splits=3)

    scores = cross_val_score(
        model, X, y,
        cv=gkf,
        groups=groups,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    maes = -scores
    print(f"Cross-val MAE by fold: {maes.round(3)}")
    print(f"Mean MAE: {maes.mean():.3f}  ±  {maes.std():.3f}")
    print("(MAE in finishing positions — e.g. 2.5 means off by ~2-3 places on average)")

    # Train final model on all data
    model.fit(X, y)
    return model


def print_feature_importance(model: xgb.XGBRegressor) -> None:
    importance = model.feature_importances_
    pairs = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)
    print("\nFeature importances:")
    for name, score in pairs:
        bar = "█" * int(score * 200)
        print(f"  {name:<28} {score:.4f}  {bar}")


def save_model(model: xgb.XGBRegressor) -> str:
    path = os.path.join(MODELS_DIR, "xgb_model.json")
    model.save_model(path)
    return path


def main():
    print("Loading features...")
    df = load_features()
    print(f"  {len(df)} rows, {len(FEATURE_COLS)} features")

    print("\nTraining model...")
    model = train(df)

    print_feature_importance(model)

    path = save_model(model)
    print(f"\nModel saved → {path}")


if __name__ == "__main__":
    main()
