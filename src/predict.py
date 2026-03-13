"""
predict.py
Simulates the 2026 F1 season using the trained XGBoost model.

Approach:
  - Seed each driver's features from their 2024 end-of-season stats
  - Assign 2026 car performance tiers (quali pace proxy) based on known
    regulation changes and team/engine moves
  - Run a Monte Carlo simulation (N_SIMS runs) to smooth out randomness
  - Award F1 points each race, accumulate, print final standings

2026 grid assumptions (confirmed as of early 2026):
  Ferrari:      Hamilton, Leclerc
  Red Bull:     Verstappen, Lawson
  Mercedes:     Russell, Antonelli
  McLaren:      Norris, Piastri
  Aston Martin: Alonso, Stroll
  Alpine:       Gasly, Doohan
  Williams:     Sainz, Albon
  Haas:         Ocon, Bearman
  Audi:         Hulkenberg, Bortoleto
  Racing Bulls: Tsunoda, Hadjar
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb

RAW_DIR      = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SIMS   = 500   # Monte Carlo runs
N_ROUNDS = 24    # 2026 calendar length

POINTS_MAP = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

FEATURE_COLS = [
    "avg_finish_3", "avg_finish_5",
    "avg_points_3", "avg_points_5",
    "dnf_rate_3", "dnf_rate_5",
    "avg_grid_to_finish_5",
    "quali_position", "gap_to_pole_s", "vs_teammate_q_gap_s",
    "team_avg_points_5",
    "cum_points", "season_progress",
]

# ── 2026 grid ─────────────────────────────────────────────────────────────────
# Each driver maps to their 2026 team
GRID_2026 = {
    # driver_code : team
    "HAM": "Ferrari",
    "LEC": "Ferrari",
    "VER": "Red Bull Racing",
    "LAW": "Red Bull Racing",
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "GAS": "Alpine",
    "DOO": "Alpine",
    "SAI": "Williams",
    "ALB": "Williams",
    "OCO": "Haas",
    "BEA": "Haas",
    "HUL": "Audi",
    "BOR": "Audi",
    "TSU": "Racing Bulls",
    "HAD": "Racing Bulls",
}

# ── 2026 car performance tiers ────────────────────────────────────────────────
# gap_to_pole_s: estimated qualifying time gap to pole (seconds)
# Reflects 2026 regulation reset — closer field expected; Ferrari/McLaren/RBR lead.
# Based on 2024 constructor order + known 2026 engine/aero changes.
TEAM_QUALI_GAP = {
    "Ferrari":       0.05,   # strong 2026 engine + car
    "McLaren":       0.08,   # dominant 2024 car, evolution
    "Red Bull Racing": 0.12, # Honda engine change uncertainty
    "Mercedes":      0.18,   # new engine concept, Q4 rebound expected
    "Aston Martin":  0.45,
    "Williams":      0.55,
    "Alpine":        0.60,
    "Racing Bulls":  0.65,
    "Haas":          0.70,
    "Audi":          0.80,   # new entry, uncertain
}

# Typical quali position range per team (min, max) — used to sample quali grid
TEAM_QUALI_POS = {
    "Ferrari":        (1, 4),
    "McLaren":        (1, 5),
    "Red Bull Racing":(2, 6),
    "Mercedes":       (3, 8),
    "Aston Martin":   (6, 12),
    "Williams":       (8, 14),
    "Alpine":         (9, 15),
    "Racing Bulls":   (9, 16),
    "Haas":           (11, 17),
    "Audi":           (13, 20),
}

# DNF probability per driver (rough historical rate)
DNF_PROB = {
    "HAM": 0.06, "LEC": 0.10, "VER": 0.07, "LAW": 0.12,
    "RUS": 0.08, "ANT": 0.15, "NOR": 0.07, "PIA": 0.08,
    "ALO": 0.09, "STR": 0.13, "GAS": 0.11, "DOO": 0.18,
    "SAI": 0.09, "ALB": 0.10, "OCO": 0.12, "BEA": 0.16,
    "HUL": 0.10, "BOR": 0.18, "TSU": 0.12, "HAD": 0.18,
}


def load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(MODELS_DIR, "xgb_model.json"))
    return model


def seed_driver_stats() -> dict:
    """
    Pull each driver's trailing stats from 2024 end-of-season.
    For new/rookie drivers, fall back to midfield averages.
    """
    races = pd.read_csv(os.path.join(RAW_DIR, "race_results.csv"))
    races["finish_position"] = pd.to_numeric(races["finish_position"], errors="coerce")
    races["points"] = pd.to_numeric(races["points"], errors="coerce").fillna(0)
    races["dnf"] = (~races["status"].str.lower().str.startswith("finish")).astype(int)

    last5 = (
        races[races["season"] == 2024]
        .sort_values(["driver_code", "round"])
        .groupby("driver_code")
        .tail(5)
    )

    stats = {}
    for code in GRID_2026:
        d = last5[last5["driver_code"] == code]
        if len(d) >= 2:
            stats[code] = {
                "avg_finish_3":  d["finish_position"].tail(3).mean(),
                "avg_finish_5":  d["finish_position"].mean(),
                "avg_points_3":  d["points"].tail(3).mean(),
                "avg_points_5":  d["points"].mean(),
                "dnf_rate_3":    d["dnf"].tail(3).mean(),
                "dnf_rate_5":    d["dnf"].mean(),
                "avg_grid_to_finish_5": (
                    (pd.to_numeric(d["grid_position"], errors="coerce") -
                     d["finish_position"]).mean()
                ),
            }
        else:
            # Rookie / no 2024 data — use midfield defaults
            stats[code] = {
                "avg_finish_3": 12.0, "avg_finish_5": 12.0,
                "avg_points_3": 3.0,  "avg_points_5": 3.0,
                "dnf_rate_3":   0.15, "dnf_rate_5":   0.15,
                "avg_grid_to_finish_5": 0.0,
            }
    return stats


def sample_quali_grid(rng: np.random.Generator) -> dict[str, int]:
    """
    Assign a qualifying position to every driver for one race.
    Uses team pace tier with Gaussian noise, then rank to get integer positions.
    """
    scores = {}
    for driver, team in GRID_2026.items():
        lo, hi = TEAM_QUALI_POS[team]
        base = (lo + hi) / 2
        noise = rng.normal(0, (hi - lo) / 4)
        scores[driver] = base + noise

    ranked = sorted(scores, key=lambda d: scores[d])
    return {driver: pos + 1 for pos, driver in enumerate(ranked)}


def simulate_season(model: xgb.XGBRegressor, driver_stats: dict,
                    rng: np.random.Generator) -> dict[str, float]:
    """Run one full 2026 season, return points per driver."""
    stats = {d: dict(s) for d, s in driver_stats.items()}  # deep copy
    season_points = {d: 0.0 for d in GRID_2026}
    cum_points = {d: 0.0 for d in GRID_2026}

    for rnd in range(1, N_ROUNDS + 1):
        quali_grid = sample_quali_grid(rng)
        season_progress = (rnd - 1) / N_ROUNDS

        # Build feature matrix for this race
        rows = []
        for driver, team in GRID_2026.items():
            qpos = quali_grid[driver]
            gap  = TEAM_QUALI_GAP[team] + rng.normal(0, 0.03)
            teammate = [d for d, t in GRID_2026.items() if t == team and d != driver][0]
            vs_tm = gap - TEAM_QUALI_GAP[team]  # approx within-team gap

            team_pts = np.mean([cum_points[d] for d, t in GRID_2026.items() if t == team])

            row = {**stats[driver],
                   "quali_position":    qpos,
                   "gap_to_pole_s":     max(0, gap),
                   "vs_teammate_q_gap_s": vs_tm,
                   "team_avg_points_5": team_pts,
                   "cum_points":        cum_points[driver],
                   "season_progress":   season_progress}
            rows.append((driver, row))

        X = pd.DataFrame([r for _, r in rows])[FEATURE_COLS]
        preds = model.predict(X) + rng.normal(0, 1.0, size=len(X))  # small noise

        # Rank predictions → finish positions (handle DNFs)
        finish_scores = {}
        for i, (driver, _) in enumerate(rows):
            if rng.random() < DNF_PROB.get(driver, 0.10):
                finish_scores[driver] = 999  # DNF
            else:
                finish_scores[driver] = preds[i]

        ranked = sorted(finish_scores, key=lambda d: finish_scores[d])
        finish_pos = {driver: pos + 1 for pos, driver in enumerate(ranked)}

        # Award points
        race_pts = {}
        for driver, pos in finish_pos.items():
            pts = POINTS_MAP.get(pos, 0)
            season_points[driver] += pts
            cum_points[driver] += pts
            race_pts[driver] = pts

        # Update rolling stats
        for driver in GRID_2026:
            pos = finish_pos[driver]
            dnf = 1 if finish_scores[driver] == 999 else 0
            grid_gain = quali_grid[driver] - pos

            s = stats[driver]
            # Shift rolling windows (simple EWMA-style update)
            s["avg_finish_5"]  = s["avg_finish_5"]  * 0.8 + pos * 0.2
            s["avg_finish_3"]  = s["avg_finish_3"]  * 0.67 + pos * 0.33
            s["avg_points_5"]  = s["avg_points_5"]  * 0.8 + race_pts[driver] * 0.2
            s["avg_points_3"]  = s["avg_points_3"]  * 0.67 + race_pts[driver] * 0.33
            s["dnf_rate_5"]    = s["dnf_rate_5"]    * 0.8 + dnf * 0.2
            s["dnf_rate_3"]    = s["dnf_rate_3"]    * 0.67 + dnf * 0.33
            s["avg_grid_to_finish_5"] = s["avg_grid_to_finish_5"] * 0.8 + grid_gain * 0.2

    return season_points


def main():
    print("Loading model...")
    model = load_model()

    print("Seeding 2026 driver stats from 2024 data...")
    driver_stats = seed_driver_stats()

    print(f"Running {N_SIMS} Monte Carlo simulations...\n")
    rng = np.random.default_rng(42)

    all_points = {d: [] for d in GRID_2026}
    for sim in range(N_SIMS):
        result = simulate_season(model, driver_stats, rng)
        for d, pts in result.items():
            all_points[d].append(pts)

    # Aggregate
    rows = []
    for driver, team in GRID_2026.items():
        pts_arr = np.array(all_points[driver])
        rows.append({
            "driver":     driver,
            "team":       team,
            "avg_points": round(pts_arr.mean(), 1),
            "std_points": round(pts_arr.std(), 1),
            "p10":        round(np.percentile(pts_arr, 10), 0),
            "p90":        round(np.percentile(pts_arr, 90), 0),
        })

    standings = pd.DataFrame(rows).sort_values("avg_points", ascending=False).reset_index(drop=True)
    standings.index += 1

    print("=" * 65)
    print(f"  2026 F1 PREDICTED CHAMPIONSHIP STANDINGS  ({N_SIMS} simulations)")
    print("=" * 65)
    print(f"  {'#':<3} {'Driver':<6} {'Team':<20} {'Avg Pts':>8}  {'±':>5}  {'P10–P90':>12}")
    print("-" * 65)
    for rank, row in standings.iterrows():
        print(f"  {rank:<3} {row['driver']:<6} {row['team']:<20} "
              f"{row['avg_points']:>8.1f}  {row['std_points']:>5.1f}  "
              f"{int(row['p10']):>5}–{int(row['p90']):<5}")
    print("=" * 65)

    # Constructor standings
    print("\n  PREDICTED CONSTRUCTOR STANDINGS")
    print("-" * 40)
    constructor = standings.groupby("team")["avg_points"].sum().sort_values(ascending=False)
    for i, (team, pts) in enumerate(constructor.items(), 1):
        print(f"  {i}. {team:<22} {pts:.1f} pts")

    # Save
    out_path = os.path.join(RESULTS_DIR, "predicted_standings_2026.csv")
    standings.to_csv(out_path, index_label="rank")
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
