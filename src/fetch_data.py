"""
fetch_data.py
Pulls historical F1 race and qualifying data using the FastF1 API.
Saves raw CSVs to data/raw/ for use in feature engineering.
"""

import os
import time
import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

# ── Cache setup (speeds up repeated runs) ────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Seasons to pull (2022–2024 confirmed; 2025 as it completes)
SEASONS = [2022, 2023, 2024]


def fetch_race_results(season: int) -> pd.DataFrame:
    """Return a DataFrame of every race result for a given season."""
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    rows = []

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        gp_name = event["EventName"]

        for attempt in range(3):
            try:
                session = fastf1.get_session(season, round_num, "R")
                session.load(telemetry=False, weather=False, messages=False)
                results = session.results

                for _, driver in results.iterrows():
                    rows.append({
                        "season":           season,
                        "round":            round_num,
                        "gp_name":          gp_name,
                        "driver_number":    driver["DriverNumber"],
                        "driver_code":      driver["Abbreviation"],
                        "full_name":        driver["FullName"],
                        "team":             driver["TeamName"],
                        "grid_position":    driver["GridPosition"],
                        "finish_position":  driver["Position"],
                        "points":           driver["Points"],
                        "status":           driver["Status"],
                        "fastest_lap":      driver.get("FastestLap", False),
                    })

                print(f"  [OK] {season} R{round_num:02d} – {gp_name}")
                time.sleep(1)
                break

            except RateLimitExceededError:
                wait = 65 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s before retry ({attempt+1}/3)...")
                time.sleep(wait)
            except Exception as e:
                print(f"  [SKIP] {season} R{round_num:02d} – {gp_name}: {e}")
                break

    return pd.DataFrame(rows)


def fetch_qualifying_results(season: int) -> pd.DataFrame:
    """Return a DataFrame of every qualifying result for a given season."""
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    rows = []

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        gp_name = event["EventName"]

        for attempt in range(3):
            try:
                session = fastf1.get_session(season, round_num, "Q")
                session.load(telemetry=False, weather=False, messages=False)
                results = session.results

                for _, driver in results.iterrows():
                    rows.append({
                        "season":           season,
                        "round":            round_num,
                        "gp_name":          gp_name,
                        "driver_code":      driver["Abbreviation"],
                        "team":             driver["TeamName"],
                        "q1_time_s":        driver["Q1"].total_seconds() if pd.notna(driver["Q1"]) else None,
                        "q2_time_s":        driver["Q2"].total_seconds() if pd.notna(driver["Q2"]) else None,
                        "q3_time_s":        driver["Q3"].total_seconds() if pd.notna(driver["Q3"]) else None,
                        "quali_position":   driver["Position"],
                    })

                print(f"  [OK] {season} Q{round_num:02d} – {gp_name}")
                time.sleep(1)
                break

            except RateLimitExceededError:
                wait = 65 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s before retry ({attempt+1}/3)...")
                time.sleep(wait)
            except Exception as e:
                print(f"  [SKIP] {season} Q{round_num:02d} – {gp_name}: {e}")
                break

    return pd.DataFrame(rows)


def main():
    all_races = []
    all_qualis = []

    for season in SEASONS:
        print(f"\n── Season {season} ─────────────────────────────")

        print("Fetching race results...")
        race_df = fetch_race_results(season)
        all_races.append(race_df)

        print("Fetching qualifying results...")
        quali_df = fetch_qualifying_results(season)
        all_qualis.append(quali_df)

    races = pd.concat(all_races, ignore_index=True)
    qualis = pd.concat(all_qualis, ignore_index=True)

    race_path = os.path.join(RAW_DIR, "race_results.csv")
    quali_path = os.path.join(RAW_DIR, "quali_results.csv")

    races.to_csv(race_path, index=False)
    qualis.to_csv(quali_path, index=False)

    print(f"\nSaved {len(races)} race rows  → {race_path}")
    print(f"Saved {len(qualis)} quali rows → {quali_path}")


if __name__ == "__main__":
    main()
