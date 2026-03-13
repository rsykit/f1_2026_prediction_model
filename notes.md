# F1 2026 Prediction — Project Notes

## Data

**Seasons fetched:** 2022, 2023, 2024
**Raw files:** `data/raw/race_results.csv` (1,200 rows), `data/raw/quali_results.csv` (1,359 rows)

---

## Feature Engineering — Missing Values

Output: `data/processed/features.csv` — 1 row per driver per race, 19 columns.

| Feature | Missing | Reason |
|---|---|---|
| `avg_finish_3/5` | 28 | Driver's debut race — no prior history to roll over |
| `avg_points_3/5` | 28 | Same as above |
| `dnf_rate_3/5` | 28 | Same as above |
| `avg_grid_to_finish_5` | 28 | Same as above |
| `gap_to_pole_s` | 9 | Qualifying sessions skipped due to API rate limit |
| `vs_teammate_q_gap_s` | 9 | Same as above |
| `team_avg_points_5` | 24 | Team's first few races with no prior history |
| `quali_position`, `finish_position`, `cum_points` | **0** | Clean |

**Note:** XGBoost handles NaN natively — no imputation needed.

---

## Model Training Results

**Algorithm:** XGBoost Regressor (predicts finish position)
**Validation:** GroupKFold (k=3, grouped by season — no future leakage)

| Fold | MAE |
|------|-----|
| 1 (hold out one season) | 3.501 |
| 2 | 3.851 |
| 3 | 3.416 |
| **Mean** | **3.589 ± 0.189** |

A MAE of ~3.6 finishing positions is reasonable — races have genuine randomness (safety cars, DNFs, rain) that no model can predict.

**Feature importances (top 5):**

| Feature | Importance | Interpretation |
|---------|-----------|----------------|
| `quali_position` | 0.2745 | Where you start largely determines where you finish in modern F1 |
| `avg_points_5` | 0.1544 | Recent driver form (last 5 races) |
| `team_avg_points_5` | 0.1107 | Constructor / car quality |
| `gap_to_pole_s` | 0.0598 | Raw pace gap to fastest qualifier |
| `cum_points` | 0.0585 | Season-long championship momentum |

**Key takeaway:** Qualifying dominates. Car quality + recent form are the next biggest signals. Individual race-over-race noise features (DNF rate, grid-to-finish delta) contribute but are minor.

---

## 2026 Prediction Results (500 Monte Carlo simulations, 24 rounds)

### Driver Championship
| # | Driver | Team | Avg Pts | ± | P10–P90 |
|---|--------|------|---------|---|---------|
| 1 | HAM | Ferrari | 383.6 | 103.4 | 263–535 |
| 2 | VER | Red Bull Racing | 332.3 | 87.7 | 216–447 |
| 3 | LEC | Ferrari | 310.0 | 71.2 | 232–409 |
| 4 | PIA | McLaren | 307.7 | 84.9 | 219–433 |
| 5 | NOR | McLaren | 291.1 | 68.4 | 224–393 |
| 6 | LAW | Red Bull Racing | 232.5 | 48.4 | 195–302 |
| 7 | RUS | Mercedes | 208.1 | 53.2 | 145–278 |
| 8 | ANT | Mercedes | 184.0 | 32.7 | 152–221 |
| 9 | ALO | Aston Martin | 51.0 | 18.8 | 31–75 |
| 10 | STR | Aston Martin | 43.5 | 15.7 | 28–63 |
| 11 | SAI | Williams | 31.0 | 18.1 | 14–52 |
| 12 | ALB | Williams | 19.6 | 10.3 | 11–29 |
| 13 | GAS | Alpine | 10.2 | 8.6 | 3–19 |
| 14 | DOO | Alpine | 5.9 | 4.5 | 2–12 |
| 15 | TSU | Racing Bulls | 5.5 | 4.0 | 1–11 |
| 16 | OCO | Haas | 3.8 | 3.7 | 0–9 |
| 17 | HAD | Racing Bulls | 3.7 | 3.4 | 0–9 |
| 18 | BEA | Haas | 0.6 | 1.1 | 0–2 |
| 19 | HUL | Audi | 0.1 | 0.4 | 0–0 |
| 20 | BOR | Audi | 0.1 | 0.4 | 0–0 |

### Constructor Championship
| # | Team | Avg Pts |
|---|------|---------|
| 1 | Ferrari | 693.6 |
| 2 | McLaren | 598.8 |
| 3 | Red Bull Racing | 564.8 |
| 4 | Mercedes | 392.1 |
| 5 | Aston Martin | 94.5 |
| 6 | Williams | 50.6 |
| 7 | Alpine | 16.1 |
| 8 | Racing Bulls | 9.2 |
| 9 | Haas | 4.4 |
| 10 | Audi | 0.2 |

### Key Takeaways
- **Hamilton predicted WDC** — strong 2024 race-craft stats + Ferrari pegged as fastest 2026 car. Wide band (263–535) means a bad season could drop him to 3rd.
- **Verstappen 2nd** — elite 2024 stats, but Red Bull's Honda engine change uncertainty costs him vs Ferrari.
- **Piastri edges Norris (4th vs 5th)** — driven purely by Piastri's better 2024 average finish positions.
- **Mercedes 4th in constructors** — entirely a function of the `TEAM_QUALI_GAP = 0.18` assumption for their new engine concept. The biggest lever to adjust if Mercedes looks stronger in testing.
- **Audi scores near zero** — realistic for a brand new constructor entry with unknown pace.
- **Rookies underrated by design** — Antonelli, Doohan, Hadjar, Bortoleto seeded at generic midfield defaults. Model will underestimate any standout rookie talent.

### Assumptions to revisit as 2026 testing data emerges
- `TEAM_QUALI_GAP` values in `src/predict.py` — the single biggest lever in the model
- Rookie driver baselines (currently all set to midfield avg finish = 12.0)
- DNF probabilities per driver
