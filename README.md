# F1 2026 Championship Prediction Model

Predicts the 2026 Formula 1 World Championship standings using historical race data and machine learning.

## How it works

1. **Fetch** — pulls 2022–2024 race and qualifying results via the [FastF1](https://docs.fastf1.dev/) API
2. **Features** — engineers rolling form, qualifying pace, DNF rate, teammate gaps, constructor strength
3. **Train** — XGBoost regression model predicts finish position per driver per race
4. **Predict** — Monte Carlo simulation runs 500 full 2026 seasons to produce a probability distribution of championship outcomes

## Results

| # | Driver | Team | Avg Pts | P10–P90 |
|---|--------|------|---------|---------|
| 1 | HAM | Ferrari | 383.6 | 263–535 |
| 2 | VER | Red Bull Racing | 332.3 | 216–447 |
| 3 | LEC | Ferrari | 310.0 | 232–409 |
| 4 | PIA | McLaren | 307.7 | 219–433 |
| 5 | NOR | McLaren | 291.1 | 224–393 |

**Constructors:** Ferrari > McLaren > Red Bull > Mercedes

Full standings in [`data/results/predicted_standings_2026.csv`](data/results/predicted_standings_2026.csv)

## Model performance

Validated with GroupKFold cross-validation (grouped by season to prevent data leakage):

- **Mean MAE: 3.59 ± 0.19 finishing positions**

Top features by importance: qualifying position (0.27), rolling points avg (0.15), constructor strength (0.11)

## Project structure

```
src/
  fetch_data.py   # pull raw data from FastF1 API
  features.py     # engineer feature matrix
  model.py        # train and evaluate XGBoost model
  predict.py      # simulate 2026 season
data/
  raw/            # race_results.csv, quali_results.csv (2022–2024)
  processed/      # features.csv
  results/        # predicted_standings_2026.csv
models/           # saved xgb_model.json (generated, not committed)
notes.md          # methodology notes and result analysis
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install fastf1 pandas numpy xgboost scikit-learn

# macOS only (XGBoost dependency)
brew install libomp
```

## Usage

```bash
# 1. Fetch historical data (~25 min first run, instant after caching)
python src/fetch_data.py

# 2. Build feature matrix
python src/features.py

# 3. Train model
python src/model.py

# 4. Generate 2026 predictions
python src/predict.py
```

## Key assumptions

The prediction is only as good as its inputs. The biggest levers to adjust in [`src/predict.py`](src/predict.py):

- **`TEAM_QUALI_GAP`** — estimated qualifying pace gap per team. Based on 2024 constructor performance + known 2026 engine/regulation changes. This is the single most impactful parameter.
- **Rookie baselines** — Antonelli, Doohan, Hadjar, Bortoleto are seeded at midfield defaults (no F1 data). Any standout rookie will be underrated.
- **`DNF_PROB`** — per-driver DNF probability, set manually from historical trends.

Re-run `predict.py` after updating these values as pre-season testing data emerges.

## 2026 grid

Ferrari (HAM, LEC) · Red Bull (VER, LAW) · Mercedes (RUS, ANT) · McLaren (NOR, PIA) · Aston Martin (ALO, STR) · Alpine (GAS, DOO) · Williams (SAI, ALB) · Haas (OCO, BEA) · Audi (HUL, BOR) · Racing Bulls (TSU, HAD)
