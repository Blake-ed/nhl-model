# NHL Forward Expected-Goals Model

Predicts a forward's **expected goals (λ)** and **P(≥1 goal)** for a given game, using only
information available before puck drop. Forwards only, regular season, 2018-19 → 2025-26.

## Run it
```bash
pip install -r requirements.txt
python collect_data.py --start 2018 --end 2025   # forward game logs + team results (~20-40 min, cached)
python collect_pbp.py                            # play-by-play -> every unblocked shot (~1 hr, resumable)
python shot_model.py                             # shot-quality (xG) model + per-player / per-team xG
python features.py                               # leakage-safe feature table (adds xG features if present)
python train.py                                  # compares 6 models + baseline, with and without xG
python predict.py --player "Auston Matthews" --opponent BOS --away --date 2026-11-20
```
`python make_synthetic_data.py` creates fake data (including shots) so you can test the pipeline offline.
The xG stage is optional: skip `collect_pbp.py` / `shot_model.py` and everything still runs on base features.

## Models compared
| Target | Models |
|---|---|
| Expected goals (Poisson) | Poisson GLM, Random Forest (Poisson criterion), XGBoost (`count:poisson`) |
| P(score ≥ 1) | Logistic regression, Random Forest, XGBoost classifier |

Poisson models convert to a scoring probability via `1 - exp(-λ)`, so all models share log loss,
Brier score and AUC. A "career goals/game" baseline is included: **beat it, or the model adds nothing.**

## Features (all computed from prior games only)
Rolling goals / shots / TOI (5, 10, 20 games), season-to-date and career rates, PP goals,
shrunk career shooting %, home/away, rest days & back-to-back, games played,
opponent goals-for/against (last 10), own team goals-for, position.

## Shot quality (xG) stage
`shot_model.py` scores every unblocked shot attempt with P(goal) from distance, angle, shot type, rebound, rush,
strength state and empty net. No player identity is used, so xG measures the *chance*, not the shooter.
Each season is scored by a model trained on the other seasons, so no shot is scored by a model that saw its result.
Per player-game it aggregates xG, power-play and even-strength xG, attempts and high-danger attempts. Per team-game
it aggregates xG for/against. New features: rolling xG (5/10/20 games), xG per 60, PP/EV xG,
attempts, high-danger chances, opponent xG allowed, and a shrunk "finishing" ratio (goals vs xG).
Actual goals stay in the model alongside xG. `train.py` runs both feature sets on identical rows and prints a
"Did shot quality help?" table.

## Validation
Time-based split: train ≤ 2023-24, early-stop on 2024-25, test on 2025-26. Outputs land in `reports/`
(`metrics.csv`, `calibration.png`, `feature_importance.csv`, `test_predictions.csv`).

## Known limitations / next steps
- Game logs lack **power-play TOI, line assignment, and shot quality (xG)**; these are the biggest upgrades
  (NHL play-by-play has shot coordinates for a true xG shot model).
- Injuries / scratches / lineup changes aren't known; add confirmed lineups for real use.
- Opponent goalie quality isn't modeled.
- Scores include shootout-winner goal in team totals (minor).
- Final step for production: refit on train+val+test with chosen hyperparameters.
- Expect modest signal: goal scoring is noisy, so AUC around 0.60–0.65 is realistic; judge by calibration.
