"""Predict expected goals for one forward in one upcoming game.

  python predict.py --player "Auston Matthews" --opponent BOS --away --date 2026-11-20
"""
import argparse
import joblib, numpy as np, pandas as pd
from pathlib import Path
from features import build_features, load_xg


def main(a):
    pg = pd.read_csv("data/processed/player_games.csv.gz")
    tg = pd.read_csv("data/processed/team_games.csv.gz")
    hit = pg[pg.player_id == a.player_id] if a.player_id else pg[pg.name.str.contains(a.player, case=False, regex=False)]
    if hit.empty:
        raise SystemExit("Player not found in data (forwards only).")
    if hit.player_id.nunique() > 1:
        raise SystemExit("Ambiguous name, matches: " + ", ".join(sorted(hit.name.unique())))
    hist = pg[pg.player_id == hit.player_id.iloc[0]].copy()
    last = hist.sort_values("game_date").iloc[-1]

    d = pd.Timestamp(a.date)
    row = last.copy()
    row[["game_id", "game_date", "opponent", "is_home", "season"]] = [-1, a.date, a.opponent.upper(), int(not a.away), d.year if d.month >= 9 else d.year - 1]
    for c in ("goals", "assists", "shots", "toi_min", "pp_goals", "pp_points", "shifts", "pim", "plus_minus"):
        row[c] = np.nan
    pxg, txg = load_xg()
    feats = build_features(pd.concat([hist, row.to_frame().T], ignore_index=True).infer_objects(), tg, pxg, txg)
    x = feats.iloc[[-1]]

    label = a.model_set or ("with_xg" if Path("models/models_with_xg.joblib").exists() else "base")
    bundle = joblib.load(f"models/models_{label}.joblib")
    X = x[bundle["features"]].fillna(feats[bundle["features"]].median())
    print(f"\n[{label} features]")
    print(f"{last['name']} ({last['team']}) vs {a.opponent.upper()} on {a.date} ({'away' if a.away else 'home'})\n")
    for name, m in bundle["reg"].items():
        lam = float(np.clip(m.predict(X)[0], 1e-4, None))
        print(f"  {name:12s} expected goals = {lam:.3f}   P(>=1 goal) = {1 - np.exp(-lam):.1%}")
    for name, m in bundle["clf"].items():
        print(f"  {name:12s}                      P(>=1 goal) = {m.predict_proba(X)[0, 1]:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default="")
    ap.add_argument("--player-id", type=int)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--away", action="store_true")
    ap.add_argument("--model-set", choices=["base", "with_xg"], help="default: with_xg if trained")
    main(ap.parse_args())
