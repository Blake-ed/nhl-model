"""Stage 2: shot-quality (xG) model.

Estimates P(goal) for every unblocked shot attempt from geometry and context only
(distance, angle, shot type, rebound, rush, strength, empty net). No player identity is used,
so xG measures the *chance*, not the shooter.

Each season's shots are scored by a model trained on the OTHER seasons (leave-one-season-out),
so no shot is scored by a model that saw its own outcome.

Outputs (data/processed/):
  shots_xg.csv.gz   shots with xg
  player_xg.csv.gz  per player-game: ixg, ixg_pp, ixg_ev, iff (attempts), hd_att (xG >= 0.15)
  team_xg.csv.gz    per team-game: xgf, xga
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

SHOT_TYPES = ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around", "bat", "poke", "cradle", "between-legs"]
HD_THRESHOLD = 0.15


def shot_features(s):
    X = pd.DataFrame(index=s.index)
    dx = 89 - s["x_att"]
    X["distance"] = np.hypot(dx, s["y"])
    X["angle"] = np.degrees(np.arctan2(s["y"].abs(), dx))
    X["x_att"], X["abs_y"] = s["x_att"], s["y"].abs()
    for t in SHOT_TYPES:
        X[f"type_{t}"] = (s["shot_type"] == t).astype(int)
    X["skater_diff"] = s["skater_diff"]
    X["is_pp"] = (s["skater_diff"] > 0).astype(int)
    X["empty_net"], X["is_rebound"], X["is_rush"], X["dt_prev"] = s["empty_net"], s["is_rebound"], s["is_rush"], s["dt_prev"]
    return X


def main():
    s = pd.read_csv("data/processed/shots.csv.gz")
    n_all = len(s)
    s = s[s.empty_net == 0].reset_index(drop=True)   # empty-net shots are late-game situations, not shot quality
    print(f"dropped {n_all - len(s):,} empty-net shots")
    X, y = shot_features(s), s["goal"]
    s["xg"] = np.nan
    for season in sorted(s.season.unique()):
        te = s.season == season
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=4, max_iter=300, min_samples_leaf=200,
                                           early_stopping=True, random_state=0)
        m.fit(X[~te], y[~te])
        s.loc[te, "xg"] = m.predict_proba(X[te])[:, 1]
    print(f"{len(s):,} shots | goal rate {y.mean():.3f} | mean xG {s.xg.mean():.3f} | "
          f"out-of-fold log loss {log_loss(y, s.xg):.4f} (naive {log_loss(y, np.full(len(y), y.mean())):.4f}) | AUC {roc_auc_score(y, s.xg):.3f}")
    s.to_csv("data/processed/shots_xg.csv.gz", index=False)

    s["xg_pp"] = s.xg.where(s.skater_diff > 0, 0.0)
    s["xg_ev"] = s.xg.where(s.skater_diff == 0, 0.0)
    s["hd"] = (s.xg >= HD_THRESHOLD).astype(int)
    p = s.groupby(["shooter_id", "game_id"]).agg(ixg=("xg", "sum"), ixg_pp=("xg_pp", "sum"), ixg_ev=("xg_ev", "sum"),
                                                 iff=("xg", "size"), hd_att=("hd", "sum")).reset_index()
    p.rename(columns={"shooter_id": "player_id"}).to_csv("data/processed/player_xg.csv.gz", index=False)

    t = s.groupby(["game_id", "team"]).agg(xgf=("xg", "sum")).reset_index()
    t["xga"] = t.groupby("game_id")["xgf"].transform("sum") - t["xgf"]
    t.to_csv("data/processed/team_xg.csv.gz", index=False)
    print(f"Saved player_xg ({len(p):,} rows) and team_xg ({len(t):,} rows)")


if __name__ == "__main__":
    main()
