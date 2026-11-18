"""Leakage-safe feature engineering.

Every player feature uses shift(1) so a game's features only contain information
from games played BEFORE it. Team (opponent) features use merge_asof with
allow_exact_matches=False, i.e. only games on earlier dates.
The same code path builds training rows and a single upcoming-game row (predict.py).

Two feature sets:
  FEATURES_BASE  goals / shots / TOI / context  (no shot-quality data needed)
  FEATURES_XG    shot-quality features from shot_model.py (player_xg.csv.gz, team_xg.csv.gz)
"""
from pathlib import Path
import numpy as np
import pandas as pd

LG_SH_PCT = 0.105   # approx league-wide forward shooting %
SHRINK_K = 100      # shots of "prior" mixed into a player's career shooting %
FIN_K = 10          # xG of "prior" mixed into the finishing ratio (goals vs xG)

FEATURES_BASE = [
    "g_r5", "g_r10", "g_r20", "g_season", "g_career",
    "s_r5", "s_r10", "s_r20", "s_career",
    "toi_r5", "toi_r20", "toi_career",
    "ppg_r20", "ppg_career", "career_sh_pct",
    "is_home", "rest_days", "back_to_back", "games_prior",
    "opp_ga10", "opp_gf10", "team_gf10",
    "pos_C", "pos_L", "pos_R",
]
FEATURES_XG = [
    "ixg_r5", "ixg_r10", "ixg_r20", "ixg_career", "ixgpp_r20", "ixgev_r20",
    "iff_r10", "hd_r20", "ixg60_r20", "fin_ratio",
    "opp_xga10", "team_xgf10",
]
FEATURES = FEATURES_BASE + FEATURES_XG

ROLLS = {  # column -> (short name, windows)
    "goals": ("g", (5, 10, 20)),
    "shots": ("s", (5, 10, 20)),
    "toi_min": ("toi", (5, 20)),
    "pp_goals": ("ppg", (20,)),
}
CAREER = {"goals": "g", "shots": "s", "toi_min": "toi", "pp_goals": "ppg"}
XG_COLS = ["ixg", "ixg_pp", "ixg_ev", "iff", "hd_att"]
XG_ROLLS = {"ixg": ("ixg", (5, 10, 20)), "ixg_pp": ("ixgpp", (20,)), "ixg_ev": ("ixgev", (20,)),
            "iff": ("iff", (10,)), "hd_att": ("hd", (20,))}


def load_xg(processed="data/processed"):
    """Return (player_xg, team_xg), or (None, None) if shot_model.py hasn't been run."""
    p, t = Path(processed, "player_xg.csv.gz"), Path(processed, "team_xg.csv.gz")
    if p.exists() and t.exists():
        return pd.read_csv(p), pd.read_csv(t)
    return None, None


def _team_rolling(team_games):
    t = team_games.copy()
    t["game_date"] = pd.to_datetime(t["game_date"]).astype("datetime64[ns]")
    t = t.sort_values(["team", "game_date"])
    gt = t.groupby("team", sort=False)
    cols = {"gf": "gf10", "ga": "ga10"}
    if "xgf" in t:
        cols.update({"xgf": "xgf10", "xga": "xga10"})
    for src, dst in cols.items():
        t[dst] = gt[src].transform(lambda s: s.rolling(10, min_periods=3).mean())
    return t[["team", "game_date", *cols.values()]].sort_values("game_date")


def _asof(left, roll, key, rename):
    right = roll.rename(columns={"team": key, **rename})
    return pd.merge_asof(left.sort_values("game_date"), right.sort_values("game_date"),
                         on="game_date", by=key, allow_exact_matches=False)


def build_features(player_games, team_games, player_xg=None, team_xg=None):
    df = player_games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).astype("datetime64[ns]")
    use_xg = player_xg is not None and team_xg is not None

    if use_xg:
        team_games = team_games.merge(team_xg[["game_id", "team", "xgf", "xga"]], on=["game_id", "team"], how="left")
        df = df.merge(player_xg[["player_id", "game_id", *XG_COLS]], on=["player_id", "game_id"], how="left")
        covered = df["game_id"].isin(set(team_xg["game_id"]))   # play-by-play exists -> no row means zero attempts
        for c in XG_COLS:
            df.loc[covered & df[c].isna(), c] = 0.0

    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    g = df.groupby("player_id", sort=False)

    def rolling(col, short, windows):
        for w in windows:
            df[f"{short}_r{w}"] = g[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=max(3, w // 2)).mean())

    for col, (short, windows) in ROLLS.items():
        rolling(col, short, windows)
    for col, short in CAREER.items():
        df[f"{short}_career"] = g[col].transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
    df["g_season"] = df.groupby(["player_id", "season"], sort=False)["goals"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean())

    # career shooting % shrunk toward league average (stable even for low-shot players)
    cum_g = g["goals"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    cum_s = g["shots"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    df["career_sh_pct"] = (cum_g + SHRINK_K * LG_SH_PCT) / (cum_s + SHRINK_K)

    df["games_prior"] = g.cumcount()
    df["rest_days"] = g["game_date"].diff().dt.days.clip(upper=10).fillna(7)
    df["back_to_back"] = (df["rest_days"] <= 1).astype(int)
    for p in ("C", "L", "R"):
        df[f"pos_{p}"] = (df["position"] == p).astype(int)

    if use_xg:
        for col, (short, windows) in XG_ROLLS.items():
            rolling(col, short, windows)
        df["ixg_career"] = g["ixg"].transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
        sx = g["ixg"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).sum())
        st = g["toi_min"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).sum())
        df["ixg60_r20"] = sx / st.where(st > 0) * 60
        cum_xg = g["ixg"].transform(lambda s: s.shift(1).fillna(0).cumsum())
        # finishing: goals above/below xG as a fraction of xG, shrunk so small samples stay near 0
        df["fin_ratio"] = (cum_g - cum_xg) / (cum_xg + FIN_K)

    roll = _team_rolling(team_games)
    opp_map = {"gf10": "opp_gf10", "ga10": "opp_ga10", "xgf10": "_opp_xgf10", "xga10": "opp_xga10"}
    own_map = {"gf10": "team_gf10", "ga10": "_t_ga10", "xgf10": "team_xgf10", "xga10": "_t_xga10"}
    df = _asof(df, roll, "opponent", {k: v for k, v in opp_map.items() if k in roll})
    df = _asof(df, roll, "team", {k: v for k, v in own_map.items() if k in roll})
    df = df.drop(columns=[c for c in df if c.startswith("_")])

    df["scored"] = np.where(df["goals"].isna(), np.nan, (df["goals"] >= 1).astype(float))
    return df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)


if __name__ == "__main__":
    pg = pd.read_csv("data/processed/player_games.csv.gz")
    tg = pd.read_csv("data/processed/team_games.csv.gz")
    pxg, txg = load_xg()
    print("shot-quality features:", "ON" if pxg is not None else "OFF (run collect_pbp.py + shot_model.py)")
    feats = build_features(pg, tg, pxg, txg)
    feats.to_csv("data/processed/features.csv.gz", index=False)
    print(f"Built features: {feats.shape}; goals/game = {feats['goals'].mean():.3f}; "
          f"P(>=1 goal) = {feats['scored'].mean():.3f}")
