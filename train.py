"""Train + compare models on a time-based split (never random: that would leak the future).

Two targets from the same features:
  * expected goals  lambda  (count regression: Poisson GLM, Random Forest, XGBoost-Poisson)
  * P(score >= 1)           (classifiers: logistic regression, Random Forest, XGBoost)
Poisson models also give P(>=1 goal) = 1 - exp(-lambda), so everything is comparable.
"""
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_poisson_deviance, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import FEATURES_BASE, FEATURES_XG

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAVE_XGB = True
except ImportError:  # graceful fallback so the project still runs
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    HAVE_XGB = False


def make_models(seed=42):
    reg = {
        "poisson_glm": make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=1000)),
        "rf_poisson": RandomForestRegressor(n_estimators=300, criterion="poisson", min_samples_leaf=100,
                                            max_features=0.5, n_jobs=-1, random_state=seed),
    }
    clf = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        "rf_clf": RandomForestClassifier(n_estimators=300, min_samples_leaf=100, max_features=0.5,
                                         n_jobs=-1, random_state=seed),
    }
    if HAVE_XGB:
        reg["xgb_poisson"] = XGBRegressor(objective="count:poisson", n_estimators=1500, learning_rate=0.02,
                                          max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                                          early_stopping_rounds=50, random_state=seed, n_jobs=-1)
        clf["xgb_clf"] = XGBClassifier(objective="binary:logistic", n_estimators=1500, learning_rate=0.02,
                                       max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                                       early_stopping_rounds=50, random_state=seed, n_jobs=-1)
    else:
        reg["xgb_poisson"] = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.03, max_depth=3,
                                                           max_iter=500, early_stopping=True, random_state=seed)
        clf["xgb_clf"] = HistGradientBoostingClassifier(learning_rate=0.03, max_depth=3, max_iter=500,
                                                        early_stopping=True, random_state=seed)
    return reg, clf


def fit(model, name, Xtr, ytr, Xva, yva):
    if HAVE_XGB and name.startswith("xgb"):
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        model.fit(Xtr, ytr)
    return model


def prob_metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(log_loss=log_loss(y, p), brier=brier_score_loss(y, p), auc=roc_auc_score(y, p))


def run_feature_set(label, feats, tr, va, te, save_plots):
    X = lambda d: d[feats]
    reg, clf = make_models()
    rows, test_pred = [], te[["player_id", "name", "game_date", "opponent", "goals", "scored"]].copy()

    base = te["g_career"].clip(lower=0.01).values
    rows.append(dict(model="baseline_career_rate", dev=mean_poisson_deviance(te.goals, base), mae=mean_absolute_error(te.goals, base),
                     **prob_metrics(te.scored, 1 - np.exp(-base))))
    test_pred["baseline_career_rate_xg"] = base

    for name, m in reg.items():
        fit(m, name, X(tr), tr.goals, X(va), va.goals)
        lam = np.clip(m.predict(X(te)), 1e-4, None)
        test_pred[f"{name}_xg"] = lam
        rows.append(dict(model=name, dev=mean_poisson_deviance(te.goals, lam), mae=mean_absolute_error(te.goals, lam),
                         **prob_metrics(te.scored, 1 - np.exp(-lam))))
    for name, m in clf.items():
        fit(m, name, X(tr), tr.scored, X(va), va.scored)
        p = m.predict_proba(X(te))[:, 1]
        test_pred[f"{name}_p"] = p
        rows.append(dict(model=name, dev=np.nan, mae=np.nan, **prob_metrics(te.scored, p)))

    res = pd.DataFrame(rows).set_index("model").round(4)
    print(f"\n=== Feature set: {label} ({len(feats)} features) ===\n", res.to_string())
    res.to_csv(f"reports/metrics_{label}.csv")
    test_pred.to_csv(f"reports/test_predictions_{label}.csv", index=False)
    joblib.dump({"features": feats, "reg": reg, "clf": clf}, f"models/models_{label}.joblib")

    if save_plots:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for col in [c for c in test_pred if c.endswith(("_p", "_xg")) and not c.startswith("baseline")]:
            p = test_pred[col] if col.endswith("_p") else 1 - np.exp(-test_pred[col])
            b = pd.qcut(p, 10, duplicates="drop")
            t = pd.DataFrame({"p": p, "y": test_pred.scored}).groupby(b, observed=True).mean()
            ax.plot(t.p, t.y, marker="o", label=col)
        ax.plot([0, .6], [0, .6], "k--", lw=1); ax.set_xlabel("predicted P(>=1 goal)"); ax.set_ylabel("actual rate")
        ax.legend(fontsize=7); ax.set_title(f"Calibration, test set ({label})"); fig.tight_layout()
        fig.savefig(f"reports/calibration_{label}.png", dpi=130)
        m = reg["xgb_poisson"]
        if hasattr(m, "feature_importances_"):
            imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
            imp.to_csv(f"reports/feature_importance_{label}.csv"); print("\nTop features:\n", imp.head(12).round(3).to_string())
    return res


def main(a):
    df = pd.read_csv("data/processed/features.csv.gz", parse_dates=["game_date"])
    have_xg = all(c in df for c in FEATURES_XG)
    sets = {"base": FEATURES_BASE}
    if have_xg:
        sets["with_xg"] = FEATURES_BASE + FEATURES_XG
    else:
        print("No shot-quality columns found: training base features only (run collect_pbp.py, shot_model.py, features.py).")
    all_feats = sorted({f for fs in sets.values() for f in fs})

    df = df[(df["games_prior"] >= a.min_games) & df["goals"].notna()].copy()
    for c in ("opp_ga10", "opp_gf10", "team_gf10", "opp_xga10", "team_xgf10"):
        if c in df:
            df[c] = df[c].fillna(df[c].median())
    df = df.dropna(subset=all_feats)   # same rows for every feature set -> fair comparison

    tr, va, te = (df[df.season <= a.val_season - 1], df[df.season == a.val_season], df[df.season >= a.test_season])
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}  (xgboost={'yes' if HAVE_XGB else 'no, using sklearn HGB'})")
    Path("models").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)

    results = {label: run_feature_set(label, feats, tr, va, te, save_plots=(label == list(sets)[-1]))
               for label, feats in sets.items()}
    if len(results) == 2:
        cmp = results["with_xg"][["log_loss", "brier", "auc"]].join(results["base"][["log_loss", "brier", "auc"]],
                                                                     lsuffix="_xg", rsuffix="_base")
        cmp["logloss_gain"] = (cmp.log_loss_base - cmp.log_loss_xg).round(4)
        print("\n=== Did shot quality help? (positive logloss_gain = yes) ===\n", cmp.drop(index="baseline_career_rate").to_string())
    json.dump({"val_season": a.val_season, "test_season": a.test_season, "sets": list(sets)}, open("models/split.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-season", type=int, default=2024, help="season start-year used for early stopping")
    ap.add_argument("--test-season", type=int, default=2025, help="first season held out for final testing")
    ap.add_argument("--min-games", type=int, default=10, help="min prior career games for a row to be used")
    main(ap.parse_args())
