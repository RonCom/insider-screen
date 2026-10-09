"""Scores for acquisition targets, development period only (spec, "Scores" and "Splits").

Rules, set before any score was computed (spec change log, 2026-10-09):
- events: features.targets, pre-event window, in the universe, day 0 in 2017-2020 (development);
- composite features, each signed so that higher means more buying before good news: abn_volume,
  last5_share, scar, short_share_z (car and short_share_abn are the unscaled forms of two of these);
- peer benchmarking: each feature becomes a robust z-score, (x - median) / (1.4826 x MAD), within events
  of the same size quintile (baseline dollar volume) and SIC division; a peer group under MIN_PEERS
  events falls back to the size quintile, then to all events;
- composite: mean of the available robust z-scores (at least 2);
- Isolation Forest on the four z-scores (missing set to 0), 500 trees, fixed seed;
- supervised: gradient boosting on the z-scores, charged vs. unlabeled, scored out-of-fold (5-fold,
  stratified) so no event is scored by a model that saw its label;
- metrics per score: lift of charged events in the top 5% and top 1%, AUC, charged events captured,
  and charged events per year; the placebo window (-70 to -51) is scored the same way (H5).

The test period (2021-2025) isn't read here; it is scored once, after the code is frozen.

Usage:
    uv run python -m insider_screen.scores dev
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from insider_screen.db import FEATURES, RELEASES

DEV_YEARS = (2017, 2020)
FEATURES_USED = ["abn_volume", "last5_share", "scar", "short_share_z"]
MIN_PEERS = 20
MIN_FEATURES = 2
SIC_DIVISIONS = [(1, 9, "A"), (10, 14, "B"), (15, 17, "C"), (20, 39, "D"), (40, 49, "E"), (50, 51, "F"),
                 (52, 59, "G"), (60, 67, "H"), (70, 89, "I"), (91, 99, "J")]


def sic_division(sic) -> str:
    try:
        two = int(str(sic)[:2]) if len(str(sic)) == 4 else int(sic) // 100
    except (TypeError, ValueError):
        return "?"
    return next((d for lo, hi, d in SIC_DIVISIONS if lo <= two <= hi), "?")


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = 1.4826 * (x - med).abs().median()
    return (x - med) / mad if mad and not math.isnan(mad) else x * np.nan


def peer_z(df: pd.DataFrame, cols: list[str] = FEATURES_USED, min_peers: int = MIN_PEERS) -> pd.DataFrame:
    """Robust z of each column within size quintile x SIC division, falling back to the quintile, then all."""
    df = df.copy()
    size = df.dollar_volume.fillna(df.dollar_volume.median())
    df["size_q"] = pd.qcut(size.rank(method="first"), 5, labels=False) + 1
    df["sector"] = df.sic.map(sic_division)
    df["peer_group"] = df.size_q.astype(str) + "/" + df.sector
    counts = df.peer_group.map(df.peer_group.value_counts())
    q_counts = df.size_q.map(df.size_q.value_counts())
    df.loc[counts < min_peers, "peer_group"] = df.size_q.astype(str) + "/all"
    df.loc[(counts < min_peers) & (q_counts < min_peers), "peer_group"] = "all"
    for c in cols:
        df[f"z_{c}"] = df.groupby("peer_group")[c].transform(robust_z)
    return df


def composite(df: pd.DataFrame, cols: list[str] = FEATURES_USED) -> pd.Series:
    z = df[[f"z_{c}" for c in cols]]
    return z.mean(axis=1).where(z.notna().sum(axis=1) >= MIN_FEATURES)


def isolation_forest(df: pd.DataFrame, cols: list[str] = FEATURES_USED, seed: int = 0) -> pd.Series:
    from sklearn.ensemble import IsolationForest
    x = df[[f"z_{c}" for c in cols]].fillna(0).clip(-10, 10)
    model = IsolationForest(n_estimators=500, random_state=seed).fit(x)
    return pd.Series(-model.score_samples(x), index=df.index)


def supervised_oof(df: pd.DataFrame, y: pd.Series, cols: list[str] = FEATURES_USED, seed: int = 0) -> pd.Series:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    x = df[[f"z_{c}" for c in cols]].clip(-10, 10)
    out = pd.Series(np.nan, index=df.index)
    folds = min(5, int(y.sum()))
    if folds < 2:
        return out
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y):
        m = HistGradientBoostingClassifier(max_depth=2, max_iter=100, learning_rate=0.05,
                                           class_weight="balanced", random_state=seed)
        m.fit(x.iloc[tr], y.iloc[tr])
        out.iloc[te] = m.predict_proba(x.iloc[te])[:, 1]
    return out


def lift(score: pd.Series, y: pd.Series, top: float) -> tuple[float, int, int]:
    """(lift, charged events in the top share, events in the top share). Events without a score rank last."""
    s = score.fillna(-np.inf)
    k = max(1, math.ceil(top * len(s)))
    idx = s.sort_values(ascending=False).index[:k]
    hit = int(y.loc[idx].sum())
    base = y.mean()
    return (hit / k) / base if base else float("nan"), hit, k


def auc(score: pd.Series, y: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score
    s = score.fillna(score.min() - 1)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def evaluate(df: pd.DataFrame, y: pd.Series, scores: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        l5, h5, k5 = lift(s, y, 0.05)
        l1, h1, k1 = lift(s, y, 0.01)
        rows.append({"score": name, "events": len(s), "charged": int(y.sum()), "top5_lift": round(l5, 2),
                     "top5_charged": f"{h5}/{k5}", "top1_lift": round(l1, 2), "top1_charged": f"{h1}/{k1}",
                     "auc": round(auc(s, y), 3)})
    return pd.DataFrame(rows)


def load(features_db: str, releases_db: str, years=DEV_YEARS) -> pd.DataFrame:
    con = duckdb.connect(features_db, read_only=True)
    df = con.execute("""SELECT * FROM features.targets WHERE in_universe
                        AND year(day0) BETWEEN ? AND ?""", list(years)).df()
    con.close()
    con = duckdb.connect(releases_db, read_only=True)
    lab = con.execute("SELECT event_id, is_charged FROM labels.charged_events").df()
    con.close()
    df = df.merge(lab, on="event_id", how="left")
    df["is_charged"] = df.is_charged.fillna(False).astype(bool)
    return df


def dev(features_db: str = FEATURES, releases_db: str = RELEASES, out: str = "reports/dev_scores.md") -> pd.DataFrame:
    df = load(features_db, releases_db)
    results = {}
    for window in ("pre", "placebo"):
        w = peer_z(df[df.window == window].reset_index(drop=True))
        y = w.is_charged.astype(int)
        w["composite"] = composite(w)
        w["isolation_forest"] = isolation_forest(w)
        w["supervised_oof"] = supervised_oof(w, y)
        results[window] = (w, evaluate(w, y, {c: w[c] for c in ("composite", "isolation_forest", "supervised_oof")}))
    pre, ev_pre = results["pre"]
    _, ev_placebo = results["placebo"]
    by_year = pre.assign(year=pd.to_datetime(pre.day0).dt.year).groupby("year").agg(
        events=("event_id", "size"), charged=("is_charged", "sum"))
    top = pre.sort_values("composite", ascending=False).head(15)[
        ["ticker", "day0", "is_charged", "composite"] + [f"z_{c}" for c in FEATURES_USED]].round(2)
    text = "\n".join([
        f"# Development scores, acquisition targets {DEV_YEARS[0]}-{DEV_YEARS[1]}", "",
        "Pre-event window (-20 to -1):", "", ev_pre.to_markdown(index=False), "",
        "Placebo window (-70 to -51), H5 expects top-5% lift at or below 1.5:", "",
        ev_placebo.to_markdown(index=False), "",
        "Events and charged events per year:", "", by_year.to_markdown(), "",
        "Top 15 by composite:", "", top.to_markdown(index=False), ""])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    print(text)
    con = duckdb.connect(features_db)
    con.register("s", pd.concat([results[w][0] for w in results]))
    con.execute("CREATE OR REPLACE TABLE features.target_scores_dev AS SELECT * FROM s")
    con.close()
    return ev_pre


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dev")
    ap.parse_args()
    dev()


if __name__ == "__main__":
    main()
