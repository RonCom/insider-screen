"""Scores for acquisition targets and earnings events (spec, "Scores", "Splits", "Expectations").

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

Earnings events orient scar by the sign of the day-0 abnormal return (day0_ar, an outcome kept out of the
features), so a run-up before good news and a drop before bad news both count as informed trading.

`dev` reads only 2017-2020. `test` reads 2021-2025 once: it runs only at a commit tagged freeze-*, with no
uncommitted changes to tracked files, and refuses if reports/test_scores.md exists. It reports H1, H3, H5
and H6; the Isolation Forest and the supervised model are fitted on development events and applied to test.

Usage:
    uv run python -m insider_screen.scores dev
    uv run python -m insider_screen.scores dev --kind earnings
    git tag freeze-1 && uv run python -m insider_screen.scores test
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from insider_screen.db import EDGAR, FEATURES, RELEASES

DEV_YEARS = (2017, 2020)
TEST_YEARS = (2021, 2025)
BUDGET_PER_MONTH = 50  # H6
TEST_REPORT = "reports/test_scores.md"
FEATURES_USED = ["abn_volume", "last5_share", "scar", "short_share_z"]
MIN_PEERS = 20
MIN_FEATURES = 2
Z_CAP = 5  # each robust z is capped at +-5, so one extreme feature (a SPAC's scar of 55) can't carry the mean
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
        df[f"z_{c}"] = df.groupby("peer_group")[c].transform(robust_z).clip(-Z_CAP, Z_CAP)
    return df


def composite(df: pd.DataFrame, cols: list[str] = FEATURES_USED) -> pd.Series:
    z = df[[f"z_{c}" for c in cols]]
    return z.mean(axis=1).where(z.notna().sum(axis=1) >= MIN_FEATURES)


def isolation_forest(df: pd.DataFrame, cols: list[str] = FEATURES_USED, seed: int = 0,
                     fit: pd.DataFrame | None = None) -> pd.Series:
    """Fitted on `fit` (the development events) when given, else on `df` itself."""
    from sklearn.ensemble import IsolationForest
    zc = [f"z_{c}" for c in cols]
    xs = df[zc].fillna(0)
    model = IsolationForest(n_estimators=500, random_state=seed).fit((fit if fit is not None else df)[zc].fillna(0))
    return pd.Series(-model.score_samples(xs), index=df.index)


def supervised_fit(fit: pd.DataFrame, y_fit: pd.Series, df: pd.DataFrame, cols: list[str] = FEATURES_USED,
                   seed: int = 0) -> pd.Series:
    """Gradient boosting fitted on development events, applied to `df`."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    zc = [f"z_{c}" for c in cols]
    if y_fit.sum() < 2:
        return pd.Series(np.nan, index=df.index)
    m = HistGradientBoostingClassifier(max_depth=2, max_iter=100, learning_rate=0.05, class_weight="balanced",
                                       random_state=seed).fit(fit[zc], y_fit)
    return pd.Series(m.predict_proba(df[zc])[:, 1], index=df.index)


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


def orient(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Earnings: scar times the sign of the day-0 abnormal return (no direction, no scar)."""
    df = df.copy()
    if kind == "earnings":
        df["scar_raw"] = df.scar
        df["scar"] = df.scar * np.sign(df.day0_ar).replace(0, np.nan)
    return df


def load(features_db: str, releases_db: str, years=DEV_YEARS, edgar_db: str = EDGAR,
         kind: str = "targets") -> pd.DataFrame:
    con = duckdb.connect(features_db, read_only=True)
    con.execute(f"ATTACH '{Path(edgar_db).as_posix()}' AS edgar (READ_ONLY)")
    if kind == "targets":
        sql = """SELECT f.* FROM features.targets f JOIN edgar.events.target_audit a USING (event_id)
                 WHERE f.in_universe AND a.in_target_set AND year(f.day0) BETWEEN ? AND ?"""
    else:
        sql = """SELECT * FROM features.earnings WHERE in_universe AND year(day0) BETWEEN ? AND ?"""
    df = con.execute(sql, list(years)).df()
    con.close()
    con = duckdb.connect(releases_db, read_only=True)
    lab = con.execute("SELECT event_id, is_charged FROM labels.charged_events").df()
    con.close()
    df = df.merge(lab, on="event_id", how="left")
    df["is_charged"] = df.is_charged.fillna(False).astype(bool)
    return orient(df, kind)


def dev(features_db: str = FEATURES, releases_db: str = RELEASES, out: str | None = None,
        kind: str = "targets") -> pd.DataFrame:
    out = out or f"reports/dev_scores{'' if kind == 'targets' else '_' + kind}.md"
    df = load(features_db, releases_db, kind=kind)
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
    pre = pre.assign(day0=pd.to_datetime(pre.day0).dt.date,
                     rank=pre.composite.rank(ascending=False, method="min").astype("Int64"))
    zcols = [f"z_{c}" for c in FEATURES_USED]
    top = pre.sort_values("composite", ascending=False).head(15)[
        ["ticker", "day0", "is_charged", "composite"] + zcols].round(2)
    charged = pre[pre.is_charged].sort_values("composite", ascending=False)[
        ["ticker", "day0", "rank", "composite"] + zcols + ["car", "abn_volume"]].round(2)
    text = "\n".join([
        f"# Development scores, {kind} {DEV_YEARS[0]}-{DEV_YEARS[1]}", "",
        "Pre-event window (-20 to -1):", "", ev_pre.to_markdown(index=False), "",
        "Placebo window (-70 to -51), H5 expects top-5% lift at or below 1.5:", "",
        ev_placebo.to_markdown(index=False), "",
        "Events and charged events per year:", "", by_year.to_markdown(), "",
        "Top 15 by composite:", "", top.to_markdown(index=False), "",
        f"Charged events, by composite rank (of {len(pre)}):", "", charged.to_markdown(index=False), ""])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    print(text)
    con = duckdb.connect(features_db)
    con.register("s", pd.concat([results[w][0] for w in results]))
    con.execute(f"CREATE OR REPLACE TABLE features.{kind}_scores_dev AS SELECT * FROM s")
    con.close()
    return ev_pre


def frozen() -> str:
    """The freeze tag at HEAD; exits if there is none or tracked files have uncommitted changes."""
    import subprocess
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True)  # noqa: E731
    dirty = run("status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        raise SystemExit(f"Uncommitted changes to tracked files; the test period is scored only on frozen code:\n{dirty}")
    tags = [t for t in run("tag", "--points-at", "HEAD").stdout.split() if t.startswith("freeze")]
    if not tags:
        raise SystemExit("HEAD has no freeze-* tag. Freeze the code first: git tag freeze-1 (and push the tag).")
    if Path(TEST_REPORT).exists():
        raise SystemExit(f"{TEST_REPORT} exists: the test period has been scored. It is scored once.")
    return f"{tags[0]} at {run('rev-parse', '--short', 'HEAD').stdout.strip()}"


def _scored(fit: pd.DataFrame, test: pd.DataFrame, supervised: bool = True) -> pd.DataFrame:
    """Peer z within each period; Isolation Forest and the supervised model fitted on development events."""
    fit, test = peer_z(fit.reset_index(drop=True)), peer_z(test.reset_index(drop=True))
    test["composite"] = composite(test)
    test["isolation_forest"] = isolation_forest(test, fit=fit)
    if supervised:
        test["supervised"] = supervised_fit(fit, fit.is_charged.astype(int), test)
    return test


def budget_capture(scored: pd.DataFrame, per_month: int = BUDGET_PER_MONTH) -> tuple[float, int, int]:
    """H6: alert on the top `per_month` events by composite each calendar month; share of charged events caught."""
    df = scored.assign(month=pd.to_datetime(scored.day0).dt.to_period("M"))
    df["r"] = df.groupby("month").composite.rank(ascending=False, method="first")
    caught = int((df.is_charged & (df.r <= per_month)).sum())
    total = int(df.is_charged.sum())
    return (caught / total if total else float("nan")), caught, total


def test(features_db: str = FEATURES, releases_db: str = RELEASES, out: str = TEST_REPORT) -> None:
    freeze = frozen()
    lines = [f"# Test period scores, {TEST_YEARS[0]}-{TEST_YEARS[1]}", "", f"Code: {freeze}.", ""]
    scored_all = []
    t_dev = load(features_db, releases_db, DEV_YEARS)
    t_test = load(features_db, releases_db, TEST_YEARS)
    tables = {}
    for window in ("pre", "placebo"):
        s = _scored(t_dev[t_dev.window == window], t_test[t_test.window == window])
        y = s.is_charged.astype(int)
        tables[window] = evaluate(s, y, {c: s[c] for c in ("composite", "isolation_forest", "supervised")})
        if window == "pre":
            scored_all.append(s.assign(kind="targets"))
    best = tables["pre"].sort_values("top5_lift", ascending=False).iloc[0]
    placebo_best = tables["placebo"].set_index("score").loc[best.score]
    lines += ["## Acquisition targets", "", "Pre-event window:", "", tables["pre"].to_markdown(index=False), "",
              "Placebo window:", "", tables["placebo"].to_markdown(index=False), "",
              f"- H1 (top-5% lift >= 3x, best score): {best.score}, {best.top5_lift}x "
              f"({best.top5_charged}): {'met' if best.top5_lift >= 3 else 'failed'}",
              f"- H5 (placebo top-5% lift <= 1.5x, same score): {placebo_best.top5_lift}x: "
              f"{'met' if placebo_best.top5_lift <= 1.5 else 'failed'}", ""]

    try:
        e_dev = load(features_db, releases_db, DEV_YEARS, kind="earnings")
        e_test = load(features_db, releases_db, TEST_YEARS, kind="earnings")
    except duckdb.CatalogException:
        e_test = None
        lines += ["## Earnings", "", "features.earnings not built: H3 not tested; H6 covers targets only.", ""]
    if e_test is not None:
        s = _scored(e_dev[e_dev.window == "pre"], e_test[e_test.window == "pre"], supervised=False)
        scored_all.append(s.assign(kind="earnings"))
        neg = s[s.day0_ar < 0]
        y = neg.is_charged.astype(int)
        stock = ["abn_volume", "last5_share", "scar"]
        a_stock = auc(composite(neg, stock), y)
        a_both = auc(composite(neg, stock + ["short_share_z"]), y)
        gain = a_both - a_stock
        lines += ["## Earnings, negative day-0 abnormal return (H3)", "",
                  f"{len(neg)} events, {int(y.sum())} charged. Composite AUC, stock features only: {a_stock:.3f}; "
                  f"with the off-exchange short share: {a_both:.3f}; gain {gain:+.3f}.",
                  f"- H3 (gain < 0.02): {'met' if gain < 0.02 else 'failed: report as a finding, re-test on 2024-2025'}",
                  ""]

    both = pd.concat(scored_all, ignore_index=True)
    share, caught, total = budget_capture(both)
    lines += ["## Alert budget (H6)", "",
              f"Top {BUDGET_PER_MONTH} events a month by composite, {' and '.join(both.kind.unique())}: "
              f"{caught} of {total} charged events caught ({share:.0%}).",
              f"- H6 (>= 25%): {'met' if share >= 0.25 else 'failed'}", "",
              "## Charged events per year", "",
              both.assign(year=pd.to_datetime(both.day0).dt.year).groupby(["kind", "year"])
                  .agg(events=("event_id", "size"), charged=("is_charged", "sum")).to_markdown(), ""]
    text = "\n".join(lines)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    print(text)
    con = duckdb.connect(features_db)
    con.register("s", both)
    con.execute("CREATE OR REPLACE TABLE features.scores_test AS SELECT * FROM s")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--kind", choices=["targets", "earnings"], default="targets")
    sub.add_parser("test")
    a = ap.parse_args()
    if a.cmd == "dev":
        dev(kind=a.kind)
    else:
        test()


if __name__ == "__main__":
    main()
