"""Pre-event features (spec, "Features"), acquisition targets first.

Windows are counted in NYSE sessions from day 0 (the final day 0 from events.target_day0):
pre-event window -20..-1, baseline -250..-31. The placebo (H5) uses -70..-51 with baseline -300..-81,
the same lengths and the same 10-session gap.

Per event and window:
- abn_volume: log of mean daily volume in the window over mean daily volume in the baseline
- last5_share: share of the window's volume traded in its last 5 sessions
- car, scar: cumulative abnormal return over the window from a market model (stock on SPY) fitted on
  the baseline, and the same divided by the residual SD x sqrt(window sessions)
- short_share_abn, short_share_z: FINRA off-exchange short share (short volume / total volume) in the
  window minus the baseline's, and that difference over its standard error from the baseline's daily SD
- abn_short_trades, small_trade_share_abn: from the monthly transaction files, computed only if they're
  loaded; dropped from the study (spec change log, 2026-10-09), so normally empty
Plus price_d30 (raw close at day -30, for the $1 floor), dollar_volume (baseline median of raw close x
volume, the size proxy: no shares-outstanding source is free, so market cap isn't available) and SIC.

No feature uses data from day 0 or later: `event_features` drops every row dated on or after day 0
before computing anything, and tests/test_features.py changes all such rows and checks nothing moves.

Usage:
    uv run python -m insider_screen.features targets
"""

from __future__ import annotations

import argparse
from datetime import date

import duckdb
import exchange_calendars as xc
import numpy as np
import pandas as pd

from insider_screen.db import EDGAR, FEATURES, FINRA, PRICES, REFERENCE

WINDOWS = {"pre": ((-20, -1), (-250, -31)), "placebo": ((-70, -51), (-300, -81))}
MIN_BASELINE = 60
MIN_WINDOW = 10
LAST_N = 5


def _span(sessions: list[date], i0: int, lo: int, hi: int) -> list[date]:
    a, b = i0 + lo, i0 + hi
    return sessions[max(a, 0):b + 1] if b >= 0 else []


def event_features(day0: date, sessions: list[date], stock: pd.DataFrame, spy: pd.Series,
                   short: pd.DataFrame | None = None, trades: pd.DataFrame | None = None,
                   window: str = "pre") -> dict:
    """stock: indexed by date, columns close (adjusted), close_raw, volume. spy: adjusted close by date.
    short: short_volume, total_volume by date. trades: short_trades, small_trades by date.
    Every input is cut to dates before day 0 first."""
    cut = lambda x: None if x is None else x[x.index < day0]  # noqa: E731
    stock, spy, short, trades = cut(stock), cut(spy), cut(short), cut(trades)
    sessions = [s for s in sessions if s < day0]
    i0 = len(sessions)  # index day 0 would have
    (wlo, whi), (blo, bhi) = WINDOWS[window]
    win, base = _span(sessions, i0, wlo, whi), _span(sessions, i0, blo, bhi)
    out: dict = {"window": window}

    w, b = stock.reindex(win), stock.reindex(base)
    out["n_window"], out["n_baseline"] = int(w.volume.notna().sum()), int(b.volume.notna().sum())
    d30 = _span(sessions, i0, -30, -30)
    out["price_d30"] = float(stock.close_raw.get(d30[0], np.nan)) if d30 else np.nan
    out["dollar_volume"] = float((b.close_raw * b.volume).median()) if len(b) else np.nan
    ok = out["n_window"] >= MIN_WINDOW and out["n_baseline"] >= MIN_BASELINE
    if ok and b.volume.mean() > 0 and w.volume.mean() > 0:
        out["abn_volume"] = float(np.log(w.volume.mean()) - np.log(b.volume.mean()))
        out["last5_share"] = float(w.volume.iloc[-LAST_N:].sum() / w.volume.sum())
    # returns on consecutive sessions only
    allp = pd.DataFrame({"r": stock.close, "m": spy}).reindex(sessions)
    rets = allp.pct_change(fill_method=None)
    rw, rb = rets.reindex(win).dropna(), rets.reindex(base).dropna()
    if ok and len(rb) >= MIN_BASELINE and len(rw) >= MIN_WINDOW:
        beta, alpha = np.polyfit(rb.m, rb.r, 1)
        resid_sd = float((rb.r - alpha - beta * rb.m).std())
        car = float((rw.r - alpha - beta * rw.m).sum())
        out.update(car=car, scar=car / (resid_sd * np.sqrt(len(rw))) if resid_sd > 0 else np.nan, beta=float(beta))
    if short is not None and len(short):
        sw, sb = short.reindex(win).dropna(), short.reindex(base).dropna()
        sw, sb = sw[sw.total_volume > 0], sb[sb.total_volume > 0]
        if len(sw) >= MIN_WINDOW and len(sb) >= MIN_BASELINE:
            share_w = sw.short_volume.sum() / sw.total_volume.sum()
            share_b = sb.short_volume.sum() / sb.total_volume.sum()
            daily_sd = (sb.short_volume / sb.total_volume).std()
            out["short_share_abn"] = float(share_w - share_b)
            out["short_share_z"] = (float((share_w - share_b) / (daily_sd / np.sqrt(len(sw))))
                                    if daily_sd > 0 else np.nan)
    if trades is not None and len(trades):
        tw = trades.reindex(win).fillna(0)
        tb = trades.reindex(base).fillna(0)
        if tb.short_trades.sum() > 0:
            out["abn_short_trades"] = float(np.log(tw.short_trades.mean() + 1) - np.log(tb.short_trades.mean() + 1))
            if tw.short_trades.sum() > 0:
                out["small_trade_share_abn"] = float(tw.small_trades.sum() / tw.short_trades.sum()
                                                     - tb.small_trades.sum() / tb.short_trades.sum())
    return out


def _load(con, sql: str, params=None) -> pd.DataFrame:
    return con.execute(sql, params or []).df()


def build_targets(edgar_db: str = EDGAR, reference_db: str = REFERENCE, prices_db: str = PRICES,
                  finra_db: str = FINRA, out_db: str = FEATURES) -> pd.DataFrame:
    con = duckdb.connect(edgar_db, read_only=True)
    ev = _load(con, """SELECT d.event_id, a.cik, c.sic, d.ticker, d.ticker_source, d.day0, d.day0_8k, d.basis
                       FROM events.target_day0 d JOIN events.announcements a USING (event_id)
                       JOIN raw.edgar_companies c USING (cik)
                       WHERE d.ticker IS NOT NULL""")
    con.close()
    ev["day0"] = pd.to_datetime(ev.day0).dt.date
    con = duckdb.connect(reference_db, read_only=True)
    types = _load(con, "SELECT DISTINCT ticker, cik, type FROM ref.ticker_cik")
    con.close()
    ev = ev.merge(types, on=["ticker", "cik"], how="left").drop_duplicates("event_id")
    syms = sorted(set(ev.ticker))

    con = duckdb.connect(prices_db, read_only=True)
    con.register("syms", pd.DataFrame({"symbol": syms + ["SPY"]}))
    bars = _load(con, """SELECT symbol, date, adjustment, close, volume FROM raw.alpaca_bars_daily
                         JOIN syms USING (symbol)""")
    con.close()
    bars["date"] = pd.to_datetime(bars.date).dt.date
    adj = bars[bars.adjustment == "all"].set_index(["symbol", "date"])
    raw = bars[bars.adjustment == "raw"].set_index(["symbol", "date"])
    spy = adj.loc["SPY"].close

    short = trades = None
    try:
        con = duckdb.connect(finra_db, read_only=True)
        con.register("syms", pd.DataFrame({"symbol": syms}))
        short = _load(con, """SELECT symbol, date, sum(short_volume) AS short_volume, sum(total_volume) AS total_volume
                              FROM raw.finra_short_daily JOIN syms USING (symbol) GROUP BY 1, 2""")
        try:
            trades = _load(con, """SELECT symbol, date, sum(short_trades) AS short_trades,
                                          sum(small_trades) AS small_trades
                                   FROM raw.finra_short_trades_daily JOIN syms USING (symbol) GROUP BY 1, 2""")
        except duckdb.CatalogException:
            print("No monthly short-sale transaction data; trade-count features left empty")
        con.close()
    except duckdb.IOException as err:
        print(f"FINRA data unavailable ({err}); off-exchange features left empty")
    for df in (short, trades):
        if df is not None:
            df["date"] = pd.to_datetime(df.date).dt.date
    short = short.set_index(["symbol", "date"]) if short is not None else None
    trades = trades.set_index(["symbol", "date"]) if trades is not None else None

    cal = xc.get_calendar("XNYS", start="2015-01-01")
    sessions = [s.date() for s in cal.sessions]
    rows = []
    for e in ev.itertuples(index=False):
        if e.ticker not in adj.index.get_level_values(0):
            continue
        st = adj.loc[e.ticker][["close", "volume"]].join(
            raw.loc[e.ticker][["close"]].rename(columns={"close": "close_raw"}) if e.ticker in raw.index.get_level_values(0)
            else pd.DataFrame(columns=["close_raw"]), how="left")
        sh = short.loc[e.ticker] if short is not None and e.ticker in short.index.get_level_values(0) else None
        tr = trades.loc[e.ticker] if trades is not None and e.ticker in trades.index.get_level_values(0) else None
        for window in WINDOWS:
            f = event_features(e.day0, sessions, st, spy, sh, tr, window)
            rows.append({"event_id": e.event_id, "cik": e.cik, "ticker": e.ticker, "ticker_type": e.type,
                         "sic": e.sic, "day0": e.day0, "day0_basis": e.basis, **f})
    out = pd.DataFrame(rows)
    out["in_universe"] = (out.price_d30 >= 1) & ~out.ticker_type.isin(["ADRC", "ADRP", "ADRS", "GDR", "NYRS"])
    con = duckdb.connect(out_db)
    con.execute("CREATE SCHEMA IF NOT EXISTS features")
    con.register("out", out)
    con.execute("CREATE OR REPLACE TABLE features.targets AS SELECT * FROM out")
    con.close()

    pre = out[out.window == "pre"]
    print(f"{len(pre)} target events with bars ({int(pre.in_universe.sum())} in the universe: price >= $1 at "
          f"day -30, not an ADR); features.targets in {out_db}")
    cols = ["abn_volume", "last5_share", "car", "scar", "short_share_abn", "short_share_z",
            "abn_short_trades", "small_trade_share_abn"]
    cols = [c for c in cols if c in out]
    print("\nPre-event window, universe events: coverage and quartiles")
    print(pre[pre.in_universe][cols].describe(percentiles=[.25, .5, .75]).T[["count", "25%", "50%", "75%"]]
          .round(3).to_string())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("targets")
    ap.parse_args()
    build_targets()


if __name__ == "__main__":
    main()
