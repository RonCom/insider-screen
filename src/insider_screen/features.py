"""Pre-event features (spec, "Features") for acquisition targets and earnings events.

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

Each row also has day0_ar, the day-0 return minus SPY's: an outcome, not a feature, used only to
select negative-return earnings events (H3) and to orient earnings scores.

Usage:
    uv run python -m insider_screen.features targets
    uv run python -m insider_screen.features earnings
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


UNIVERSE_EXCLUDED_TYPES = ["ADRC", "ADRP", "ADRS", "GDR", "NYRS"]
SUMMARY_COLS = ["abn_volume", "last5_share", "car", "scar", "short_share_abn", "short_share_z",
                "abn_short_trades", "small_trade_share_abn"]


def day0_ar(day0: date, sessions: list[date], close: pd.Series, spy: pd.Series) -> float:
    """Day-0 return minus SPY's. An outcome, not a feature: it only says which way the news went (H3 picks
    negative-return earnings events by it, and earnings scores orient scar by its sign)."""
    if day0 not in close.index:
        return np.nan
    i = sessions.index(day0) if day0 in sessions else None
    if not i:
        return np.nan
    prev = sessions[i - 1]
    try:
        return float(close[day0] / close[prev] - 1 - (spy[day0] / spy[prev] - 1))
    except (KeyError, ZeroDivisionError):
        return np.nan


def compute(ev: pd.DataFrame, prices_db: str = PRICES, finra_db: str = FINRA, chunk: int = 400,
            every: int = 5000) -> pd.DataFrame:
    """Features for events with columns event_id, cik, sic, ticker, type, day0 (date), day0_basis; tickers
    are loaded `chunk` at a time to bound memory."""
    cal = xc.get_calendar("XNYS", start="2015-01-01")
    sessions = [s.date() for s in cal.sessions]
    con = duckdb.connect(prices_db, read_only=True)
    spy = _load(con, """SELECT date, close FROM raw.alpaca_bars_daily WHERE symbol = 'SPY' AND adjustment = 'all'
                        ORDER BY date""")
    spy = pd.Series(spy.close.values, index=pd.to_datetime(spy.date).dt.date)
    try:
        fcon = duckdb.connect(finra_db, read_only=True)
    except duckdb.IOException as err:
        print(f"FINRA data unavailable ({err}); off-exchange features left empty")
        fcon = None
    has_trades = fcon is not None and fcon.execute(
        "SELECT count(*) FROM duckdb_tables() WHERE table_name = 'finra_short_trades_daily'").fetchone()[0]
    if fcon is not None and not has_trades:
        print("No monthly short-sale transaction data; trade-count features left empty")

    syms = sorted(set(ev.ticker))
    rows, done = [], 0
    for k in range(0, len(syms), chunk):
        part = syms[k:k + chunk]
        con.register("syms", pd.DataFrame({"symbol": part}))
        bars = _load(con, """SELECT symbol, date, adjustment, close, volume FROM raw.alpaca_bars_daily
                             JOIN syms USING (symbol)""")
        con.unregister("syms")
        bars["date"] = pd.to_datetime(bars.date).dt.date
        short = trades = None
        if fcon is not None:
            fcon.register("syms", pd.DataFrame({"symbol": part}))
            short = _load(fcon, """SELECT symbol, date, sum(short_volume) AS short_volume,
                                          sum(total_volume) AS total_volume
                                   FROM raw.finra_short_daily JOIN syms USING (symbol) GROUP BY 1, 2""")
            if has_trades:
                trades = _load(fcon, """SELECT symbol, date, sum(short_trades) AS short_trades,
                                               sum(small_trades) AS small_trades
                                        FROM raw.finra_short_trades_daily JOIN syms USING (symbol) GROUP BY 1, 2""")
            fcon.unregister("syms")
        for df in (short, trades):
            if df is not None:
                df["date"] = pd.to_datetime(df.date).dt.date
        by = lambda df: {} if df is None else {s: g.set_index("date").drop(columns="symbol")  # noqa: E731
                                                for s, g in df.groupby("symbol")}
        adj = by(bars[bars.adjustment == "all"].drop(columns="adjustment"))
        raw = by(bars[bars.adjustment == "raw"].drop(columns="adjustment"))
        sh_by, tr_by = by(short), by(trades)
        for e in ev[ev.ticker.isin(part)].itertuples(index=False):
            if e.ticker not in adj:
                continue
            st = adj[e.ticker][["close", "volume"]].join(
                raw[e.ticker][["close"]].rename(columns={"close": "close_raw"}) if e.ticker in raw
                else pd.DataFrame(columns=["close_raw"]), how="left")
            ar0 = day0_ar(e.day0, sessions, st.close, spy)
            for window in WINDOWS:
                f = event_features(e.day0, sessions, st, spy, sh_by.get(e.ticker), tr_by.get(e.ticker), window)
                rows.append({"event_id": e.event_id, "cik": e.cik, "ticker": e.ticker, "ticker_type": e.type,
                             "sic": e.sic, "day0": e.day0, "day0_basis": e.day0_basis, "day0_ar": ar0, **f})
            done += 1
            if every and done % every == 0:
                print(f"  {done}/{len(ev)} events", flush=True)
    con.close()
    if fcon is not None:
        fcon.close()
    out = pd.DataFrame(rows)
    if len(out):
        out["in_universe"] = (out.price_d30 >= 1) & ~out.ticker_type.isin(UNIVERSE_EXCLUDED_TYPES)
    return out


def _write(out: pd.DataFrame, table: str, out_db: str, label: str) -> None:
    con = duckdb.connect(out_db)
    con.execute("CREATE SCHEMA IF NOT EXISTS features")
    con.register("out", out)
    con.execute(f"CREATE OR REPLACE TABLE features.{table} AS SELECT * FROM out")
    con.close()
    pre = out[out.window == "pre"]
    print(f"{len(pre)} {label} with bars ({int(pre.in_universe.sum())} in the universe: price >= $1 at "
          f"day -30, not an ADR); features.{table} in {out_db}")
    cols = [c for c in SUMMARY_COLS if c in out]
    print("\nPre-event window, universe events: coverage and quartiles")
    print(pre[pre.in_universe][cols].describe(percentiles=[.25, .5, .75]).T[["count", "25%", "50%", "75%"]]
          .round(3).to_string())


def _with_types(ev: pd.DataFrame, reference_db: str) -> pd.DataFrame:
    con = duckdb.connect(reference_db, read_only=True)
    types = _load(con, "SELECT DISTINCT ticker, cik, type FROM ref.ticker_cik")
    con.close()
    ev = ev.merge(types, on=["ticker", "cik"], how="left").drop_duplicates("event_id")
    ev["day0"] = pd.to_datetime(ev.day0).dt.date
    return ev


def build_targets(edgar_db: str = EDGAR, reference_db: str = REFERENCE, prices_db: str = PRICES,
                  finra_db: str = FINRA, out_db: str = FEATURES) -> pd.DataFrame:
    con = duckdb.connect(edgar_db, read_only=True)
    ev = _load(con, """SELECT d.event_id, a.cik, c.sic, d.ticker, d.day0, d.basis AS day0_basis
                       FROM events.target_day0 d JOIN events.announcements a USING (event_id)
                       JOIN raw.edgar_companies c USING (cik)
                       WHERE d.ticker IS NOT NULL""")
    con.close()
    out = compute(_with_types(ev, reference_db), prices_db, finra_db)
    _write(out, "targets", out_db, "target events")
    return out


def build_earnings(edgar_db: str = EDGAR, reference_db: str = REFERENCE, prices_db: str = PRICES,
                   finra_db: str = FINRA, out_db: str = FEATURES) -> pd.DataFrame:
    """Earnings events (Item 2.02) with a ticker on day 0. Day 0 is the 8-K day 0 (earliest day consistent
    with EDGAR's hours); the day-0 rule is for acquisition targets only."""
    from insider_screen import tickers
    con = duckdb.connect(reference_db, read_only=True)
    tickers.attach(con, edgar_db, finra_db)
    ev = _load(con, f"""SELECT e.event_id, e.cik, c.sic, e.ticker, e.day0, e.day0_basis
                        FROM ({tickers.event_tickers_sql(con)}) e JOIN edgar.raw.edgar_companies c USING (cik)
                        WHERE e.event_type = 'earnings' AND e.ticker IS NOT NULL""")
    con.close()
    print(f"{len(ev)} earnings events with a ticker", flush=True)
    out = compute(_with_types(ev, reference_db), prices_db, finra_db)
    _write(out, "earnings", out_db, "earnings events")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("targets")
    sub.add_parser("earnings")
    a = ap.parse_args()
    build_targets() if a.cmd == "targets" else build_earnings()


if __name__ == "__main__":
    main()
