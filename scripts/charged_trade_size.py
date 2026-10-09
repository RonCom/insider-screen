"""How big were the charged trades next to normal trading? For each charged target in the audited set,
the largest share count, option-contract count and profit figure stated in its releases, against the
stock's mean daily volume in the pre-event window (-20 to -1) and the baseline (-250 to -31).

shares_of_window = largest stated share count / total volume over the 20 window sessions. A trade worth a
few percent of that can't move daily volume or returns by itself.

    uv run python scripts/charged_trade_size.py
"""

import re

import duckdb
import exchange_calendars as xc
import pandas as pd

from insider_screen.db import EDGAR, FEATURES, PRICES, RELEASES

NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
SHARES_RE = re.compile(NUM + r"\s+(?:shares|ADSs?)\b", re.I)
CONTRACTS_RE = re.compile(NUM + r"\s+(?:call\s+|put\s+)?(?:option\s+)?contracts\b", re.I)
PROFIT_RE = re.compile(r"\$\s?" + NUM + r"(\s*million)?\s+(?:in\s+)?(?:combined\s+|total\s+|illicit\s+|illegal\s+|"
                       r"ill-gotten\s+|unlawful\s+)*(?:profits?|gains?)", re.I)


def biggest(rx, text, million_group=False):
    vals = []
    for m in rx.finditer(text or ""):
        v = float(m.group(1).replace(",", ""))
        if million_group and m.group(2):
            v *= 1e6
        vals.append(v)
    return max(vals) if vals else None


con = duckdb.connect(FEATURES, read_only=True)
con.execute(f"ATTACH '{EDGAR}' AS edgar (READ_ONLY)")
con.execute(f"ATTACH '{RELEASES}' AS rel (READ_ONLY)")
ev = con.execute("""
    SELECT f.event_id, f.ticker, f.day0, f.in_universe, e.lr_numbers
    FROM features.targets f JOIN edgar.events.target_audit a USING (event_id)
    JOIN rel.labels.charged_events e USING (event_id)
    WHERE f.window = 'pre' AND a.in_target_set AND e.is_charged ORDER BY f.day0""").df()
texts = dict(con.execute("SELECT CAST(lr_no AS VARCHAR), text FROM rel.raw.sec_litigation_releases").fetchall())
con.close()

con = duckdb.connect(PRICES, read_only=True)
con.register("syms", pd.DataFrame({"symbol": sorted(set(ev.ticker))}))
bars = con.execute("""SELECT symbol, date, volume FROM raw.alpaca_bars_daily JOIN syms USING (symbol)
                      WHERE adjustment = 'raw'""").df()
con.close()
bars["date"] = pd.to_datetime(bars.date).dt.date
vol = {s: g.set_index("date").volume for s, g in bars.groupby("symbol")}
sessions = [s.date() for s in xc.get_calendar("XNYS", start="2015-01-01").sessions]

rows = []
for e in ev.itertuples(index=False):
    d0 = pd.Timestamp(e.day0).date()
    i0 = sum(s < d0 for s in sessions)
    v = vol.get(e.ticker, pd.Series(dtype=float))
    win = v.reindex(sessions[i0 - 20:i0])
    base = v.reindex(sessions[max(0, i0 - 250):i0 - 30])
    text = " ".join(texts.get(lr, "") for lr in (e.lr_numbers or "").split("|"))
    shares = biggest(SHARES_RE, text)
    rows.append({"ticker": e.ticker, "day0": d0, "universe": e.in_universe,
                 "window_adv": round(win.mean()) if win.notna().any() else None,
                 "baseline_adv": round(base.mean()) if base.notna().any() else None,
                 "max_shares": shares, "max_contracts": biggest(CONTRACTS_RE, text),
                 "max_profit": biggest(PROFIT_RE, text, million_group=True),
                 "shares_of_window": round(shares / win.sum(), 4) if shares and win.notna().any() and win.sum() else None})
out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
s = out.shares_of_window.dropna()
print(f"\n{len(out)} charged targets with features; {len(s)} with a share count in the release.")
if len(s):
    print(f"Largest stated share count as a share of window volume: median {s.median():.2%}, "
          f"{(s >= 0.05).sum()} at 5% or more, {(s >= 0.20).sum()} at 20% or more")
