"""Events priced on a thinly traded share class when a sibling class trades far more (DISCB chosen for
Discovery, whose DISCA and DISCK carry most of the volume). A sibling is a ticker with the same length and
all but the last letter in common (DISCA, DISCB, DISCK) whose map name is the same company once class and
series words are dropped (so FRPH and FRPT, two companies, aren't siblings); flagged when its FINRA volume
that month is at least 10x the chosen ticker's.

    uv run python scripts/share_class_check.py
"""

import re

import duckdb
import pandas as pd

from insider_screen import tickers
from insider_screen.db import EDGAR, FINRA, REFERENCE

con = duckdb.connect(REFERENCE, read_only=True)
tickers.attach(con, EDGAR, FINRA)
ev = con.execute(f"""SELECT event_id, event_type, ticker, date_trunc('month', day0) AS month
                     FROM ({tickers.event_tickers_sql(con)}) WHERE ticker IS NOT NULL
                       AND event_type IN ('earnings', 'acquisition_target')""").df()
vol = con.execute("SELECT symbol, month, vol FROM ticker_volume").df()
names = con.execute("SELECT ticker, any_value(name) AS name FROM ref.ticker_cik GROUP BY 1").df()
con.close()


def company(name) -> str:
    if not isinstance(name, str):
        return ""
    name = re.sub(r"(?i)\b(series|class)\s+[a-z]\b.*$", "", name)
    return tickers._name_key(name)


key = dict(zip(names.ticker, names.name.map(company)))
vol["root"], vol["n"] = vol.symbol.str[:-1], vol.symbol.str.len()
ev["root"], ev["n"] = ev.ticker.str[:-1], ev.ticker.str.len()
ev = ev[ev.n >= 4]
own = ev.merge(vol.rename(columns={"symbol": "ticker", "vol": "own_vol"})[["ticker", "month", "own_vol"]],
               on=["ticker", "month"], how="left")
sib = (vol.merge(own[["event_id", "ticker", "root", "n", "month"]], on=["root", "n", "month"])
          .query("symbol != ticker"))
sib = sib[[bool(key.get(a)) and key.get(a) == key.get(b) for a, b in zip(sib.symbol, sib.ticker)]]
sib = (sib.groupby("event_id").agg(sibling=("symbol", "first"), sib_vol=("vol", "max")))
df = own.merge(sib, on="event_id")
df["ratio"] = df.sib_vol / df.own_vol.fillna(0).clip(lower=1)
bad = df[df.ratio >= 10]
print(f"{len(bad)} of {len(ev)} events on a share class with a sibling trading 10x more")
print(bad.groupby("event_type").size().to_string())
top = (bad.groupby(["ticker", "sibling"]).agg(events=("event_id", "size"), median_ratio=("ratio", "median"))
          .sort_values("events", ascending=False).head(25))
pd.set_option("display.width", 200)
print(top.round(0).to_string())
