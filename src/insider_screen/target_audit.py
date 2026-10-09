"""Audit of the acquisition-target events and their tickers (day-0 spot check, 2026-10-09).

The target rule (edgar.py) takes the latest 8-K with Item 1.01 within 120 days before the filer's first
PREM14A, DEFM14A, SC 14D9, SC14D9C or SC 13E3. Merger proxies are also filed by acquirers that issue
shares (Ribbon buying ECI), by SPACs for their business combination (Clover Leaf), and by companies
selling a major asset (Seres selling VOWST). Each target event is put in one category:

  spac            SIC 6770 (blank check): a SPAC buys, it isn't bought
  tender_or_13e3  the filer also filed SC 14D9 or SC 13E3 (only targets and going-private issuers do)
  delisted_after  the filer has a 25-NSE or 15-12B within DELIST_DAYS after day 0: it stopped trading
  unconfirmed     none of these: an acquirer, an asset seller, or a deal that broke

and each event's day-0 ticker is listed with its type in the map, to find notes and preferred shares
picked in place of the common stock (DHCNI for Diversified Healthcare Trust).

Usage:
    uv run python -m insider_screen.target_audit
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from insider_screen.db import EDGAR, FINRA, REFERENCE

DELIST_DAYS = 540
TARGET_ONLY_FORMS = ("SC 14D9", "SC 13E3")
EXIT_FORMS = ("25-NSE", "15-12B")


def audit(edgar_db: str = EDGAR, reference_db: str = REFERENCE, finra_db: str = FINRA,
          out: str = "data/target_audit.csv", seed: int = 5) -> pd.DataFrame:
    from insider_screen import tickers

    con = duckdb.connect(reference_db, read_only=True)
    tickers.attach(con, edgar_db, finra_db)
    df = con.execute(f"""
        WITH t AS (SELECT * FROM ({tickers.event_tickers_sql(con)}) WHERE event_type = 'acquisition_target'),
        f AS (SELECT cik, form, filing_date FROM edgar.raw.edgar_filings
              WHERE form IN {TARGET_ONLY_FORMS + EXIT_FORMS})
        SELECT t.event_id, t.cik, c.name AS company, c.sic, CAST(t.day0 AS DATE) AS day0, t.evidence,
               t.ticker, t.ticker_source, m.type AS ticker_type, m.name AS ticker_name,
               EXISTS (SELECT 1 FROM f WHERE f.cik = t.cik AND f.form IN {TARGET_ONLY_FORMS}
                       AND f.filing_date BETWEEN CAST(t.day0 AS DATE) - 30 AND CAST(t.day0 AS DATE) + 365)
                 AS tender_or_13e3,
               (SELECT min(f.filing_date) FROM f WHERE f.cik = t.cik AND f.form IN {EXIT_FORMS}
                       AND f.filing_date BETWEEN CAST(t.day0 AS DATE) AND CAST(t.day0 AS DATE) + {DELIST_DAYS})
                 AS exit_date
        FROM t JOIN edgar.raw.edgar_companies c USING (cik)
        LEFT JOIN ref.ticker_cik m
          ON m.ticker = t.ticker AND m.cik = t.cik
         AND (m.valid_from IS NULL OR t.day0 >= m.valid_from) AND (m.valid_to IS NULL OR t.day0 <= m.valid_to)
        QUALIFY row_number() OVER (PARTITION BY t.event_id ORDER BY m.type = 'CS' DESC) = 1""").df()
    con.close()
    df["category"] = "unconfirmed"
    df.loc[df.exit_date.notna(), "category"] = "delisted_after"
    df.loc[df.tender_or_13e3, "category"] = "tender_or_13e3"
    df.loc[df.sic.astype(str) == "6770", "category"] = "spac"
    df["year"] = pd.to_datetime(df.day0).dt.year
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"{len(df)} acquisition-target events by category (written to {out}):")
    print(df.category.value_counts().to_string())
    print("\nBy year:")
    print(df.pivot_table(index="year", columns="category", values="event_id", aggfunc="size", fill_value=0)
          .to_string())
    unc = df[df.category == "unconfirmed"]
    print("\n15 unconfirmed, at random:")
    print(unc.sample(n=min(15, len(unc)), random_state=seed)[["company", "day0", "evidence", "ticker"]]
          .to_string(index=False))

    has = df[df.ticker.notna()]
    print(f"\nTicker type in the map, {len(has)} target events with a ticker:")
    print(has.ticker_type.fillna("(supplement, no map row)").value_counts().to_string())
    odd = has[has.ticker_type.notna() & (has.ticker_type != "CS")]
    if len(odd):
        print(f"\n{len(odd)} not typed CS:")
        print(odd[["company", "day0", "ticker", "ticker_type", "ticker_name"]].head(30).to_string(index=False))
    return df


def main() -> None:
    audit()


if __name__ == "__main__":
    main()
