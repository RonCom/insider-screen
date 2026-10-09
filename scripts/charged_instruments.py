"""Charged acquisition targets in the audited target set, with what each release says was traded.

A release charging several trades lists several issuers; the row whose issuer_name matches the
charged company is the one that counts. NULLs in the extraction columns mean the release has no row
in the extraction table.

    uv run python scripts/charged_instruments.py
"""

import duckdb

from insider_screen.db import EDGAR, RELEASES
from insider_screen.extract import TABLE

con = duckdb.connect(RELEASES, read_only=True)
con.execute(f"ATTACH '{EDGAR}' AS edgar (READ_ONLY)")
df = con.execute(f"""
    WITH ch AS (
        SELECT e.event_id, e.cik, CAST(e.day0 AS DATE) AS day0,
               CAST(unnest(string_split(e.lr_numbers, '|')) AS INTEGER) AS lr_no
        FROM labels.charged_events e JOIN edgar.events.target_audit a USING (event_id)
        WHERE e.is_charged AND a.in_target_set)
    SELECT DISTINCT co.name, ch.day0, ch.lr_no, t.issuer_name, t.event_type, t.instruments, t.direction
    FROM ch JOIN edgar.raw.edgar_companies co USING (cik)
    LEFT JOIN {TABLE} t USING (lr_no)
    ORDER BY ch.day0, ch.lr_no""").df()
con.close()
print(df.to_string(index=False, max_colwidth=40))
print(f"\n{df[['name', 'day0']].drop_duplicates().shape[0]} charged target events, {df.lr_no.nunique()} releases")
