"""Write the release texts for the rows of a hand-check CSV to one JSON file, for review away from sec.gov.

    uv run python scripts/export_handcheck_texts.py data/handcheck.csv data/handcheck_texts.json
"""

import sys
from pathlib import Path

import duckdb

csv, out = sys.argv[1], sys.argv[2]
db = sys.argv[3] if len(sys.argv) > 3 else ("data/releases.duckdb" if Path("data/releases.duckdb").exists()
                                             else "data/insider.duckdb")
con = duckdb.connect(db, read_only=True)
n = con.execute(
    f"""COPY (SELECT lr_no, release_date, respondents, text FROM raw.sec_litigation_releases
              WHERE lr_no IN (SELECT TRY_CAST(lr_no AS INTEGER)
                              FROM read_csv('{Path(csv).as_posix()}', header=true, all_varchar=true))
              ORDER BY lr_no)
        TO '{Path(out).as_posix()}' (FORMAT JSON, ARRAY true)""").fetchone()
con.close()
print(f"Wrote {n[0] if n else '?'} releases from {db} to {out}")
