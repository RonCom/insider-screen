import duckdb
import pandas as pd

from insider_screen import day0check


def _db(tmp_path, n=3):
    db = str(tmp_path / "e.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw; CREATE SCHEMA events")
    con.execute("CREATE TABLE raw.edgar_companies AS SELECT * FROM (VALUES (1, 'Target Co'), (2, 'Other Co')) t(cik, name)")
    ev = pd.DataFrame({
        "event_id": [f"acq-000{i}-21-00000{i}" for i in range(n)], "cik": [1] * n,
        "accession": [f"000{i}-21-00000{i}" for i in range(n)],
        "accepted_et": pd.to_datetime(["2021-06-16 07:10"] * n), "day0": pd.to_datetime(["2021-06-16"] * n),
        "evidence": [""] * n, "event_type": ["acquisition_target"] * n, "day0_basis": ["index_header"] * n,
    })
    con.register("ev", ev)
    con.execute("CREATE TABLE events.announcements AS SELECT * FROM ev")
    con.close()
    return db


def test_sample_and_score(tmp_path, capsys):
    db, csv = _db(tmp_path), str(tmp_path / "d.csv")
    df = day0check.sample(db, csv)
    assert len(df) == 3 and df.filing_index.iloc[0].endswith("-index.htm")
    df = pd.read_csv(csv, dtype=str).fillna("")
    df.loc[0, "press_release_et"] = "2021-06-16 06:30"   # same session
    df.loc[1, "press_release_et"] = "2021-06-15 16:30"   # after the prior close: day 0 is still June 16
    df.loc[2, "press_release_et"] = "2021-06-15 08:00"   # a session earlier
    df.to_csv(csv, index=False)
    out = day0check.score(csv)
    assert out.session_diff.tolist() == [0, 0, 1]
    assert "1 differ" in capsys.readouterr().out
