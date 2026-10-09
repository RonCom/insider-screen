import duckdb

from insider_screen import target_audit


def test_audit_categories(tmp_path):
    ref, edgar = str(tmp_path / "reference.duckdb"), str(tmp_path / "edgar.duckdb")
    con = duckdb.connect(ref)
    con.execute("CREATE SCHEMA ref")
    con.execute("""CREATE TABLE ref.ticker_cik AS SELECT * FROM (VALUES
        ('TGT', 1::BIGINT, 'Target Co', 'CS', NULL::DATE, NULL::DATE),
        ('DHCNI', 4::BIGINT, 'Diversified Healthcare Trust', 'untyped', NULL::DATE, NULL::DATE))
        t(ticker, cik, name, type, valid_from, valid_to)""")
    con.close()
    con = duckdb.connect(edgar)
    con.execute("CREATE SCHEMA events; CREATE SCHEMA raw")
    con.execute("""CREATE TABLE raw.edgar_companies AS SELECT * FROM (VALUES
        (1, 'Target Co', '2834'), (2, 'Tender Co', '2834'), (3, 'Clover Leaf Capital', '6770'),
        (4, 'DIVERSIFIED HEALTHCARE TRUST', '6798'), (5, 'Ribbon', '7373')) t(cik, name, sic)""")
    con.execute("""CREATE TABLE raw.edgar_filings AS SELECT * FROM (VALUES
        (1, '25-NSE', DATE '2021-06-01'), (2, 'SC 14D9', DATE '2021-02-01'),
        (5, '25-NSE', DATE '2015-01-01')) t(cik, form, filing_date)""")
    con.execute("""CREATE TABLE events.announcements AS SELECT * FROM (VALUES
        ('a1', 1, 'acquisition_target', TIMESTAMP '2021-01-10', 'DEFM14A'),
        ('a2', 2, 'acquisition_target', TIMESTAMP '2021-01-20', 'SC 14D9'),
        ('a3', 3, 'acquisition_target', TIMESTAMP '2021-01-20', 'PREM14A'),
        ('a4', 4, 'acquisition_target', TIMESTAMP '2023-04-12', 'PREM14A'),
        ('a5', 5, 'acquisition_target', TIMESTAMP '2019-11-14', 'PREM14A'),
        ('e1', 1, 'earnings', TIMESTAMP '2021-01-10', '')) t(event_id, cik, event_type, day0, evidence)""")
    con.close()
    df = target_audit.audit(edgar, ref, str(tmp_path / "finra.duckdb"), out=str(tmp_path / "a.csv"))
    got = dict(zip(df.event_id, df.category))
    assert got == {"a1": "delisted_after", "a2": "tender_or_13e3", "a3": "spac", "a4": "unconfirmed",
                   "a5": "unconfirmed"}
    assert dict(zip(df.event_id, df.ticker_type))["a4"] == "untyped"
    con = duckdb.connect(edgar, read_only=True)
    kept = {r[0] for r in con.execute("SELECT event_id FROM events.target_audit WHERE in_target_set").fetchall()}
    assert kept == {"a1", "a2"}


def test_audit_table_and_test_symbols(tmp_path):
    from insider_screen import tickers as tk
    db = str(tmp_path / "reference.duckdb")
    con = duckdb.connect(db)
    tk._setup(con)
    con.execute("""INSERT INTO raw.massive_tickers VALUES
        ('NTEST.B', 'NASDAQ TEST STOCK', '0000001', NULL, FALSE, 'XNAS', NULL, NULL, '2016-06-01T00:00:00Z', NULL, now()),
        ('CSH', 'Cash America International', '0000001', 'CS', FALSE, 'XNYS', NULL, NULL, '2016-09-01T00:00:00Z', NULL, now())""")
    con.close()
    out = tk.build(db)
    assert list(out.ticker) == ["CSH"]
