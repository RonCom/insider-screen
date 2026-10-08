from datetime import date

from insider_screen.sec_releases import parse_release

PAGE = """<html><head><title>x</title></head><body>
<nav>Menu Enforcement Litigation Release No. 1 / bogus</nav>
<main>
<h1>Jane Q. Example</h1>
<p>U.S. SECURITIES AND EXCHANGE COMMISSION</p>
<p>Litigation Release No. 25999 / March 3, 2024</p>
<p>Securities and Exchange Commission v. Example, No. 1:24-cv-00001 (S.D.N.Y. filed Mar. 1, 2024)</p>
<p>SEC Charges Former Analyst with Insider Trading</p>
<p>The complaint alleges Example bought out-of-the-money call options in Target Co. ahead of the
June 5, 2023 announcement that Buyer Inc. would acquire Target Co., using material nonpublic information.</p>
<p>The case originated from the Market Abuse Unit's Analysis and Detection Center.
The SEC appreciates the assistance of the Financial Industry Regulatory Authority.</p>
<p>Last Reviewed or Updated: March 4, 2024</p>
</main></body></html>"""

SPLIT_HEADER = PAGE.replace(
    "<p>Litigation Release No. 25999 / March 3, 2024</p>",
    "<p>Litigation Release No. 25999</p><p>March 3, 2024</p>",
)


def test_parse_fields():
    r = parse_release(PAGE, 25999)
    assert r.lr_no == 25999
    assert r.release_date == date(2024, 3, 3)
    assert r.respondents == "Jane Q. Example"
    assert r.is_insider_candidate
    assert r.mentions_finra and r.mentions_detection_center and r.mentions_market_abuse_unit
    assert r.mentions_options
    assert "Last Reviewed" not in r.text
    assert r.text.startswith("Litigation Release No. 25999")


def test_date_on_separate_line():
    assert parse_release(SPLIT_HEADER, 25999).release_date == date(2024, 3, 3)


def test_non_release_page_returns_none():
    assert parse_release("<html><body><h1>Page not found</h1></body></html>", 1) is None


def test_non_insider_release():
    page = PAGE.replace("Insider Trading", "Offering Fraud").replace(
        "using material nonpublic information", "while misstating revenue"
    ).replace("ahead of the\nJune 5, 2023 announcement", "after the June 5, 2023 offering")
    r = parse_release(page, 25999)
    assert not r.is_insider_candidate


def test_tipping_without_insider_phrases():
    page = PAGE.replace("SEC Charges Former Analyst with Insider Trading", "SEC Charges Former Analyst").replace(
        "using material nonpublic information", "after a friend tipped him about the deal"
    ).replace("ahead of the\nJune 5, 2023 announcement", "before June 5, 2023")
    assert "insider" not in page.lower() and "nonpublic" not in page.lower()
    assert parse_release(page, 25999).is_insider_candidate


def test_header_date_variants():
    from insider_screen.sec_releases import date_from_text
    assert date_from_text("Litigation Release No. 24498 / June 11. 2019\nSEC v. X") == date(2019, 6, 11)
    assert date_from_text("Litigation Release No. 24713 / January 13 2020\nSEC v. X\n, No. 3:18-cv-01135 (D. Conn. filed July 10, 2018)") == date(2020, 1, 13)
    assert date_from_text("Litigation Release No. 26197 / Dec. 18, 2024\nSEC v. X") == date(2024, 12, 18)
    assert date_from_text("Litigation Release No. 24035/ January 26, 2018") == date(2018, 1, 26)
    assert date_from_text("Litigation Release No. 25001 / Sept. 5, 2019") == date(2019, 9, 5)
    assert date_from_text("Litigation Release No. 25999\nMarch 3, 2024\nSEC v. X") == date(2024, 3, 3)
    # a filing date in the caption is not the release date
    assert date_from_text("Litigation Release No. 25999\nSEC v. X (S.D.N.Y. filed Mar. 1, 2024)") is None


def test_date_anomalies_flags_outliers_and_gaps():
    import pandas as pd

    from insider_screen.sec_releases import date_anomalies
    days = pd.date_range("2019-01-01", periods=60, freq="D")
    df = pd.DataFrame({"lr_no": range(1000, 1060), "release_date": [d.date() for d in days]})
    assert date_anomalies(df).empty
    df.loc[20, "release_date"] = date(2017, 7, 10)  # complaint filing date read as release date
    df.loc[40, "release_date"] = None
    bad = date_anomalies(df)
    assert dict(zip(bad.lr_no, bad.problem)) == {1020: "out_of_sequence", 1040: "no_date"}


def test_fix_dates_updates_from_stored_text(tmp_path):
    import duckdb

    from insider_screen.sec_releases import fix_dates
    db = str(tmp_path / "r.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw")
    con.execute("""CREATE TABLE raw.sec_litigation_releases AS SELECT * FROM (VALUES
        (24498, NULL::DATE, 'Litigation Release No. 24498 / June 11. 2019'),
        (24713, DATE '2018-07-10', 'Litigation Release No. 24713 / January 13 2020'),
        (24714, DATE '2020-01-14', 'Litigation Release No. 24714 / January 14, 2020')) t(lr_no, release_date, text)""")
    con.close()
    fix_dates(db)
    con = duckdb.connect(db, read_only=True)
    got = con.execute("SELECT lr_no, release_date::VARCHAR FROM raw.sec_litigation_releases ORDER BY 1").fetchall()
    assert got == [(24498, "2019-06-11"), (24713, "2020-01-13"), (24714, "2020-01-14")]
