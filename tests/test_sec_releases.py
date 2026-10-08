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
