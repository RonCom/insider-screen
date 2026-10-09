import base64
from datetime import date

import pandas as pd

from insider_screen import day0check
from insider_screen import press_release as pr

INDEX_URL = "https://www.sec.gov/Archives/edgar/data/701374/000119312523268712/0001193125-23-268712-index.htm"
INDEX = """<html><body><table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/ix?doc=/Archives/edgar/data/701374/000119312523268712/d8k.htm">d8k.htm</a></td><td>8-K</td><td>50</td></tr>
<tr><td>2</td><td>EX-2.1</td><td><a href="/Archives/edgar/data/701374/000119312523268712/dex21.htm">dex21.htm</a></td><td>EX-2.1</td><td>900</td></tr>
<tr><td>3</td><td>EX-99.2</td><td><a href="/Archives/edgar/data/701374/000119312523268712/dex992.htm">dex992.htm</a></td><td>EX-99.2</td><td>40</td></tr>
<tr><td>4</td><td>EX-99.1</td><td><a href="/Archives/edgar/data/701374/000119312523268712/dex991.htm">dex991.htm</a></td><td>EX-99.1</td><td>30</td></tr>
</table></body></html>"""
EXHIBIT_URL = "https://www.sec.gov/Archives/edgar/data/701374/000119312523268712/dex991.htm"
EXHIBIT = """<html><body><p>Exhibit 99.1</p>
<p><b>Six Flags and Cedar Fair to Combine in Merger of Equals, Creating a Leading Amusement Park Operator</b></p>
<p>Combined company to operate 42 parks</p>
<p>ARLINGTON, Texas and SANDUSKY, Ohio, Nov. 2, 2023 /PRNewswire/ -- Six Flags Entertainment Corporation (NYSE: SIX)
and Cedar Fair, L.P. (NYSE: FUN) today announced a definitive merger agreement.</p></body></html>"""
ARTICLE_URL = "https://www.prnewswire.com/news-releases/six-flags-and-cedar-fair-to-combine-301975271.html"
ARTICLE = """<html><head>
<meta property="og:title" content="Six Flags and Cedar Fair to Combine in Merger of Equals, Creating a Leading Amusement Park Operator">
<script type="application/ld+json">{"@context":"https://schema.org","@type":"NewsArticle",
 "datePublished":"2023-11-02T10:00:00Z"}</script></head>
<body><p class="mb-no">Nov 02, 2023, 06:00 ET</p></body></html>"""
DDG = f"""<html><body>
<a class="result__a" href="//duckduckgo.com/l/?uddg={ARTICLE_URL.replace(':', '%3A').replace('/', '%2F')}&rut=x">Six Flags</a>
<a class="result__a" href="https://www.example.com/other">Other</a></body></html>"""


def test_exhibit_url_prefers_991():
    assert pr.exhibit_url(INDEX, INDEX_URL) == EXHIBIT_URL


def test_read_exhibit():
    wire, d, head = pr.read_exhibit(pr.exhibit_text(EXHIBIT))
    assert (wire, d) == ("PR Newswire", date(2023, 11, 2))
    assert head.startswith("Six Flags and Cedar Fair to Combine")


def test_business_wire_dateline():
    text = pr.exhibit_text("<p>Acme to Be Acquired by Beta for $40 per Share in Cash</p>"
                           "<p>BOSTON--(BUSINESS WIRE)--June 3, 2024-- Acme Inc. (NASDAQ: ACME) today...</p>")
    assert pr.read_exhibit(text)[:2] == ("Business Wire", date(2024, 6, 3))


def test_search_urls_ddg_and_bing():
    assert pr.search_urls(DDG) == [ARTICLE_URL, "https://www.example.com/other"]
    enc = base64.urlsafe_b64encode(ARTICLE_URL.encode()).decode().rstrip("=")
    bing = f'<ol><li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&&p=1&u=a1{enc}&ntb=1">x</a></h2></li></ol>'
    assert pr.search_urls(bing) == [ARTICLE_URL]


def test_page_time_sources():
    assert pr.page_time(ARTICLE) == (pd.Timestamp("2023-11-02 06:00", tz=pr.EASTERN), "json-ld")
    text_only = "<html><body><p>November 02, 2023 04:05 PM Eastern Daylight Time</p></body></html>"
    assert pr.page_time(text_only) == (pd.Timestamp("2023-11-02 16:05", tz=pr.EASTERN), "page text")
    gnw = "<html><body><span>January 13, 2025 07:00 ET</span></body></html>"
    assert pr.page_time(gnw)[0] == pd.Timestamp("2025-01-13 07:00", tz=pr.EASTERN)
    # a date with no time, or a time with no zone, is not used
    assert pr.page_time('<meta property="article:published_time" content="2023-11-02">')[0] is None
    assert pr.page_time('<meta property="article:published_time" content="2023-11-02T06:00:00">')[0] is None


class Fake:
    def __init__(self, pages):
        self.pages, self.urls = pages, []

    def get(self, url, use_cache=True, store=True):
        self.urls.append(url)
        for key, body in self.pages.items():
            if url.startswith(key):
                return 200, body.encode()
        return 404, b""


def test_find_release_end_to_end():
    sec = Fake({INDEX_URL: INDEX, EXHIBIT_URL: EXHIBIT})
    web = Fake({"https://html.duckduckgo.com/html/": DDG, ARTICLE_URL: ARTICLE})
    got = pr.find_release(sec, web, INDEX_URL)
    assert got["press_release_et"] == "2023-11-02 06:00"
    assert got["press_release_source"] == ARTICLE_URL
    assert got["notes"].startswith("auto: PR Newswire")


def test_find_release_rejects_wrong_date():
    sec = Fake({INDEX_URL: INDEX, EXHIBIT_URL: EXHIBIT.replace("Nov. 2, 2023", "Nov. 3, 2023")})
    web = Fake({"https://html.duckduckgo.com/html/": DDG, ARTICLE_URL: ARTICLE})
    got = pr.find_release(sec, web, INDEX_URL)
    assert got["press_release_et"] == "" and "dateline 2023-11-03" in got["notes"]
    assert "duckduckgo.com" in got["notes"]


def test_find_release_no_exhibit():
    sec = Fake({INDEX_URL: INDEX.replace("EX-99", "EX-10")})
    got = pr.find_release(sec, Fake({}), INDEX_URL)
    assert got["press_release_et"] == "" and "no EX-99 exhibit" in got["notes"]


def test_fill_skips_filled_rows_and_saves(tmp_path):
    csv = tmp_path / "day0_check.csv"
    pd.DataFrame({
        "event_id": ["a", "b"], "company": ["Six Flags", "Done Co"], "filing_index": [INDEX_URL, "x"],
        "press_release_et": ["", "2020-01-01 07:00"], "press_release_source": ["", "u"], "notes": ["", ""],
    }).to_csv(csv, index=False, encoding="utf-8-sig")
    sec = Fake({INDEX_URL: INDEX, EXHIBIT_URL: EXHIBIT})
    web = Fake({"https://html.duckduckgo.com/html/": DDG, ARTICLE_URL: ARTICLE})
    day0check.fill(str(csv), sec, web)
    out = pd.read_csv(csv, dtype=str, encoding="utf-8-sig").fillna("")
    assert out.press_release_et.tolist() == ["2023-11-02 06:00", "2020-01-01 07:00"]
    assert "x" not in sec.urls
    assert open(csv, "rb").read(3) == b"\xef\xbb\xbf"  # UTF-8 with BOM, as Excel's "CSV UTF-8"


def test_headline_is_first_line_not_longest():
    text = pr.exhibit_text("""<p>Filed by Six Flags Entertainment Corporation pursuant to Rule 425 under the Securities Act of 1933</p>
<p>Six Flags and Cedar Fair to Combine in Merger of Equals</p>
<p>Combined Company Will Benefit from Expanded and Complementary Portfolio of 42 Iconic Parks and 9 Resort Properties</p>
<p>ARLINGTON, Texas, Nov. 2, 2023 -- Six Flags today announced...</p>""")
    wire, d, head = pr.read_exhibit(text)
    assert (wire, d, head) == (None, date(2023, 11, 2), "Six Flags and Cedar Fair to Combine in Merger of Equals")


PRN_SEARCH = f"""<html><body><div class="row newsCards">
<a class="newsreleaseconsolidatelink" href="/news-releases/six-flags-and-cedar-fair-to-combine-301975271.html">Six Flags</a>
<a href="/news/six-flags-entertainment-corporation/">company page</a></div></body></html>"""


def test_wire_site_search_used_first():
    sec = Fake({INDEX_URL: INDEX, EXHIBIT_URL: EXHIBIT})
    web = Fake({"https://www.prnewswire.com/search/news/": PRN_SEARCH, ARTICLE_URL: ARTICLE})
    got = pr.find_release(sec, web, INDEX_URL)
    assert got["press_release_et"] == "2023-11-02 06:00"
    assert not any("duckduckgo" in u for u in web.urls)


def test_press_release_in_separate_filing():
    import json
    index_no_ex = INDEX.replace("EX-99", "EX-10")
    other_idx = "https://www.sec.gov/Archives/edgar/data/701374/000119312523268800/0001193125-23-268800-index.htm"
    doc_425 = "https://www.sec.gov/Archives/edgar/data/701374/000119312523268800/d425.htm"
    subs = {"filings": {"recent": {
        "accessionNumber": ["0001193125-23-268712", "0001193125-23-268800", "0001193125-23-100000"],
        "filingDate": ["2023-11-02", "2023-11-02", "2023-05-01"],
        "form": ["8-K", "425", "425"],
        "primaryDocument": ["d8k.htm", "d425.htm", "old.htm"]}, "files": []}}
    sec = Fake({INDEX_URL: index_no_ex, "https://data.sec.gov/submissions/CIK0000701374.json": json.dumps(subs),
                other_idx: "<html></html>", doc_425: EXHIBIT})
    web = Fake({"https://www.prnewswire.com/search/news/": PRN_SEARCH, ARTICLE_URL: ARTICLE})
    got = pr.find_release(sec, web, INDEX_URL, filed=date(2023, 11, 2))
    assert got["press_release_et"] == "2023-11-02 06:00" and doc_425 in got["notes"]
    assert not any("old.htm" in u for u in sec.urls)


def test_no_dateline_needs_close_title_and_filing_window():
    exhibit = EXHIBIT.replace("Nov. 2, 2023 /PRNewswire/ --", "--")
    sec = Fake({INDEX_URL: INDEX, EXHIBIT_URL: exhibit})
    web = Fake({"https://www.globenewswire.com/search/": "", "https://www.prnewswire.com/search/news/": PRN_SEARCH,
                ARTICLE_URL: ARTICLE})
    assert pr.find_release(sec, web, INDEX_URL, filed=date(2023, 11, 2))["press_release_et"] == "2023-11-02 06:00"
    got = pr.find_release(sec, web, INDEX_URL, filed=date(2023, 12, 1))
    assert got["press_release_et"] == "" and "filed 2023-12-01" in got["notes"]


def test_only_submission_pages_covering_the_date_are_fetched():
    import json
    subs = {"filings": {"recent": {"accessionNumber": [], "filingDate": [], "form": [], "primaryDocument": []},
                        "files": [{"name": "CIK0000000001-submissions-001.json", "filingFrom": "2019-01-01", "filingTo": "2021-12-31"},
                                  {"name": "CIK0000000001-submissions-002.json", "filingFrom": "2001-01-01", "filingTo": "2018-12-31"}]}}
    page = {"accessionNumber": ["0000000001-19-000001"], "filingDate": ["2019-06-10"], "form": ["425"],
            "primaryDocument": ["a.htm"]}
    sec = Fake({"https://data.sec.gov/submissions/CIK0000000001.json": json.dumps(subs),
                "https://data.sec.gov/submissions/CIK0000000001-submissions-001.json": json.dumps(page)})
    df = pr._filings(sec, 1, date(2019, 6, 10))
    assert df.accessionNumber.tolist() == ["0000000001-19-000001"]
    assert not any("submissions-002" in u for u in sec.urls)
