"""Find the newswire time of the press release behind an 8-K, for the day-0 check (day0check fill).

For each filing: the EX-99 exhibit in the filing index is the press release. Its dateline names the wire
("/PRNewswire/", "(BUSINESS WIRE)", "(GLOBE NEWSWIRE)") and the date; its first lines hold the headline.
The headline is searched on that wire's site (DuckDuckGo, then Bing), and the article's published time is
read from its structured data (JSON-LD, meta tags) or its visible timestamp line. A page is accepted only
if its title matches the headline and its Eastern date matches the dateline date; anything else is left
blank with a note and a search link for checking by hand.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

EASTERN = "America/New_York"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36")
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
}
MIN_TITLE_SCORE = 75

# wire name -> (site domains, dateline marker)
WIRES = {
    "PR Newswire": (("prnewswire.com",), re.compile(r"/\s*PRNewswire(?:-[A-Za-z]+)?\s*/|\bPRNewswire\b", re.I)),
    "Business Wire": (("businesswire.com",), re.compile(r"\(\s*BUSINESS\s+WIRE\s*\)", re.I)),
    "GlobeNewswire": (("globenewswire.com",), re.compile(r"\(\s*GLOBE\s+NEWSWIRE\s*\)|GlobeNewswire", re.I)),
    "Accesswire": (("accessnewswire.com", "accesswire.com"), re.compile(r"\bACCESS\s*(?:NEWS)?WIRE\b", re.I)),
    "Newsfile": (("newsfilecorp.com",), re.compile(r"\bNewsfile\s+Corp\b", re.I)),
}
ALL_DOMAINS = tuple(d for domains, _ in WIRES.values() for d in domains)

MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
         r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
DATE_RE = re.compile(rf"{MONTH}\s+\d{{1,2}},?\s+\d{{4}}")
# visible timestamps: PR Newswire "Nov 02, 2023, 06:00 ET"; GlobeNewswire "November 02, 2023 06:00 ET";
# Business Wire "November 02, 2023 06:00 AM Eastern Daylight Time"
TIME_TEXT_RE = re.compile(
    rf"({MONTH}\s+\d{{1,2}},?\s+\d{{4}}),?\s+(\d{{1,2}}:\d{{2}})\s*(AM|PM)?\s*"
    r"(?:ET|EST|EDT|Eastern(?:\s+(?:Daylight|Standard))?(?:\s+Time)?)\b", re.I)
BOILERPLATE_RE = re.compile(
    r"pursuant to Rule|Securities Act of 1933|Exchange Act of 1934|Subject Company|Commission File|"
    r"Filed by|Form 8-K|Exhibit 99|Registration (?:No|Statement)|^Page \d|^\(Translation", re.I)
SKIP_LINE_RE = re.compile(r"^(?:exhibit|ex-?\s*99|press release|news release|for immediate release|"
                          r"contacts?|media|investors?|source)\b", re.I)


@dataclass
class Exhibit:
    url: str | None
    wire: str | None = None
    dateline_date: date | None = None
    headline: str | None = None


def _parse_date(s: str) -> date | None:
    try:
        return pd.Timestamp(re.sub(r"(?<=\w)\.", "", s).replace("Sept", "Sep")).date()
    except (ValueError, TypeError):
        return None


def exhibit_url(index_html: str | bytes, index_url: str) -> str | None:
    """The press-release exhibit from an EDGAR filing index: EX-99.1 if present, else the first EX-99."""
    soup = BeautifulSoup(index_html, "lxml")
    found = []
    for tr in soup.select("table.tableFile tr"):
        cells = tr.find_all("td")
        link = tr.find("a", href=True)
        if len(cells) < 4 or link is None:
            continue
        doc_type = cells[3].get_text(strip=True).upper()
        if doc_type.startswith("EX-99"):
            href = link["href"].replace("/ix?doc=", "")
            found.append((doc_type != "EX-99.1", urljoin(index_url, href)))
    return min(found)[1] if found else None


def exhibit_text(html: str | bytes) -> str:
    text = BeautifulSoup(html, "lxml").get_text("\n")
    lines = (re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines())
    return "\n".join(ln for ln in lines if ln)


def read_exhibit(text: str) -> tuple[str | None, date | None, str | None]:
    """(wire, dateline date, headline) from a press-release exhibit."""
    wire, pos = None, None
    for name, (_, marker) in WIRES.items():
        m = marker.search(text[:6000])
        if m and (pos is None or m.start() < pos):
            wire, pos = name, m.start()
    head_end = pos if pos is not None else None
    dl_date = None
    if pos is not None:
        dates = list(DATE_RE.finditer(text[max(0, pos - 250):pos + 120]))
        dl_date = _parse_date(dates[0].group(0)) if dates else None
    else:  # no wire: the first date near the top is the dateline
        m = DATE_RE.search(text[:5000])
        if m:
            dl_date, head_end = _parse_date(m.group(0)), m.start()
    top = text[:head_end] if head_end is not None else text[:3000]
    return wire, dl_date, headline_of(top)


def headline_of(top: str) -> str | None:
    """The headline is the first substantial line of the release; sub-headings follow it and are often longer.
    Filing boilerplate above it ("Filed by ... pursuant to Rule 425") is skipped."""
    candidates = [ln for ln in top.splitlines()
                  if not SKIP_LINE_RE.match(ln) and not DATE_RE.search(ln) and not BOILERPLATE_RE.search(ln)]
    for min_words in (5, 3):
        for ln in candidates:
            if len(ln.split()) >= min_words and not ln.isupper() or len(ln.split()) >= 8:
                return ln[:200]
    return None


def search_urls(html: str | bytes) -> list[str]:
    """Result links from a DuckDuckGo HTML or Bing results page."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.select("a.result__a[href], li.b_algo h2 a[href]"):
        href = a["href"]
        if "uddg=" in href:
            href = parse_qs(urlparse(href if "//" in href else "https:" + href).query).get("uddg", [href])[0]
        if href.startswith("//"):
            href = "https:" + href
        if "bing.com/ck/" in href:  # Bing redirect: u=a1<base64 of the target URL>
            u = parse_qs(urlparse(href).query).get("u", [""])[0]
            if u.startswith("a1"):
                try:
                    href = base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode()
                except (ValueError, UnicodeDecodeError):
                    continue
        out.append(href)
    return out


# Each wire's own search page, and the path its article links use.
SITE_SEARCH = {
    "PR Newswire": ("https://www.prnewswire.com/search/news/?keyword={q}&pagesize=25", "/news-releases/"),
    "GlobeNewswire": ("https://www.globenewswire.com/search/keyword/{q}", "/news-release/"),
}


def site_search_url(wire: str, query: str) -> str:
    return SITE_SEARCH[wire][0].format(q=quote_plus(query))


def article_links(html: str | bytes, base: str, path: str) -> list[str]:
    """Article links on a wire's own search results page."""
    out = []
    for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
        url = urljoin(base, a["href"].split("#")[0])
        if path in urlparse(url).path and url not in out:
            out.append(url)
    return out


def ddg_url(query: str) -> str:
    return "https://html.duckduckgo.com/html/?q=" + quote_plus(query)


def bing_url(query: str) -> str:
    return "https://www.bing.com/search?q=" + quote_plus(query)


def _iso_eastern(value: str) -> pd.Timestamp | None:
    """An ISO timestamp with a time and an offset, in Eastern time. Dates without a time or offset are refused."""
    if not re.search(r"T?\d{1,2}:\d{2}", value) or not re.search(r"(?:Z|[+-]\d{2}:?\d{2})\s*$", value.strip()):
        return None
    try:
        return pd.Timestamp(value.strip()).tz_convert(EASTERN)
    except (ValueError, TypeError):
        return None


def _ld_dates(obj) -> list[str]:
    if isinstance(obj, list):
        return [d for o in obj for d in _ld_dates(o)]
    if isinstance(obj, dict):
        found = [obj["datePublished"]] if isinstance(obj.get("datePublished"), str) else []
        return found + _ld_dates(obj.get("@graph", []))
    return []


def page_time(html: str | bytes) -> tuple[pd.Timestamp | None, str]:
    """Published time of a newswire article in Eastern time, and where it was read from."""
    soup = BeautifulSoup(html, "lxml")
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
        except json.JSONDecodeError:
            continue
        for v in _ld_dates(data):
            t = _iso_eastern(v)
            if t is not None:
                return t, "json-ld"
    for attrs in ({"property": "article:published_time"}, {"itemprop": "datePublished"},
                  {"name": "article:published_time"}, {"name": "date"}, {"name": "DC.date.issued"}):
        m = soup.find("meta", attrs=attrs)
        if m and m.get("content"):
            t = _iso_eastern(m["content"])
            if t is not None:
                return t, "meta"
    m = TIME_TEXT_RE.search(soup.get_text(" "))
    if m:
        d = _parse_date(m.group(1))
        if d is not None:
            t = pd.Timestamp(f"{d} {m.group(2)} {m.group(3) or ''}".strip())
            return t.tz_localize(EASTERN), "page text"
    for tm in soup.find_all("time", datetime=True):
        t = _iso_eastern(tm["datetime"])
        if t is not None:
            return t, "time tag"
    return None, "no time on page"


def page_title(html: str | bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return og["content"].strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    return soup.title.get_text(strip=True) if soup.title else ""


def on_wire(url: str, domains: tuple[str, ...] = ALL_DOMAINS) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in domains)


def _get(client, url: str, **kw) -> tuple[int | str, bytes]:
    """client.get that never raises: a timeout or dropped connection comes back as the error's name
    (e.g. 'ReadTimeout') in place of a status code, so one slow site doesn't end the lookup."""
    try:
        return client.get(url, **kw)
    except httpx.HTTPError as err:
        return f"{urlparse(url).netloc} {type(err).__name__}", b""


SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{acc}-index.htm"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{doc}"
RELATED_FORMS = {"8-K", "425", "DEFA14A", "SC14D9C", "SC 14D9-C", "SC TO-C", "8-K12B", "6-K"}


def _filings(sec, cik: int, day: date | None = None) -> pd.DataFrame:
    """A filer's filings from the submissions API. Older filings sit on extra pages, each covering a date
    range; only the pages covering `day` are fetched (long-time filers have many, megabytes each)."""
    status, body = _get(sec, SUBMISSIONS_URL.format(cik=cik))
    if status != 200:
        return pd.DataFrame()
    data = json.loads(body)
    blocks = [data.get("filings", {}).get("recent", {})]
    for f in data.get("filings", {}).get("files", []):
        lo, hi = f.get("filingFrom"), f.get("filingTo")
        if day is not None and lo and hi and not (pd.Timestamp(lo).date() <= day <= pd.Timestamp(hi).date()
                                                  + pd.Timedelta(days=1)):
            continue
        st, b = _get(sec, "https://data.sec.gov/submissions/" + f["name"])
        if st == 200:
            blocks.append(json.loads(b))
    cols = ["accessionNumber", "filingDate", "form", "primaryDocument"]
    frames = [pd.DataFrame({c: blk.get(c, []) for c in cols}) for blk in blocks if blk]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def related_documents(sec, cik: int, accession: str, day: date) -> list[str]:
    """Press-release candidates the same filer filed on `day` or the day after: EX-99 exhibits of each
    filing, or the primary document of a 425, DEFA14A or SC14D9C (those are often the release itself)."""
    df = _filings(sec, cik, day)
    if df.empty:
        return []
    when = pd.to_datetime(df.filingDate).dt.date
    near = df[(when >= day) & (when <= day + pd.Timedelta(days=1)) & df.form.isin(RELATED_FORMS)
              & (df.accessionNumber != accession)]
    out = []
    for acc, form, doc in near[["accessionNumber", "form", "primaryDocument"]].itertuples(index=False):
        folder = acc.replace("-", "")
        idx = INDEX_URL.format(cik=cik, folder=folder, acc=acc)
        st, body = _get(sec, idx)
        ex = exhibit_url(body, idx) if st == 200 else None
        if ex:
            out.append(ex)
        elif form != "8-K" and doc:
            out.append(DOC_URL.format(cik=cik, folder=folder, doc=doc))
    return out


def _accession_cik(index_url: str) -> tuple[int, str]:
    m = re.search(r"/data/(\d+)/\d+/(\d{10}-\d{2}-\d{6})-index", index_url)
    return int(m.group(1)), m.group(2)


def _candidates(web, wire: str | None, headline: str) -> tuple[list[str], list[str]]:
    """Article URLs for a headline: the wire's own search first (both searchable wires when the exhibit
    doesn't name one), then DuckDuckGo and Bing. Returns (urls, what each search returned)."""
    log, urls = [], []
    q = " ".join(headline.split()[:14])
    wires = [wire] if wire in SITE_SEARCH else ([] if wire else list(SITE_SEARCH))
    for w in wires:
        url = site_search_url(w, q)
        status, page = _get(web, url, use_cache=False, store=False)
        found = article_links(page, url, SITE_SEARCH[w][1]) if status == 200 else []
        log.append(f"{w} search {status}: {len(found)} links")
        urls += [u for u in found if u not in urls]
    if urls:
        return urls, log
    domains = WIRES[wire][0] if wire else ALL_DOMAINS
    site = " OR ".join(f"site:{d}" for d in domains)
    for name, url in (("DuckDuckGo", ddg_url(f'"{headline[:150]}" {site}')),
                      ("Bing", bing_url(f'"{headline[:150]}" {site}'))):
        status, page = _get(web, url, use_cache=False, store=False)
        found = [u for u in search_urls(page) if on_wire(u, domains)] if status == 200 else []
        log.append(f"{name} {status}: {len(found)} links")
        urls += [u for u in found if u not in urls]
        if urls:
            break
    return urls, log


def _check_page(web, url: str, headline: str, dl_date: date | None, filed: date | None):
    """(time, how, title score) if the page passes, else (None, reason, score)."""
    status, page = _get(web, url)
    if status != 200:
        return None, f"page {status}", 0
    t, how = page_time(page)
    score = fuzz.token_set_ratio(headline.lower(), page_title(page).lower())
    if t is None:
        return None, how, score
    if dl_date is not None:
        if score < MIN_TITLE_SCORE:
            return None, f"title match {score:.0f}", score
        if t.date() != dl_date:
            return None, f"published {t:%Y-%m-%d %H:%M} ET, dateline {dl_date}", score
    else:  # no dateline: stricter title match, and within 3 days before or on the filing date
        if score < 85:
            return None, f"title match {score:.0f} (no dateline, 85 needed)", score
        if filed is not None and not (filed - pd.Timedelta(days=3) <= t.date() <= filed):
            return None, f"published {t:%Y-%m-%d} ET, filed {filed}", score
    return t, how, score


def find_release(sec, web, index_url: str, filed: date | None = None, max_candidates: int = 5) -> dict:
    """Look up one filing. `sec` and `web` are PoliteClient-like objects with get(url, ...) -> (status, body).
    `filed` is the 8-K's acceptance date (Eastern). Returns press_release_et ('YYYY-MM-DD HH:MM' or ''),
    press_release_source and notes."""
    def blank(note: str) -> dict:
        return {"press_release_et": "", "press_release_source": "", "notes": note}

    status, body = _get(sec, index_url)
    if status != 200:
        return blank(f"filing index returned {status}")
    docs = [u for u in [exhibit_url(body, index_url)] if u]
    if filed is not None:
        cik, acc = _accession_cik(index_url)
        docs += related_documents(sec, cik, acc, filed)
    if not docs:
        return blank("no EX-99 exhibit in this 8-K and no press release filed the same or next day")
    for doc in docs:
        status, ex_body = _get(sec, doc)
        if status != 200:
            continue
        wire, dl_date, headline = read_exhibit(exhibit_text(ex_body))
        if headline:
            break
    else:
        return blank(f"no headline found in {' | '.join(docs)}")
    candidates, log = _candidates(web, wire, headline)
    rejected = []
    for url in candidates[:max_candidates]:
        t, how, score = _check_page(web, url, headline, dl_date, filed)
        if t is None:
            rejected.append(f"{url} ({how})")
            continue
        return {"press_release_et": f"{t:%Y-%m-%d %H:%M}", "press_release_source": url,
                "notes": f"auto: {wire or 'wire not named'}; time from {how}; title match {score:.0f}; "
                         f"dateline {dl_date}; release {doc}"}
    tried = f"; rejected: {' | '.join(rejected)}" if rejected else ""
    return blank(f"not found automatically ({wire or 'wire not named'}, dateline {dl_date}; "
                 f"headline '{headline[:80]}'; {'; '.join(log)}){tried}; release {doc}; "
                 f"search {ddg_url(headline[:150])}")
