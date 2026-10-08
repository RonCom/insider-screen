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

import pandas as pd
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

EASTERN = "America/New_York"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36")
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
        m = DATE_RE.search(text[:2000])
        if m:
            dl_date, head_end = _parse_date(m.group(0)), m.start()
    top = text[:head_end] if head_end is not None else text[:1500]
    lines = [ln for ln in top.splitlines() if len(ln.split()) >= 5 and not SKIP_LINE_RE.match(ln)
             and not DATE_RE.search(ln)]
    headline = max(lines[:6], key=len) if lines else None
    return wire, dl_date, headline


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


def find_release(sec, web, index_url: str, max_candidates: int = 4) -> dict:
    """Look up one filing. `sec` and `web` are PoliteClient-like objects with get(url, ...) -> (status, body).
    Returns press_release_et ('YYYY-MM-DD HH:MM' or ''), press_release_source and notes."""
    status, body = sec.get(index_url)
    if status != 200:
        return {"press_release_et": "", "press_release_source": "", "notes": f"filing index returned {status}"}
    ex_url = exhibit_url(body, index_url)
    if ex_url is None:
        return {"press_release_et": "", "press_release_source": "",
                "notes": "no EX-99 exhibit in this 8-K; the press release may be in a separate filing (425 or 8-K)"}
    status, ex_body = sec.get(ex_url)
    if status != 200:
        return {"press_release_et": "", "press_release_source": "", "notes": f"exhibit {ex_url} returned {status}"}
    wire, dl_date, headline = read_exhibit(exhibit_text(ex_body))
    if not headline:
        return {"press_release_et": "", "press_release_source": "", "notes": f"no headline found in {ex_url}"}
    domains = WIRES[wire][0] if wire else ALL_DOMAINS
    site = " OR ".join(f"site:{d}" for d in domains)
    query = f'"{headline[:150]}" {site}'
    candidates: list[str] = []
    for url in (ddg_url(query), bing_url(query), ddg_url(f"{headline[:150]} {site}")):
        status, page = web.get(url, use_cache=False, store=False)
        if status == 200:
            candidates += [u for u in search_urls(page) if on_wire(u, domains) and u not in candidates]
        if candidates:
            break
    rejected = []
    for url in candidates[:max_candidates]:
        status, page = web.get(url)
        if status != 200:
            rejected.append(f"{url} ({status})")
            continue
        t, how = page_time(page)
        score = fuzz.token_set_ratio(headline.lower(), page_title(page).lower())
        if t is None:
            rejected.append(f"{url} ({how})")
            continue
        if score < MIN_TITLE_SCORE:
            rejected.append(f"{url} (title match {score:.0f})")
            continue
        if dl_date is not None and t.date() != dl_date:
            rejected.append(f"{url} (published {t:%Y-%m-%d %H:%M} ET, dateline {dl_date})")
            continue
        return {"press_release_et": f"{t:%Y-%m-%d %H:%M}", "press_release_source": url,
                "notes": f"auto: {wire or 'wire not named'}; time from {how}; title match {score:.0f}; "
                         f"dateline {dl_date}; exhibit {ex_url}"}
    tried = "; rejected: " + " | ".join(rejected) if rejected else ""
    return {"press_release_et": "", "press_release_source": "",
            "notes": f"not found automatically ({wire or 'wire not named'}, dateline {dl_date}){tried}; "
                     f"exhibit {ex_url}; search {ddg_url(headline[:150])}"}
