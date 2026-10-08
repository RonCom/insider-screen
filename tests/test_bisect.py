from datetime import date, timedelta

from insider_screen import sec_releases as sr


class FakeClient:
    """Release n is dated 2010-01-01 + (n - 22000) days; every 7th number is missing."""
    def __init__(self):
        self.calls = 0
    def get(self, url):
        self.calls += 1
        n = int(url.rsplit("-", 1)[1])
        if n % 7 == 0 or n > 26500:
            return 404, b""
        d = date(2010, 1, 1) + timedelta(days=n - 22000)
        html = f"<main><h1>X</h1><p>Litigation Release No. {n} / {d:%B} {d.day}, {d.year}</p></main>"
        return 200, html.encode()


def test_find_start():
    c = FakeClient()
    since = date(2016, 1, 1)
    n = sr.find_start(c, since)
    assert sr.fetch(c, n).release_date >= since
    prev = next(k for k in range(n - 1, n - 10, -1) if k % 7)
    assert sr.fetch(c, prev).release_date < since
    assert c.calls < 200


def test_crawl_stops_after_misses():
    out = sr.crawl(FakeClient(), 26400, stop_after_misses=40)
    assert out[-1].lr_no == 26500
