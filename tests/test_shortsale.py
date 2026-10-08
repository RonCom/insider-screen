import io
from datetime import date

import duckdb

from insider_screen import shortsale as ss

CNMS = """Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
20230605|AAA|1000|10|2500|B,Q,N
20230605|BBB|50|0|400|Q,N
"""
FNSQ = """Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
20160301|AAA|600|0|1000|Q
2
"""
FNYX = """Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
20160301|AAA|100|0|300|N
20160301|CCC|5|0|10|N
"""


class FakeClient:
    def __init__(self):
        self.urls = []

    def get(self, url, use_cache=True, store=True):
        self.urls.append(url)
        if "CNMSshvol20230605" in url:
            return 200, CNMS.encode()
        if "FNSQshvol20160301" in url:
            return 200, FNSQ.encode()
        if "FNYXshvol20160301" in url:
            return 200, FNYX.encode()
        return 404, b""


def test_parse_daily_drops_trailer():
    df = ss.parse_daily(FNSQ)
    assert len(df) == 1 and df.short_volume.iloc[0] == 600


def test_facilities_switch():
    assert ss.facilities_for(date(2018, 7, 31)) == ["FNSQ", "FNYX"]
    assert ss.facilities_for(date(2018, 8, 1)) == ["CNMS"]


def test_fetch_day_sums_facilities():
    df, got = ss.fetch_day(FakeClient(), date(2016, 3, 1))
    assert got == ["FNSQ", "FNYX"]
    aaa = df[df.symbol == "AAA"].iloc[0]
    assert (aaa.short_volume, aaa.total_volume) == (700, 1300)


def test_load_daily_resumes(tmp_path):
    db = str(tmp_path / "s.duckdb")
    c = FakeClient()
    ss.load_daily(c, db, "2023-06-05", "2023-06-06")
    con = duckdb.connect(db)
    assert con.execute("SELECT count(*) FROM raw.finra_short_daily").fetchone()[0] == 2
    con.close()
    c2 = FakeClient()
    ss.load_daily(c2, db, "2023-06-05", "2023-06-06")
    assert not any("20230605" in u for u in c2.urls)  # loaded day skipped


def test_aggregate_monthly():
    text = """MarketCenter|Symbol|Date|Time|ShortType|Size|Price|LinkIndicator
Q|AAA|20160301|09:30:01|S|100|10.0|A
Q|AAA|20160301|09:31:00|S|500|10.1|A
Q|BBB|20160302|10:00:00|E|50|5.0|A
"""
    df = ss.aggregate_monthly(io.StringIO(text))
    aaa = df[df.symbol == "AAA"].iloc[0]
    assert (aaa.short_trades, aaa.short_shares, aaa.small_trades) == (2, 600, 1)


def test_find_parts_and_labels():
    existing = {
        "https://cdn.finra.org/equity/regsho/monthly/FNYXsh201603.zip",
        "https://cdn.finra.org/equity/regsho/monthly/FNSQsh202608_1.zip",
        "https://cdn.finra.org/equity/regsho/monthly/FNSQsh202608_2.zip",
    }
    head = lambda url: 200 if url in existing else 403  # noqa: E731
    assert ss.find_parts(head, "FNYX", 2016, 3) == ["https://cdn.finra.org/equity/regsho/monthly/FNYXsh201603.zip"]
    parts = ss.find_parts(head, "FNSQ", 2026, 8)
    assert [ss.part_label(u) for u in parts] == ["FNSQ_1", "FNSQ_2"]
    assert ss.find_parts(head, "FNQC", 2016, 3) == []


def test_aggregate_zip_matches_streaming(tmp_path):
    import zipfile
    text = """MarketCenter|Symbol|Date|Time|ShortType|Size|Price|LinkIndicator
Q|AAA|20160301|09:30:01|S|100|10.0|A
Q|AAA|20160301|09:31:00|S|500|10.1|A
Q|BBB|20160302|10:00:00|E|50|5.0|A
3
"""
    z = tmp_path / "FNSQsh201603_1.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("FNSQsh201603_1.txt", text)
    a = ss.aggregate_zip(z, tmp_path).sort_values(["date", "symbol"]).reset_index(drop=True)
    b = ss.aggregate_monthly(io.StringIO(text)).sort_values(["date", "symbol"]).reset_index(drop=True)
    assert a[["symbol", "short_trades", "short_shares", "small_trades"]].values.tolist() == \
        b[["symbol", "short_trades", "short_shares", "small_trades"]].values.tolist()
    assert not (tmp_path / "FNSQsh201603_1.txt").exists()


def test_bad_file_is_skipped_not_fatal():
    class Bad(FakeClient):
        def get(self, url, use_cache=True, store=True):
            if "FNYXshvol20160301" in url:
                return 200, b"<?xml version='1.0'?><Error>Something</Error>"
            return super().get(url, use_cache, store)
    df, got = ss.fetch_day(Bad(), date(2016, 3, 1))
    assert got == ["FNSQ"] and len(df) == 1
    assert ss.BAD_FILES and "FNYXshvol20160301" in ss.BAD_FILES[-1][0]
