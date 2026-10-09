"""How many insider-trading releases are cut short at a given Ollama context size. extract.py keeps the
first (num_ctx - 1500) x 4 characters of each release (1500 tokens held back for the answer, about
4 characters per token).

    uv run python scripts/release_lengths.py
"""

import duckdb

from insider_screen.db import RELEASES

con = duckdb.connect(RELEASES, read_only=True)
lengths = [r[0] for r in con.execute(
    "SELECT length(text) FROM raw.sec_litigation_releases WHERE is_insider_candidate").fetchall()]
con.close()
lengths.sort()
n = len(lengths)
print(f"{n} insider-trading candidate releases; median {lengths[n // 2]:,} characters, "
      f"90th percentile {lengths[int(n * 0.9)]:,}, longest {lengths[-1]:,}")
for ctx in (4096, 5120, 6144, 8192, 12288, 16384):
    keep = (ctx - 1500) * 4
    cut = [x for x in lengths if x > keep]
    lost = sum(x - keep for x in cut)
    print(f"  num_ctx {ctx:>6}: keeps {keep:>6,} characters; {len(cut):>4} releases cut ({len(cut) / n:.0%}), "
          f"{lost / max(1, sum(lengths)):.1%} of all release text lost")
