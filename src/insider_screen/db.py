"""Database files. Each long-running loader writes its own file, because DuckDB lets one process
write a file at a time: a price load and an extraction can then run side by side.

    data/releases.duckdb  sec_releases, extract, match
    data/edgar.duckdb   edgar
    data/finra.duckdb   shortsale daily and monthly
    data/prices.duckdb  prices

Steps that read another file attach it read-only, which fails while that file's loader is running.

`split` copies the tables of the old single file (data/insider.duckdb) into these files and leaves
the old file in place:
    uv run python -m insider_screen.db split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

RELEASES = "data/releases.duckdb"
EDGAR = "data/edgar.duckdb"
FINRA = "data/finra.duckdb"
PRICES = "data/prices.duckdb"
OLD = "data/insider.duckdb"

# (schema, table prefix) -> file; first match wins
ROUTES = [
    ("raw", "sec_", RELEASES), ("extracted", "", RELEASES), ("labels", "", RELEASES),
    ("raw", "edgar_", EDGAR), ("events", "", EDGAR),
    ("raw", "finra_", FINRA),
    ("raw", "alpaca_", PRICES),
]


def route(schema: str, name: str, files: dict[str, str] | None = None) -> str | None:
    for s, prefix, target in ROUTES:
        if schema == s and name.startswith(prefix):
            return (files or {}).get(target, target)
    return None


def split(old: str = OLD, files: dict[str, str] | None = None) -> list[tuple[str, str, str]]:
    """Copy every table and view of `old` to the file `route` gives it. Tables already present in
    the target are skipped, so it can be rerun. A view whose tables now live in another file is
    copied as a table. Returns (object, target, action) rows."""
    if not Path(old).exists():
        raise SystemExit(f"{old} not found; nothing to split")
    src = duckdb.connect(old, read_only=True)
    tables = src.execute(
        "SELECT schema_name, table_name FROM duckdb_tables() WHERE database_name = current_database()").fetchall()
    views = src.execute(
        """SELECT schema_name, view_name, sql FROM duckdb_views()
           WHERE database_name = current_database() AND NOT internal""").fetchall()
    src.close()

    log = []
    targets: dict[str, list] = {}
    for schema, name in tables:
        targets.setdefault(route(schema, name, files) or "", []).append((schema, name, None))
    for schema, name, sql in views:
        targets.setdefault(route(schema, name, files) or "", []).append((schema, name, sql))
    for schema, name, _ in targets.pop("", []):
        log.append((f"{schema}.{name}", "-", "left in old file (no route)"))

    for target, objs in targets.items():
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(target)
        con.execute(f"ATTACH '{Path(old).as_posix()}' AS old (READ_ONLY)")
        cat = con.execute("SELECT current_database()").fetchone()[0]
        have = {(s, t) for s, t in con.execute(
            "SELECT schema_name, table_name FROM duckdb_tables() WHERE database_name = current_database() "
            "UNION ALL SELECT schema_name, view_name FROM duckdb_views() "
            "WHERE database_name = current_database() AND NOT internal").fetchall()}
        for schema, name, sql in sorted(objs, key=lambda o: o[2] is not None):  # tables before views
            full = f"{schema}.{name}"
            if (schema, name) in have:
                log.append((full, target, "already there"))
                continue
            con.execute(f'CREATE SCHEMA IF NOT EXISTS "{cat}".{schema}')
            if sql is None:
                con.execute(f'CREATE TABLE "{cat}".{full} AS SELECT * FROM old.{full}')
                log.append((full, target, "copied"))
                continue
            try:
                con.execute(sql)
                log.append((full, target, "view recreated"))
            except (duckdb.CatalogException, duckdb.BinderException):
                con.execute(f'CREATE TABLE "{cat}".{full} AS SELECT * FROM old.{full}')
                log.append((full, target, "view copied as table"))
        con.close()
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--old", default=OLD)
    a = ap.parse_args()
    for obj, target, action in split(a.old):
        print(f"{obj:40s} {target:22s} {action}")
    print(f"Done. Check the new files, then delete {a.old}.")


if __name__ == "__main__":
    main()
