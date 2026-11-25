"""Check the raw downloads: every file matches its manifest checksum, and each one parses.

Usage (from the project root):
    .venv/bin/python -I src/check_raw.py

Prints one line per file with its coverage (countries, years, share of non-missing values),
so problems are visible before any cleaning starts. It reads the raw files and changes nothing.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "MANIFEST.csv"


def check_manifest() -> tuple[list[dict], list[str]]:
    problems = []
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    listed = {row["path"] for row in rows}
    on_disk = {p.relative_to(ROOT).as_posix() for p in RAW.rglob("*")
               if p.is_file() and p.name not in ("MANIFEST.csv", ".DS_Store")}
    for path in sorted(on_disk - listed):
        problems.append(f"on disk but not in manifest: {path}")
    for row in rows:
        path = ROOT / row["path"]
        if not path.exists():
            problems.append(f"in manifest but missing: {row['path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            problems.append(f"checksum changed: {row['path']}")
    return rows, problems


def check_wdi(rows: list[dict]) -> None:
    print("\nWorld Bank WDI  (real countries only; aggregates such as 'World' excluded)")
    countries_row = next(r for r in rows if r["source"] == "World Bank WDI" and r["series"] == "countries")
    countries = json.loads(gzip.decompress((ROOT / countries_row["path"]).read_bytes()))[1]
    real = {c["id"] for c in countries if c["region"]["value"] != "Aggregates"}
    print(f"  country list: {len(countries)} entries, {len(real)} real countries")
    print(f"  {'series':22s} {'countries':>9s} {'years':>11s} {'filled':>7s}  name")
    for row in rows:
        if row["source"] != "World Bank WDI" or row["series"] == "countries":
            continue
        meta, data = json.loads(gzip.decompress((ROOT / row["path"]).read_bytes()))
        obs = [d for d in data if d["countryiso3code"] in real]
        filled = [d for d in obs if d["value"] is not None]
        years = sorted({int(d["date"]) for d in filled})
        share = len(filled) / len(obs) if obs else 0
        span = f"{years[0]}-{years[-1]}" if years else "none"
        print(f"  {row['series']:22s} {len({d['countryiso3code'] for d in filled}):>9d} {span:>11s} "
              f"{share:>6.0%}  {row['description'][:55]}")
    print(f"  vintage: {rows and next(r['vintage'] for r in rows if r['source'] == 'World Bank WDI' and r['series'] != 'countries')}")


def check_gdd(rows: list[dict]) -> None:
    print("\nIMF Global Debt Database")
    print(f"  {'series':18s} {'countries':>9s} {'years':>11s}  description")
    for row in rows:
        if row["source"] != "IMF GDD" or row["series"] in ("catalogue", "countries"):
            continue
        values = json.loads((ROOT / row["path"]).read_bytes())["values"][row["series"]]
        years = sorted({int(y) for series in values.values() for y in series})
        print(f"  {row['series']:18s} {len(values):>9d} {years[0]}-{years[-1]:>4d}   {row['description'][:60]}")
    print(f"  vintage: {next(r['vintage'] for r in rows if r['source'] == 'IMF GDD' and r['series'] not in ('catalogue', 'countries'))}")


def check_fred(rows: list[dict]) -> None:
    print("\nFRED")
    print(f"  {'series':20s} {'rows':>7s} {'first':>11s} {'last':>11s}  description")
    for row in rows:
        if row["source"] != "FRED":
            continue
        lines = (ROOT / row["path"]).read_text(encoding="utf-8").strip().splitlines()
        header, body = lines[0], lines[1:]
        if "," not in header or not body:
            print(f"  {row['series']:20s} NOT A CSV")
            continue
        print(f"  {row['series']:20s} {len(body):>7d} {body[0].split(',')[0]:>11s} "
              f"{body[-1].split(',')[0]:>11s}  {row['description'][:50]}")


def check_excel(rows: list[dict]) -> None:
    for row in rows:
        if not row["path"].endswith(".xlsx"):
            continue
        print(f"\n{row['source']}: {Path(row['path']).name}  [{row['vintage']}]")
        workbook = openpyxl.load_workbook(ROOT / row["path"], read_only=True, data_only=True)
        for sheet in workbook.worksheets:
            n_rows = 0
            header = None
            for values in sheet.iter_rows(values_only=True):
                if any(v is not None for v in values):
                    n_rows += 1
                    if header is None:
                        header = [str(v) for v in values if v is not None]
            print(f"  sheet {sheet.title!r}: {n_rows:,} non-empty rows, {len(header or [])} header fields")
            if header:
                print(f"    first fields: {', '.join(h[:22] for h in header[:8])}")


def main() -> int:
    rows, problems = check_manifest()
    print(f"manifest: {len(rows)} files listed")
    check_excel(rows)
    check_wdi(rows)
    check_gdd(rows)
    check_fred(rows)
    print()
    if problems:
        print("PROBLEMS")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("all files present and unchanged since download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
