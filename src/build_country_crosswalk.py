"""Build the country crosswalk: one row per country, with the code each source uses for it.

Usage (from the project root):
    .venv/bin/python -I src/build_country_crosswalk.py

Why this table exists
    The sources identify countries differently. WDI, WEO and the Global Debt Database use
    three-letter codes that mostly agree; the Laeven-Valencia crisis database uses country
    names, spelled differently on different sheets. Every later merge goes through this table,
    so that a mismatch is caught here, once, instead of silently dropping rows in a join.

How names are matched
    1. Names are normalised: accents removed, everything except letters dropped, lower case.
    2. A Laeven-Valencia name that equals a World Bank, WEO or DataMapper name is matched
       automatically.
    3. Anything else must be listed in data/reference/country_aliases.csv, which is written
       by hand. If a name is in neither place, the script stops with an error.

The standard key is the World Bank's ISO3 code. Taiwan is added from the IMF sources
because the World Bank does not publish data for it.

Output: data/reference/country_crosswalk.csv
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
import tomllib
import unicodedata
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REFERENCE = ROOT / "data" / "reference"
VINTAGES = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))["vintages"]


def norm(name: str | None) -> str:
    """Comparison key for a country name: no accents, letters only, lower case."""
    ascii_name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", ascii_name.lower())


def strip_footnote(name: str) -> str:
    """'Argentina 8/' -> 'Argentina' (the outcomes sheet marks footnotes this way)."""
    return re.sub(r"\s*\d+/.*$", "", name.strip())


# --------------------------------------------------------------------------- read sources
def read_wdi() -> dict[str, dict]:
    path = RAW / "wdi" / VINTAGES["wdi"] / "countries.json.gz"
    entries = json.loads(gzip.decompress(path.read_bytes()))[1]
    return {e["id"]: {"name": e["name"], "region": e["region"]["value"],
                      "income_group": e["incomeLevel"]["value"]}
            for e in entries if e["region"]["value"] != "Aggregates"}


def read_weo() -> dict[str, str]:
    workbook = openpyxl.load_workbook(RAW / "weo" / VINTAGES["weo"], read_only=True, data_only=True)
    rows = workbook["Countries"].iter_rows(values_only=True)
    header = list(next(rows))
    code_col, name_col = header.index("COUNTRY.ID"), header.index("COUNTRY")
    countries: dict[str, str] = {}
    for row in rows:
        if row[code_col]:
            countries.setdefault(row[code_col], row[name_col])
    return countries


def read_gdd() -> tuple[set[str], dict[str, str]]:
    folder = RAW / "gdd" / VINTAGES["gdd"]
    labels = {code: info.get("label") for code, info in
              json.loads((folder / "datamapper_countries.json").read_bytes())["countries"].items()}
    codes: set[str] = set()
    for path in folder.glob("*.json"):
        if not path.name.startswith("datamapper"):
            codes |= set(json.loads(path.read_bytes())["values"][path.stem])
    return codes, labels


def read_lv() -> tuple[list[str], list[str]]:
    """Country names on the two sheets we use: the full country list, and the episode list."""
    workbook = openpyxl.load_workbook(RAW / "laeven_valencia" / VINTAGES["lv"],
                                      read_only=True, data_only=True)
    years = [str(r[0]).strip() for r in workbook["Crisis Years"].iter_rows(values_only=True) if r[0]]
    outcomes = [strip_footnote(str(r[0])) for r in
                workbook["Crisis Resolution and Outcomes"].iter_rows(values_only=True)
                if r[0] and len(str(r[0])) < 60]          # long cells are the footnote text
    return years[1:], sorted(set(outcomes) - {"Country"})


def read_aliases() -> dict[tuple[str, str], str]:
    with (REFERENCE / "country_aliases.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    aliases = {}
    for row in rows:
        key = row["key"] if row["source"] in ("WEO", "GDD") else norm(row["key"])
        aliases[(row["source"], key)] = row["iso3"]
    return aliases


# --------------------------------------------------------------------------- build
def main() -> int:
    wdi, weo, (gdd_codes, dm_labels) = read_wdi(), read_weo(), read_gdd()
    lv_years, lv_outcomes = read_lv()
    aliases = read_aliases()
    problems: list[str] = []

    # Countries in the table: every World Bank country, plus IMF-only economies.
    table: dict[str, dict] = {}
    for iso3, info in wdi.items():
        table[iso3] = {"iso3": iso3, "country": info["name"], "region": info["region"],
                       "income_group": info["income_group"], "in_wdi": 1}

    def to_iso3(source: str, code: str) -> str:
        return aliases.get((source, code), code)

    for code, name in weo.items():
        iso3 = to_iso3("WEO", code)
        row = table.setdefault(iso3, {"iso3": iso3, "country": name, "region": "",
                                      "income_group": "", "in_wdi": 0})
        row["weo_code"], row["in_weo"] = code, 1
    for code in sorted(gdd_codes):
        iso3 = to_iso3("GDD", code)
        if iso3 not in table:
            problems.append(f"GDD code {code} ({dm_labels.get(code)}) has no country row")
            continue
        table[iso3]["gdd_code"], table[iso3]["in_gdd"] = code, 1

    # Laeven-Valencia names -> ISO3
    by_name: dict[str, str] = {}
    for source in (wdi, {k: {"name": v} for k, v in weo.items()},
                   {k: {"name": v} for k, v in dm_labels.items()}):
        for code, info in source.items():
            if info["name"]:
                by_name.setdefault(norm(info["name"]), code)

    def lv_to_iso3(name: str) -> str | None:
        key = norm(name)
        if ("LV", key) in aliases:
            return aliases[("LV", key)] or None       # blank iso3 = deliberately unmapped
        if key in by_name:
            return to_iso3("WEO", by_name[key])
        problems.append(f"Laeven-Valencia name not matched: {name!r}")
        return None

    unmapped = []
    for name in lv_years:
        iso3 = lv_to_iso3(name)
        if iso3 is None:
            unmapped.append(name)
        elif iso3 not in table:
            problems.append(f"{name!r} maps to {iso3}, which has no country row")
        elif table[iso3].get("lv_name"):
            problems.append(f"two Laeven-Valencia countries map to {iso3}: "
                            f"{table[iso3]['lv_name']!r} and {name!r}")
        else:
            table[iso3]["lv_name"], table[iso3]["in_lv"] = name, 1
    for name in lv_outcomes:                          # every spelling must resolve
        iso3 = lv_to_iso3(name)
        if iso3 and not table.get(iso3, {}).get("in_lv"):
            problems.append(f"outcomes-sheet country {name!r} ({iso3}) is not on the country list")

    # Notes for rows whose codes differ between sources
    for row in table.values():
        codes = {c for c in (row.get("weo_code"), row.get("gdd_code")) if c and c != row["iso3"]}
        notes = []
        if codes:
            notes.append(f"other codes in use: {', '.join(sorted(codes))}")
        if not row["in_wdi"]:
            notes.append("not in World Bank data")
        row["note"] = "; ".join(notes)

    if problems:
        print("PROBLEMS - nothing written")
        for problem in problems:
            print(f"  {problem}")
        return 1

    fields = ["iso3", "country", "region", "income_group", "in_wdi", "in_weo", "weo_code",
              "in_gdd", "gdd_code", "in_lv", "lv_name", "note"]
    out = REFERENCE / "country_crosswalk.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for iso3 in sorted(table):
            row = table[iso3]
            writer.writerow({f: row.get(f, 0 if f.startswith("in_") else "") for f in fields})

    rows = list(table.values())
    count = lambda flag: sum(1 for r in rows if r.get(flag))
    print(f"wrote {out.relative_to(ROOT).as_posix()}: {len(rows)} countries")
    print(f"  in WDI {count('in_wdi')} | in WEO {count('in_weo')} | in GDD {count('in_gdd')} "
          f"| in Laeven-Valencia {count('in_lv')}")
    print(f"  in all four sources: {sum(1 for r in rows if all(r.get(f) for f in ('in_wdi', 'in_weo', 'in_gdd', 'in_lv')))}")
    print(f"  Laeven-Valencia names left unmapped on purpose: {unmapped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
