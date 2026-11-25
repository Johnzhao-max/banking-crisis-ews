"""Read Table A1 of Laeven and Valencia (2026) out of the PDF and save it as a CSV.

Usage (from the project root):
    .venv/bin/python -I src/extract_lv2026_episodes.py

Why: the 2026 edition of the Systemic Banking Crises Database (coverage 1970-2025) was published
as a paper only, without a data file. Table A1 in its appendix lists every episode with its start
and end year. This script turns that table into data/reference/lv2026_episodes.csv and then
checks it against the 2020 edition, which does have an Excel file.

Reading the footnote markers
    A marker after the END year ("1994 5/") means the duration was truncated at five years.
    A marker after the COUNTRY name ("Argentina 5/ 1995") flags the episode itself. The table's
    own footnote list numbers these inconsistently, so the script does not trust the number:
    it records the raw marker, and the comparison with the 2020 edition (where borderline cases
    are marked 8/) shows what the marker means. Result: every episode marked "5/" after the
    country name is a borderline case in the 2020 file, and the three borderline cases the 2026
    text names (Nicaragua 2018, Vietnam 2022, Sri Lanka 2023) carry the same marker. So
    borderline = (country marker is "5/").

Start years: Table 1 in the text and Table A1 in the appendix disagree for some new episodes.
    The CSV keeps Table A1's year as start_year and records Table 1's year next to it.

The output is committed to Git: it is small, and it is the hand-checkable record of the labels.
"""

from __future__ import annotations

import csv
import re
import sys
import tomllib
import unicodedata
from pathlib import Path

import openpyxl
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "laeven_valencia"
REFERENCE = ROOT / "data" / "reference"
PDF = RAW / "lv2026_wp2026094.pdf"
VINTAGES = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))["vintages"]

YEAR = r"(?:19|20)\d{2}"
ROW = re.compile(
    rf"^(?P<country>[^\d]+?)\s*(?P<cmark>\d/)?\s+(?P<start>{YEAR})\s+"
    rf"(?P<end>{YEAR}|…|\.\.\.)\s*(?P<emark>\d/)?(?:\s|$)")


def norm(name: str | None) -> str:
    ascii_name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", ascii_name.lower())


def name_lookup() -> dict[str, str]:
    """Normalised country name -> ISO3, from the crosswalk and the alias table."""
    lookup: dict[str, str] = {}
    with (REFERENCE / "country_crosswalk.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for name in (row["country"], row["lv_name"]):
                if name:
                    lookup.setdefault(norm(name), row["iso3"])
    with (REFERENCE / "country_aliases.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source"] == "LV" and row["iso3"]:
                lookup[norm(row["key"])] = row["iso3"]
    return lookup


def table_lines() -> list[tuple[int, str]]:
    """(page number, line) for every line of Table A1."""
    reader = PdfReader(PDF)
    lines, inside = [], False
    for number, page in enumerate(reader.pages, start=1):
        for line in page.extract_text().splitlines():
            line = line.strip()
            if line.startswith("Table A1."):
                inside = True
                continue
            if line.startswith("Table A2.") or line.startswith("1/ Output losses"):
                return lines
            if inside and line and not line.startswith(("IMF WORKING PAPERS", "INTERNATIONAL MONETARY",
                                                        "Country Start End")):
                lines.append((number, line))
    return lines


def parse() -> list[dict]:
    episodes, pending = [], ""
    for page, line in table_lines():
        text = f"{pending} {line}".strip()
        match = ROW.match(text)
        if not match:                      # first half of a country name that wraps over two lines
            pending = text
            continue
        pending = ""
        end = match["end"]
        episodes.append({
            "country_lv": match["country"].strip(),
            "start_year": int(match["start"]),
            "end_year": int(end) if end.isdigit() else "",
            "end_truncated_at_5y": 1 if match["emark"] else 0,
            "country_marker": match["cmark"] or "",
            "pdf_page": page,
        })
    if pending:
        raise RuntimeError(f"unparsed text left over: {pending!r}")
    return episodes


def read_table1() -> dict[str, int]:
    """Start years of the new episodes as listed in Table 1 of the text: country name -> year."""
    starts, inside = {}, False
    for page in PdfReader(PDF).pages:
        for line in page.extract_text().splitlines():
            line = line.strip()
            if line.startswith("Table 1. New episodes"):
                inside = True
            elif inside and line.startswith("* Denotes"):
                return starts
            elif inside and (match := re.fullmatch(rf"(.+?)\*?\s+({YEAR})", line)):
                starts[match[1].strip()] = int(match[2])
    return starts


def read_lv2020() -> dict[tuple[str, int], dict]:
    """Episodes in the 2020 Excel file, keyed by (country name as printed, start year)."""
    workbook = openpyxl.load_workbook(RAW / VINTAGES["lv"], read_only=True, data_only=True)
    episodes = {}
    for row in workbook["Crisis Resolution and Outcomes"].iter_rows(values_only=True):
        name, start, end = (str(v).strip() if v is not None else "" for v in row[:3])
        if not re.fullmatch(YEAR, start[:4]) or len(name) > 60:
            continue
        marker = re.search(r"(\d)/", name)
        episodes[(re.sub(r"\s*\d+/.*$", "", name), int(start[:4]))] = {
            "end_year": int(end[:4]) if re.match(YEAR, end) else "",
            "borderline": 1 if marker and marker[1] == "8" else 0,
        }
    return episodes


def main() -> int:
    lookup = name_lookup()
    episodes = parse()
    unmatched = sorted({e["country_lv"] for e in episodes if norm(e["country_lv"]) not in lookup})
    if unmatched:
        print("country names not in the crosswalk or alias table - add them to country_aliases.csv:")
        for name in unmatched:
            print(f"  {name!r}")
        return 1
    table1 = {lookup[norm(name)]: year for name, year in read_table1().items()}
    for episode in episodes:
        episode["iso3"] = lookup[norm(episode["country_lv"])]
        episode["borderline"] = 1 if episode["country_marker"] == "5/" else 0

    # ---- compare with the 2020 edition
    old = {(lookup.get(norm(name), name), start): info for (name, start), info in read_lv2020().items()}
    new = {(e["iso3"], e["start_year"]): e for e in episodes}
    for key, episode in new.items():
        episode["in_lv2020"] = 1 if key in old else 0
        episode["borderline_lv2020"] = old[key]["borderline"] if key in old else ""
        episode["start_year_table1"] = "" if key in old else table1.get(key[0], "")
    only_new = sorted(set(new) - set(old))
    only_old = sorted(set(old) - set(new))
    marker_vs_old = {}
    for key in set(new) & set(old):
        pair = (new[key]["country_marker"] or "none", old[key]["borderline"])
        marker_vs_old[pair] = marker_vs_old.get(pair, 0) + 1
    end_changed = sorted((k, old[k]["end_year"], new[k]["end_year"]) for k in set(new) & set(old)
                         if old[k]["end_year"] != new[k]["end_year"])

    fields = ["iso3", "country_lv", "start_year", "start_year_table1", "end_year",
              "end_truncated_at_5y", "borderline", "country_marker", "in_lv2020",
              "borderline_lv2020", "pdf_page"]
    out = REFERENCE / "lv2026_episodes.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in sorted(episodes, key=lambda e: (e["iso3"], e["start_year"])):
            writer.writerow({f: episode[f] for f in fields})

    print(f"wrote {out.relative_to(ROOT).as_posix()}: {len(episodes)} episodes, "
          f"{len({e['iso3'] for e in episodes})} countries (the paper reports 164 episodes)")
    print(f"2020 edition: {len(old)} episodes; in both editions: {len(set(new) & set(old))}")
    print(f"\nin 2026 only ({len(only_new)}):")
    for key in only_new:
        e = new[key]
        flag = "" if e["start_year_table1"] == e["start_year"] else f"   <-- Table 1 says {e['start_year_table1']}"
        print(f"  {key[0]} {e['country_lv']:26s} {e['start_year']}-{e['end_year'] or '...':<5} "
              f"borderline={e['borderline']}{flag}")
    print(f"\nin 2020 only ({len(only_old)}): {only_old}")
    print("\ncountry marker in 2026 table vs borderline flag in 2020 file (episodes in both):")
    for (marker, borderline), count in sorted(marker_vs_old.items()):
        print(f"  marker {marker:5s} borderline_2020={borderline}: {count}")
    print(f"\nend year differs between editions ({len(end_changed)}):")
    for key, old_end, new_end in end_changed:
        print(f"  {key[0]} {key[1]}: 2020 file {old_end or '...'} -> 2026 table {new_end or '...'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
