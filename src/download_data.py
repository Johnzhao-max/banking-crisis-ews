"""Download every raw input for the banking-crisis early-warning project.

Usage (from the project root):
    .venv/bin/python src/download_data.py all        # everything
    .venv/bin/python src/download_data.py wdi fred   # only some sources

Rules this script follows:
  * Raw files are never modified or overwritten. A file that already exists is skipped.
  * Sources that are revised over time (WDI, GDD, FRED) go into a folder named after the
    retrieval date, so a later download sits next to the earlier one instead of replacing it.
  * Every file is logged in data/raw/MANIFEST.csv with its URL, vintage, retrieval time,
    size, and SHA-256 checksum.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "MANIFEST.csv"
MANIFEST_FIELDS = ["path", "source", "series", "description", "vintage", "url",
                   "retrieved_utc", "bytes", "sha256"]
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# --------------------------------------------------------------------------- series lists
# World Bank World Development Indicators: candidate predictors and scaling variables.
# This is deliberately a generous list; which ones enter the model is decided at the cleaning stage.
WDI_INDICATORS = [
    # real economy
    "NY.GDP.MKTP.KD.ZG",     # GDP growth (annual %)
    "NY.GDP.PCAP.KD",        # GDP per capita (constant US$)
    "NY.GDP.MKTP.CD",        # GDP (current US$)
    "NY.GDP.DEFL.KD.ZG",     # inflation, GDP deflator
    "FP.CPI.TOTL.ZG",        # inflation, consumer prices
    "NE.GDI.TOTL.ZS",        # gross capital formation (% of GDP)
    "NY.GNS.ICTR.ZS",        # gross savings (% of GDP)
    "SP.POP.TOTL",           # population
    # credit, money, banking
    "FS.AST.PRVT.GD.ZS",     # domestic credit to private sector (% of GDP)
    "FD.AST.PRVT.GD.ZS",     # domestic credit to private sector by banks (% of GDP)
    "FM.LBL.BMNY.GD.ZS",     # broad money (% of GDP)
    "FM.LBL.BMNY.IR.ZS",     # broad money to total reserves ratio
    "FR.INR.RINR",           # real interest rate
    "FR.INR.LNDP",           # interest rate spread (lending minus deposit)
    "FB.AST.NPER.ZS",        # bank nonperforming loans (% of gross loans)
    "FB.BNK.CAPA.ZS",        # bank capital to assets ratio
    # external sector
    "BN.CAB.XOKA.GD.ZS",     # current account balance (% of GDP)
    "NE.TRD.GNFS.ZS",        # trade (% of GDP)
    "NE.EXP.GNFS.ZS",        # exports of goods and services (% of GDP)
    "BX.KLT.DINV.WD.GD.ZS",  # FDI net inflows (% of GDP)
    "FI.RES.TOTL.MO",        # total reserves in months of imports
    "FI.RES.TOTL.CD",        # total reserves incl. gold (current US$)
    "DT.DOD.DECT.GN.ZS",     # external debt stocks (% of GNI)
    "DT.DOD.DSTC.IR.ZS",     # short-term debt (% of total reserves)
    "DT.DOD.DSTC.ZS",        # short-term debt (% of total external debt)
    "DT.TDS.DECT.EX.ZS",     # total debt service (% of exports)
    "PA.NUS.FCRF",           # official exchange rate (LCU per US$)
    "PX.REX.REER",           # real effective exchange rate index
    # fiscal
    "GC.DOD.TOTL.GD.ZS",     # central government debt (% of GDP)
]
WDI_FIRST_YEAR, WDI_LAST_YEAR = 1970, 2025

# IMF Global Debt Database, served through the IMF DataMapper API.
GDD_INDICATORS = ["Privatedebt_all", "HH_ALL", "NFC_ALL", "PVD_LS", "HH_LS", "NFC_LS",
                  "PS_DEBT_GDP", "NFPS_DEBT_GDP", "GG_DEBT_GDP", "CG_DEBT_GDP"]

# FRED: global financial conditions (all U.S. or world-market series).
FRED_SERIES = {
    "FEDFUNDS": "Effective federal funds rate, monthly",
    "TB3MS": "3-month U.S. Treasury bill rate, monthly",
    "GS10": "10-year U.S. Treasury yield, monthly",
    "REAINTRATREARAT10Y": "10-year U.S. real interest rate (Cleveland Fed), monthly",
    "CPIAUCSL": "U.S. consumer price index, monthly, seasonally adjusted",
    "BAA10YM": "Moody's Baa corporate yield minus 10-year Treasury, monthly",
    "VIXCLS": "CBOE volatility index (VIX), daily",
    "WTISPLC": "Spot crude oil price, West Texas Intermediate, monthly",
}

# Financial-sector variables added for the IMF-aligned variable set (version 2).
# (World Bank source id, indicator code). Source 32 is the Global Financial Development Database,
# which stopped being updated in September 2022, so its series end in 2021.
WB_FINANCIAL = [
    (32, "GFDD.OI.09"),   # loans from nonresident banks, amounts outstanding (% of GDP)
    (32, "GFDD.SI.04"),   # bank credit to bank deposits (%)
    (32, "GFDD.SI.05"),   # bank regulatory capital to risk-weighted assets (%)
    (32, "GFDD.SI.01"),   # bank Z-score (distance to default)
    (2, "FR.INR.DPST"),   # deposit interest rate (%), from WDI
]
# IMF Financial Development Index, read from the DBnomics mirror because the IMF portal
# requires a login for bulk downloads.
FDI_INDICATORS = {"FD_FIA_IX": "Financial Institutions Access Index",
                  "FD_FIE_IX": "Financial Institutions Efficiency Index"}

# Single files with a fixed edition in their name.
WEO_URL = ("https://data.imf.org/-/media/iData/External-Storage/Documents/"
           "0FA3DC49A04D446E9D69DBEB29979E1C/en/WEOOct2026all.xlsx")
LV_FILES = [
    {
        "name": "lv2020_imfer_database.xlsx",
        "url": ("https://static-content.springer.com/esm/art%3A10.1057%2Fs41308-020-00107-3/"
                "MediaObjects/41308_2020_107_MOESM1_ESM.xlsx"),
        "series": "LV2020",
        "description": ("Laeven and Valencia (2020), Systemic Banking Crises Database II, "
                        "IMF Economic Review: data supplement, crises 1970-2017"),
        "vintage": "2020 edition (coverage 1970-2017)",
    },
    {
        "name": "lv2026_wp2026094.pdf",
        "url": "https://www.imf.org/-/media/files/publications/wp/2026/english/wpiea2026094-source-pdf.pdf",
        "series": "LV2026",
        "description": ("Laeven and Valencia (2026), Systemic Banking Crises Database: 1970-2025, "
                        "IMF WP 2026/094: paper only, episode list is in Appendix Table A1"),
        "vintage": "2026 edition (coverage 1970-2025)",
    },
]


# --------------------------------------------------------------------------- helpers
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http_get(url: str, params: dict | None = None, tries: int = 4) -> requests.Response:
    """GET with simple retries; raises if the last attempt fails."""
    last_error: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            response = requests.get(url, params=params, timeout=120)
            if response.status_code == 200:
                return response
            last_error = RuntimeError(f"HTTP {response.status_code} for {response.url}")
        except requests.RequestException as error:
            last_error = error
        time.sleep(2 * attempt)
    raise RuntimeError(f"giving up on {url}: {last_error}")


def load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return {row["path"]: row for row in csv.DictReader(handle)}


def write_manifest(rows: dict[str, dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for path in sorted(rows):
            writer.writerow(rows[path])


def store(manifest: dict[str, dict], dest: Path, content: bytes, *, source: str, series: str,
          description: str, vintage: str, url: str) -> bool:
    """Write `content` to `dest` unless the file exists, and record it. Returns True if written."""
    rel = dest.relative_to(ROOT).as_posix()
    if dest.exists():
        print(f"  exists, skipped   {rel}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    manifest[rel] = {
        "path": rel, "source": source, "series": series, "description": description,
        "vintage": vintage, "url": url, "retrieved_utc": utc_now(),
        "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
    }
    write_manifest(manifest)
    print(f"  saved {len(content):>10,} B  {rel}")
    return True


# --------------------------------------------------------------------------- sources
def download_lv(manifest: dict[str, dict]) -> None:
    print("Laeven-Valencia systemic banking crises")
    folder = RAW / "laeven_valencia"
    for item in LV_FILES:
        dest = folder / item["name"]
        if dest.exists():
            print(f"  exists, skipped   {dest.relative_to(ROOT).as_posix()}")
            continue
        content = http_get(item["url"]).content
        store(manifest, dest, content, source="Laeven-Valencia", series=item["series"],
              description=item["description"], vintage=item["vintage"], url=item["url"])


def download_wdi(manifest: dict[str, dict]) -> None:
    print("World Bank World Development Indicators")
    folder = RAW / "wdi" / f"retrieved_{TODAY}"
    api = "https://api.worldbank.org/v2"

    dest = folder / "countries.json.gz"
    if dest.exists():
        print(f"  exists, skipped   {dest.relative_to(ROOT).as_posix()}")
    else:
        response = http_get(f"{api}/country", {"format": "json", "per_page": 400})
        payload = response.json()
        if len(payload) < 2 or payload[0].get("pages", 1) != 1:
            raise RuntimeError("unexpected country list response")
        store(manifest, dest, gzip.compress(response.content), source="World Bank WDI",
              series="countries",
              description="Country and aggregate list with ISO codes, region, income group",
              vintage=f"retrieved {TODAY}", url=response.url)

    for code in WDI_INDICATORS:
        dest = folder / f"{code}.json.gz"
        if dest.exists():
            print(f"  exists, skipped   {dest.relative_to(ROOT).as_posix()}")
            continue
        response = http_get(f"{api}/country/all/indicator/{code}",
                            {"format": "json", "per_page": 20000, "source": 2,
                             "date": f"{WDI_FIRST_YEAR}:{WDI_LAST_YEAR}"})
        payload = response.json()
        if len(payload) < 2 or not payload[1]:
            print(f"  NO DATA           {code}: {json.dumps(payload)[:160]}")
            continue
        meta, rows = payload
        if meta.get("pages", 1) != 1:
            raise RuntimeError(f"{code}: response was paginated ({meta.get('pages')} pages)")
        name = rows[0]["indicator"]["value"]
        store(manifest, dest, gzip.compress(response.content), source="World Bank WDI",
              series=code, description=name,
              vintage=f"WDI last updated {meta.get('lastupdated', 'unknown')}", url=response.url)
        time.sleep(0.5)


def download_weo(manifest: dict[str, dict]) -> None:
    print("IMF World Economic Outlook")
    dest = RAW / "weo" / "WEOOct2026all.xlsx"
    if dest.exists():
        print(f"  exists, skipped   {dest.relative_to(ROOT).as_posix()}")
        return
    content = http_get(WEO_URL).content
    store(manifest, dest, content, source="IMF WEO", series="all",
          description="World Economic Outlook database, full download (countries, groups, commodity prices)",
          vintage="October 2026 edition", url=WEO_URL)


def download_gdd(manifest: dict[str, dict]) -> None:
    print("IMF Global Debt Database (via DataMapper API)")
    folder = RAW / "gdd" / f"retrieved_{TODAY}"
    api = "https://www.imf.org/external/datamapper/api/v1"

    catalogue = http_get(f"{api}/indicators")
    indicators = catalogue.json()["indicators"]
    store(manifest, folder / "datamapper_indicators.json", catalogue.content,
          source="IMF GDD", series="catalogue",
          description="DataMapper indicator catalogue: labels, units, and source edition",
          vintage=f"retrieved {TODAY}", url=catalogue.url)
    store_if_missing = folder / "datamapper_countries.json"
    if not store_if_missing.exists():
        countries = http_get(f"{api}/countries")
        store(manifest, store_if_missing, countries.content, source="IMF GDD", series="countries",
              description="DataMapper country codes and names", vintage=f"retrieved {TODAY}",
              url=countries.url)

    for code in GDD_INDICATORS:
        dest = folder / f"{code}.json"
        if dest.exists():
            print(f"  exists, skipped   {dest.relative_to(ROOT).as_posix()}")
            continue
        info = indicators.get(code, {})
        response = http_get(f"{api}/{code}")
        if not response.json().get("values", {}).get(code):
            print(f"  NO DATA           {code}")
            continue
        store(manifest, dest, response.content, source="IMF GDD", series=code,
              description=f"{info.get('label', code)} ({info.get('unit', '')})",
              vintage=info.get("source", "unknown"), url=response.url)
        time.sleep(0.5)


def download_fred(manifest: dict[str, dict]) -> None:
    print("FRED global financial conditions")
    folder = RAW / "fred" / f"retrieved_{TODAY}"
    for code, description in FRED_SERIES.items():
        dest = folder / f"{code}.csv"
        if dest.exists():
            print(f"  exists, skipped   {dest.relative_to(ROOT).as_posix()}")
            continue
        response = http_get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": code})
        if response.content.count(b"\n") < 10:
            print(f"  NO DATA           {code}")
            continue
        store(manifest, dest, response.content, source="FRED", series=code,
              description=description, vintage=f"retrieved {TODAY}", url=response.url)
        time.sleep(0.5)


def download_wb_financial(manifest: dict[str, dict]) -> None:
    print("World Bank financial-sector indicators (GFDD and WDI)")
    folder = RAW / "wb_financial" / f"retrieved_{TODAY}"
    for source_id, code in WB_FINANCIAL:
        dest = folder / f"{code}.json.gz"
        if dest.exists():
            print(f"  exists, skipped   {dest.relative_to(ROOT).as_posix()}")
            continue
        response = http_get(f"https://api.worldbank.org/v2/country/all/indicator/{code}",
                            {"format": "json", "per_page": 20000, "source": source_id,
                             "date": f"{WDI_FIRST_YEAR}:{WDI_LAST_YEAR}"})
        payload = response.json()
        if len(payload) < 2 or not payload[1]:
            print(f"  NO DATA           {code}: {json.dumps(payload)[:160]}")
            continue
        meta, rows = payload
        if meta.get("pages", 1) != 1:
            raise RuntimeError(f"{code}: response was paginated ({meta.get('pages')} pages)")
        label = "World Bank GFDD" if source_id == 32 else "World Bank WDI"
        store(manifest, dest, gzip.compress(response.content), source=label, series=code,
              description=rows[0]["indicator"]["value"],
              vintage=f"{label.split()[-1]} last updated {meta.get('lastupdated', 'unknown')}",
              url=response.url)
        time.sleep(0.5)


def download_fdi(manifest: dict[str, dict]) -> None:
    print("IMF Financial Development Index (via DBnomics)")
    folder = RAW / "imf_fdi" / f"retrieved_{TODAY}"
    api = "https://api.db.nomics.world/v22"
    indexed = http_get(f"{api}/datasets/IMF/FDI").json()["datasets"]["docs"][0].get("indexed_at", "unknown")
    for code, label in FDI_INDICATORS.items():
        dest = folder / f"{code}.json"
        if dest.exists():
            print(f"  exists, skipped   {dest.relative_to(ROOT).as_posix()}")
            continue
        response = http_get(f"{api}/series/IMF/FDI",
                            {"dimensions": json.dumps({"FREQ": ["A"], "INDICATOR": [code]}),
                             "observations": 1, "limit": 1000})
        series = response.json()["series"]
        if series["num_found"] > len(series["docs"]):
            raise RuntimeError(f"{code}: {series['num_found']} series found but only {len(series['docs'])} returned")
        store(manifest, dest, response.content, source="IMF FDI", series=code, description=label,
              vintage=f"IMF FDI as mirrored by DBnomics on {indexed[:10]}", url=response.url)
        time.sleep(0.5)


SOURCES = {"lv": download_lv, "wdi": download_wdi, "weo": download_weo,
           "gdd": download_gdd, "fred": download_fred,
           "wbfin": download_wb_financial, "fdi": download_fdi}


def main(argv: list[str]) -> int:
    wanted = argv or ["all"]
    if "all" in wanted:
        wanted = list(SOURCES)
    unknown = [name for name in wanted if name not in SOURCES]
    if unknown:
        print(f"unknown source(s): {unknown}; choose from {list(SOURCES)} or 'all'")
        return 2
    manifest = load_manifest()
    failed = []
    for name in wanted:
        try:
            SOURCES[name](manifest)
        except Exception as error:  # keep going so one source cannot block the rest
            failed.append(name)
            print(f"  FAILED            {name}: {error}")
    print(f"\nmanifest: {MANIFEST.relative_to(ROOT).as_posix()} ({len(manifest)} files)")
    if failed:
        print(f"sources with errors: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
