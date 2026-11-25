"""Task 10: turn the raw downloads into one tidy "long" table.

Usage (from the project root):
    .venv/bin/python src/build_long_table.py

What a long table is
    One row per observation: which country, which year, which series, what value.

        iso3  year  series_id              value   source  vintage              status
        ARG   2001  WDI.FS.AST.PRVT.GD.ZS  20.8    WDI     WDI 2026-11-11       actual
        ARG   2001  WEO.GGXWDG_NGDP        48.0    WEO     WEO October 2026     actual

    The three columns (iso3, year, series_id) together are the PRIMARY KEY: no two rows may
    share them. Adding a new series means adding rows, never columns, so the table's shape
    does not change when the list of variables does. The wide table the model needs
    (one column per variable) is built from this one later.

How the script is organised
    The four `read_*` functions parse one raw format each and return the same columns.
    `annualise` turns the monthly and daily FRED series into annual values.
    `stack` combines the sources and enforces the primary key.
    The `check_*` functions then test the result and print PASS or FAIL for each property.

Outputs
    data/clean/country_long.parquet   country-year observations
    data/clean/global_long.parquet    world-wide series by year (interest rates, VIX, oil)
    data/clean/series_summary.csv     one line per series: coverage, first and last year
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
REFERENCE = ROOT / "data" / "reference"
VINTAGES = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))["vintages"]

KEY = ["iso3", "year", "series_id"]
COUNTRY_COLUMNS = KEY + ["value", "source", "vintage", "status"]
GLOBAL_COLUMNS = ["year", "series_id", "value", "source", "vintage"]


# =========================================================================== transformations
def annualise(fred: pd.DataFrame) -> pd.DataFrame:
    """Turn the FRED series, which are monthly or daily, into one value per year.

    Input columns:  date (datetime), series_id, value (may be NaN), source, vintage
    Output columns: year (int), series_id, value, source, vintage

    Two decisions are made here:
      1. A year is summarised by its AVERAGE, not its last value. The model predicts from
         annual data, and the average describes conditions over the whole year instead of on
         one day in December. (Example: the VIX averaged 32.7 in 2008 but ended the year at 40.)
      2. The current year is DROPPED because it is incomplete: an average of ten months is not
         comparable with an average of twelve.
    """
    monthly = fred.assign(year=fred["date"].dt.year)
    monthly = monthly[monthly["year"] < pd.Timestamp.now().year]
    annual = monthly.groupby(["year", "series_id", "source", "vintage"], as_index=False)["value"].mean()
    annual = annual.dropna(subset=["value"])          # a year with no observations at all
    return annual.sort_values(["series_id", "year"]).reset_index(drop=True)[GLOBAL_COLUMNS]


def stack(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine the per-source tables into one long table and enforce the primary key.

    1. Append the rows of all sources.
    2. Drop rows whose value is missing: in a long table, "no data" means "no row".
    3. Refuse to continue if two rows share (iso3, year, series_id). Duplicates are never
       dropped quietly, because a duplicate means something upstream is wrong (for example,
       two source codes mapped to the same country).
    4. Sort by the key.
    """
    table = pd.concat(frames, ignore_index=True)
    table = table.dropna(subset=["value"])
    twins = table.duplicated(subset=KEY, keep=False)
    if twins.any():
        raise ValueError(f"{int(twins.sum())} rows share a primary key, for example:\n"
                         f"{table[twins].sort_values(KEY).head(10).to_string(index=False)}")
    return table.sort_values(KEY).reset_index(drop=True)[COUNTRY_COLUMNS]


# =========================================================================== readers
def load_code_maps() -> tuple[set[str], dict[tuple[str, str], str]]:
    """ISO3 codes in the crosswalk, and the source-specific codes that differ from them."""
    with (REFERENCE / "country_crosswalk.csv").open(newline="", encoding="utf-8") as handle:
        iso3 = {row["iso3"] for row in csv.DictReader(handle)}
    with (REFERENCE / "country_aliases.csv").open(newline="", encoding="utf-8") as handle:
        overrides = {(row["source"], row["key"]): row["iso3"] for row in csv.DictReader(handle)
                     if row["source"] in ("WEO", "GDD")}
    return iso3, overrides


def manifest_vintages() -> dict[str, str]:
    """Raw file path -> vintage string recorded when it was downloaded."""
    with (RAW / "MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        return {row["path"]: row["vintage"] for row in csv.DictReader(handle)}


def read_wdi(iso3_codes: set[str], vintages: dict[str, str]) -> pd.DataFrame:
    """World Bank WDI: one gzipped JSON file per indicator."""
    folder = RAW / "wdi" / VINTAGES["wdi"]
    records = []
    for path in sorted(folder.glob("*.json.gz")):
        if path.name == "countries.json.gz":
            continue
        code = path.name.removesuffix(".json.gz")
        vintage = vintages[path.relative_to(ROOT).as_posix()]
        for item in json.loads(gzip.decompress(path.read_bytes()))[1]:
            if item["countryiso3code"] in iso3_codes:       # skips aggregates such as "World"
                records.append((item["countryiso3code"], int(item["date"]), f"WDI.{code}",
                                item["value"], "WDI", vintage, "actual"))
    return pd.DataFrame.from_records(records, columns=COUNTRY_COLUMNS)


def read_weo(overrides: dict[tuple[str, str], str]) -> pd.DataFrame:
    """IMF WEO: one wide Excel sheet, a row per country-indicator and a column per year.

    The WEO contains projections. Each row says which year is its last actual observation
    (LATEST_ACTUAL_ANNUAL_DATA); later years are marked status = "projection" here so that
    they can be excluded later. Two details of that field, both handled conservatively:
      * Countries that report by fiscal year write it as "FY2025/26". The first year (2025)
        is used, which may treat one genuinely observed year as a projection, never the reverse.
      * About a quarter of the rows leave it blank (PPP series, output gaps, and others).
        They get the most common value among the same country's other series.
    """
    sheet = pd.read_excel(RAW / "weo" / VINTAGES["weo"], sheet_name="Countries", engine="openpyxl")
    sheet = sheet[sheet["COUNTRY.ID"].notna()].copy()
    stated = sheet["LATEST_ACTUAL_ANNUAL_DATA"].astype("string").str.extract(r"((?:19|20)\d{2})")[0]
    sheet["last_actual"] = pd.to_numeric(stated, errors="coerce")
    usual = sheet.dropna(subset=["last_actual"]).groupby("COUNTRY.ID")["last_actual"].agg(
        lambda years: years.mode().iloc[0])
    sheet["last_actual"] = sheet["last_actual"].fillna(sheet["COUNTRY.ID"].map(usual))
    year_columns = [c for c in sheet.columns if str(c).isdigit()]
    long = sheet.melt(id_vars=["COUNTRY.ID", "INDICATOR.ID", "last_actual"],
                      value_vars=year_columns, var_name="year", value_name="value")
    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")       # "n/a" -> NaN
    long["iso3"] = [overrides.get(("WEO", code), code) for code in long["COUNTRY.ID"]]
    long["series_id"] = "WEO." + long["INDICATOR.ID"].astype(str)
    long["status"] = "actual"
    long.loc[long["last_actual"].isna() | (long["year"] > long["last_actual"]), "status"] = "projection"
    long["source"], long["vintage"] = "WEO", "WEO October 2026"
    return long[COUNTRY_COLUMNS]


def read_gdd(overrides: dict[tuple[str, str], str], vintages: dict[str, str]) -> pd.DataFrame:
    """IMF Global Debt Database: one JSON file per indicator, {country: {year: value}}."""
    folder = RAW / "gdd" / VINTAGES["gdd"]
    records = []
    for path in sorted(folder.glob("*.json")):
        if path.name.startswith("datamapper"):
            continue
        vintage = vintages[path.relative_to(ROOT).as_posix()]
        for code, series in json.loads(path.read_bytes())["values"][path.stem].items():
            for year, value in series.items():
                records.append((overrides.get(("GDD", code), code), int(year), f"GDD.{path.stem}",
                                value, "GDD", vintage, "actual"))
    return pd.DataFrame.from_records(records, columns=COUNTRY_COLUMNS)


def read_fred(vintages: dict[str, str]) -> pd.DataFrame:
    """FRED: one CSV per series at its native frequency (monthly, or daily for the VIX)."""
    folder = RAW / "fred" / VINTAGES["fred"]
    frames = []
    for path in sorted(folder.glob("*.csv")):
        frame = pd.read_csv(path)
        frame.columns = ["date", "value"]
        frame["date"] = pd.to_datetime(frame["date"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")  # FRED writes "." for missing
        frame["series_id"] = f"FRED.{path.stem}"
        frame["source"] = "FRED"
        frame["vintage"] = vintages[path.relative_to(ROOT).as_posix()]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[["date", "series_id", "value", "source", "vintage"]]


def read_wb_financial(iso3_codes: set[str], vintages: dict[str, str]) -> pd.DataFrame:
    """World Bank financial-sector series: GFDD indicators and one extra WDI indicator."""
    folder = RAW / "wb_financial" / VINTAGES["wb_financial"]
    records = []
    for path in sorted(folder.glob("*.json.gz")):
        code = path.name.removesuffix(".json.gz")
        source = "GFDD" if code.startswith("GFDD.") else "WDI"
        series_id = code if source == "GFDD" else f"WDI.{code}"
        vintage = vintages[path.relative_to(ROOT).as_posix()]
        for item in json.loads(gzip.decompress(path.read_bytes()))[1]:
            if item["countryiso3code"] in iso3_codes:
                records.append((item["countryiso3code"], int(item["date"]), series_id,
                                item["value"], source, vintage, "actual"))
    return pd.DataFrame.from_records(records, columns=COUNTRY_COLUMNS)


def read_fdi(iso3_codes: set[str], vintages: dict[str, str]) -> pd.DataFrame:
    """IMF Financial Development Index (DBnomics JSON). Countries are coded with ISO2 there."""
    countries = json.loads(gzip.decompress(
        (RAW / "wdi" / VINTAGES["wdi"] / "countries.json.gz").read_bytes()))[1]
    iso2_to_iso3 = {c["iso2Code"]: c["id"] for c in countries if c["id"] in iso3_codes}
    folder = RAW / "imf_fdi" / VINTAGES["imf_fdi"]
    records = []
    for path in sorted(folder.glob("*.json")):
        vintage = vintages[path.relative_to(ROOT).as_posix()]
        for series in json.loads(path.read_bytes())["series"]["docs"]:
            iso3 = iso2_to_iso3.get(series["dimensions"]["REF_AREA"])      # skips country groups
            if iso3 is None:
                continue
            for period, value in zip(series["period"], series["value"]):
                records.append((iso3, int(period), f"FDI.{path.stem}",
                                pd.to_numeric(value, errors="coerce"), "FDI", vintage, "actual"))
    return pd.DataFrame.from_records(records, columns=COUNTRY_COLUMNS)


# =========================================================================== checks
def check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}{'  - ' + detail if detail and not passed else ''}")
    return passed


def check_country_long(table: pd.DataFrame, frames: list[pd.DataFrame], iso3_codes: set[str]) -> bool:
    expected_rows = sum(int(frame["value"].notna().sum()) for frame in frames)
    ok = check("has exactly the expected columns", list(table.columns) == COUNTRY_COLUMNS,
               f"got {list(table.columns)}")
    if not ok:
        return False
    results = [
        check("no missing values", bool(table["value"].notna().all())),
        check("primary key is unique", not table.duplicated(subset=KEY).any()),
        check("row count = non-missing observations in the sources", len(table) == expected_rows,
              f"{len(table):,} rows, expected {expected_rows:,}"),
        check("every country code is in the crosswalk", set(table["iso3"]) <= iso3_codes,
              f"unknown: {sorted(set(table['iso3']) - iso3_codes)[:5]}"),
        check("sorted by key", table[KEY].equals(table[KEY].sort_values(KEY).reset_index(drop=True))),
    ]
    return all(results)


def check_global_long(table: pd.DataFrame, fred: pd.DataFrame) -> bool:
    ok = check("has exactly the expected columns", list(table.columns) == GLOBAL_COLUMNS,
               f"got {list(table.columns)}")
    if not ok:
        return False
    this_year = pd.Timestamp.now().year
    results = [
        check("one row per series and year", not table.duplicated(subset=["year", "series_id"]).any()),
        check("all eight series are present", table["series_id"].nunique() == fred["series_id"].nunique()),
        check("no missing values", bool(table["value"].notna().all())),
        check("the incomplete current year is not included", int(table["year"].max()) < this_year,
              f"latest year in table: {int(table['year'].max())}"),
        check("years are whole numbers", pd.api.types.is_integer_dtype(table["year"])),
    ]
    vix_2008 = table.loc[(table["series_id"] == "FRED.VIXCLS") & (table["year"] == 2008), "value"]
    results.append(check("VIX for 2008 is a plausible number (average about 33, year-end 40)",
                         len(vix_2008) == 1 and 25 < float(vix_2008.iloc[0]) < 45,
                         f"got {vix_2008.round(1).tolist()}"))
    return all(results)


def summarise(country: pd.DataFrame, world: pd.DataFrame) -> pd.DataFrame:
    actual = country[country["status"] == "actual"]
    by_series = actual.groupby("series_id").agg(
        source=("source", "first"), vintage=("vintage", "first"),
        countries=("iso3", "nunique"), first_year=("year", "min"), last_year=("year", "max"),
        observations=("value", "size")).reset_index()
    world_summary = world.groupby("series_id").agg(
        source=("source", "first"), vintage=("vintage", "first"),
        first_year=("year", "min"), last_year=("year", "max"),
        observations=("value", "size")).reset_index()
    world_summary["countries"] = 0
    return pd.concat([by_series, world_summary], ignore_index=True)[
        ["series_id", "source", "vintage", "countries", "first_year", "last_year", "observations"]]


# =========================================================================== main
def main() -> int:
    iso3_codes, overrides = load_code_maps()
    vintages = manifest_vintages()

    print("reading raw files ...")
    frames = [read_wdi(iso3_codes, vintages), read_weo(overrides), read_gdd(overrides, vintages),
              read_wb_financial(iso3_codes, vintages), read_fdi(iso3_codes, vintages)]
    fred = read_fred(vintages)
    for frame in frames:
        print(f"  {'/'.join(sorted(frame['source'].unique())):8s} {len(frame):>9,} rows, {frame['series_id'].nunique():>3} series, "
              f"{int(frame['value'].isna().sum()):>8,} missing values")
    print(f"  {'FRED':8s} {len(fred):>9,} rows, {fred['series_id'].nunique():>3} series (monthly or daily)")

    world = annualise(fred)
    print("\nchecking annualise() ...")
    if not check_global_long(world, fred):
        print("annualise() is not right yet - see the FAIL lines above.")
        return 1

    country = stack(frames)
    print("\nchecking stack() ...")
    if not check_country_long(country, frames, iso3_codes):
        print("stack() is not right yet - see the FAIL lines above.")
        return 1

    CLEAN.mkdir(parents=True, exist_ok=True)
    country.to_parquet(CLEAN / "country_long.parquet", index=False)
    world.to_parquet(CLEAN / "global_long.parquet", index=False)
    summary = summarise(country, world)
    summary.to_csv(CLEAN / "series_summary.csv", index=False)
    projections = int((country["status"] == "projection").sum())
    print(f"\nwrote data/clean/country_long.parquet  {len(country):,} rows, "
          f"{country['series_id'].nunique()} series, {country['iso3'].nunique()} countries "
          f"({projections:,} rows are WEO projections)")
    print(f"wrote data/clean/global_long.parquet   {len(world):,} rows, {world['series_id'].nunique()} series")
    print(f"wrote data/clean/series_summary.csv    {len(summary)} series")
    return 0


if __name__ == "__main__":
    sys.exit(main())
