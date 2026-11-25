"""Tasks 11-14: load the clean data into a database, check its quality, write the data dictionary.

Usage (from the project root):
    .venv/bin/python src/build_database.py

What it builds: data/ews.duckdb, a single-file SQL database with a "star" layout.

    dim_country        one row per country      (key: iso3)
    dim_series         one row per data series  (key: series_id)
    fact_country_year  the observations         (key: iso3, year, series_id)
    fact_global_year   world-wide series        (key: year, series_id)
    crisis_episode     banking crises           (key: iso3, start_year)
    sample_country_year (a view)  the observations that belong to the estimation sample

    "Fact" tables hold numbers; "dimension" tables describe what the numbers are about.
    The keys are declared as PRIMARY KEY and REFERENCES, so the database itself refuses a
    duplicate observation or an observation for a country or series it does not know.
    Power BI uses the same layout, so these tables can be loaded there as they are.

What it checks: value ranges, agreement between sources, and how complete each candidate
predictor is inside the sample. Candidates that are missing too often are marked as dropped.

What it writes: docs/data_dictionary.md, generated from the database and from the three
hand-written files in data/reference (model_variables.csv, decisions.csv, country_aliases.csv).
Edit those files, not the dictionary.
"""

from __future__ import annotations

import csv
import json
import sys
import tomllib
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, CLEAN, REFERENCE = ROOT / "data" / "raw", ROOT / "data" / "clean", ROOT / "data" / "reference"
DATABASE = ROOT / "data" / "ews.duckdb"
CONFIG = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
VINTAGES, SAMPLE = CONFIG["vintages"], CONFIG["sample"]
FIRST, LAST = SAMPLE["first_year"], SAMPLE["last_predictor_year"]

SCHEMA = """
CREATE TABLE dim_country (
    iso3 VARCHAR PRIMARY KEY, country VARCHAR NOT NULL, region VARCHAR, income_group VARCHAR,
    in_sample BOOLEAN NOT NULL, weo_code VARCHAR, gdd_code VARCHAR, lv_name VARCHAR);
CREATE TABLE dim_series (
    series_id VARCHAR PRIMARY KEY, source VARCHAR NOT NULL, source_code VARCHAR NOT NULL,
    name VARCHAR, unit VARCHAR, vintage VARCHAR, scope VARCHAR NOT NULL);
CREATE TABLE fact_country_year (
    iso3 VARCHAR NOT NULL REFERENCES dim_country (iso3), year INTEGER NOT NULL,
    series_id VARCHAR NOT NULL REFERENCES dim_series (series_id),
    value DOUBLE NOT NULL, status VARCHAR NOT NULL,
    PRIMARY KEY (iso3, year, series_id));
CREATE TABLE fact_global_year (
    year INTEGER NOT NULL, series_id VARCHAR NOT NULL REFERENCES dim_series (series_id),
    value DOUBLE NOT NULL,
    PRIMARY KEY (year, series_id));
CREATE TABLE crisis_episode (
    iso3 VARCHAR NOT NULL REFERENCES dim_country (iso3), start_year INTEGER NOT NULL,
    end_year INTEGER, borderline BOOLEAN NOT NULL, in_lv2020 BOOLEAN NOT NULL,
    start_year_table1 INTEGER,
    PRIMARY KEY (iso3, start_year));
"""


# --------------------------------------------------------------------------- series metadata
def series_metadata(country: pd.DataFrame, world: pd.DataFrame) -> pd.DataFrame:
    """Name and unit of every series, taken from each source's own metadata."""
    with (RAW / "MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        manifest = {(row["source"], row["series"]): row["description"] for row in csv.DictReader(handle)}
    weo = pd.read_excel(RAW / "weo" / VINTAGES["weo"], sheet_name="Countries", engine="openpyxl",
                        usecols=["INDICATOR.ID", "INDICATOR", "UNIT", "SCALE"]).dropna(subset=["INDICATOR.ID"])
    weo = weo.drop_duplicates("INDICATOR.ID").set_index("INDICATOR.ID")
    gdd = json.loads((RAW / "gdd" / VINTAGES["gdd"] / "datamapper_indicators.json").read_bytes())["indicators"]

    rows = []
    both = pd.concat([country.assign(scope="country"), world.assign(scope="global")])
    for (series_id, source, vintage, scope), _ in both.groupby(["series_id", "source", "vintage", "scope"]):
        code = series_id.split(".", 1)[1]
        if source == "WDI":
            name, unit = manifest[("World Bank WDI", code)], ""
        elif source == "WEO":
            parts = [weo.at[code, "UNIT"], weo.at[code, "SCALE"]]      # either may be blank
            unit = ", ".join(str(p) for p in parts if not pd.isna(p) and p != "Units")
            name = weo.at[code, "INDICATOR"]
        elif source == "GDD":
            name, unit = gdd[code]["label"], gdd[code]["unit"]
        elif source == "GFDD":
            code = series_id                                          # GFDD codes keep their prefix
            name, unit = manifest[("World Bank GFDD", code)], ""
        elif source == "FDI":
            name, unit = manifest[("IMF FDI", code)], "index, 0 to 1"
        else:
            name, unit = manifest[("FRED", code)], ""
        rows.append((series_id, source, code, name, unit, vintage, scope))
    return pd.DataFrame(rows, columns=["series_id", "source", "source_code", "name", "unit", "vintage", "scope"])


# --------------------------------------------------------------------------- load
def load(con: duckdb.DuckDBPyConnection) -> None:
    country = pd.read_parquet(CLEAN / "country_long.parquet")
    world = pd.read_parquet(CLEAN / "global_long.parquet")
    crosswalk = pd.read_csv(REFERENCE / "country_crosswalk.csv", keep_default_na=False)
    episodes = pd.read_csv(REFERENCE / "lv2026_episodes.csv")
    series = series_metadata(country, world)
    crosswalk["in_sample"] = (crosswalk[["in_wdi", "in_weo", "in_gdd", "in_lv"]].astype(int).sum(axis=1) == 4)

    con.execute(SCHEMA)
    for name, frame in {"crosswalk": crosswalk, "series": series, "country": country,
                        "world": world, "episodes": episodes}.items():
        con.register(name, frame)
    con.execute("""INSERT INTO dim_country
        SELECT iso3, country, nullif(region, ''), nullif(income_group, ''), in_sample,
               nullif(weo_code, ''), nullif(gdd_code, ''), nullif(lv_name, '') FROM crosswalk""")
    con.execute("INSERT INTO dim_series SELECT * FROM series")
    con.execute("INSERT INTO fact_country_year SELECT iso3, year, series_id, value, status FROM country")
    con.execute("INSERT INTO fact_global_year SELECT year, series_id, value FROM world")
    con.execute("""INSERT INTO crisis_episode
        SELECT iso3, start_year, end_year, borderline = 1, in_lv2020 = 1, start_year_table1 FROM episodes""")
    con.execute(f"""CREATE VIEW sample_country_year AS
        SELECT f.iso3, f.year, f.series_id, f.value
        FROM fact_country_year f JOIN dim_country c USING (iso3)
        WHERE c.in_sample AND f.status = 'actual' AND f.year BETWEEN {FIRST} AND {LAST}""")


# --------------------------------------------------------------------------- quality checks
def quality_checks(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return findings as text lines; they are printed and copied into the data dictionary."""
    findings = []
    q = lambda sql: con.execute(sql).fetchall()

    counts = {t: q(f"SELECT count(*) FROM {t}")[0][0] for t in
              ("dim_country", "dim_series", "fact_country_year", "fact_global_year", "crisis_episode")}
    findings.append("Rows loaded: " + ", ".join(f"{t} {n:,}" for t, n in counts.items())
                    + ". Primary and foreign keys were enforced by the database on insert.")

    n_sample = q("SELECT count(*) FROM dim_country WHERE in_sample")[0][0]
    n_crises = q(f"""SELECT count(*), sum(borderline::INT) FROM crisis_episode e JOIN dim_country c USING (iso3)
                     WHERE c.in_sample AND e.start_year BETWEEN {FIRST + 1} AND {LAST + SAMPLE['horizon_years']}""")[0]
    findings.append(f"Sample: {n_sample} countries x {FIRST}-{LAST} = {n_sample * (LAST - FIRST + 1):,} country-years. "
                    f"Crises that can be predicted from it (starting {FIRST + 1}-{LAST + SAMPLE['horizon_years']}): "
                    f"{n_crises[0]}, of which {n_crises[1]} borderline.")

    negative = q("""SELECT series_id, count(*) FROM sample_country_year
                    WHERE value < 0 AND series_id IN ('WDI.FD.AST.PRVT.GD.ZS', 'WDI.FM.LBL.BMNY.GD.ZS',
                          'GDD.PVD_LS', 'GDD.CG_DEBT_GDP', 'WDI.FI.RES.TOTL.MO', 'WDI.NE.TRD.GNFS.ZS')
                    GROUP BY 1""")
    findings.append("Stock ratios that cannot be negative (credit, money, debt, reserves, trade): "
                    + ("no negative values." if not negative else f"NEGATIVE VALUES FOUND {negative}"))

    extreme = q("""SELECT c.country, f.year, round(f.value) FROM sample_country_year f JOIN dim_country c USING (iso3)
                   WHERE f.series_id = 'WEO.PCPIPCH' ORDER BY f.value DESC LIMIT 5""")
    n_hyper = q("SELECT count(*) FROM sample_country_year WHERE series_id = 'WEO.PCPIPCH' AND value > 100")[0][0]
    findings.append(f"Inflation above 100 percent: {n_hyper} country-years. Largest: "
                    + "; ".join(f"{c} {y} ({v:,.0f}%)" for c, y, v in extreme)
                    + ". These are real hyperinflations, not errors, but they will dominate a logit model "
                      "unless the variable is winsorised or transformed.")

    growth = q("""SELECT count(*), sum((value < -30 OR value > 50)::INT) FROM sample_country_year
                  WHERE series_id = 'WEO.NGDP_RPCH'""")[0]
    findings.append(f"GDP growth outside -30 to +50 percent: {growth[1]} of {growth[0]:,} country-years "
                    "(wars, oil discoveries, post-conflict rebounds).")

    agree = q("""SELECT count(*), corr(a.value, b.value), avg((abs(a.value - b.value) > 2)::INT)
                 FROM sample_country_year a JOIN sample_country_year b USING (iso3, year)
                 WHERE a.series_id = 'WDI.NY.GDP.MKTP.KD.ZG' AND b.series_id = 'WEO.NGDP_RPCH'""")[0]
    findings.append(f"Agreement between sources, GDP growth in WDI and WEO: {agree[0]:,} common country-years, "
                    f"correlation {agree[1]:.3f}, {agree[2]:.0%} differ by more than 2 percentage points "
                    "(different vintages and base years).")
    return findings


def variable_coverage(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Share of the sample's country-years for which each candidate variable can be computed."""
    variables = pd.read_csv(REFERENCE / "model_variables.csv")
    n_sample = con.execute("SELECT count(*) FROM dim_country WHERE in_sample").fetchone()[0]
    grid = n_sample * (LAST - FIRST + 1)
    known = {row[0] for row in con.execute("SELECT series_id FROM dim_series").fetchall()}
    fills, eighties = [], []
    for row in variables.itertuples():
        inputs = row.series_id.split("|")
        missing_series = [s for s in inputs if s not in known]
        if missing_series:
            raise ValueError(f"{row.variable}: series not in the database: {missing_series}")
        if row.group == "global":
            sql = f"""SELECT count(*) FILTER (WHERE n = {len(inputs)}) * 1.0 / {LAST - FIRST + 1},
                             count(*) FILTER (WHERE n = {len(inputs)} AND year < 1990) / 10.0
                      FROM (SELECT year, count(*) AS n FROM fact_global_year
                            WHERE series_id IN ({', '.join(repr(s) for s in inputs)})
                              AND year BETWEEN {FIRST} AND {LAST} GROUP BY year)"""
        else:
            # a variable built from two series needs both of them in the same country-year
            sql = f"""SELECT count(*) FILTER (WHERE n = {len(inputs)}) * 1.0 / {grid},
                             count(*) FILTER (WHERE n = {len(inputs)} AND year < 1990) * 1.0 / {n_sample * 10}
                      FROM (SELECT iso3, year, count(*) AS n FROM sample_country_year
                            WHERE series_id IN ({', '.join(repr(s) for s in inputs)}) GROUP BY iso3, year)"""
        fill, fill_80s = con.execute(sql).fetchone()
        fills.append(fill)
        eighties.append(fill_80s)
    variables["fill"], variables["fill_1980s"] = fills, eighties
    variables["kept"] = variables["fill"] >= 1 - SAMPLE["max_missing_share"]
    return variables


# --------------------------------------------------------------------------- data dictionary
def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns)
    body = "\n".join("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |"
                     for row in frame.itertuples(index=False))
    return f"{header}\n{body}"


def write_dictionary(con: duckdb.DuckDBPyConnection, findings: list[str], coverage: pd.DataFrame) -> Path:
    names = dict(con.execute("SELECT series_id, name || CASE WHEN unit <> '' THEN ' (' || unit || ')' ELSE '' END FROM dim_series").fetchall())
    decisions = pd.read_csv(REFERENCE / "decisions.csv")
    sources = con.execute("""
        SELECT s.source AS Source, s.vintage AS Vintage, count(DISTINCT s.series_id) AS Series
        FROM dim_series s GROUP BY 1, 2 ORDER BY 1""").df()
    catalogue = con.execute(f"""
        SELECT s.series_id AS "Series", s.name AS "Name", s.unit AS "Unit",
               count(DISTINCT f.iso3) AS "Countries", min(f.year) || '-' || max(f.year) AS "Years"
        FROM dim_series s JOIN fact_country_year f USING (series_id)
        WHERE f.status = 'actual' GROUP BY 1, 2, 3
        UNION ALL
        SELECT s.series_id, s.name, s.unit, NULL, min(g.year) || '-' || max(g.year)
        FROM dim_series s JOIN fact_global_year g USING (series_id) GROUP BY 1, 2, 3
        ORDER BY 1""").df()
    catalogue["Countries"] = ["global" if pd.isna(c) else int(c) for c in catalogue["Countries"]]
    catalogue[["Name", "Unit"]] = catalogue[["Name", "Unit"]].fillna("")

    table = pd.DataFrame({
        "Variable": coverage["variable"],
        "Group": coverage["group"],
        "Built from": [" and ".join(f"`{s}`" for s in ids.split("|")) for ids in coverage["series_id"]],
        "Source definition": [" / ".join(names[s] for s in ids.split("|")) for ids in coverage["series_id"]],
        "Transformation": coverage["transformation"],
        "Why it is a candidate": coverage["why"],
        "Available": [f"{v:.0%}" for v in coverage["fill"]],
        "In 1980s": [f"{v:.0%}" for v in coverage["fill_1980s"]],
        "Kept": ["yes" if k else "no" for k in coverage["kept"]],
    })
    spec_v2 = pd.read_csv(REFERENCE / "model_variables_v2.csv")
    summary_v2 = ROOT / "data" / "model" / "feature_summary_v2.csv"       # exists once build_features.py v2 has run
    missing_v2 = pd.read_csv(summary_v2).set_index("variable")["missing"] if summary_v2.exists() else {}
    table_v2 = pd.DataFrame({
        "Variable": spec_v2["variable"],
        "IMF variable": spec_v2["imf_variable"],
        "Built from": [" and ".join(f"`{s}`" for s in ids.split("|")) for ids in spec_v2["series_id"]],
        "Transformation": spec_v2["transformation"],
        "Note": spec_v2["why"],
        "Missing": [f"{missing_v2[v]:.0%}" if v in missing_v2 else "" for v in spec_v2["variable"]],
    })
    text = f"""# Data dictionary

Generated by `src/build_database.py`. Do not edit this file: change `config.toml` or the files in
`data/reference/` and run the script again.

## Sources and vintages

{markdown_table(sources)}

Every raw file, with its URL, retrieval time and checksum, is listed in `data/raw/MANIFEST.csv`.
Crisis dates: Laeven and Valencia (2026), IMF WP 2026/094, Appendix Table A1
(`data/reference/lv2026_episodes.csv`).

## Sample

- Countries: {con.execute("SELECT count(*) FROM dim_country WHERE in_sample").fetchone()[0]} (in all four sources)
- Predictor years: {FIRST}-{LAST}
- Target: a systemic banking crisis starting in the following {SAMPLE['horizon_years']} years
- A candidate variable is kept if it is available for at least {1 - SAMPLE['max_missing_share']:.0%} of country-years

## Decisions

{markdown_table(decisions.rename(columns=str.capitalize))}

## Quality checks

{chr(10).join('- ' + line for line in findings)}

## Variable set 1 (first, self-chosen list)

"Available" is the share of the sample's country-years for which the underlying series exist.
{int(coverage['kept'].sum())} of {len(coverage)} candidates are kept.

{markdown_table(table)}

## Variable set 2 (aligned with the IMF's list)

Each variable corresponds to an entry in Table 3 of the IMF's financial sector model.
"Missing" is the share of rows in the model table without a value; those gaps are filled
with the training-window median, as in the IMF's model.

{markdown_table(table_v2)}

## All series in the database

{markdown_table(catalogue)}
"""
    path = ROOT / "docs" / "data_dictionary.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- main
def main() -> int:
    DATABASE.unlink(missing_ok=True)                 # always rebuilt from the clean files
    con = duckdb.connect(str(DATABASE))
    load(con)
    findings = quality_checks(con)
    coverage = variable_coverage(con)
    path = write_dictionary(con, findings, coverage)
    coverage.to_csv(CLEAN / "variable_coverage.csv", index=False)   # read by build_features.py
    con.close()

    print(f"built {DATABASE.relative_to(ROOT).as_posix()}\n")
    for line in findings:
        print(f"- {line}")
    print(f"\ncandidate predictors (kept if available for at least {1 - SAMPLE['max_missing_share']:.0%} of the sample)")
    print(f"  {'variable':28s} {'group':9s} {'available':>9s} {'1980s':>6s}  kept")
    for row in coverage.sort_values(["kept", "fill"], ascending=[False, False]).itertuples():
        print(f"  {row.variable:28s} {row.group:9s} {row.fill:>9.0%} {row.fill_1980s:>6.0%}  {'yes' if row.kept else 'NO'}")
    print(f"\nwrote {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
