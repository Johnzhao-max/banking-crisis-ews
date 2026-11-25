"""Tasks 15-19: build the table the models are trained on.

Usage (from the project root):
    .venv/bin/python src/build_features.py        # version 1: first, self-chosen variable set
    .venv/bin/python src/build_features.py v2     # version 2: variables aligned with the IMF's list

Version 2 was designed after version 1 had been backtested. Its variables follow Table 3 of the
IMF's financial sector model (TNM 21/03) wherever public data exist; they are listed in
data/reference/model_variables_v2.csv. It uses exactly the same country-years as version 1, so
that the two versions differ only in their predictors.

One row per country-year t, with
    target      1 if a systemic banking crisis starts in t+1 or t+2, else 0
    predictors  the kept variables from data/reference/model_variables.csv, measured in year t

Steps
    1. Pull the needed series from the database and lay them out on a complete
       country x year grid, so that "three years earlier" always means three calendar years
       earlier, even where a country has gaps in its data.
    2. Apply each variable's transformation (level, change, log, ratio, ...).
    3. Build the target from the crisis episodes.
    4. Remove years during a crisis, and country-years that are mostly empty.
    5. Check the result and save it.

No leakage by construction: every predictor for year t uses data from year t or earlier, and the
target uses crisis dates from t+1 and t+2 only. Missing values are NOT filled here; that happens
inside each training window of the backtest.

Outputs (the suffix is the version)
    data/model/model_table_v1.parquet    the model table
    data/model/feature_summary_v1.csv    per predictor: missing share and distribution
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN, MODEL, REFERENCE = ROOT / "data" / "clean", ROOT / "data" / "model", ROOT / "data" / "reference"
DATABASE = ROOT / "data" / "ews.duckdb"
SAMPLE = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))["sample"]
FIRST, LAST, HORIZON = SAMPLE["first_year"], SAMPLE["last_predictor_year"], SAMPLE["horizon_years"]
HISTORY = 20                    # extra years loaded before FIRST so that changes and trends can be computed
LAST_CRISIS_YEAR = 2025         # coverage of the crisis database


# --------------------------------------------------------------------------- transformations
def lag(series: pd.Series, years: int) -> pd.Series:
    """Value `years` calendar years earlier. The grid is complete, so shifting rows shifts years."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="iso3").shift(years)
    return series.shift(years)


def positive(series: pd.Series) -> pd.Series:
    return series.where(series > 0)


_HP_WEIGHTS: dict[tuple[int, float], np.ndarray] = {}


def hp_trend_at_end(values: np.ndarray, lam: float) -> float:
    """Hodrick-Prescott trend at the LAST point of `values`.

    The HP trend solves (I + lam * D'D) trend = values, with D the second-difference matrix.
    Only the last trend value is needed, and it is a fixed weighted sum of the data for a given
    length, so the weights are computed once per length and reused.
    """
    n = len(values)
    if (n, lam) not in _HP_WEIGHTS:
        second_diff = np.diff(np.eye(n), n=2, axis=0)
        system = np.eye(n) + lam * second_diff.T @ second_diff
        _HP_WEIGHTS[(n, lam)] = np.linalg.solve(system, np.eye(n)[-1])
    return float(_HP_WEIGHTS[(n, lam)] @ values)


def hp_gap(series: pd.Series, lam: float, min_years: int = 10) -> pd.Series:
    """One-sided HP gap: value in year t minus the trend estimated on data up to year t only.

    A normal (two-sided) HP filter uses later years to place the trend, which would let the
    future leak into the predictor. Re-estimating the trend for every year on its own past
    is what "one-sided" means, and it is how the IMF and the BIS compute such gaps.
    """
    def one_country(group: pd.Series) -> pd.Series:
        values = group.to_numpy(dtype=float)
        gap = np.full(len(values), np.nan)
        seen = np.flatnonzero(~np.isnan(values))
        for count in range(min_years, len(seen) + 1):
            history = values[seen[:count]]
            gap[seen[count - 1]] = history[-1] - hp_trend_at_end(history, lam)
        return pd.Series(gap, index=group.index)

    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="iso3", group_keys=False).apply(one_country)
    return one_country(series)


def log_level_from_growth(growth: pd.Series) -> pd.Series:
    """100 x log of a level index built by cumulating annual growth rates (percent)."""
    step = 100 * np.log(positive(1 + growth / 100))
    if isinstance(growth.index, pd.MultiIndex):
        return step.groupby(level="iso3").cumsum()
    return step.cumsum()


def real_rate(nominal: pd.Series, price_index: pd.Series) -> pd.Series:
    return nominal - 100 * (price_index / lag(price_index, 1) - 1)


def log_real(nominal_rate: pd.Series, inflation: pd.Series) -> pd.Series:
    """Real interest rate from log rates, so that hyperinflation years stay on a usable scale."""
    return 100 * np.log(positive(1 + nominal_rate / 100)) - 100 * np.log(positive(1 + inflation / 100))


TRANSFORMS = {
    "level":     lambda a: a,
    "chg1":      lambda a: a - lag(a, 1),
    "chg3":      lambda a: a - lag(a, 3),
    "log":       lambda a: np.log(positive(a)),
    "logpct":    lambda a: 100 * np.log(positive(1 + a / 100)),
    "logchg1":   lambda a: 100 * np.log(positive(a) / positive(lag(a, 1))),
    "pct_chg1":  lambda a: 100 * (a / lag(a, 1) - 1),
    "ratio_pct": lambda a, b: 100 * a / b,
    "real_rate": real_rate,
    # added for version 2
    "real_rate_hpgap": lambda a, b: hp_gap(real_rate(a, b), 100),
    "hpgap100":        lambda a: hp_gap(a, 100),
    "hpgap1600":       lambda a: hp_gap(a, 1600),
    "growth_hpgap100": lambda a: hp_gap(log_level_from_growth(a), 100),
    "sum_chg3":        lambda a, b: (a + b) - lag(a + b, 3),
    "logreal_chg1":    lambda a, b: log_real(a, b) - lag(log_real(a, b), 1),
}


# --------------------------------------------------------------------------- steps
def load_grid(con: duckdb.DuckDBPyConnection, variables: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Base series on a complete grid: (iso3, year) for country series, year for global ones."""
    needed = sorted({s for ids in variables["series_id"] for s in ids.split("|")})
    countries = [r[0] for r in con.execute("SELECT iso3 FROM dim_country WHERE in_sample ORDER BY iso3").fetchall()]
    years = range(FIRST - HISTORY, LAST + 1)

    long = con.execute("""
        SELECT f.iso3, f.year, f.series_id, f.value
        FROM fact_country_year f JOIN dim_country c USING (iso3)
        WHERE c.in_sample AND f.status = 'actual' AND f.year BETWEEN ? AND ?
          AND f.series_id IN (SELECT unnest(?))""", [years.start, years[-1], needed]).df()
    country = long.pivot(index=["iso3", "year"], columns="series_id", values="value")
    country = country.reindex(pd.MultiIndex.from_product([countries, years], names=["iso3", "year"]))

    world = con.execute("""
        SELECT year, series_id, value FROM fact_global_year
        WHERE year BETWEEN ? AND ? AND series_id IN (SELECT unnest(?))""",
        [years.start, years[-1], needed]).df()
    world = world.pivot(index="year", columns="series_id", values="value").reindex(list(years))
    world.index.name = "year"
    return country, world


def build_predictors(variables: pd.DataFrame, country: pd.DataFrame, world: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=country.index)
    years = features.index.get_level_values("year")
    for row in variables.itertuples():
        base = world if row.group == "global" else country
        inputs = [base[s] for s in row.series_id.split("|")]
        values = TRANSFORMS[row.transform](*inputs)
        features[row.variable] = values.reindex(years).to_numpy() if row.group == "global" else values
    return features.replace([np.inf, -np.inf], np.nan)


def build_target(con: duckdb.DuckDBPyConnection, index: pd.MultiIndex) -> pd.DataFrame:
    episodes = con.execute("""SELECT e.iso3, e.start_year, e.start_year_table1, e.end_year, e.borderline
                              FROM crisis_episode e JOIN dim_country c USING (iso3) WHERE c.in_sample""").df()
    if SAMPLE["crisis_start_source"] == "table1":
        episodes["start_year"] = episodes["start_year_table1"].fillna(episodes["start_year"]).astype(int)
    if not SAMPLE["include_borderline"]:
        episodes = episodes[~episodes["borderline"]]
    episodes["end_year"] = episodes["end_year"].fillna(LAST_CRISIS_YEAR).astype(int)   # no end yet = ongoing

    starts = {(r.iso3, r.start_year): bool(r.borderline) for r in episodes.itertuples()}
    during = {(r.iso3, year) for r in episodes.itertuples() for year in range(r.start_year, r.end_year + 1)}

    rows = []
    for iso3, year in index:
        ahead = [year + k for k in range(1, HORIZON + 1) if (iso3, year + k) in starts]
        rows.append({
            "target": int(bool(ahead)),
            "crisis_start_ahead": ahead[0] if ahead else pd.NA,
            "ahead_is_borderline": int(starts[(iso3, ahead[0])]) if ahead else 0,
            "in_crisis": (iso3, year) in during,
        })
    return pd.DataFrame(rows, index=index).astype({"crisis_start_ahead": "Int64"})


def check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}{'  - ' + detail if detail and not passed else ''}")
    return passed


# --------------------------------------------------------------------------- main
def main(argv: list[str]) -> int:
    version = argv[0] if argv else "v1"
    if version == "v1":
        variables = pd.read_csv(CLEAN / "variable_coverage.csv")
        kept = variables[variables["kept"]]
    elif version == "v2":
        kept = pd.read_csv(REFERENCE / "model_variables_v2.csv")   # no coverage rule: gaps are imputed, as in the IMF model
    else:
        print("usage: build_features.py [v1|v2]")
        return 2
    con = duckdb.connect(str(DATABASE), read_only=True)

    country, world = load_grid(con, kept)
    features = build_predictors(kept, country, world)
    table = build_target(con, features.index).join(features)
    names = dict(con.execute("SELECT iso3, country FROM dim_country").fetchall())
    con.close()

    table = table[table.index.get_level_values("year") >= FIRST]
    n_grid = len(table)
    local = list(kept.loc[kept["group"] != "global", "variable"])
    in_crisis = int(table["in_crisis"].sum())
    if SAMPLE["drop_crisis_years"]:
        table = table[~table["in_crisis"]]
    missing_share = table[local].isna().mean(axis=1)
    too_empty = missing_share > SAMPLE["max_missing_row"]
    lost_positives = int(table.loc[too_empty, "target"].sum())
    if version == "v1":
        table = table[~too_empty]
    else:                           # same country-years as version 1, whatever their missing share
        base_rows = pd.read_parquet(MODEL / "model_table_v1.parquet", columns=["iso3", "year"])
        table = table[table.index.isin(pd.MultiIndex.from_frame(base_rows))]
        too_empty = pd.Series(False, index=table.index)
        lost_positives = 0
    rows_in_crisis = int(table["in_crisis"].sum())
    table = table.drop(columns="in_crisis").reset_index()
    table.insert(1, "country", table["iso3"].map(names))

    predictors = list(kept["variable"])
    print(f"grid {n_grid:,} country-years -> removed {in_crisis:,} during a crisis and {int(too_empty.sum()):,} "
          f"mostly empty (these held {lost_positives} positives) -> {len(table):,} rows")
    print(f"positives: {int(table['target'].sum())} ({table['target'].mean():.1%}), "
          f"{table.loc[table['target'] == 1, 'crisis_start_ahead'].groupby(table['iso3']).nunique().sum()} distinct crises, "
          f"{table['iso3'].nunique()} countries, {len(predictors)} predictors\n")

    tha = table[(table["iso3"] == "THA") & table["year"].between(1994, 1996)]
    usa = table[(table["iso3"] == "USA") & table["year"].between(2004, 2006)]
    results = [
        check("one row per country-year", not table.duplicated(["iso3", "year"]).any()),
        check("years within the sample window", table["year"].between(FIRST, LAST).all()),
        check("target is 0 or 1", set(table["target"].unique()) <= {0, 1}),
        check("no infinite values", bool(np.isfinite(table[predictors].to_numpy(dtype=float, na_value=0)).all())),
        check("global variables have no gaps", bool(table[list(kept.loc[kept['group'] == 'global', 'variable'])].notna().all().all())),
        check("same country-years as version 1", version == "v1" or len(table) == len(base_rows), f"{len(table):,} rows"),
        check("Thailand 1995 and 1996 are flagged (crisis in 1997), 1994 is not",
              tha.set_index("year")["target"].to_dict() == {1994: 0, 1995: 1, 1996: 1}, str(tha.set_index("year")["target"].to_dict())),
        check("United States 2005 and 2006 are flagged (crisis in 2007), 2004 is not",
              usa.set_index("year")["target"].to_dict() == {2004: 0, 2005: 1, 2006: 1}, str(usa.set_index("year")["target"].to_dict())),
        check("Thailand 1996: bank credit far above its own trend (boom before the 1997 crisis)",
              float(tha.loc[tha["year"] == 1996, "bank_credit_gdp_chg3" if version == "v1" else "credit_gap"].iloc[0]) > 15),
        check("no row lies inside a crisis", not SAMPLE["drop_crisis_years"] or rows_in_crisis == 0,
              f"{rows_in_crisis} rows"),
    ]
    if not all(results):
        return 1

    summary = table[predictors].describe(percentiles=[0.01, 0.5, 0.99]).T[["mean", "std", "min", "1%", "50%", "99%", "max"]]
    summary.insert(0, "missing", table[predictors].isna().mean())
    summary.insert(1, "mean_before_crisis", table.loc[table["target"] == 1, predictors].mean())
    summary.insert(2, "mean_otherwise", table.loc[table["target"] == 0, predictors].mean())
    MODEL.mkdir(parents=True, exist_ok=True)
    table.to_parquet(MODEL / f"model_table_{version}.parquet", index=False)
    summary.round(3).to_csv(MODEL / f"feature_summary_{version}.csv", index_label="variable")

    print("\npredictor                  missing   mean before a crisis   mean otherwise")
    for name, row in summary.iterrows():
        print(f"  {name:26s} {row['missing']:>5.0%} {row['mean_before_crisis']:>18.2f} {row['mean_otherwise']:>16.2f}")
    print("\npositives by decade:", table.groupby((table["year"] // 10) * 10)["target"].sum().to_dict())
    print("rows by decade:     ", table.groupby((table["year"] // 10) * 10).size().to_dict())
    print(f"\nwrote data/model/model_table_{version}.parquet and data/model/feature_summary_{version}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
