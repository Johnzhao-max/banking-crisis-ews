"""Export the tables the Power BI dashboard is built from.

Usage (from the project root):
    .venv/bin/python src/export_powerbi.py

Writes CSV files to output/powerbi/ in a star layout: three "dim" tables that describe
countries, years and variables, and "fact" tables that hold the numbers. In Power BI the fact
tables are linked to the dim tables on iso3, year and variable. See docs/powerbi_guide.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR, OUTPUT, REFERENCE = ROOT / "data" / "model", ROOT / "output", ROOT / "data" / "reference"
TARGET = OUTPUT / "powerbi"
VERSION = "v2"

LABELS = {
    "us_real_10y": "U.S. real 10-year yield", "us_real_10y_gap": "U.S. real 10-year yield, gap",
    "us_tbill_gap": "U.S. 3-month T-bill rate, gap", "ext_bank_loans_gdp": "Loans from foreign banks / GDP",
    "loan_to_deposit": "Loan-to-deposit ratio", "capital_adequacy": "Capital adequacy ratio",
    "bank_zscore": "Bank Z-score", "fin_access": "Financial access index",
    "fin_efficiency": "Financial efficiency index", "real_deposit_rate_chg": "Real deposit rate, change",
    "bank_credit_gdp": "Bank credit / GDP", "total_debt_chg3": "Total debt / GDP, 3-year change",
    "credit_gap": "Credit gap", "gdp_per_capita_ppp_log": "GDP per capita (PPP, log)",
    "gdp_growth": "Real GDP growth", "output_gap": "Output gap",
    "external_debt_exports": "External debt / exports", "reserves_gdp": "Reserves / GDP",
    "reserves_months_imports": "Reserves, months of imports", "debt_service_exports": "Debt service / exports",
}


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ROOT / "data" / "ews.duckdb"), read_only=True)
    scores = pd.read_parquet(MODEL_DIR / f"risk_scores_{VERSION}.parquet")
    shap_long = pd.read_parquet(MODEL_DIR / f"shap_{VERSION}.parquet")

    tables = {}
    tables["dim_country"] = con.execute("""
        SELECT iso3, country, region, income_group FROM dim_country WHERE in_sample ORDER BY iso3""").df()
    tables["dim_year"] = pd.DataFrame({"year": range(int(scores["year"].min()), int(scores["year"].max()) + 1)})
    spec = pd.read_csv(REFERENCE / "model_variables_v2.csv")
    importance = pd.read_csv(OUTPUT / f"shap_importance_{VERSION}.csv").set_index("variable")
    tables["dim_variable"] = pd.DataFrame({
        "variable": spec["variable"], "label": spec["variable"].map(LABELS), "group": spec["group"],
        "imf_variable": spec["imf_variable"], "transformation": spec["transformation"],
        "importance_rank": spec["variable"].map(importance["rank"]),
    })

    risk = scores.rename(columns={"target": "crisis_within_2y"}).drop(columns=["country"])
    # rank within each year: 1 = highest fitted score that year
    risk["rank_in_year_fitted"] = risk.groupby("year")["score_fitted"].rank(ascending=False, method="min").astype(int)
    risk["rank_in_year_block_cv"] = risk.groupby("year")["score_block_cv"].rank(ascending=False, method="min").astype(int)
    tables["fact_risk"] = risk.round(4)

    tables["fact_contribution"] = shap_long[["iso3", "year", "variable", "value", "contribution"]].round(5)
    tables["fact_crisis"] = con.execute("""
        SELECT e.iso3, e.start_year, e.end_year, e.borderline
        FROM crisis_episode e JOIN dim_country c USING (iso3) WHERE c.in_sample ORDER BY 1, 2""").df()
    metrics = pd.read_csv(OUTPUT / "model_comparison.csv")
    tables["fact_metrics"] = metrics[["version", "evaluation", "model", "auc", "auc_ci_low", "auc_ci_high", "brier"]]
    con.close()

    for name, frame in tables.items():
        frame.to_csv(TARGET / f"{name}.csv", index=False)
        print(f"  {name:20s} {len(frame):>8,} rows  {', '.join(frame.columns[:7])}{' ...' if len(frame.columns) > 7 else ''}")
    print(f"wrote {len(tables)} files to {TARGET.relative_to(ROOT).as_posix()}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
