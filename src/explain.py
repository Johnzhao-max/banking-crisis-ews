"""Tasks 28-30: explain what the model has learned, with Shapley values.

Usage (from the project root):
    .venv/bin/python src/explain.py          # variable set v2 (aligned with the IMF's list)
    .venv/bin/python src/explain.py v1

What this is, and what it is not
    A random forest is fitted on the WHOLE sample and its fitted risk scores are decomposed
    into contributions of each variable. This is how the IMF presents its financial sector
    model, and it answers "which patterns did the model find in 1980-2023?".
    It is descriptive. It says nothing about forecasting ability: the backtest shows that
    these patterns did not hold up out of sample (see output/model_comparison.csv).

Shapley values in one paragraph
    For one country-year, the model's risk score equals the sample-average score plus one
    number per variable. That number is the variable's Shapley value: how much knowing this
    variable's value moved the score up or down, averaged over all the orders in which the
    variables could be revealed. The contributions add up exactly to the score, which is what
    makes a "waterfall" breakdown possible. A regression coefficient gives one number per
    variable for the whole sample; a Shapley value gives one per variable per country-year.

Outputs
    data/model/shap_v2.parquet        one row per country-year and variable: value, contribution
    data/model/risk_scores_v2.parquet one row per country-year: fitted score and out-of-sample scores
    output/shap_importance_v2.csv     mean absolute contribution per variable, with the IMF's ranking
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR, OUTPUT, REFERENCE = ROOT / "data" / "model", ROOT / "output", ROOT / "data" / "reference"
ID_COLUMNS = ["iso3", "country", "year", "target", "crisis_start_ahead", "ahead_is_borderline"]
SEED = 20261119
TREES = 300

# Order of the variables in Figure 21 of the IMF's technical note (most important first).
IMF_RANKING = ["Real 10yr US Yield", "External Bank Liabilities to GDP", "3mo T-bill Rate HP gap",
               "Gross Reserves to GDP", "External Debt to Exports", "Total Debt Growth",
               "Corp Debt Sub-Inv. Grade", "Real Deposit Rate Growth", "Loan to Deposit Ratio",
               "CB Liquidity Support", "Real Output Growth", "HP Output Gap", "House Price-to-Income",
               "BIS Credit Gap", "Capital Adequacy Ratio", "Int. Rate on New Pub. Debt"]


def main(argv: list[str]) -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    version = argv[0] if argv else "v2"
    table = pd.read_parquet(MODEL_DIR / f"model_table_{version}.parquet").sort_values(["iso3", "year"]).reset_index(drop=True)
    predictors = [c for c in table.columns if c not in ID_COLUMNS]
    params = json.loads((OUTPUT / f"tuned_params_{version}.json").read_text())["random_forest"]

    # ---- fit on the whole sample (descriptive use)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(table[predictors].to_numpy(dtype=float))
    forest = RandomForestClassifier(n_estimators=TREES, class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=SEED, **params)
    forest.fit(X, table["target"].to_numpy())
    fitted = forest.predict_proba(X)[:, 1]

    # ---- Shapley values for the "crisis ahead" class
    explainer = shap.TreeExplainer(forest)
    values = np.asarray(explainer.shap_values(X, check_additivity=False))
    contributions = values[:, :, 1] if values.ndim == 3 else values[1]
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    gap = float(np.abs(base + contributions.sum(axis=1) - fitted).max())
    print(f"{version}: {len(table):,} country-years, {len(predictors)} variables, {TREES} trees")
    print(f"  {'PASS' if gap < 1e-6 else 'FAIL'}  contributions add up to the fitted score "
          f"(largest gap {gap:.1e}); sample-average score {base:.3f}")

    # ---- long table of contributions
    long = pd.DataFrame({
        "iso3": np.repeat(table["iso3"].to_numpy(), len(predictors)),
        "year": np.repeat(table["year"].to_numpy(), len(predictors)),
        "variable": np.tile(predictors, len(table)),
        "value": table[predictors].to_numpy(dtype=float).ravel(),          # as observed; NaN if missing
        "value_used": X.ravel(),                                           # after median fill
        "contribution": contributions.ravel(),
    })
    long.to_parquet(MODEL_DIR / f"shap_{version}.parquet", index=False)

    # ---- risk scores: fitted (in sample) next to the honest out-of-sample ones
    scores = table[ID_COLUMNS].copy()
    scores["score_fitted"] = fitted
    scores["score_average"] = base
    predictions = pd.read_parquet(MODEL_DIR / f"predictions_{version}.parquet")
    for evaluation in ("block_cv", "backtest"):
        part = predictions[(predictions["model"] == "random_forest") & (predictions["evaluation"] == evaluation)]
        scores = scores.merge(part[["iso3", "year", "probability"]].rename(columns={"probability": f"score_{evaluation}"}),
                              on=["iso3", "year"], how="left")
    scores.to_parquet(MODEL_DIR / f"risk_scores_{version}.parquet", index=False)

    # ---- importance, with the IMF's ranking where the variable has a counterpart
    importance = pd.DataFrame({"variable": predictors, "mean_abs_contribution": np.abs(contributions).mean(axis=0)})
    importance = importance.sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)
    importance.insert(0, "rank", importance.index + 1)
    spec_file = REFERENCE / ("model_variables_v2.csv" if version == "v2" else "model_variables.csv")
    spec = pd.read_csv(spec_file).set_index("variable")
    importance["group"] = importance["variable"].map(spec["group"])
    if "imf_variable" in spec.columns:
        importance["imf_variable"] = importance["variable"].map(spec["imf_variable"])
        lookup = {name.lower(): i + 1 for i, name in enumerate(IMF_RANKING)}
        importance["imf_rank"] = [lookup.get(str(n).split(",")[0].strip().lower().replace(" - ", " "), pd.NA)
                                  for n in importance["imf_variable"]]
    importance.round(5).to_csv(OUTPUT / f"shap_importance_{version}.csv", index=False)

    print("\n  rank  variable                    mean |contribution|   IMF rank")
    for row in importance.itertuples():
        imf = getattr(row, "imf_rank", pd.NA)
        print(f"  {row.rank:>4d}  {row.variable:26s} {row.mean_abs_contribution:>12.4f}        {'' if pd.isna(imf) else int(imf)}")

    # ---- the cases used in the figures
    print("\n  cases (fitted score vs sample average; largest contributions)")
    for iso3, year in [("THA", 1996), ("IRL", 2006), ("LBN", 2018)]:
        mask = (long["iso3"] == iso3) & (long["year"] == year)
        if not mask.any():
            print(f"  {iso3} {year}: not in the sample")
            continue
        row = scores[(scores["iso3"] == iso3) & (scores["year"] == year)].iloc[0]
        top = long[mask].reindex(long[mask]["contribution"].abs().sort_values(ascending=False).index).head(4)
        print(f"  {row['country']} {year}: fitted {row['score_fitted']:.2f}, block-CV {row['score_block_cv']:.2f}, "
              f"average {base:.2f} | " + ", ".join(f"{r.variable} {r.contribution:+.3f}" for r in top.itertuples()))
    print(f"\nwrote data/model/shap_{version}.parquet, data/model/risk_scores_{version}.parquet, "
          f"output/shap_importance_{version}.csv")
    return 0 if gap < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
