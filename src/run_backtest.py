"""Tasks 20-27: out-of-sample backtest of the crisis models.

Usage (from the project root):
    .venv/bin/python src/run_backtest.py          # both variable sets, then a comparison
    .venv/bin/python src/run_backtest.py v2       # one variable set
    .venv/bin/python src/run_backtest.py compare  # reprint the comparison from saved results

The question: standing at the end of year T with only what was known then, how well does each
model rank countries by the risk of a banking crisis starting in T+1 or T+2?

How one step of the backtest works (expanding window)
    test rows   predictor year T
    train rows  predictor years up to T - 2
    The two-year gap matters. The target of a row from year T-1 depends on whether a crisis
    starts in T or T+1, which is not known at the end of year T. Training on it would use the
    future. So the last training year is T - 2, whose target was fully observed by year T.
    The step is repeated for T = 2001, 2002, ..., 2023, each time with one more year of data.

Everything that learns from data is fitted inside the training rows only: the model, the
medians used to fill gaps, the winsorising cut-offs, and the scaling. The hyperparameters are
tuned once, on the first training window (years up to 1999), and then kept fixed.

Second evaluation: time-block cross-validation
    The backtest's test years are dominated by one event, the 2007-08 crisis, which is unlike
    the crises before 2000. The IMF notes the same risk and complements its backtest with
    cross-validation on groups of consecutive years. Here: hold out each 5-year block in turn,
    drop the 2 years on either side of it, train on the rest, predict the block. Unlike the
    backtest this is not a real-time test, because later years help predict earlier ones; it
    asks whether the patterns hold across periods.

Models
    rule of thumb   past crisis frequency: average of the global and the country's own rate
    logit           the traditional benchmark (gaps filled, winsorised, standardised)
    random forest   500 trees, classes re-weighted
    balanced RF     each tree sees as many calm as pre-crisis rows (the IMF's choice)
    XGBoost         gradient-boosted trees; handles missing values itself, so nothing is filled

Reference point: shuffled cross-validation (NOT a valid test)
    The default in many tutorials: split the rows at random into 5 folds. It is reported here
    only to show how much it overstates performance. With random folds, the two pre-crisis
    years of one crisis land on both sides of the split, and so do countries hit by the same
    global wave in the same year, so the model is tested on events it has in effect already seen.

Outputs (the suffix is the variable-set version)
    data/model/predictions_v1.parquet   every out-of-sample prediction, both evaluations
    output/metrics_v1.csv               AUC with confidence interval, by period, and Brier score
    output/tuning_v1.csv                the hyperparameter search
    output/roc_curves_v1.png            ROC curves of the backtest
    output/model_comparison.csv         both versions side by side
"""

from __future__ import annotations

import itertools
import json
import sys
import tomllib
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR, OUTPUT = ROOT / "data" / "model", ROOT / "output"
SAMPLE = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))["sample"]
HORIZON = SAMPLE["horizon_years"]
FIRST_TEST_YEAR = 2001
SEED = 20261119
ID_COLUMNS = ["iso3", "country", "year", "target", "crisis_start_ahead", "ahead_is_borderline"]


# --------------------------------------------------------------------------- building blocks
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to its 1st-99th percentile range, learned from the training rows."""

    def fit(self, X, y=None):
        self.low_, self.high_ = np.nanpercentile(X, [1, 99], axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.low_, self.high_)


def make_models(params: dict) -> dict:
    """Fresh, unfitted models. `params` holds the tuned hyperparameters."""
    return {
        "logit": make_pipeline(
            SimpleImputer(strategy="median"), Winsorizer(), StandardScaler(),
            LogisticRegression(C=1e6, max_iter=5000)),                # C this large = no penalty
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=SEED, **params["random_forest"])),
        "balanced_rf": make_pipeline(
            SimpleImputer(strategy="median"),
            BalancedRandomForestClassifier(n_estimators=500, sampling_strategy="all", replacement=True,
                                           bootstrap=False, n_jobs=-1, random_state=SEED,
                                           **params["random_forest"])),
        "xgboost": XGBClassifier(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=5, n_jobs=-1, random_state=SEED,
                                 eval_metric="logloss", **params["xgboost"]),
    }


def rule_of_thumb(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Past frequency of pre-crisis years: mean of the global rate and the country's own rate."""
    overall = train["target"].mean()
    own = train.groupby("iso3")["target"].mean()
    return 0.5 * overall + 0.5 * test["iso3"].map(own).fillna(overall).to_numpy()


def split(table: pd.DataFrame, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = table[table["year"] <= test_year - HORIZON]
    test = table[table["year"] == test_year]
    assert train["year"].max() + HORIZON <= test_year, "training rows whose target is not yet known"
    return train, test


# --------------------------------------------------------------------------- tuning
GRIDS = {
    "random_forest": [{"min_samples_leaf": leaf, "max_features": feats}
                      for leaf, feats in itertools.product([1, 5, 20], ["sqrt", 0.5])],
    "xgboost": [{"max_depth": depth, "n_estimators": trees}
                for depth, trees in itertools.product([2, 3, 4], [100, 300, 600])],
}


def time_blocks(years: pd.Series, block: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-validation folds made of consecutive years, with a gap around the held-out block.

    Rows are never shuffled across time: a random split would put 1997 in training and 1996 in
    validation for the same country, and the two rows describe the same crisis.
    """
    folds, start = [], int(years.min())
    while start <= years.max():
        held = (years >= start) & (years < start + block)
        near = (years >= start - HORIZON) & (years < start + block + HORIZON)
        if held.sum():
            folds.append((np.where(~near)[0], np.where(held)[0]))
        start += block
    return folds


def tune(table: pd.DataFrame, predictors: list[str]) -> tuple[dict, pd.DataFrame]:
    """Pick hyperparameters using only the first training window."""
    first_window, _ = split(table, FIRST_TEST_YEAR)
    X, y = first_window[predictors].to_numpy(dtype=float), first_window["target"].to_numpy()
    folds = time_blocks(first_window["year"].reset_index(drop=True))
    records, best = [], {}
    for family, grid in GRIDS.items():
        for candidate in grid:
            params = {"random_forest": GRIDS["random_forest"][0], "xgboost": GRIDS["xgboost"][0]} | {family: candidate}
            scores = []
            for train_idx, valid_idx in folds:
                if y[valid_idx].sum() == 0 or y[train_idx].sum() == 0:
                    continue
                model = make_models(params)[family]
                model.fit(X[train_idx], y[train_idx])
                scores.append(roc_auc_score(y[valid_idx], model.predict_proba(X[valid_idx])[:, 1]))
            records.append({"model": family, **candidate, "mean_auc": np.mean(scores), "folds": len(scores)})
        family_records = [r for r in records if r["model"] == family]
        winner = max(family_records, key=lambda r: r["mean_auc"])
        best[family] = {k: v for k, v in winner.items() if k not in ("model", "mean_auc", "folds")}
    return best, pd.DataFrame(records)


# --------------------------------------------------------------------------- evaluations
MODELS = ["rule_of_thumb", "logit", "random_forest", "balanced_rf", "xgboost"]


def predict_fold(train: pd.DataFrame, test: pd.DataFrame, predictors: list[str], params: dict,
                 y_train: np.ndarray | None = None) -> pd.DataFrame:
    """Fit every model on `train` and return its predictions for `test`."""
    labels = train["target"].to_numpy() if y_train is None else y_train
    result = test[ID_COLUMNS].copy()
    result["rule_of_thumb"] = rule_of_thumb(train, test)
    for name, model in make_models(params).items():
        model.fit(train[predictors].to_numpy(dtype=float), labels)
        result[name] = model.predict_proba(test[predictors].to_numpy(dtype=float))[:, 1]
    return result


def backtest(table: pd.DataFrame, predictors: list[str], params: dict, shuffle_target: bool = False) -> pd.DataFrame:
    """Expanding-window predictions for every test year. `shuffle_target` is used by the leakage test."""
    rng = np.random.default_rng(SEED)
    pieces = []
    for test_year in range(FIRST_TEST_YEAR, int(table["year"].max()) + 1):
        train, test = split(table, test_year)
        if test.empty:
            continue
        shuffled = rng.permutation(train["target"].to_numpy()) if shuffle_target else None
        result = predict_fold(train, test, predictors, params, shuffled)
        result["train_rows"], result["train_last_year"] = len(train), int(train["year"].max())
        pieces.append(result)
    return pd.concat(pieces, ignore_index=True)


def block_cv(table: pd.DataFrame, predictors: list[str], params: dict) -> pd.DataFrame:
    """Time-block cross-validation over the whole sample (see the module docstring)."""
    pieces = [predict_fold(table.iloc[train_idx], table.iloc[test_idx], predictors, params)
              for train_idx, test_idx in time_blocks(table["year"])]
    return pd.concat(pieces, ignore_index=True)


def shuffled_cv(table: pd.DataFrame, predictors: list[str], params: dict) -> pd.DataFrame:
    """Random 5-fold cross-validation. Leaks across time by design; a reference point only."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(table, table["target"])
    pieces = [predict_fold(table.iloc[train_idx], table.iloc[test_idx], predictors, params)
              for train_idx, test_idx in folds]
    return pd.concat(pieces, ignore_index=True)


def evaluate(predictions: pd.DataFrame, periods: list[tuple[int, int]], draws: int = 1000) -> pd.DataFrame:
    """AUC with a confidence interval from resampling whole countries, sub-period AUCs, Brier score."""
    rng = np.random.default_rng(SEED)
    by_country = {iso3: frame for iso3, frame in predictions.groupby("iso3")}
    codes = list(by_country)
    boot = {m: [] for m in MODELS}
    for _ in range(draws):
        sample = pd.concat([by_country[c] for c in rng.choice(codes, size=len(codes), replace=True)])
        for m in MODELS:
            boot[m].append(roc_auc_score(sample["target"], sample[m]))
    rows = []
    for m in MODELS:
        draws_m, draws_logit = np.array(boot[m]), np.array(boot["logit"])
        row = {"model": m,
               "auc": roc_auc_score(predictions["target"], predictions[m]),
               "auc_ci_low": np.percentile(draws_m, 2.5), "auc_ci_high": np.percentile(draws_m, 97.5),
               "share_of_draws_beating_logit": np.nan if m == "logit" else float((draws_m > draws_logit).mean())}
        for first, last in periods:
            part = predictions[predictions["year"].between(first, last)]
            row[f"auc_{first}_{last}"] = roc_auc_score(part["target"], part[m])
        row["brier"] = brier_score_loss(predictions["target"], predictions[m].clip(0, 1))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_roc(predictions: pd.DataFrame, metrics: pd.DataFrame, title: str, path: Path) -> None:
    labels = {"rule_of_thumb": "Rule of thumb", "logit": "Logit", "random_forest": "Random forest",
              "balanced_rf": "Balanced random forest", "xgboost": "XGBoost"}
    colours = {"rule_of_thumb": "#9a9a9a", "logit": "#1f77b4", "random_forest": "#2ca02c",
               "balanced_rf": "#17becf", "xgboost": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for row in metrics.itertuples():
        fpr, tpr, _ = roc_curve(predictions["target"], predictions[row.model])
        ax.plot(fpr, tpr, color=colours[row.model], lw=1.8, label=f"{labels[row.model]} (AUC {row.auc:.2f})")
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls=":")
    ax.set_xlabel("False alarm rate (share of calm country-years flagged)")
    ax.set_ylabel("Hit rate (share of pre-crisis country-years flagged)")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_metrics(metrics: pd.DataFrame) -> None:
    period_columns = [c for c in metrics.columns if c.startswith("auc_") and c[4:8].isdigit()]
    print(f"   {'model':16s} {'AUC':>6s}  {'95% interval':>14s}  "
          + " ".join(f"{c[4:].replace('_', '-'):>9s}" for c in period_columns) + f"  {'Brier':>7s}  beats logit in")
    for _, row in metrics.iterrows():
        beats = "" if np.isnan(row["share_of_draws_beating_logit"]) else f"{row['share_of_draws_beating_logit']:.0%} of resamples"
        print(f"   {row['model']:16s} {row['auc']:>6.3f}  [{row['auc_ci_low']:.3f}, {row['auc_ci_high']:.3f}]  "
              + " ".join(f"{row[c]:>9.3f}" for c in period_columns) + f"  {row['brier']:>7.4f}  {beats}")


# --------------------------------------------------------------------------- main
def run_version(version: str) -> bool:
    table = pd.read_parquet(MODEL_DIR / f"model_table_{version}.parquet").sort_values(["year", "iso3"]).reset_index(drop=True)
    predictors = [c for c in table.columns if c not in ID_COLUMNS]
    last_year = int(table["year"].max())
    print(f"\n================ variable set {version}: {len(table):,} rows, "
          f"{int(table['target'].sum())} positives, {len(predictors)} predictors ================")

    print(f"1. tuning on the first training window (years up to {FIRST_TEST_YEAR - HORIZON}) ...")
    params, tuning = tune(table, predictors)
    tuning.to_csv(OUTPUT / f"tuning_{version}.csv", index=False)
    (OUTPUT / f"tuned_params_{version}.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    for family, chosen in params.items():
        scores = tuning[tuning["model"] == family]["mean_auc"]
        print(f"   {family:14s} {chosen}  (cross-validated AUC {scores.max():.3f}; grid range {scores.min():.3f}-{scores.max():.3f})")

    print(f"2. backtest, test years {FIRST_TEST_YEAR}-{last_year} ...")
    backtested = backtest(table, predictors, params)
    print(f"   {len(backtested):,} predictions, {int(backtested['target'].sum())} positives, "
          f"{backtested.loc[backtested['target'] == 1, ['iso3', 'crisis_start_ahead']].drop_duplicates().shape[0]} crises")
    gap_ok = bool(((backtested["year"] - backtested["train_last_year"]) >= HORIZON).all())
    print(f"   {'PASS' if gap_ok else 'FAIL'}  every prediction comes from a model trained on years at least {HORIZON} before it")
    shuffled = backtest(table, predictors, params, shuffle_target=True)
    placebo = {m: roc_auc_score(shuffled["target"], shuffled[m]) for m in MODELS[1:]}
    placebo_ok = all(0.38 < v < 0.62 for v in placebo.values())
    print(f"   {'PASS' if placebo_ok else 'FAIL'}  with shuffled training labels the AUC is near chance: "
          + ", ".join(f"{m} {v:.2f}" for m, v in placebo.items()))
    backtest_metrics = evaluate(backtested, [(FIRST_TEST_YEAR, 2007), (2008, last_year)])
    print_metrics(backtest_metrics)

    print("3. time-block cross-validation, whole sample ...")
    crossval = block_cv(table, predictors, params)
    crossval_metrics = evaluate(crossval, [(int(table["year"].min()), 1999), (2000, last_year)])
    print_metrics(crossval_metrics)

    print("4. shuffled cross-validation (reference point, not a valid test) ...")
    shuffled_folds = shuffled_cv(table, predictors, params)
    shuffled_metrics = evaluate(shuffled_folds, [(int(table["year"].min()), 1999), (2000, last_year)])
    print_metrics(shuffled_metrics)

    metrics = pd.concat([backtest_metrics.assign(evaluation="backtest"),
                         crossval_metrics.assign(evaluation="block_cv"),
                         shuffled_metrics.assign(evaluation="shuffled_cv")], ignore_index=True)
    metrics.insert(0, "version", version)
    metrics.round(4).to_csv(OUTPUT / f"metrics_{version}.csv", index=False)
    predictions = pd.concat([backtested.assign(evaluation="backtest"), crossval.assign(evaluation="block_cv"),
                             shuffled_folds.assign(evaluation="shuffled_cv")])
    long = predictions.melt(id_vars=ID_COLUMNS + ["evaluation"], value_vars=MODELS,
                            var_name="model", value_name="probability")
    long.to_parquet(MODEL_DIR / f"predictions_{version}.parquet", index=False)
    plot_roc(backtested, backtest_metrics, f"Backtest {FIRST_TEST_YEAR}-{last_year}, variable set {version}",
             OUTPUT / f"roc_curves_{version}.png")
    return gap_ok and placebo_ok


def compare() -> None:
    """Both variable sets and all three evaluations side by side."""
    files = sorted(OUTPUT.glob("metrics_v*.csv"))
    if not files:
        return
    both = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    both.to_csv(OUTPUT / "model_comparison.csv", index=False)
    order = ["shuffled_cv", "block_cv", "backtest"]
    titles = {"shuffled_cv": "shuffled CV (invalid)", "block_cv": "time-block CV", "backtest": "real-time backtest"}
    wide = both.pivot_table(index="model", columns=["evaluation", "version"], values="auc").reindex(MODELS)
    versions = sorted(both["version"].unique())
    print("\n================ AUC by evaluation design ================")
    print(f"   {'':16s} " + "   ".join(f"{titles[e]:>{9 * len(versions) - 1}s}" for e in order))
    print(f"   {'model':16s} " + "   ".join(" ".join(f"{v:>8s}" for v in versions) for _ in order))
    for model, row in wide.iterrows():
        print(f"   {model:16s} " + "   ".join(" ".join(f"{row[(e, v)]:>8.3f}" for v in versions) for e in order))


def main(argv: list[str]) -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    OUTPUT.mkdir(exist_ok=True)
    if argv == ["compare"]:                 # only rebuild the comparison from saved metrics
        compare()
        return 0
    versions = argv or ["v1", "v2"]
    ok = all([run_version(version) for version in versions])
    compare()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
