"""Draw the project's figures from saved results.

Usage (from the project root):
    .venv/bin/python src/make_figures.py

Run src/run_backtest.py and src/explain.py first. Figures written to output/:
    auc_by_evaluation.png     the main result: AUC under three evaluation designs
    shap_importance_v2.png    which variables the full-sample model relies on
    shap_cases_v2.png         contribution breakdown for three country-years
    shap_interaction_v2.png   external debt matters more when U.S. real rates are high

Colours follow one fixed scheme: blue / orange / aqua for up to three series, blue and red
for "lowers" and "raises", grey ink for all text. The numbers behind every figure are in the
CSV files next to them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR, OUTPUT, REFERENCE = ROOT / "data" / "model", ROOT / "output", ROOT / "data" / "reference"

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "savefig.facecolor": SURFACE,
})

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


def clean(ax, grid_axis: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis)


# --------------------------------------------------------------------------- figure 1
def auc_by_evaluation() -> None:
    metrics = pd.read_csv(OUTPUT / "model_comparison.csv")
    designs = [("shuffled_cv", "Shuffled\ncross-validation\n(not a valid test)"),
               ("block_cv", "Time-block\ncross-validation"),
               ("backtest", "Real-time backtest\n2001-2023")]
    series = [("logit", "Logit", BLUE, -0.07), ("random_forest", "Random forest", ORANGE, 0.0),
              ("xgboost", "XGBoost", AQUA, 0.07)]
    titles = {"v1": "Variable set 1: first, self-chosen list (24 variables)",
              "v2": "Variable set 2: aligned with the IMF's list (20 variables)"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True)
    for ax, version in zip(axes, ["v1", "v2"]):
        part = metrics[metrics["version"] == version].set_index(["evaluation", "model"])
        ax.axhline(0.5, color=AXIS, lw=1.0, zorder=1)
        for model, label, colour, shift in series:
            xs = np.arange(len(designs)) + shift
            rows = [part.loc[(design, model)] for design, _ in designs]
            ax.vlines(xs, [r["auc_ci_low"] for r in rows], [r["auc_ci_high"] for r in rows],
                      color=colour, lw=1.4, alpha=0.45, zorder=2)
            ax.plot(xs, [r["auc"] for r in rows], color=colour, lw=2, marker="o", ms=8.5,
                    mec=SURFACE, mew=2, solid_capstyle="round", label=label, zorder=3)
        ax.set_xticks(range(len(designs)), [text for _, text in designs])
        ax.set_xlim(-0.45, 2.45)
        ax.set_ylim(0.30, 0.95)
        ax.set_title(titles[version])
        clean(ax)
    axes[0].set_ylabel("Area under the ROC curve (AUC)")
    axes[0].text(-0.42, 0.508, "0.5 = no better than chance", color=MUTED, fontsize=9, va="bottom")
    axes[0].legend(loc="upper right", ncols=1, handlelength=1.6)
    fig.suptitle("The same models look strong or useless depending on how they are tested",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.012, 0.915, "Out-of-sample AUC for predicting a systemic banking crisis one to two years ahead. "
             "Vertical bars: 95% intervals from resampling countries.", color=INK_2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT / "auc_by_evaluation.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- figure 2
def shap_importance() -> None:
    importance = pd.read_csv(OUTPUT / "shap_importance_v2.csv").iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.barh(range(len(importance)), importance["mean_abs_contribution"], height=0.56, color=BLUE)
    ax.set_yticks(range(len(importance)), [LABELS[v] for v in importance["variable"]])
    limit = importance["mean_abs_contribution"].max() * 1.02
    ax.set_xlim(0, limit * 1.16)
    first_use = ~importance.iloc[::-1]["imf_rank"].duplicated().iloc[::-1]      # keep the rank on the higher bar
    for y, rank in enumerate(importance["imf_rank"]):
        if not pd.isna(rank) and first_use.iloc[y]:
            ax.text(limit * 1.15, y, f"{int(rank)}", ha="right", va="center", color=INK_2, fontsize=9.5)
    ax.text(limit * 1.15, len(importance) - 0.25, "Rank in the\nIMF's model", ha="right", va="bottom", color=MUTED, fontsize=9)
    ax.set_xlabel("Average absolute contribution to the risk score (Shapley value)")
    ax.set_title("What the full-sample model relies on", pad=30)
    ax.text(0, 1.035, "Random forest fitted on all 6,058 country-years, 1980-2023. Descriptive, not a forecast test.",
            transform=ax.transAxes, color=INK_2, fontsize=9.5)
    clean(ax, grid_axis="x")
    ax.spines["left"].set_color(AXIS)
    fig.tight_layout()
    fig.savefig(OUTPUT / "shap_importance_v2.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- figure 3
def shap_cases() -> None:
    long = pd.read_parquet(MODEL_DIR / "shap_v2.parquet")
    scores = pd.read_parquet(MODEL_DIR / "risk_scores_v2.parquet")
    cases = [("THA", 1996, "crisis in 1997"), ("IRL", 2006, "crisis in 2008"), ("LBN", 2018, "crisis in 2019")]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.0))
    headings = []
    for ax, (iso3, year, note) in zip(axes, cases):
        part = long[(long["iso3"] == iso3) & (long["year"] == year)]
        part = part.reindex(part["contribution"].abs().sort_values(ascending=False).index).head(8).iloc[::-1]
        score = scores[(scores["iso3"] == iso3) & (scores["year"] == year)].iloc[0]
        colours = [RED if c > 0 else BLUE for c in part["contribution"]]
        ax.barh(range(len(part)), part["contribution"], height=0.56, color=colours)
        ax.axvline(0, color=AXIS, lw=1.0)
        names = [f"{LABELS[v]}" + ("" if np.isnan(x) else f"  ({x:,.1f})") for v, x in zip(part["variable"], part["value"])]
        ax.set_yticks(range(len(part)), names, fontsize=9)
        reach = max(0.16, float(part["contribution"].abs().max()) * 1.15)
        ax.set_xlim(-reach, reach)
        headings.append((f"{score['country']} {year}  ({note})",
                         f"Fitted score {score['score_fitted']:.2f}   |   sample average {score['score_average']:.2f}\n"
                         f"Out-of-sample score (time-block CV) {score['score_block_cv']:.2f}"))
        ax.set_xlabel("Contribution to the fitted risk score")
        clean(ax, grid_axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=RED), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    fig.legend(handles, ["Raises the score", "Lowers the score"], loc="upper right", ncols=2,
               bbox_to_anchor=(0.995, 0.995), handlelength=1.2, fontsize=9.5)
    fig.suptitle("Why the fitted model rates these country-years as risky - and what it said out of sample",
                 x=0.008, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.008, 0.915, "Eight largest contributions per case. The observed value of each variable is in brackets.",
             color=INK_2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.79), w_pad=2.5)
    fig.canvas.draw()                                  # needed to measure where the labels start
    for ax, (title, subtitle) in zip(axes, headings):
        left_px = min(label.get_window_extent().x0 for label in ax.get_yticklabels())
        left = ax.transAxes.inverted().transform((left_px, 0))[0]
        ax.text(left, 1.155, title, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
        ax.text(left, 1.03, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
    fig.savefig(OUTPUT / "shap_cases_v2.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- figure 4
def shap_interaction() -> None:
    long = pd.read_parquet(MODEL_DIR / "shap_v2.parquet")
    debt = long[(long["variable"] == "external_debt_exports") & long["value"].notna()][["iso3", "year", "value", "contribution"]]
    rate = long[long["variable"] == "us_real_10y"][["iso3", "year", "value"]].rename(columns={"value": "us_rate"})
    data = debt.merge(rate, on=["iso3", "year"])
    data = data[data["value"] <= 600]
    cut = float(rate["us_rate"].median())
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    for mask, colour, label in [(data["us_rate"] <= cut, BLUE, f"U.S. real 10-year yield at or below {cut:.1f}%"),
                                (data["us_rate"] > cut, ORANGE, f"U.S. real 10-year yield above {cut:.1f}%")]:
        ax.scatter(data.loc[mask, "value"], data.loc[mask, "contribution"], s=26, color=colour, alpha=0.55,
                   edgecolors=SURFACE, linewidths=0.6, label=label, zorder=2)
    ax.set_xlabel("External debt, percent of exports")
    ax.set_ylabel("Contribution of external debt to the fitted risk score")
    ax.set_title("External debt adds more risk when global interest rates are high", pad=24)
    ax.text(0, 1.03, f"Each dot is a country-year with external debt data ({len(data):,} dots; developing countries only).",
            transform=ax.transAxes, color=INK_2, fontsize=9.5)
    ax.legend(loc="lower right", handletextpad=0.3)
    clean(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT / "shap_interaction_v2.png", dpi=200)
    plt.close(fig)


def main() -> int:
    for draw in (auc_by_evaluation, shap_importance, shap_cases, shap_interaction):
        draw()
        print(f"drew {draw.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
