"""
Generate all figures for the manuscript.

Figure 1: Depression MR -- OLS vs IV comparison (wealth + claiming)
Figure 2: Claiming vs retirement age dissociation (dual panel)
Figure 3: PGS-wealth associations (paired bar chart, with/without education)
Figure 4: Summary heatmap (all PGS x all outcomes, depression-first)

All saved as PDF + PNG.
"""

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, PGS_LABELS

logger = logging.getLogger(__name__)

# NHB-appropriate styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOUR_OLS = "#2166AC"
COLOUR_IV = "#B2182B"
COLOUR_POSITIVE = "#2166AC"
COLOUR_NEGATIVE = "#B2182B"
COLOUR_NS = "#999999"
COLOUR_WITHOUT_EDU = "#2166AC"
COLOUR_WITH_EDU = "#B2182B"

# Narrative ordering: depression first, then neuroticism, then education/cognition
NARRATIVE_ORDER = ["Depression", "Neuroticism", "Education", "Cognition", "Well-being"]


def _save_figure(fig: plt.Figure, stem: str) -> None:
    """Save figure as PDF and PNG."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(path)
        logger.info("Saved: %s", path)
    plt.close(fig)


def figure1_depression_mr(sensitivity_df: pd.DataFrame) -> None:
    """
    Figure 1: Depression MR headline.
    Two panels showing OLS vs IV point estimates + 95% CIs for
    depression -> IHS(wealth) and depression -> SS claiming age.
    """
    if sensitivity_df.empty:
        logger.warning("No sensitivity data for Figure 1")
        return

    dep = sensitivity_df[sensitivity_df["trait"] == "Depression"]
    if dep.empty:
        logger.warning("No depression rows in sensitivity data for Figure 1")
        return

    outcomes = ["IHS(wealth)", "SS claiming age"]
    labels = ["IHS(wealth)", "SS claiming age (years)"]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    for i, (outcome, ylabel) in enumerate(zip(outcomes, labels)):
        ax = axes[i]
        subset = dep[dep["outcome"] == outcome]
        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        ols_row = subset[subset["method"] == "OLS"]
        iv_row = subset[subset["method"] == "IV"]

        y_pos = [1, 0]
        method_labels = ["OLS", "IV/2SLS"]
        colours = [COLOUR_OLS, COLOUR_IV]

        for j, (row_df, yp, colour) in enumerate(
            [(ols_row, y_pos[0], colours[0]), (iv_row, y_pos[1], colours[1])]
        ):
            if row_df.empty:
                continue
            row = row_df.iloc[0]
            coef = row["coef"]
            ci_lo = row["ci_lower"]
            ci_hi = row["ci_upper"]

            ax.barh(yp, coef, height=0.5, color=colour, edgecolor="white",
                    linewidth=0.5, alpha=0.85)
            ax.errorbar(coef, yp,
                        xerr=[[coef - ci_lo], [ci_hi - coef]],
                        fmt="none", color="black", capsize=4, linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(method_labels)
        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlabel(ylabel)
        panel_label = chr(ord("A") + i)
        ax.set_title(f"{panel_label}", fontweight="bold", loc="left")

    fig.suptitle("Depression and financial outcomes", fontsize=11, y=1.02)
    fig.tight_layout()
    _save_figure(fig, "figure1_depression_mr")


def figure2_claiming_vs_retirement(
    table3: pd.DataFrame,
    ed_table2: pd.DataFrame,
) -> None:
    """
    Figure 2: Claiming vs retirement age dissociation.
    Panel A: PGS -> SS claiming age (significant for 4 of 5 traits).
    Panel B: PGS -> retirement age (all null).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

    panels = [
        (ax1, table3, "SS claiming age (years per SD)", "A"),
        (ax2, ed_table2, "Retirement age (years per SD)", "B"),
    ]

    for ax, data, xlabel, panel_label in panels:
        if data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # Use the education-controlled specification
        plot_data = data[data["model"] == "With education"].copy()
        if plot_data.empty:
            plot_data = data.copy()

        plot_data = plot_data.sort_values("beta", ascending=True)
        y_pos = np.arange(len(plot_data))

        colours = [
            COLOUR_POSITIVE if row["beta"] > 0 and row["p"] < 0.05
            else COLOUR_NEGATIVE if row["beta"] < 0 and row["p"] < 0.05
            else COLOUR_NS
            for _, row in plot_data.iterrows()
        ]

        ax.barh(
            y_pos,
            plot_data["beta"],
            xerr=[(plot_data["beta"] - plot_data["ci_lower"]).values,
                  (plot_data["ci_upper"] - plot_data["beta"]).values],
            color=colours,
            edgecolor="white",
            linewidth=0.5,
            capsize=3,
            height=0.6,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_data["trait"].values)
        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{panel_label}", fontweight="bold", loc="left")

    fig.tight_layout()
    _save_figure(fig, "figure2_claiming_vs_retirement")


def figure3_pgs_wealth(table2: pd.DataFrame) -> None:
    """
    Figure 3: Paired bar chart showing PGS-wealth coefficients
    with and without education control.
    """
    if table2.empty:
        logger.warning("No data for Figure 3")
        return

    traits = list(PGS_LABELS.values())
    without_edu = table2[table2["model"] == "Without education"].set_index("trait")
    with_edu = table2[table2["model"] == "With education"].set_index("trait")

    common = [t for t in traits if t in without_edu.index and t in with_edu.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(common))
    width = 0.35

    ax.bar(
        x - width / 2,
        [without_edu.loc[t, "beta"] for t in common],
        width,
        label="Without education control",
        color=COLOUR_WITHOUT_EDU,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        [with_edu.loc[t, "beta"] for t in common],
        width,
        label="With education control",
        color=COLOUR_WITH_EDU,
        edgecolor="white",
        linewidth=0.5,
    )

    for i, t in enumerate(common):
        ci_a = (without_edu.loc[t, "ci_upper"] - without_edu.loc[t, "ci_lower"]) / 2
        ci_b = (with_edu.loc[t, "ci_upper"] - with_edu.loc[t, "ci_lower"]) / 2
        ax.errorbar(i - width / 2, without_edu.loc[t, "beta"], yerr=ci_a,
                     fmt="none", color="black", capsize=2, linewidth=0.8)
        ax.errorbar(i + width / 2, with_edu.loc[t, "beta"], yerr=ci_b,
                     fmt="none", color="black", capsize=2, linewidth=0.8)

    ax.set_ylabel("Coefficient on PGS (IHS wealth)")
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=30, ha="right")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    _save_figure(fig, "figure3_pgs_wealth")


def figure4_summary_heatmap(all_results: dict[str, pd.DataFrame]) -> None:
    """
    Figure 4: Summary heatmap of all PGS x all outcomes.
    Rows ordered by narrative emphasis (depression first).
    Colour = standardised coefficient or log(OR); asterisks = significance.
    """
    outcomes = ["IHS(wealth)", "SS claiming age", "Retirement age",
                "Stock (current)", "IRA (current)"]

    # Build matrix from available results using narrative ordering
    matrix = pd.DataFrame(np.nan, index=NARRATIVE_ORDER, columns=outcomes)
    pvals = pd.DataFrame(np.nan, index=NARRATIVE_ORDER, columns=outcomes)

    # Wealth (with education)
    if "table2" in all_results and not all_results["table2"].empty:
        t2 = all_results["table2"]
        t2_edu = t2[t2["model"] == "With education"]
        for _, row in t2_edu.iterrows():
            if row["trait"] in matrix.index:
                matrix.loc[row["trait"], "IHS(wealth)"] = row["beta"]
                pvals.loc[row["trait"], "IHS(wealth)"] = row["p"]

    # Claiming
    if "table3" in all_results and not all_results["table3"].empty:
        t3 = all_results["table3"]
        t3_edu = t3[t3["model"] == "With education"]
        for _, row in t3_edu.iterrows():
            if row["trait"] in matrix.index:
                matrix.loc[row["trait"], "SS claiming age"] = row["beta"]
                pvals.loc[row["trait"], "SS claiming age"] = row["p"]

    # Retirement
    if "ed_table2" in all_results and not all_results["ed_table2"].empty:
        ed2 = all_results["ed_table2"]
        ed2_edu = ed2[ed2["model"] == "With education"]
        for _, row in ed2_edu.iterrows():
            if row["trait"] in matrix.index:
                matrix.loc[row["trait"], "Retirement age"] = row["beta"]
                pvals.loc[row["trait"], "Retirement age"] = row["p"]

    # Investment (log OR)
    if "table4" in all_results and not all_results["table4"].empty:
        t4 = all_results["table4"]
        for _, row in t4.iterrows():
            outcome = row["outcome"]
            if row["trait"] in matrix.index and outcome in matrix.columns:
                matrix.loc[row["trait"], outcome] = np.log(row["or"])
                pvals.loc[row["trait"], outcome] = row["p"]

    # Drop columns that are all NaN
    matrix = matrix.dropna(axis=1, how="all")
    if matrix.empty:
        logger.warning("No data for Figure 4 heatmap")
        return

    pvals = pvals.loc[matrix.index, matrix.columns]

    fig, ax = plt.subplots(figsize=(6, 4))
    vmax = max(abs(matrix.min().min()), abs(matrix.max().max()))
    im = ax.imshow(
        matrix.values.astype(float),
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.iloc[i, j]
            pv = pvals.iloc[i, j]
            if pd.isna(val):
                continue
            stars = ""
            if pd.notna(pv):
                if pv < 0.001:
                    stars = "***"
                elif pv < 0.01:
                    stars = "**"
                elif pv < 0.05:
                    stars = "*"
            text_colour = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}{stars}", ha="center", va="center",
                    fontsize=7, color=text_colour)

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Coefficient / log(OR)")

    fig.tight_layout()
    _save_figure(fig, "figure4_summary_heatmap")


def generate_all_figures(all_results: dict[str, pd.DataFrame]) -> None:
    """Generate all manuscript figures from analysis results."""
    logger.info("Generating figures")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up old figure files from prior naming scheme
    for old_stem in ["figure1_pgs_wealth", "figure2_pgs_claiming",
                     "figure3_pgs_investment"]:
        for ext in ["pdf", "png"]:
            old_path = FIGURES_DIR / f"{old_stem}.{ext}"
            if old_path.exists():
                old_path.unlink()
                logger.info("Removed old figure: %s", old_path)

    # Figure 1: Depression MR (OLS vs IV)
    if "sensitivity" in all_results and not all_results["sensitivity"].empty:
        figure1_depression_mr(all_results["sensitivity"])

    # Figure 2: Claiming vs retirement dissociation
    if "table3" in all_results and "ed_table2" in all_results:
        figure2_claiming_vs_retirement(
            all_results["table3"],
            all_results.get("ed_table2", pd.DataFrame()),
        )

    # Figure 3: PGS -> wealth with/without education
    if "table2" in all_results:
        figure3_pgs_wealth(all_results["table2"])

    # Figure 4: Summary heatmap
    figure4_summary_heatmap(all_results)

    logger.info("All figures generated")
