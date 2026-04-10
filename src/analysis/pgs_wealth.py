"""
OLS regressions: PGS -> IHS(wealth), with and without education control.

All models control for birth year, gender, and PCs 1-10.
Produces: table2_pgs_wealth.csv and data for Figure 1.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import (
    COVARIATES_BASIC,
    COVARIATES_EDUCATION,
    OUTPUT_MAP,
    PGS_LABELS,
)

logger = logging.getLogger(__name__)


def run_ols_wealth(
    df: pd.DataFrame,
    pgs_col: str,
    covariates: list[str],
    outcome: str = "ihs_wealth",
) -> dict:
    """
    Run OLS: outcome ~ PGS + covariates.

    Returns dict with coefficient, SE, t-stat, p-value, CI, N, R-squared.
    """
    cols = [outcome, pgs_col] + covariates
    sample = df[cols + ["hhid"]].dropna(subset=cols)
    assert not sample[cols].columns.duplicated().any(), f"Duplicate columns in wealth model for {pgs_col}"
    n = len(sample)
    logger.info("Wealth model %s (%s) sample: N=%d", pgs_col, outcome, n)
    if n < 100:
        logger.warning("Small sample (N=%d) for %s", n, pgs_col)
        return None

    y = sample[outcome]
    X = sm.add_constant(sample[[pgs_col] + covariates])
    model = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": sample["hhid"]},
    )

    ci = model.conf_int().loc[pgs_col]
    return {
        "beta": model.params[pgs_col],
        "se": model.bse[pgs_col],
        "t": model.tvalues[pgs_col],
        "p": model.pvalues[pgs_col],
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "n": n,
        "r2": model.rsquared,
        "r2_adj": model.rsquared_adj,
    }


def run_pgs_wealth(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run PGS -> wealth regressions for all PGS traits, with and without
    education control. Save Table 2 and Figure 1 data.
    """
    logger.info("Running PGS -> wealth regressions")

    results = []
    for trait, label in PGS_LABELS.items():
        pgs_col = f"pgs_{trait}"
        if pgs_col not in df.columns:
            logger.warning("PGS column %s not found; skipping", pgs_col)
            continue

        # Model A: without education control
        res_a = run_ols_wealth(df, pgs_col, COVARIATES_BASIC)
        if res_a:
            results.append({
                "trait": label,
                "model": "Without education",
                **res_a,
            })

        # Model B: with education control
        res_b = run_ols_wealth(df, pgs_col, COVARIATES_EDUCATION)
        if res_b:
            results.append({
                "trait": label,
                "model": "With education",
                **res_b,
            })

    results_df = pd.DataFrame(results)

    # Save Table 2
    outpath = OUTPUT_MAP["table2_pgs_wealth"]
    outpath.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(outpath, index=False, float_format="%.4f")
    logger.info("Saved: %s (N rows=%d)", outpath, len(results_df))

    # Compute attenuation (% change when adding education)
    for trait in PGS_LABELS.values():
        rows_a = results_df[
            (results_df["trait"] == trait) &
            (results_df["model"] == "Without education")
        ]
        rows_b = results_df[
            (results_df["trait"] == trait) &
            (results_df["model"] == "With education")
        ]
        if len(rows_a) == 1 and len(rows_b) == 1:
            beta_a = rows_a["beta"].values[0]
            beta_b = rows_b["beta"].values[0]
            if abs(beta_a) > 0.001:
                attenuation = (1 - beta_b / beta_a) * 100
                logger.info(
                    "%s: attenuation = %.1f%% when adding education",
                    trait, attenuation,
                )

    return {"table2": results_df}
