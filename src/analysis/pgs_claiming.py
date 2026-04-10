"""
OLS regressions: PGS -> Social Security claiming age.

All models control for birth year, gender, education, and PCs 1-10.
Produces: table3_pgs_claiming.csv and data for Figure 2.
"""

import logging

import pandas as pd
import statsmodels.api as sm

from src.config import (
    COVARIATES_BASIC,
    COVARIATES_EDUCATION,
    OUTPUT_MAP,
    PGS_LABELS,
)

logger = logging.getLogger(__name__)


def run_ols_claiming(
    df: pd.DataFrame,
    pgs_col: str,
    covariates: list[str],
) -> dict | None:
    """
    Run OLS: ss_claiming_age ~ PGS + covariates.

    Returns dict with coefficient, SE, t-stat, p-value, CI, N, R-squared.
    """
    outcome = "ss_claiming_age"
    cols = [outcome, pgs_col] + covariates
    sample = df[cols].dropna()
    assert not sample.columns.duplicated().any(), f"Duplicate columns in claiming model for {pgs_col}"
    n = len(sample)
    logger.info("Claiming model %s sample: N=%d", pgs_col, n)
    if n < 100:
        logger.warning("Small sample (N=%d) for %s", n, pgs_col)
        return None

    y = sample[outcome]
    X = sm.add_constant(sample[[pgs_col] + covariates])
    model = sm.OLS(y, X).fit(cov_type="HC3")

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


def run_pgs_claiming(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run PGS -> SS claiming age regressions for all PGS traits.
    Reports both with and without education control.
    Save Table 3 and Figure 2 data.
    """
    logger.info("Running PGS -> SS claiming age regressions")

    results = []
    for trait, label in PGS_LABELS.items():
        pgs_col = f"pgs_{trait}"
        if pgs_col not in df.columns:
            continue

        # Model A: without education
        res_a = run_ols_claiming(df, pgs_col, COVARIATES_BASIC)
        if res_a:
            results.append({
                "trait": label,
                "model": "Without education",
                **res_a,
            })

        # Model B: with education (primary specification)
        res_b = run_ols_claiming(df, pgs_col, COVARIATES_EDUCATION)
        if res_b:
            results.append({
                "trait": label,
                "model": "With education",
                **res_b,
            })

    results_df = pd.DataFrame(results)

    # Save Table 3
    outpath = OUTPUT_MAP["table3_pgs_claiming"]
    outpath.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(outpath, index=False, float_format="%.4f")
    logger.info("Saved: %s (N rows=%d)", outpath, len(results_df))

    # Log key findings
    primary = results_df[results_df["model"] == "With education"]
    for _, row in primary.iterrows():
        sig = "***" if row["p"] < 0.001 else ("**" if row["p"] < 0.01 else ("*" if row["p"] < 0.05 else ""))
        logger.info(
            "SS claiming: %s PGS -> %.3f years (p=%.4f) %s",
            row["trait"], row["beta"], row["p"], sig,
        )

    return {"table3": results_df}
