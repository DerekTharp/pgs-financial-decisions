"""
OLS regressions: PGS -> retirement age.

Expected null result: PGS does not predict retirement age after controls.
Produces: ed_table2_pgs_retirement.csv (Extended Data Table 2).
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


def run_ols_retirement(
    df: pd.DataFrame,
    pgs_col: str,
    covariates: list[str],
) -> dict | None:
    """
    Run OLS: retirement_age ~ PGS + covariates.
    """
    outcome = "retirement_age"
    cols = [outcome, pgs_col] + covariates
    sample = df[cols].dropna()
    assert not sample.columns.duplicated().any(), f"Duplicate columns in retirement model for {pgs_col}"
    n = len(sample)
    logger.info("Retirement model %s sample: N=%d", pgs_col, n)
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


def run_pgs_retirement(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run PGS -> retirement age regressions for all PGS traits.
    Save Extended Data Table 2.
    """
    logger.info("Running PGS -> retirement age regressions")

    results = []
    for trait, label in PGS_LABELS.items():
        pgs_col = f"pgs_{trait}"
        if pgs_col not in df.columns:
            continue

        # Model A: without education
        res_a = run_ols_retirement(df, pgs_col, COVARIATES_BASIC)
        if res_a:
            results.append({
                "trait": label,
                "model": "Without education",
                **res_a,
            })

        # Model B: with education
        res_b = run_ols_retirement(df, pgs_col, COVARIATES_EDUCATION)
        if res_b:
            results.append({
                "trait": label,
                "model": "With education",
                **res_b,
            })

    results_df = pd.DataFrame(results)

    # Save Extended Data Table 2
    outpath = OUTPUT_MAP["ed_table2_pgs_retirement"]
    outpath.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(outpath, index=False, float_format="%.4f")
    logger.info("Saved: %s (N rows=%d)", outpath, len(results_df))

    # Log null finding
    primary = results_df[results_df["model"] == "With education"]
    n_sig = (primary["p"] < 0.05).sum()
    logger.info(
        "Retirement age: %d of %d PGS traits significant at p<0.05",
        n_sig, len(primary),
    )

    return {"ed_table2": results_df}
