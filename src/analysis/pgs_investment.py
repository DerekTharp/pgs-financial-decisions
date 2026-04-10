"""
Logistic regressions: PGS -> stock participation and IRA participation.

Reports odds ratios with 95% CIs.
Produces: table4_pgs_investment.csv and data for Figure 3.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import (
    COVARIATES_EDUCATION,
    OUTPUT_MAP,
    PGS_LABELS,
)

logger = logging.getLogger(__name__)

# Investment outcomes to analyse
INVESTMENT_OUTCOMES = {
    "stock_participation": "Stock (current)",
    "ira_participation": "IRA (current)",
    "ever_stock": "Stock (ever)",
    "ever_ira": "IRA (ever)",
}


def run_logistic(
    df: pd.DataFrame,
    pgs_col: str,
    outcome: str,
    covariates: list[str],
) -> dict | None:
    """
    Run logistic regression: outcome ~ PGS + covariates.

    Returns dict with OR, CI, z-stat, p-value, N.
    """
    cols = [outcome, pgs_col] + covariates
    sample = df[cols].dropna()
    assert not sample.columns.duplicated().any(), f"Duplicate columns in investment model for {pgs_col} -> {outcome}"

    # Require both outcomes present
    n_pos = (sample[outcome] == 1).sum()
    n_neg = (sample[outcome] == 0).sum()
    if n_pos < 50 or n_neg < 50:
        logger.warning(
            "Insufficient variation for %s (pos=%d, neg=%d)",
            outcome, n_pos, n_neg,
        )
        return None

    n = len(sample)
    logger.info("Investment model %s -> %s sample: N=%d", pgs_col, outcome, n)
    y = sample[outcome]
    X = sm.add_constant(sample[[pgs_col] + covariates])
    try:
        model = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=100, cov_type="HC3")
    except Exception as exc:
        logger.warning(
            "Logit failed for %s -> %s (N=%d): %s",
            pgs_col, outcome, n, exc,
        )
        return None

    coef = model.params[pgs_col]
    ci = model.conf_int().loc[pgs_col]

    return {
        "or": np.exp(coef),
        "or_ci_lower": np.exp(ci[0]),
        "or_ci_upper": np.exp(ci[1]),
        "coef": coef,
        "se": model.bse[pgs_col],
        "z": model.tvalues[pgs_col],
        "p": model.pvalues[pgs_col],
        "n": n,
        "pseudo_r2": model.prsquared,
    }


def run_pgs_investment(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run PGS -> investment participation regressions (logistic) for all
    PGS traits and investment outcomes. Save Table 4 and Figure 3 data.
    """
    logger.info("Running PGS -> investment participation regressions")

    results = []
    for trait, trait_label in PGS_LABELS.items():
        pgs_col = f"pgs_{trait}"
        if pgs_col not in df.columns:
            continue

        for outcome_var, outcome_label in INVESTMENT_OUTCOMES.items():
            if outcome_var not in df.columns:
                continue

            res = run_logistic(df, pgs_col, outcome_var, COVARIATES_EDUCATION)
            if res:
                results.append({
                    "trait": trait_label,
                    "outcome": outcome_label,
                    **res,
                })

    results_df = pd.DataFrame(results)

    # Save Table 4
    outpath = OUTPUT_MAP["table4_pgs_investment"]
    outpath.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(outpath, index=False, float_format="%.4f")
    logger.info("Saved: %s (N rows=%d)", outpath, len(results_df))

    # Log key findings
    for _, row in results_df.iterrows():
        sig = "***" if row["p"] < 0.001 else ("**" if row["p"] < 0.01 else ("*" if row["p"] < 0.05 else ""))
        logger.info(
            "%s -> %s: OR=%.3f [%.3f, %.3f] p=%.4f %s",
            row["trait"], row["outcome"],
            row["or"], row["or_ci_lower"], row["or_ci_upper"],
            row["p"], sig,
        )

    return {"table4": results_df}
