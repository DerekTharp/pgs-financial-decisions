"""
Supplementary and sensitivity analyses.

- African-ancestry and Hispanic-ancestry replication
- Robustness to additional controls (self-rated health, marital status, income)
- Exclusion of respondents with possible cognitive impairment

Produces: supp_ancestry_replication.csv, ed_table3_robustness.csv,
supp_cognitive_exclusion.csv
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import (
    COVARIATES_EDUCATION,
    COVARIATES_ROBUSTNESS,
    OUTPUT_MAP,
    PGS_LABELS,
)
from src.data.merge import merge_pgs_rand

logger = logging.getLogger(__name__)


def _run_ols(
    df: pd.DataFrame,
    pgs_col: str,
    outcome: str,
    covariates: list[str],
) -> dict | None:
    """Run OLS and return results dict."""
    available_covs = [c for c in covariates if c in df.columns]
    cols = [outcome, pgs_col] + available_covs
    sample = df[cols].dropna()
    assert not sample.columns.duplicated().any(), f"Duplicate columns in supplementary OLS for {pgs_col} -> {outcome}"
    n = len(sample)
    if n < 50:
        return None

    y = sample[outcome]
    X = sm.add_constant(sample[[pgs_col] + available_covs])
    model = sm.OLS(y, X).fit(cov_type="HC3")

    ci = model.conf_int().loc[pgs_col]
    return {
        "beta": model.params[pgs_col],
        "se": model.bse[pgs_col],
        "t": model.tvalues[pgs_col],
        "p": model.pvalues[pgs_col],
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "ci_scale": "beta",
        "n": n,
        "r2": model.rsquared,
    }


def _run_logistic(
    df: pd.DataFrame,
    pgs_col: str,
    outcome: str,
    covariates: list[str],
) -> dict | None:
    """Run logistic regression and return results dict."""
    available_covs = [c for c in covariates if c in df.columns]
    cols = [outcome, pgs_col] + available_covs
    sample = df[cols].dropna()
    assert not sample.columns.duplicated().any(), f"Duplicate columns in supplementary logit for {pgs_col} -> {outcome}"
    n = len(sample)
    if n < 50 or (sample[outcome] == 1).sum() < 20:
        return None

    y = sample[outcome]
    X = sm.add_constant(sample[[pgs_col] + available_covs])
    try:
        model = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=100, cov_type="HC3")
    except Exception as exc:
        logger.warning(
            "Supplementary logit failed for %s -> %s (N=%d): %s",
            pgs_col, outcome, n, exc,
        )
        return None

    coef = model.params[pgs_col]
    ci = model.conf_int().loc[pgs_col]
    return {
        "beta": coef,
        "or": np.exp(coef),
        "se": model.bse[pgs_col],
        "z": model.tvalues[pgs_col],
        "p": model.pvalues[pgs_col],
        "ci_lower": np.exp(ci[0]),
        "ci_upper": np.exp(ci[1]),
        "ci_scale": "OR",
        "n": n,
    }


def run_ancestry_replication() -> pd.DataFrame:
    """
    Replicate primary analyses in African-ancestry and Hispanic-ancestry samples.
    Report directional consistency, not formal replication.
    """
    logger.info("Running ancestry replication analyses")

    outcomes_continuous = {
        "ihs_wealth": "IHS(wealth)",
        "ss_claiming_age": "SS claiming age",
    }
    outcomes_binary = {
        "stock_participation": "Stock participation",
        "ira_participation": "IRA participation",
    }

    rows = []
    for ancestry in ["AFR", "HIS"]:
        try:
            df = merge_pgs_rand(ancestry)
        except Exception as e:
            logger.warning("Failed to load %s sample: %s", ancestry, e)
            continue

        logger.info("Ancestry %s: N = %d", ancestry, len(df))

        for trait, label in PGS_LABELS.items():
            pgs_col = f"pgs_{trait}"
            if pgs_col not in df.columns:
                logger.info(
                    "%s: PGS %s not available for %s", ancestry, trait, ancestry
                )
                continue

            # Continuous outcomes
            for var, var_label in outcomes_continuous.items():
                if var not in df.columns:
                    continue
                res = _run_ols(df, pgs_col, var, COVARIATES_EDUCATION)
                if res:
                    rows.append({
                        "ancestry": ancestry,
                        "trait": label,
                        "outcome": var_label,
                        "type": "OLS",
                        **res,
                    })

            # Binary outcomes
            for var, var_label in outcomes_binary.items():
                if var not in df.columns:
                    continue
                res = _run_logistic(df, pgs_col, var, COVARIATES_EDUCATION)
                if res:
                    rows.append({
                        "ancestry": ancestry,
                        "trait": label,
                        "outcome": var_label,
                        "type": "Logistic",
                        **res,
                    })

    return pd.DataFrame(rows)


def run_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test robustness of primary results to additional controls
    (self-rated health, marital status, household income).
    """
    logger.info("Running robustness analyses with additional controls")

    rows = []
    for trait, label in PGS_LABELS.items():
        pgs_col = f"pgs_{trait}"
        if pgs_col not in df.columns:
            continue

        continuous_outcomes = {
            "ihs_wealth": "IHS(wealth)",
            "ss_claiming_age": "SS claiming age",
            "retirement_age": "Retirement age",
        }
        binary_outcomes = {
            "stock_participation": "Stock participation",
            "ira_participation": "IRA participation",
        }

        for model_name, covs in [
            ("Base (educ)", COVARIATES_EDUCATION),
            ("+ Health", COVARIATES_EDUCATION + ["shlt_latest"]),
            ("+ Marital", COVARIATES_EDUCATION + ["mstat_latest"]),
            ("+ Income", COVARIATES_EDUCATION + ["hitot_latest"]),
            ("All controls", COVARIATES_ROBUSTNESS),
        ]:
            for outcome_var, outcome_label in continuous_outcomes.items():
                if outcome_var not in df.columns:
                    continue
                res = _run_ols(df, pgs_col, outcome_var, covs)
                if res:
                    rows.append({
                        "trait": label,
                        "outcome": outcome_label,
                        "model": model_name,
                        "method": "OLS",
                        **res,
                    })

            for outcome_var, outcome_label in binary_outcomes.items():
                if outcome_var not in df.columns:
                    continue
                res = _run_logistic(df, pgs_col, outcome_var, covs)
                if res:
                    rows.append({
                        "trait": label,
                        "outcome": outcome_label,
                        "model": model_name,
                        "method": "Logistic",
                        **res,
                    })

    return pd.DataFrame(rows)


def run_cognitive_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude respondents with possible cognitive impairment (low cognition scores)
    and re-run primary analyses as sensitivity check.
    """
    logger.info("Running cognitive impairment exclusion analysis")

    # Exclude bottom 5% of cognition scores
    if "cogtot_avg" not in df.columns:
        logger.warning("Cognition variable not available; skipping exclusion")
        return pd.DataFrame()

    threshold = df["cogtot_avg"].quantile(0.05)
    df_excl = df[df["cogtot_avg"] > threshold].copy()
    n_excluded = len(df) - len(df_excl)
    logger.info(
        "Excluded %d respondents with cognition <= %.1f (bottom 5%%)",
        n_excluded, threshold,
    )

    rows = []
    for trait, label in PGS_LABELS.items():
        pgs_col = f"pgs_{trait}"
        if pgs_col not in df_excl.columns:
            continue

        for outcome_var, outcome_label in [
            ("ihs_wealth", "IHS(wealth)"),
            ("ss_claiming_age", "SS claiming age"),
        ]:
            # Full sample
            res_full = _run_ols(df, pgs_col, outcome_var, COVARIATES_EDUCATION)
            if res_full:
                rows.append({
                    "trait": label,
                    "outcome": outcome_label,
                    "sample": "Full",
                    **res_full,
                })

            # Excluding cognitively impaired
            res_excl = _run_ols(df_excl, pgs_col, outcome_var, COVARIATES_EDUCATION)
            if res_excl:
                rows.append({
                    "trait": label,
                    "outcome": outcome_label,
                    "sample": "Excl. cognitive impairment",
                    **res_excl,
                })

    return pd.DataFrame(rows)


def run_supplementary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run all supplementary analyses and save outputs.
    """
    logger.info("Running supplementary analyses")

    # Ancestry replication
    ancestry_rep = run_ancestry_replication()
    outpath_anc = OUTPUT_MAP["supp_ancestry_replication"]
    outpath_anc.parent.mkdir(parents=True, exist_ok=True)
    ancestry_rep.to_csv(outpath_anc, index=False, float_format="%.4f")
    logger.info("Saved: %s (N rows=%d)", outpath_anc, len(ancestry_rep))

    # Robustness to additional controls
    robustness = run_robustness(df)
    outpath_rob = OUTPUT_MAP["ed_table3_robustness"]
    outpath_rob.parent.mkdir(parents=True, exist_ok=True)
    robustness.to_csv(outpath_rob, index=False, float_format="%.4f")
    logger.info("Saved: %s (N rows=%d)", outpath_rob, len(robustness))

    # Cognitive exclusion
    cog_excl = run_cognitive_exclusion(df)
    outpath_cog = OUTPUT_MAP["supp_cognitive_exclusion"]
    outpath_cog.parent.mkdir(parents=True, exist_ok=True)
    cog_excl.to_csv(outpath_cog, index=False, float_format="%.4f")
    logger.info("Saved: %s (N rows=%d)", outpath_cog, len(cog_excl))

    return {
        "ancestry_replication": ancestry_rep,
        "robustness": robustness,
        "cognitive_exclusion": cog_excl,
    }
