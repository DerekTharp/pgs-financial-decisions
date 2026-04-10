"""
Mendelian randomisation / instrumental variables (2SLS) analysis.

For each trait-outcome pair:
  - First stage: PGS -> trait (education years, cognition, CESD)
  - Second stage: instrumented trait -> financial outcome
  - Report first-stage F, 2SLS coefficient, SE, CI

MR models control only for pre-treatment variables (birth year, gender, PCs).
Do NOT control for traits influenced by the PGS (e.g., education when using
education PGS) as this would block the causal pathway.

Produces: table5_mr_estimates.csv
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from src.config import (
    COVARIATES_MR,
    FIRST_STAGE_F_THRESHOLD,
    OUTPUT_MAP,
    PGS_LABELS,
)

logger = logging.getLogger(__name__)

# Trait-to-endogenous-variable mapping
TRAIT_ENDOGENOUS = {
    "education":    "raedyrs",
    "cognition":    "cogtot_avg",
    "depression":   "cesd_avg",
}

# Financial outcomes for MR
MR_OUTCOMES = {
    "ihs_wealth":         "IHS(wealth)",
    "ss_claiming_age":    "SS claiming age",
    "stock_participation": "Stock participation",
    "ira_participation":  "IRA participation",
}


def prepare_model_sample(
    df: pd.DataFrame,
    required_cols: list[str],
    label: str,
    min_n: int = 100,
) -> pd.DataFrame | None:
    """Build a common estimation sample and log the drop from the full frame."""
    # Keep hhid for clustering even though it's not a model variable
    keep_cols = required_cols + (["hhid"] if "hhid" in df.columns else [])
    sample = df[keep_cols].dropna(subset=required_cols)
    assert not sample[required_cols].columns.duplicated().any(), f"Duplicate columns in sample for {label}"
    n = len(sample)
    logger.info("%s sample: N = %d (dropped %d)", label, n, len(df) - n)
    if n < min_n:
        logger.warning("Insufficient sample for %s: N=%d", label, n)
        return None
    return sample


def run_first_stage(
    sample: pd.DataFrame,
    pgs_col: str,
    endogenous: str,
    covariates: list[str],
) -> dict | None:
    """
    Run first-stage OLS: endogenous trait ~ PGS + covariates.
    Report coefficient, F-statistic, R-squared.
    """
    n = len(sample)

    y = sample[endogenous]
    X = sm.add_constant(sample[[pgs_col] + covariates])
    if "hhid" in sample.columns:
        model = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": sample["hhid"]},
        )
    else:
        model = sm.OLS(y, X).fit(cov_type="HC3")

    # Partial F-statistic for the instrument (robust Wald)
    f_stat = model.tvalues[pgs_col] ** 2

    return {
        "first_stage_coef": model.params[pgs_col],
        "first_stage_se": model.bse[pgs_col],
        "first_stage_t": model.tvalues[pgs_col],
        "first_stage_f": f_stat,
        "first_stage_p": model.pvalues[pgs_col],
        "first_stage_r2": model.rsquared,
        "first_stage_n": n,
    }


def run_iv_2sls(
    sample: pd.DataFrame,
    pgs_col: str,
    endogenous: str,
    outcome: str,
    covariates: list[str],
) -> dict | None:
    """
    Run IV/2SLS: outcome ~ [endogenous ~ PGS] + covariates.

    Uses linearmodels.iv.IV2SLS for proper 2SLS estimation with correct
    standard errors.
    """
    n = len(sample)

    y = sample[outcome]
    # Exogenous controls (including constant via add_constant)
    exog = sm.add_constant(sample[covariates])
    # Endogenous variable
    endog = sample[[endogenous]]
    # Instruments
    instruments = sample[[pgs_col]]

    try:
        clusters = sample["hhid"] if "hhid" in sample.columns else None
        cov_args = (
            {"cov_type": "clustered", "clusters": clusters}
            if clusters is not None
            else {"cov_type": "robust"}
        )
        model = IV2SLS(y, exog, endog, instruments).fit(**cov_args)
    except Exception as e:
        logger.warning("IV2SLS failed for %s -> %s -> %s: %s",
                       pgs_col, endogenous, outcome, e)
        return None

    # Extract results for the endogenous variable
    ci = model.conf_int().loc[endogenous]

    # First-stage F from the model
    try:
        first_stage_diag = model.first_stage
        fs_f = first_stage_diag.diagnostics["f.stat"].iloc[0]
    except Exception:
        # Fall back to manual calculation
        fs_f = np.nan

    return {
        "iv_coef": model.params[endogenous],
        "iv_se": model.std_errors[endogenous],
        "iv_t": model.tstats[endogenous],
        "iv_p": model.pvalues[endogenous],
        "iv_ci_lower": ci.iloc[0],
        "iv_ci_upper": ci.iloc[1],
        "iv_n": n,
        "first_stage_f_from_model": fs_f,
    }


def run_reduced_form(
    sample: pd.DataFrame,
    pgs_col: str,
    outcome: str,
    covariates: list[str],
) -> dict | None:
    """
    Run reduced-form OLS: outcome ~ PGS + covariates.
    Valid regardless of exclusion restriction assumptions.
    """
    n = len(sample)

    y = sample[outcome]
    X = sm.add_constant(sample[[pgs_col] + covariates])
    if "hhid" in sample.columns:
        model = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": sample["hhid"]},
        )
    else:
        model = sm.OLS(y, X).fit(cov_type="HC3")

    ci = model.conf_int().loc[pgs_col]
    return {
        "rf_coef": model.params[pgs_col],
        "rf_se": model.bse[pgs_col],
        "rf_t": model.tvalues[pgs_col],
        "rf_p": model.pvalues[pgs_col],
        "rf_ci_lower": ci[0],
        "rf_ci_upper": ci[1],
        "rf_n": n,
        "rf_r2": model.rsquared,
    }


def run_mr_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run full MR/IV analysis for each trait-outcome pair.
    Save Table 5 (MR estimates) with first-stage F, reduced-form, and IV results.
    """
    logger.info("Running Mendelian randomisation / IV analysis")

    results = []
    for trait, trait_label in PGS_LABELS.items():
        pgs_col = f"pgs_{trait}"
        endogenous = TRAIT_ENDOGENOUS.get(trait)

        if pgs_col not in df.columns:
            continue
        if endogenous is None or endogenous not in df.columns:
            logger.info("Skipping MR for %s: no endogenous variable", trait)
            continue

        for outcome_var, outcome_label in MR_OUTCOMES.items():
            if outcome_var not in df.columns:
                continue

            sample = prepare_model_sample(
                df,
                [outcome_var, endogenous, pgs_col] + COVARIATES_MR,
                label=f"MR {trait_label} -> {outcome_label}",
            )
            if sample is None:
                continue

            fs = run_first_stage(sample, pgs_col, endogenous, COVARIATES_MR)
            assert fs is not None
            strong_instrument = fs["first_stage_f"] > FIRST_STAGE_F_THRESHOLD

            row = {
                "trait": trait_label,
                "endogenous": endogenous,
                "outcome": outcome_label,
                **fs,
            }

            # Reduced form (always valid)
            rf = run_reduced_form(sample, pgs_col, outcome_var, COVARIATES_MR)
            if rf:
                row.update(rf)

            # IV/2SLS (only if instrument is strong)
            if strong_instrument:
                iv = run_iv_2sls(
                    sample, pgs_col, endogenous, outcome_var, COVARIATES_MR,
                )
                if iv:
                    row.update(iv)
                else:
                    row["iv_coef"] = np.nan
                    logger.warning(
                        "IV estimation failed for %s -> %s -> %s",
                        trait_label, endogenous, outcome_label,
                    )
            else:
                row["iv_coef"] = np.nan
                logger.info(
                    "Weak instrument (F=%.1f): reporting only reduced-form "
                    "for %s -> %s",
                    fs["first_stage_f"], trait_label, outcome_label,
                )

            results.append(row)

    results_df = pd.DataFrame(results)

    # Save Table 5
    outpath = OUTPUT_MAP["table5_mr_estimates"]
    outpath.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(outpath, index=False, float_format="%.4f")
    logger.info("Saved: %s (N rows=%d)", outpath, len(results_df))

    return {"table5": results_df}
