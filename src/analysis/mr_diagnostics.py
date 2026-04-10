"""
MR diagnostic tests and sensitivity analyses.

- First-stage F-statistics for all instruments
- OLS vs IV comparison using the same outcome-specific samples

Produces: table6_first_stage.csv, supp_mr_sensitivity.csv
"""

import logging

import pandas as pd
import statsmodels.api as sm

from src.config import (
    COVARIATES_MR,
    FIRST_STAGE_F_THRESHOLD,
    OUTPUT_MAP,
    PGS_LABELS,
)
from src.analysis.mr_analysis import MR_OUTCOMES, TRAIT_ENDOGENOUS, prepare_model_sample

logger = logging.getLogger(__name__)


def compute_first_stage_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute first-stage diagnostics for each PGS instrument.

    Reports: coefficient, SE, t, F, p, partial R-squared, N.
    """
    rows = []
    for trait, label in PGS_LABELS.items():
        pgs_col = f"pgs_{trait}"
        endogenous = TRAIT_ENDOGENOUS.get(trait)

        if pgs_col not in df.columns or endogenous is None:
            continue
        if endogenous not in df.columns:
            continue

        cols = [endogenous, pgs_col] + COVARIATES_MR
        sample = df[cols].dropna()
        n = len(sample)
        if n < 100:
            continue

        y = sample[endogenous]
        X_full = sm.add_constant(sample[[pgs_col] + COVARIATES_MR])
        X_restricted = sm.add_constant(sample[COVARIATES_MR])

        model_full = sm.OLS(y, X_full).fit(cov_type="HC3")
        model_restricted = sm.OLS(y, X_restricted).fit(cov_type="HC3")

        # Partial R-squared
        r2_full = model_full.rsquared
        r2_restricted = model_restricted.rsquared
        partial_r2 = (r2_full - r2_restricted) / (1 - r2_restricted)

        # F-statistic (equals t^2 for single instrument)
        f_stat = model_full.tvalues[pgs_col] ** 2

        rows.append({
            "trait": label,
            "endogenous": endogenous,
            "coef": model_full.params[pgs_col],
            "se": model_full.bse[pgs_col],
            "t": model_full.tvalues[pgs_col],
            "f_stat": f_stat,
            "p": model_full.pvalues[pgs_col],
            "partial_r2": partial_r2,
            "r2_full": r2_full,
            "r2_restricted": r2_restricted,
            "n": n,
            "strong_instrument": f_stat > FIRST_STAGE_F_THRESHOLD,
        })

    return pd.DataFrame(rows)


def compute_ols_vs_iv_comparison(
    df: pd.DataFrame,
    mr_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compare OLS and IV estimates for the same trait-outcome pair using the
    same common estimation sample.
    """
    if mr_results is None or mr_results.empty:
        return pd.DataFrame()

    outcome_lookup = {label: var for var, label in MR_OUTCOMES.items()}
    rows = []
    for _, mr_row in mr_results.iterrows():
        trait_label = mr_row["trait"]
        outcome_label = mr_row["outcome"]
        trait = next((key for key, value in PGS_LABELS.items() if value == trait_label), None)
        outcome_var = outcome_lookup.get(outcome_label)
        if trait is None or outcome_var is None:
            continue

        endogenous = TRAIT_ENDOGENOUS.get(trait)
        pgs_col = f"pgs_{trait}"
        if endogenous is None or endogenous not in df.columns or pgs_col not in df.columns:
            continue

        ols_covariates = [cov for cov in COVARIATES_MR if cov != endogenous]
        sample = prepare_model_sample(
            df,
            [outcome_var, endogenous, pgs_col] + ols_covariates,
            label=f"OLS/IV comparison {trait_label} -> {outcome_label}",
        )
        if sample is None:
            continue

        y = sample[outcome_var]
        X = sm.add_constant(sample[[endogenous] + ols_covariates])
        ols_model = sm.OLS(y, X).fit(cov_type="HC3")

        ols_ci = ols_model.conf_int().loc[endogenous]
        rows.append({
            "trait": trait_label,
            "outcome": outcome_label,
            "method": "OLS",
            "coef": ols_model.params[endogenous],
            "se": ols_model.bse[endogenous],
            "ci_lower": ols_ci[0],
            "ci_upper": ols_ci[1],
            "p": ols_model.pvalues[endogenous],
            "n": len(sample),
        })

        if pd.notna(mr_row.get("iv_coef", float("nan"))):
            rows.append({
                "trait": trait_label,
                "outcome": outcome_label,
                "method": "IV",
                "coef": mr_row["iv_coef"],
                "se": mr_row["iv_se"],
                "ci_lower": mr_row["iv_ci_lower"],
                "ci_upper": mr_row["iv_ci_upper"],
                "p": mr_row["iv_p"],
                "n": int(mr_row["iv_n"]),
            })

    return pd.DataFrame(rows)


def run_mr_diagnostics(
    df: pd.DataFrame,
    mr_results: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run all MR diagnostic analyses and save outputs.
    """
    logger.info("Running MR diagnostics")

    # Table 6: First-stage diagnostics
    first_stage = compute_first_stage_table(df)
    outpath_fs = OUTPUT_MAP["table6_first_stage"]
    outpath_fs.parent.mkdir(parents=True, exist_ok=True)
    first_stage.to_csv(outpath_fs, index=False, float_format="%.4f")
    logger.info("Saved: %s", outpath_fs)

    for _, row in first_stage.iterrows():
        logger.info(
            "First stage %s -> %s: F=%.1f, partial R2=%.4f, N=%d [%s]",
            row["trait"], row["endogenous"], row["f_stat"],
            row["partial_r2"], row["n"],
            "STRONG" if row["strong_instrument"] else "WEAK",
        )

    # Supplementary: OLS vs IV comparison
    sensitivity = compute_ols_vs_iv_comparison(df, mr_results)
    outpath_sens = OUTPUT_MAP["supp_mr_sensitivity"]
    outpath_sens.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(outpath_sens, index=False, float_format="%.4f")
    logger.info("Saved: %s", outpath_sens)

    return {"table6": first_stage, "sensitivity": sensitivity}
