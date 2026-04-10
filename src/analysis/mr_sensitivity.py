"""
MR sensitivity and pleiotropy tests.

1. Multi-instrument IV with overidentification (Sargan/Hansen J-test)
2. PGS-level Egger-style analysis (intercept test for directional pleiotropy)
3. Multivariable MR (control for correlated-trait pleiotropy)
4. Steiger directionality test (partial R-squared, adjusted for covariates)
5. Auxiliary PGS benchmark (height)

Produces: supp_mr_pleiotropy.csv
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from src.config import (
    ALT_PGS_SUFFIXES,
    COVARIATES_MR,
    OUTPUT_MAP,
    PGS_LABELS,
)
from src.analysis.mr_analysis import TRAIT_ENDOGENOUS

logger = logging.getLogger(__name__)

# Outcomes for sensitivity tests (wealth and claiming are the headline results)
SENSITIVITY_OUTCOMES = {
    "ihs_wealth": "IHS(wealth)",
    "ss_claiming_age": "SS claiming age",
}


def _get_instrument_cols(df: pd.DataFrame, trait: str) -> list[str]:
    """
    Return all available instrument columns for a trait: primary PGS + alts.
    Each column is a genuinely distinct PGS (alts exclude the primary suffix).
    """
    cols = []
    primary = f"pgs_{trait}"
    if primary in df.columns:
        cols.append(primary)
    if trait in ALT_PGS_SUFFIXES:
        for i in range(len(ALT_PGS_SUFFIXES[trait])):
            col = f"pgs_{trait}_alt{i}"
            if col in df.columns:
                cols.append(col)
    return cols


def run_overidentification(
    df: pd.DataFrame,
    trait: str,
    endogenous: str,
    outcome_var: str,
    outcome_label: str,
) -> dict | None:
    """
    Multi-instrument IV with Sargan/Hansen J-test for overidentification.
    Uses primary PGS + alternative PGS as instruments.
    Requires at least 2 instruments (1 overidentifying restriction).
    """
    instruments = _get_instrument_cols(df, trait)
    if len(instruments) < 2:
        return None

    required = [outcome_var, endogenous] + instruments + COVARIATES_MR
    keep = required + (["hhid"] if "hhid" in df.columns else [])
    sample = df[keep].dropna(subset=required)
    n = len(sample)
    if n < 100:
        return None

    y = sample[outcome_var]
    exog = sm.add_constant(sample[COVARIATES_MR])
    endog = sample[[endogenous]]
    inst = sample[instruments]

    try:
        clusters = sample["hhid"] if "hhid" in sample.columns else None
        cov_args = (
            {"cov_type": "clustered", "clusters": clusters}
            if clusters is not None
            else {"cov_type": "robust"}
        )
        model = IV2SLS(y, exog, endog, inst).fit(**cov_args)
    except Exception as e:
        logger.warning("Overidentification IV failed for %s -> %s: %s",
                       trait, outcome_label, e)
        return None

    try:
        overid = model.wooldridge_overid
        j_stat = overid.stat
        j_pval = overid.pval
    except Exception:
        j_stat = np.nan
        j_pval = np.nan

    return {
        "test": "Overidentification (J-test)",
        "trait": PGS_LABELS[trait],
        "outcome": outcome_label,
        "n_instruments": len(instruments),
        "iv_coef": model.params[endogenous],
        "iv_se": model.std_errors[endogenous],
        "iv_p": model.pvalues[endogenous],
        "j_stat": j_stat,
        "j_pval": j_pval,
        "n": n,
    }


def run_pgs_egger(
    df: pd.DataFrame,
    trait: str,
    endogenous: str,
    outcome_var: str,
    outcome_label: str,
) -> dict | None:
    """
    PGS-level Egger-style analysis using multiple PGS as instruments.

    Regresses each instrument's reduced-form coefficient on its first-stage
    coefficient, weighted by precision. The intercept captures directional
    pleiotropy. This is an adaptation of MR-Egger at the PGS level rather
    than the SNP level; with 3 instruments, statistical power is limited.
    """
    instruments = _get_instrument_cols(df, trait)
    if len(instruments) < 3:
        return None

    required = [outcome_var, endogenous] + instruments + COVARIATES_MR
    keep = required + (["hhid"] if "hhid" in df.columns else [])
    sample = df[keep].dropna(subset=required)
    n = len(sample)
    if n < 100:
        return None

    fs_coefs = []
    rf_coefs = []
    rf_ses = []

    for inst_col in instruments:
        X = sm.add_constant(sample[[inst_col] + COVARIATES_MR])

        if "hhid" in sample.columns:
            fs_model = sm.OLS(sample[endogenous], X).fit(
                cov_type="cluster", cov_kwds={"groups": sample["hhid"]},
            )
        else:
            fs_model = sm.OLS(sample[endogenous], X).fit(cov_type="HC3")
        fs_coefs.append(fs_model.params[inst_col])

        if "hhid" in sample.columns:
            rf_model = sm.OLS(sample[outcome_var], X).fit(
                cov_type="cluster", cov_kwds={"groups": sample["hhid"]},
            )
        else:
            rf_model = sm.OLS(sample[outcome_var], X).fit(cov_type="HC3")
        rf_coefs.append(rf_model.params[inst_col])
        rf_ses.append(rf_model.bse[inst_col])

    fs_coefs = np.array(fs_coefs)
    rf_coefs = np.array(rf_coefs)
    rf_ses = np.array(rf_ses)

    weights = 1.0 / (rf_ses ** 2)
    X_egger = sm.add_constant(fs_coefs)
    wls = sm.WLS(rf_coefs, X_egger, weights=weights).fit()

    params = np.asarray(wls.params)
    bse = np.asarray(wls.bse)
    pvals = np.asarray(wls.pvalues)

    return {
        "test": "PGS-level Egger",
        "trait": PGS_LABELS[trait],
        "outcome": outcome_label,
        "n_instruments": len(instruments),
        "egger_intercept": params[0],
        "egger_intercept_se": bse[0],
        "egger_intercept_p": pvals[0],
        "egger_slope": params[1],
        "egger_slope_se": bse[1],
        "egger_slope_p": pvals[1],
        "n": n,
    }


def run_mvmr(
    df: pd.DataFrame,
    outcome_var: str,
    outcome_label: str,
) -> list[dict]:
    """
    Multivariable MR: instrument each trait simultaneously to control for
    correlated-trait pleiotropy.
    """
    mr_traits = {t: e for t, e in TRAIT_ENDOGENOUS.items()
                 if f"pgs_{t}" in df.columns and e in df.columns}
    if len(mr_traits) < 2:
        return []

    pgs_cols = [f"pgs_{t}" for t in mr_traits]
    endog_cols = list(mr_traits.values())

    required = [outcome_var] + endog_cols + pgs_cols + COVARIATES_MR
    keep = required + (["hhid"] if "hhid" in df.columns else [])
    sample = df[keep].dropna(subset=required)
    n = len(sample)
    if n < 100:
        return []

    y = sample[outcome_var]
    exog = sm.add_constant(sample[COVARIATES_MR])
    endog = sample[endog_cols]
    instruments = sample[pgs_cols]

    try:
        model = IV2SLS(y, exog, endog, instruments).fit(cov_type="robust")
    except Exception as e:
        logger.warning("MVMR failed for %s: %s", outcome_label, e)
        return []

    rows = []
    for trait, endog_var in mr_traits.items():
        rows.append({
            "test": "Multivariable MR",
            "trait": PGS_LABELS[trait],
            "outcome": outcome_label,
            "iv_coef": model.params[endog_var],
            "iv_se": model.std_errors[endog_var],
            "iv_p": model.pvalues[endog_var],
            "n": n,
        })

    return rows


def run_steiger(
    df: pd.DataFrame,
    trait: str,
    endogenous: str,
    outcome_var: str,
    outcome_label: str,
) -> dict | None:
    """
    Steiger directionality test using partial R-squared (residualised for
    covariates), consistent with the main MR specification.
    """
    pgs_col = f"pgs_{trait}"
    if pgs_col not in df.columns:
        return None

    required = [outcome_var, endogenous, pgs_col] + COVARIATES_MR
    keep = required + (["hhid"] if "hhid" in df.columns else [])
    sample = df[keep].dropna(subset=required)
    n = len(sample)
    if n < 100:
        return None

    # Residualise exposure and outcome for covariates
    X_cov = sm.add_constant(sample[COVARIATES_MR])
    resid_exposure = sm.OLS(sample[endogenous], X_cov).fit().resid
    resid_outcome = sm.OLS(sample[outcome_var], X_cov).fit().resid

    r2_exposure = sample[pgs_col].corr(resid_exposure) ** 2
    r2_outcome = sample[pgs_col].corr(resid_outcome) ** 2

    correct_direction = r2_exposure > r2_outcome

    return {
        "test": "Steiger directionality",
        "trait": PGS_LABELS[trait],
        "outcome": outcome_label,
        "r2_exposure": r2_exposure,
        "r2_outcome": r2_outcome,
        "correct_direction": correct_direction,
        "n": n,
    }


def run_height_benchmark(
    df: pd.DataFrame,
    outcome_var: str,
    outcome_label: str,
) -> dict | None:
    """
    Auxiliary PGS benchmark: height PGS association with financial outcomes.
    Height is socially patterned (Case & Paxson 2008), so a significant
    association is expected and does not by itself indicate stratification.
    """
    if "pgs_height" not in df.columns:
        return None

    required = [outcome_var, "pgs_height"] + COVARIATES_MR
    keep = required + (["hhid"] if "hhid" in df.columns else [])
    sample = df[keep].dropna(subset=required)
    n = len(sample)
    if n < 100:
        return None

    y = sample[outcome_var]
    X = sm.add_constant(sample[["pgs_height"] + COVARIATES_MR])
    if "hhid" in sample.columns:
        model = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": sample["hhid"]},
        )
    else:
        model = sm.OLS(y, X).fit(cov_type="HC3")

    return {
        "test": "Height PGS benchmark",
        "trait": "Height",
        "outcome": outcome_label,
        "coef": model.params["pgs_height"],
        "se": model.bse["pgs_height"],
        "p": model.pvalues["pgs_height"],
        "n": n,
    }


def run_mr_sensitivity(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run all MR sensitivity and pleiotropy tests. Save results.
    """
    logger.info("Running MR sensitivity / pleiotropy tests")

    rows = []

    for trait, endogenous in TRAIT_ENDOGENOUS.items():
        if f"pgs_{trait}" not in df.columns or endogenous not in df.columns:
            continue

        for outcome_var, outcome_label in SENSITIVITY_OUTCOMES.items():
            if outcome_var not in df.columns:
                continue

            # 1. Overidentification test
            overid = run_overidentification(
                df, trait, endogenous, outcome_var, outcome_label,
            )
            if overid:
                rows.append(overid)
                logger.info(
                    "Overidentification %s -> %s: J=%.3f, p=%.4f (%d instruments)",
                    overid["trait"], outcome_label,
                    overid.get("j_stat", np.nan),
                    overid.get("j_pval", np.nan),
                    overid["n_instruments"],
                )

            # 2. PGS-level Egger
            egger = run_pgs_egger(
                df, trait, endogenous, outcome_var, outcome_label,
            )
            if egger:
                rows.append(egger)
                logger.info(
                    "PGS-Egger %s -> %s: intercept=%.4f (p=%.4f), slope=%.4f",
                    egger["trait"], outcome_label,
                    egger["egger_intercept"],
                    egger["egger_intercept_p"],
                    egger["egger_slope"],
                )

            # 3. Steiger directionality
            steiger = run_steiger(
                df, trait, endogenous, outcome_var, outcome_label,
            )
            if steiger:
                rows.append(steiger)
                logger.info(
                    "Steiger %s -> %s: R²(exposure)=%.4f, R²(outcome)=%.4f [%s]",
                    steiger["trait"], outcome_label,
                    steiger["r2_exposure"], steiger["r2_outcome"],
                    "CORRECT" if steiger["correct_direction"] else "WRONG",
                )

    # 4. MVMR (one run per outcome, all traits simultaneously)
    for outcome_var, outcome_label in SENSITIVITY_OUTCOMES.items():
        if outcome_var not in df.columns:
            continue
        mvmr_rows = run_mvmr(df, outcome_var, outcome_label)
        rows.extend(mvmr_rows)
        for r in mvmr_rows:
            logger.info(
                "MVMR %s -> %s: IV=%.4f (p=%.4f)",
                r["trait"], outcome_label, r["iv_coef"], r["iv_p"],
            )

    # 5. Height benchmark
    for outcome_var, outcome_label in SENSITIVITY_OUTCOMES.items():
        if outcome_var not in df.columns:
            continue
        hb = run_height_benchmark(df, outcome_var, outcome_label)
        if hb:
            rows.append(hb)
            logger.info(
                "Height benchmark -> %s: coef=%.4f, p=%.4f",
                outcome_label, hb["coef"], hb["p"],
            )

    results_df = pd.DataFrame(rows)

    outpath = OUTPUT_MAP.get("supp_mr_pleiotropy")
    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(outpath, index=False, float_format="%.4f")
        logger.info("Saved: %s (N rows=%d)", outpath, len(results_df))

    return {"pleiotropy": results_df}
