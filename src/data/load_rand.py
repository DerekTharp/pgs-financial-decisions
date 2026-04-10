"""
Load and prepare RAND HRS longitudinal data.

Derives key outcome variables:
- Latest wealth (IHS-transformed)
- Social Security claiming age, with explicit source tracking
- Retirement age
- Stock and IRA participation (binary)
- Average CESD and cognition scores (for MR first stage)
- Robustness covariates (latest self-rated health, marital status, income)
"""

import logging

import numpy as np
import pandas as pd

from src.config import (
    AGE_VARS,
    CESD_VARS,
    COGTOT_VARS,
    DEMO_VARS,
    HITOT_VARS,
    IRA_VARS,
    MSTAT_VARS,
    RAND_HRS_FULL_DTA,
    RAND_HRS_PARQUET,
    RETYR_VARS,
    SAYRET_VARS,
    SHLT_VARS,
    SS_CLAIMING_AGE_MAX,
    SS_CLAIMING_AGE_MIN,
    SS_CLAIMING_AGE_VAR,
    STOCK_VARS,
    WAVES,
    WEALTH_VARS,
)

logger = logging.getLogger(__name__)


def ihs(x: pd.Series) -> pd.Series:
    """Inverse hyperbolic sine transformation. Handles zeros and negatives."""
    return np.log(x + np.sqrt(x**2 + 1))


def _latest_nonmissing(df: pd.DataFrame, var_list: list[str]) -> pd.Series:
    """Return the latest (highest-wave) non-missing value for each respondent."""
    result = pd.Series(np.nan, index=df.index)
    # var_list is ordered earliest to latest; iterate forward so later waves
    # overwrite earlier values, leaving the latest non-missing.
    for col in var_list:
        if col in df.columns:
            mask = df[col].notna()
            result[mask] = df.loc[mask, col]
    return result


def _average_across_waves(df: pd.DataFrame, var_list: list[str]) -> pd.Series:
    """Return the person-level mean across all available waves."""
    cols_present = [c for c in var_list if c in df.columns]
    if not cols_present:
        return pd.Series(np.nan, index=df.index)
    return df[cols_present].mean(axis=1, skipna=True)


def _build_staged_parquet(needed_cols: list[str]) -> pd.DataFrame:
    """Build the staged RAND parquet from the full .dta file."""
    logger.info("Building staged RAND parquet from %s", RAND_HRS_FULL_DTA)
    available_cols = list(
        pd.read_stata(
            str(RAND_HRS_FULL_DTA),
            iterator=True,
            convert_categoricals=False,
        ).variable_labels().keys()
    )
    read_cols = [c for c in needed_cols if c in available_cols]
    missing = sorted(set(needed_cols) - set(read_cols))
    if missing:
        logger.warning("Columns not in RAND .dta (will be absent): %s", missing)

    df = pd.read_stata(
        str(RAND_HRS_FULL_DTA),
        columns=read_cols,
        convert_categoricals=False,
    )
    RAND_HRS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAND_HRS_PARQUET, index=False)
    logger.info("Staged RAND parquet: %s", RAND_HRS_PARQUET)
    return df


def _load_rand_source(needed_cols: list[str]) -> pd.DataFrame:
    """
    Load the staged RAND extract when available; otherwise build it from the
    full RAND .dta using only the needed columns and stage it as parquet.

    If the staged parquet exists but is missing needed columns that may be
    available in the raw .dta, rebuild it automatically.
    """
    if RAND_HRS_PARQUET.exists():
        logger.info("Loading RAND HRS core from %s", RAND_HRS_PARQUET)
        staged = pd.read_parquet(RAND_HRS_PARQUET)
        missing_needed = sorted(set(needed_cols) - set(staged.columns))
        if not missing_needed or not RAND_HRS_FULL_DTA.exists():
            if missing_needed:
                logger.warning("Columns not in parquet (will be NaN): %s", missing_needed)
            return staged
        logger.info(
            "Staged parquet is missing %d needed column(s); "
            "rebuilding from raw RAND .dta",
            len(missing_needed),
        )
        return _build_staged_parquet(needed_cols)

    if not RAND_HRS_FULL_DTA.exists():
        raise FileNotFoundError(
            "RAND HRS input not found. Expected either a staged parquet at "
            f"{RAND_HRS_PARQUET} or a full RAND .dta at {RAND_HRS_FULL_DTA}. "
            "Set RAND_HRS_PARQUET and/or RAND_HRS_FULL_DTA to valid paths."
        )

    return _build_staged_parquet(needed_cols)



def derive_ss_claiming_age(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Derive SS claiming age from `rassageb` (RAND cross-wave "age r start
    receiving Social Security").

    Returns `(claiming_age, source_label)`.
    """
    if SS_CLAIMING_AGE_VAR not in df.columns:
        raise KeyError(
            f"{SS_CLAIMING_AGE_VAR} not found in RAND extract. "
            "Rebuild the staged parquet from the full RAND .dta."
        )

    claiming_age = df[SS_CLAIMING_AGE_VAR].copy()
    n_valid = claiming_age.notna().sum()
    logger.info(
        "Using %s for SS claiming age: %d non-missing values",
        SS_CLAIMING_AGE_VAR, n_valid,
    )
    return claiming_age, "rassageb"


def derive_retirement_age(df: pd.DataFrame) -> pd.Series:
    """
    Derive retirement age from r{w}retyr (self-reported retirement year)
    or, as fallback, age at first wave where r{w}sayret == 1.

    Prefer retyr because it gives a specific year; use birth year to compute age.
    """
    ret_age = pd.Series(np.nan, index=df.index, dtype=float)

    # Method 1: use reported retirement year (latest non-missing)
    for col in reversed(RETYR_VARS):
        if col in df.columns:
            valid = df[col].notna() & (df[col] > 1900) & ret_age.isna()
            ret_age[valid] = df.loc[valid, col] - df.loc[valid, "rabyear"]

    # Method 2: fallback to first wave self-reporting retired
    for w, sayret_col, age_col in zip(WAVES, SAYRET_VARS, AGE_VARS):
        if sayret_col not in df.columns or age_col not in df.columns:
            continue
        is_retired = (df[sayret_col] == 1) & df[sayret_col].notna()
        newly_retired = is_retired & ret_age.isna()
        ret_age[newly_retired] = df.loc[newly_retired, age_col]

    return ret_age


def load_rand_hrs() -> pd.DataFrame:
    """
    Load RAND HRS core extract and derive all outcome and control variables.

    Returns a DataFrame indexed by hhidpn with derived variables.
    """
    # Determine columns to load
    needed_cols = (
        ["hhidpn", "hhid", "pn", SS_CLAIMING_AGE_VAR]
        + DEMO_VARS
        + WEALTH_VARS
        + STOCK_VARS
        + IRA_VARS
        + AGE_VARS
        + RETYR_VARS
        + SAYRET_VARS
        + CESD_VARS
        + COGTOT_VARS
        + SHLT_VARS
        + MSTAT_VARS
        + HITOT_VARS
    )
    df_full = _load_rand_source(needed_cols)
    available = [c for c in needed_cols if c in df_full.columns]
    missing = set(needed_cols) - set(available)
    if missing:
        logger.warning("Columns not in parquet (will be NaN): %s", missing)
    df = df_full[available].copy()
    del df_full

    logger.info("RAND HRS loaded: N = %d", len(df))

    # -----------------------------------------------------------------------
    # Derive outcome variables
    # -----------------------------------------------------------------------

    # Latest wealth (IHS-transformed)
    df["wealth_latest"] = _latest_nonmissing(df, WEALTH_VARS)
    df["ihs_wealth"] = ihs(df["wealth_latest"])

    # Stock participation (binary, latest wave)
    stock_latest = _latest_nonmissing(df, STOCK_VARS)
    df["stock_participation"] = (stock_latest > 0).astype(float)
    df.loc[stock_latest.isna(), "stock_participation"] = np.nan

    # IRA participation (binary, latest wave)
    ira_latest = _latest_nonmissing(df, IRA_VARS)
    df["ira_participation"] = (ira_latest > 0).astype(float)
    df.loc[ira_latest.isna(), "ira_participation"] = np.nan

    # Ever held stocks
    stock_cols = [c for c in STOCK_VARS if c in df.columns]
    df["ever_stock"] = (df[stock_cols] > 0).any(axis=1).astype(float)
    df.loc[df[stock_cols].isna().all(axis=1), "ever_stock"] = np.nan

    # Ever held IRA
    ira_cols = [c for c in IRA_VARS if c in df.columns]
    df["ever_ira"] = (df[ira_cols] > 0).any(axis=1).astype(float)
    df.loc[df[ira_cols].isna().all(axis=1), "ever_ira"] = np.nan

    # SS claiming age
    df["ss_claiming_age"], claiming_source = derive_ss_claiming_age(df)
    df["ss_claiming_age_source"] = claiming_source
    # Filter to retirement-benefit range: exclude likely SSDI/survivor claims
    # (retirement benefits cannot be claimed before age 62)
    df.loc[
        (df["ss_claiming_age"] < SS_CLAIMING_AGE_MIN) |
        (df["ss_claiming_age"] > SS_CLAIMING_AGE_MAX),
        "ss_claiming_age",
    ] = np.nan

    # Retirement age
    df["retirement_age"] = derive_retirement_age(df)
    # Filter to plausible range (50-85)
    df.loc[
        (df["retirement_age"] < 50) | (df["retirement_age"] > 85),
        "retirement_age",
    ] = np.nan

    # Average CESD across waves (for MR first stage)
    df["cesd_avg"] = _average_across_waves(df, CESD_VARS)

    # Average cognition across waves (for MR first stage)
    df["cogtot_avg"] = _average_across_waves(df, COGTOT_VARS)

    # Robustness covariates: latest self-rated health, marital status, income
    df["shlt_latest"] = _latest_nonmissing(df, SHLT_VARS)
    df["mstat_latest"] = _latest_nonmissing(df, MSTAT_VARS)
    df["hitot_latest"] = _latest_nonmissing(df, HITOT_VARS)

    # Female indicator (ragender: 1=male, 2=female in RAND HRS)
    df["female"] = (df["ragender"] == 2).astype(float)

    # Household ID for clustering (hhidpn // 1000)
    df["hhid"] = df["hhidpn"] // 1000

    # Set hhidpn as index
    df = df.set_index("hhidpn")
    assert df.index.is_unique, "RAND HRS extract has duplicate hhidpn values"

    logger.info(
        "Derived variables: wealth N=%d, SS claiming N=%d [%s], retirement N=%d, "
        "stock N=%d, IRA N=%d, CESD N=%d, cognition N=%d",
        df["ihs_wealth"].notna().sum(),
        df["ss_claiming_age"].notna().sum(),
        claiming_source,
        df["retirement_age"].notna().sum(),
        df["stock_participation"].notna().sum(),
        df["ira_participation"].notna().sum(),
        df["cesd_avg"].notna().sum(),
        df["cogtot_avg"].notna().sum(),
    )

    return df
