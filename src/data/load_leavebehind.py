"""
Load selected Leave-Behind style psychosocial variables when available.

This project does not currently use these variables in the main pipeline, but
the loader is provided so the repository structure matches the documented
layout and future mechanism analyses can reuse it.
"""

import logging

import numpy as np
import pandas as pd

from src.config import RAND_HRS_FULL_DTA

logger = logging.getLogger(__name__)

# Risk-preference style variables observed in the RAND/HRS full extract.
RISK_VARS = [
    "r1risk", "r4risk", "r5risk", "r6risk", "r7risk", "r8risk",
    "s1risk", "s4risk", "s5risk", "s6risk", "s7risk", "s8risk",
]


def _latest_nonmissing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=df.index)
    for column in columns:
        if column in df.columns:
            mask = df[column].notna()
            result[mask] = df.loc[mask, column]
    return result


def load_leavebehind() -> pd.DataFrame:
    """
    Load a minimal set of psychosocial variables from the full RAND/HRS file.

    Returns an HHIDPN-indexed DataFrame. If the full .dta is unavailable, an
    empty DataFrame is returned and the caller can skip mechanism analysis.
    """
    if not RAND_HRS_FULL_DTA.exists():
        logger.warning("Leave-behind input unavailable: %s", RAND_HRS_FULL_DTA)
        return pd.DataFrame()

    reader = pd.read_stata(
        str(RAND_HRS_FULL_DTA),
        iterator=True,
        convert_categoricals=False,
    )
    available = set(reader.variable_labels().keys())
    read_cols = [col for col in ["hhidpn"] + RISK_VARS if col in available]
    if read_cols == ["hhidpn"]:
        logger.warning("No leave-behind style risk variables found in %s", RAND_HRS_FULL_DTA)
        return pd.DataFrame()

    df = pd.read_stata(
        str(RAND_HRS_FULL_DTA),
        columns=read_cols,
        convert_categoricals=False,
    )
    df["risk_pref_latest"] = _latest_nonmissing(df, [col for col in RISK_VARS if col in df.columns])
    df = df.set_index("hhidpn")
    logger.info("Loaded leave-behind style variables: N = %d", len(df))
    return df[["risk_pref_latest"]]
