"""
Merge PGS data with RAND HRS longitudinal data.

Performs inner join on hhidpn and runs integrity assertions at each step.
"""

import logging

import pandas as pd

from src.config import EXPECTED_N, MERGE_TOLERANCE, PC_VARS
from src.data.load_pgs import load_pgs
from src.data.load_rand import load_rand_hrs

logger = logging.getLogger(__name__)


def merge_pgs_rand(ancestry: str = "EUR") -> pd.DataFrame:
    """
    Merge PGS data with RAND HRS outcomes for a given ancestry.

    Parameters
    ----------
    ancestry : str
        One of 'EUR', 'AFR', 'HIS'.

    Returns
    -------
    pd.DataFrame
        Merged dataset indexed by hhidpn containing PGS, PCs, demographics,
        and derived financial outcome variables.
    """
    # Load components
    pgs_df = load_pgs(ancestry)
    rand_df = load_rand_hrs()

    logger.info(
        "Pre-merge: PGS %s N=%d, RAND HRS N=%d",
        ancestry, len(pgs_df), len(rand_df),
    )

    # Inner join on hhidpn
    merged = pgs_df.join(rand_df, how="inner")

    logger.info("Post-merge: N = %d", len(merged))

    # -----------------------------------------------------------------------
    # Integrity assertions
    # -----------------------------------------------------------------------

    # No duplicate hhidpn
    assert not merged.index.duplicated().any(), "Duplicate hhidpn after merge"

    # Sample size within tolerance of expected
    expected = EXPECTED_N[ancestry]
    pct_diff = abs(len(merged) - expected) / expected
    assert pct_diff < MERGE_TOLERANCE, (
        f"Merged N={len(merged)}, expected ~{expected} "
        f"(difference: {pct_diff:.1%})"
    )

    # PCs present and non-missing for all merged respondents
    for pc in PC_VARS:
        assert pc in merged.columns, f"PC variable {pc} missing after merge"
        n_missing = merged[pc].isna().sum()
        assert n_missing == 0, (
            f"PC variable {pc} has {n_missing} missing values after merge"
        )

    # PGS columns present
    pgs_cols = [c for c in merged.columns if c.startswith("pgs_")]
    assert len(pgs_cols) > 0, "No PGS columns found after merge"
    for col in pgs_cols:
        n_missing = merged[col].isna().sum()
        logger.info("PGS column %s: %d missing (%.1f%%)",
                     col, n_missing, 100 * n_missing / len(merged))

    # Key demographics present
    assert "rabyear" in merged.columns, "Birth year missing"
    assert "female" in merged.columns, "Female indicator missing"
    assert "raedyrs" in merged.columns, "Education years missing"

    logger.info(
        "Merge complete for %s: N=%d, columns=%d",
        ancestry, len(merged), len(merged.columns),
    )

    return merged


def load_analysis_sample(ancestry: str = "EUR") -> pd.DataFrame:
    """
    Load the fully merged analysis sample for a given ancestry.
    Convenience wrapper around merge_pgs_rand.
    """
    return merge_pgs_rand(ancestry)
