"""
Load HRS Polygenic Score Release 5 data by ancestry.

Each ancestry file (European, African, Hispanic) contains PGS variables with
ancestry-specific prefixes (E5_, A5_, H5_) plus principal components 1-10.
Constructs hhidpn from HHID + PN to match RAND HRS identifiers.
Verifies that PGS variables are standardised (mean ~ 0, SD ~ 1).
"""

import logging

import pandas as pd

from src.config import (
    ALT_PGS_SUFFIXES,
    ANCESTRY_PREFIX,
    AUX_PGS_SUFFIX,
    EXPECTED_N,
    PC_VARS,
    PGS_FILES,
    PGS_SUFFIXES,
)

logger = logging.getLogger(__name__)

STANDARDISATION_TOL_MEAN = 0.05
STANDARDISATION_TOL_SD = 0.10
_ANCESTRY_INDEX_CACHE: dict[str, set[int]] = {}


def _build_hhidpn(df: pd.DataFrame) -> pd.Series:
    """
    Construct numeric hhidpn from HHID and PN columns.

    HHID is typically 6 digits, PN is typically 3 digits (sometimes with
    leading zeros). The formula is: int(HHID) * 1000 + int(PN).
    """
    hhid = df["HHID"].astype(str).str.strip()
    pn = df["PN"].astype(str).str.strip()
    return (hhid + pn.str.zfill(3)).astype(int)


def _verify_standardisation(
    df: pd.DataFrame, pgs_cols: list[str], ancestry: str
) -> None:
    """Assert that PGS variables are approximately standardised."""
    for col in pgs_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            logger.warning("PGS column %s is all missing for %s", col, ancestry)
            continue
        mean = vals.mean()
        sd = vals.std()
        assert abs(mean) < STANDARDISATION_TOL_MEAN, (
            f"PGS {col} ({ancestry}): mean = {mean:.4f}, expected ~0"
        )
        assert abs(sd - 1.0) < STANDARDISATION_TOL_SD, (
            f"PGS {col} ({ancestry}): SD = {sd:.4f}, expected ~1"
        )
        logger.info(
            "PGS %s (%s): mean=%.4f, SD=%.4f, N=%d",
            col, ancestry, mean, sd, len(vals),
        )


def _load_hhidpn_index(ancestry: str) -> set[int]:
    """Load only respondent identifiers for a given ancestry file."""
    if ancestry in _ANCESTRY_INDEX_CACHE:
        return _ANCESTRY_INDEX_CACHE[ancestry]

    filepath = PGS_FILES[ancestry]
    if not filepath.exists():
        raise FileNotFoundError(
            f"Missing PGS file for {ancestry}: {filepath}. "
            "Set HRS_PGS_DTA or stage the restricted files under "
            "data/raw/restricted/."
        )

    id_df = pd.read_stata(str(filepath), columns=["HHID", "PN"])
    ids = set(_build_hhidpn(id_df))
    _ANCESTRY_INDEX_CACHE[ancestry] = ids
    return ids


def assert_ancestry_nonoverlap(ancestries: tuple[str, ...] = ("EUR", "AFR", "HIS")) -> None:
    """Assert that ancestry-specific HRS PGS samples do not overlap."""
    for i, left in enumerate(ancestries):
        for right in ancestries[i + 1:]:
            if not PGS_FILES[left].exists() or not PGS_FILES[right].exists():
                continue
            overlap = _load_hhidpn_index(left) & _load_hhidpn_index(right)
            assert not overlap, (
                f"PGS ancestry overlap detected between {left} and {right}: "
                f"{len(overlap)} shared HHIDPN values"
            )
            logger.info("Verified no ancestry overlap between %s and %s", left, right)


def load_pgs(ancestry: str = "EUR") -> pd.DataFrame:
    """
    Load PGS data for a given ancestry.

    Parameters
    ----------
    ancestry : str
        One of 'EUR', 'AFR', 'HIS'.

    Returns
    -------
    pd.DataFrame
        Indexed by hhidpn with PGS trait columns (renamed to trait names)
        and PC1-PC10 columns.
    """
    filepath = PGS_FILES[ancestry]
    prefix = ANCESTRY_PREFIX[ancestry]
    if not filepath.exists():
        raise FileNotFoundError(
            f"Missing PGS file for {ancestry}: {filepath}. "
            "Set HRS_PGS_DTA or stage the restricted files under "
            "data/raw/restricted/."
        )
    logger.info("Loading PGS data for %s from %s", ancestry, filepath)

    df = pd.read_stata(str(filepath))
    logger.info("PGS %s raw shape: %s", ancestry, df.shape)

    # Construct hhidpn
    df["hhidpn"] = _build_hhidpn(df)

    # Assert no duplicate hhidpn
    n_dup = df["hhidpn"].duplicated().sum()
    assert n_dup == 0, f"PGS {ancestry}: {n_dup} duplicate hhidpn values"

    # Assert sample size within tolerance
    expected = EXPECTED_N[ancestry]
    actual = len(df)
    assert abs(actual - expected) / expected < 0.02, (
        f"PGS {ancestry}: N={actual}, expected ~{expected}"
    )
    logger.info("PGS %s: N = %d (expected %d)", ancestry, actual, expected)

    # Identify PGS columns for our traits
    pgs_cols = {}
    for trait, suffix in PGS_SUFFIXES.items():
        col_name = f"{prefix}_{suffix}"
        if col_name in df.columns:
            pgs_cols[trait] = col_name
        else:
            # Some traits may not be available in all ancestries
            # Check if the African prefix is used (Hispanic file has some A5_ vars)
            alt_col = f"A5_{suffix}"
            if alt_col in df.columns and ancestry == "HIS":
                pgs_cols[trait] = alt_col
                logger.info(
                    "PGS %s: using %s for trait %s (A5 prefix)",
                    ancestry, alt_col, trait,
                )
            else:
                logger.warning(
                    "PGS %s: trait '%s' column '%s' not found",
                    ancestry, trait, col_name,
                )

    # Verify standardisation for available PGS columns
    available_pgs = list(pgs_cols.values())
    _verify_standardisation(df, available_pgs, ancestry)

    # Assert PCs present and non-missing
    for pc in PC_VARS:
        assert pc in df.columns, f"PC variable {pc} missing from {ancestry} file"
        n_missing = df[pc].isna().sum()
        assert n_missing == 0, (
            f"PC variable {pc} has {n_missing} missing values in {ancestry}"
        )

    # Build output dataframe: hhidpn + renamed PGS + PCs
    out_cols = {"hhidpn": df["hhidpn"]}
    for trait, col in pgs_cols.items():
        out_cols[f"pgs_{trait}"] = df[col].values
    for pc in PC_VARS:
        out_cols[pc] = df[pc].values

    # Add alternative PGS for multi-instrument sensitivity analyses
    alt_pgs_cols = []
    for trait, alt_list in ALT_PGS_SUFFIXES.items():
        for i, (suffix, label) in enumerate(alt_list):
            col_name = f"{prefix}_{suffix}"
            if col_name in df.columns:
                out_name = f"pgs_{trait}_alt{i}"
                out_cols[out_name] = df[col_name].values
                alt_pgs_cols.append(col_name)
                logger.info("Loaded alt PGS: %s -> %s (%s)", col_name, out_name, label)
    _verify_standardisation(df, alt_pgs_cols, ancestry)

    # Add auxiliary PGS (height) for benchmarking
    height_col = f"{prefix}_{AUX_PGS_SUFFIX}"
    if height_col in df.columns:
        out_cols["pgs_height"] = df[height_col].values
        _verify_standardisation(df, [height_col], ancestry)
        logger.info("Loaded auxiliary PGS: %s", height_col)

    result = pd.DataFrame(out_cols).set_index("hhidpn")
    logger.info("PGS %s output: %d rows, %d columns", ancestry, len(result), len(result.columns))

    return result


def load_pgs_european() -> pd.DataFrame:
    """Load European-ancestry PGS (primary analysis sample)."""
    return load_pgs("EUR")


def load_pgs_african() -> pd.DataFrame:
    """Load African-ancestry PGS (supplementary analysis)."""
    return load_pgs("AFR")


def load_pgs_hispanic() -> pd.DataFrame:
    """Load Hispanic-ancestry PGS (supplementary analysis)."""
    return load_pgs("HIS")
