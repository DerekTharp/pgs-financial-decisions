"""
Format analysis output DataFrames into publication-ready table CSVs.

Applies consistent formatting: rounding, significance stars, column ordering.
This module reads the raw CSV outputs from analysis scripts and writes
formatted versions suitable for inclusion in the manuscript.
"""

import logging

import numpy as np
import pandas as pd

from src.config import TABLES_DIR

logger = logging.getLogger(__name__)


def _stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _format_p(p: float) -> str:
    """Format p-value for manuscript: exact unless <0.001."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _format_ci(lower: float, upper: float, decimals: int = 3) -> str:
    """Format confidence interval as [lower, upper]."""
    if pd.isna(lower) or pd.isna(upper):
        return ""
    return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"


def format_wealth_table(table2: pd.DataFrame) -> pd.DataFrame:
    """Format Table 2 (PGS -> wealth) for publication."""
    if table2.empty:
        return table2

    out = table2.copy()
    out["Coefficient"] = out["beta"].apply(lambda x: f"{x:.3f}")
    out["SE"] = out["se"].apply(lambda x: f"{x:.3f}")
    out["95% CI"] = out.apply(
        lambda r: _format_ci(r["ci_lower"], r["ci_upper"]), axis=1
    )
    out["p-value"] = out["p"].apply(_format_p)
    out[""] = out["p"].apply(_stars)
    out["N"] = out["n"].astype(int)
    out["R-squared"] = out["r2"].apply(lambda x: f"{x:.4f}")

    cols = ["trait", "model", "Coefficient", "SE", "95% CI", "p-value", "", "N", "R-squared"]
    return out[cols].rename(columns={"trait": "PGS trait", "model": "Specification"})


def format_sample_characteristics(table1: pd.DataFrame) -> pd.DataFrame:
    """Format sample characteristics for publication."""
    if table1.empty:
        return table1

    out = table1.copy()
    if "N" in out.columns:
        out["N"] = out["N"].astype(int)
    for column in ["Mean", "SD", "Median", "Min", "Max"]:
        if column in out.columns:
            out[column] = out[column].apply(
                lambda x: "" if pd.isna(x) else f"{x:.3f}"
            )
    return out


def format_claiming_table(table3: pd.DataFrame) -> pd.DataFrame:
    """Format Table 3 (PGS -> claiming age) for publication."""
    if table3.empty:
        return table3

    out = table3.copy()
    out["Coefficient (years)"] = out["beta"].apply(lambda x: f"{x:.3f}")
    out["SE"] = out["se"].apply(lambda x: f"{x:.3f}")
    out["95% CI"] = out.apply(
        lambda r: _format_ci(r["ci_lower"], r["ci_upper"]), axis=1
    )
    out["p-value"] = out["p"].apply(_format_p)
    out[""] = out["p"].apply(_stars)
    out["N"] = out["n"].astype(int)

    cols = ["trait", "model", "Coefficient (years)", "SE", "95% CI", "p-value", "", "N"]
    return out[cols].rename(columns={"trait": "PGS trait", "model": "Specification"})


def format_investment_table(table4: pd.DataFrame) -> pd.DataFrame:
    """Format Table 4 (PGS -> investment, logistic) for publication."""
    if table4.empty:
        return table4

    out = table4.copy()
    out["OR"] = out["or"].apply(lambda x: f"{x:.3f}")
    out["95% CI"] = out.apply(
        lambda r: _format_ci(r.get("or_ci_lower", np.nan),
                             r.get("or_ci_upper", np.nan)),
        axis=1,
    )
    out["p-value"] = out["p"].apply(_format_p)
    out[""] = out["p"].apply(_stars)
    out["N"] = out["n"].astype(int)

    cols = ["trait", "outcome", "OR", "95% CI", "p-value", "", "N"]
    return out[cols].rename(columns={"trait": "PGS trait", "outcome": "Outcome"})


def format_mr_table(table5: pd.DataFrame) -> pd.DataFrame:
    """Format Table 5 (MR estimates) for publication."""
    if table5.empty:
        return table5

    out = table5.copy()
    out["First-stage F"] = out["first_stage_f"].apply(lambda x: f"{x:.1f}")
    out["RF coefficient"] = out.get("rf_coef", pd.Series(dtype=float)).apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else ""
    )
    out["IV coefficient"] = out.get("iv_coef", pd.Series(dtype=float)).apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "---"
    )
    out["IV SE"] = out.get("iv_se", pd.Series(dtype=float)).apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else ""
    )
    out["IV 95% CI"] = out.apply(
        lambda r: _format_ci(
            r.get("iv_ci_lower", np.nan),
            r.get("iv_ci_upper", np.nan),
            decimals=4,
        ),
        axis=1,
    )
    out["IV p"] = out.get("iv_p", pd.Series(dtype=float)).apply(_format_p)
    n_series = out.get("iv_n", out.get("rf_n", out.get("first_stage_n", pd.Series(dtype=float))))
    out["N"] = n_series.apply(
        lambda x: f"{int(x)}" if pd.notna(x) else ""
    )

    cols = [
        "trait", "outcome", "First-stage F",
        "RF coefficient", "IV coefficient", "IV SE", "IV 95% CI", "IV p", "N",
    ]
    available_cols = [c for c in cols if c in out.columns]
    return out[available_cols].rename(
        columns={"trait": "Instrument (PGS)", "outcome": "Outcome"}
    )


def format_first_stage_table(table6: pd.DataFrame) -> pd.DataFrame:
    """Format first-stage diagnostics for publication."""
    if table6.empty:
        return table6

    out = table6.copy()
    out["Coefficient"] = out["coef"].apply(lambda x: f"{x:.4f}")
    out["SE"] = out["se"].apply(lambda x: f"{x:.4f}")
    out["F-statistic"] = out["f_stat"].apply(lambda x: f"{x:.1f}")
    out["Partial R-squared"] = out["partial_r2"].apply(lambda x: f"{x:.4f}")
    out["p-value"] = out["p"].apply(_format_p)
    out["N"] = out["n"].astype(int)
    out["Instrument strength"] = out["strong_instrument"].map(
        {True: "Strong", False: "Weak"}
    )

    cols = [
        "trait",
        "endogenous",
        "Coefficient",
        "SE",
        "F-statistic",
        "Partial R-squared",
        "p-value",
        "N",
        "Instrument strength",
    ]
    return out[cols].rename(
        columns={"trait": "PGS trait", "endogenous": "Endogenous trait"}
    )


def format_all_tables(all_results: dict[str, pd.DataFrame]) -> None:
    """Format and save all publication-ready tables."""
    logger.info("Formatting publication tables")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    formatters = {
        "descriptives": ("table1_sample_characteristics_formatted.csv", format_sample_characteristics),
        "table2": ("table2_pgs_wealth_formatted.csv", format_wealth_table),
        "table3": ("table3_pgs_claiming_formatted.csv", format_claiming_table),
        "table4": ("table4_pgs_investment_formatted.csv", format_investment_table),
        "table5": ("table5_mr_estimates_formatted.csv", format_mr_table),
        "table6": ("table6_first_stage_formatted.csv", format_first_stage_table),
    }

    for key, (filename, formatter) in formatters.items():
        if key in all_results and not all_results[key].empty:
            formatted = formatter(all_results[key])
            path = TABLES_DIR / filename
            formatted.to_csv(path, index=False)
            logger.info("Saved formatted table: %s", path)
