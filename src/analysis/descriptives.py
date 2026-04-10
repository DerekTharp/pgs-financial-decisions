"""
Generate Table 1: sample characteristics and PGS quintile distributions.

Produces: table1_sample_characteristics.csv
"""

import logging

import numpy as np
import pandas as pd

from src.config import OUTPUT_MAP, PGS_LABELS

logger = logging.getLogger(__name__)


def compute_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sample characteristics for the analysis sample.

    Returns a DataFrame with variable names as rows and summary statistics
    as columns (N, Mean/%, SD, Min, Max).
    """
    stats = []

    # Continuous variables
    continuous = {
        "Birth year": "rabyear",
        "Education (years)": "raedyrs",
        "Wealth (latest, $)": "wealth_latest",
        "IHS(wealth)": "ihs_wealth",
        "SS claiming age": "ss_claiming_age",
        "Retirement age": "retirement_age",
        "CESD (average)": "cesd_avg",
        "Cognition (average)": "cogtot_avg",
    }
    for label, var in continuous.items():
        if var not in df.columns:
            continue
        vals = df[var].dropna()
        stats.append({
            "Variable": label,
            "N": len(vals),
            "Mean": vals.mean(),
            "SD": vals.std(),
            "Median": vals.median(),
            "Min": vals.min(),
            "Max": vals.max(),
        })

    # Binary variables (report as %)
    binary = {
        "Female (%)": "female",
        "Stock participation (%)": "stock_participation",
        "IRA participation (%)": "ira_participation",
        "Ever held stocks (%)": "ever_stock",
        "Ever held IRA (%)": "ever_ira",
    }
    for label, var in binary.items():
        if var not in df.columns:
            continue
        vals = df[var].dropna()
        stats.append({
            "Variable": label,
            "N": len(vals),
            "Mean": vals.mean() * 100,
            "SD": np.nan,
            "Median": np.nan,
            "Min": 0,
            "Max": 100,
        })

    # PGS distributions
    for trait, label in PGS_LABELS.items():
        col = f"pgs_{trait}"
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        stats.append({
            "Variable": f"PGS {label}",
            "N": len(vals),
            "Mean": vals.mean(),
            "SD": vals.std(),
            "Median": vals.median(),
            "Min": vals.min(),
            "Max": vals.max(),
        })

    return pd.DataFrame(stats)


def compute_quintile_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean financial outcomes by education PGS quintile.

    Returns a long-format DataFrame with quintile, outcome, and mean value.
    """
    pgs_col = "pgs_education"
    if pgs_col not in df.columns:
        logger.warning("Education PGS not found; skipping quintile table")
        return pd.DataFrame()

    df = df.copy()
    df["pgs_edu_quintile"] = pd.qcut(
        df[pgs_col], q=5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]
    )

    outcomes = {
        "Wealth ($)": "wealth_latest",
        "IHS(wealth)": "ihs_wealth",
        "Stock participation (%)": "stock_participation",
        "IRA participation (%)": "ira_participation",
        "SS claiming age": "ss_claiming_age",
    }

    rows = []
    for q_label in ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]:
        q_data = df[df["pgs_edu_quintile"] == q_label]
        row = {"Quintile": q_label, "N": len(q_data)}
        for outcome_label, var in outcomes.items():
            if var in q_data.columns:
                vals = q_data[var].dropna()
                if "(%)" in outcome_label:
                    row[outcome_label] = vals.mean() * 100
                else:
                    row[outcome_label] = vals.mean()
        rows.append(row)

    return pd.DataFrame(rows)


def run_descriptives(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run all descriptive analyses and save outputs.

    Returns dict of output name -> DataFrame.
    """
    logger.info("Computing sample descriptives (N=%d)", len(df))

    desc = compute_descriptives(df)
    quintiles = compute_quintile_table(df)

    # Combine into one output
    outpath = OUTPUT_MAP["table1_sample_characteristics"]
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Write main descriptives
    desc.to_csv(outpath, index=False, float_format="%.3f")
    logger.info("Saved: %s", outpath)

    # Write quintile table as a separate sheet-like section
    quintile_path = outpath.parent / "table1_quintile_breakdown.csv"
    if not quintiles.empty:
        quintiles.to_csv(quintile_path, index=False, float_format="%.3f")
        logger.info("Saved: %s", quintile_path)

    return {"descriptives": desc, "quintiles": quintiles}
