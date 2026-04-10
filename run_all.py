#!/usr/bin/env python3
"""
Master driver script: regenerates all tables, figures, and results summary.

Usage:
    python run_all.py              # run everything
    python run_all.py --skip-supp  # skip supplementary (ancestry replication)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR,
    OUTPUT_DIR,
    OUTPUT_MAP,
    PGS_FILES,
    RAND_HRS_FULL_DTA,
    RAND_HRS_PARQUET,
    TABLES_DIR,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "run_all.log", mode="w"),
    ],
)
logger = logging.getLogger("run_all")


def validate_inputs(skip_supp: bool) -> None:
    """Validate required inputs before the pipeline starts."""
    if not RAND_HRS_PARQUET.exists() and not RAND_HRS_FULL_DTA.exists():
        raise FileNotFoundError(
            "RAND HRS input not found. Stage a parquet at "
            f"{RAND_HRS_PARQUET} or set RAND_HRS_FULL_DTA to the raw RAND .dta."
        )

    required_ancestries = ["EUR"] if skip_supp else ["EUR", "AFR", "HIS"]
    missing = [ancestry for ancestry in required_ancestries if not PGS_FILES[ancestry].exists()]
    if missing:
        raise FileNotFoundError(
            "Missing PGS file(s) for "
            f"{', '.join(missing)}. Stage them under data/raw/restricted/ "
            "or set HRS_PGS_DTA to the containing directory."
        )


def validate_outputs(skip_supp: bool) -> None:
    """Validate that the expected outputs were written."""
    expected_keys = [
        "table1_sample_characteristics",
        "table2_pgs_wealth",
        "table3_pgs_claiming",
        "table4_pgs_investment",
        "table5_mr_estimates",
        "table6_first_stage",
        "ed_table2_pgs_retirement",
        "supp_mr_sensitivity",
        "supp_mr_pleiotropy",
        "ed_table7_survival",
        "figure1_depression_mr",
        "figure2_claiming_vs_retirement",
        "figure3_pgs_wealth",
        "figure4_summary_heatmap",
    ]
    if not skip_supp:
        expected_keys.extend([
            "ed_table3_robustness",
            "supp_ancestry_replication",
            "supp_cognitive_exclusion",
        ])

    missing = []
    for key in expected_keys:
        path = OUTPUT_MAP[key]
        if path.suffix:
            if not path.exists():
                missing.append(str(path))
        else:
            for ext in [".pdf", ".png"]:
                candidate = path.with_suffix(ext)
                if not candidate.exists():
                    missing.append(str(candidate))
    assert not missing, f"Missing expected outputs: {missing}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all analyses for project 26.41")
    parser.add_argument(
        "--skip-supp",
        action="store_true",
        help="Skip supplementary analyses (ancestry replication)",
    )
    args = parser.parse_args()

    start = time.time()
    validate_inputs(args.skip_supp)

    # Ensure output directories exist
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load and merge data (European ancestry primary sample)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and merging data")
    logger.info("=" * 60)
    from src.data.load_pgs import assert_ancestry_nonoverlap
    from src.data.merge import load_analysis_sample

    assert_ancestry_nonoverlap()
    df = load_analysis_sample("EUR")
    logger.info("Analysis sample: N = %d", len(df))

    # Collect all results for figures and formatting
    all_results = {}

    # ------------------------------------------------------------------
    # Step 2: Descriptive statistics
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Descriptive statistics")
    logger.info("=" * 60)
    from src.analysis.descriptives import run_descriptives

    desc_results = run_descriptives(df)
    all_results.update(desc_results)

    # ------------------------------------------------------------------
    # Step 3: PGS -> Wealth (OLS)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: PGS -> Wealth")
    logger.info("=" * 60)
    from src.analysis.pgs_wealth import run_pgs_wealth

    wealth_results = run_pgs_wealth(df)
    all_results.update(wealth_results)

    # ------------------------------------------------------------------
    # Step 4: PGS -> SS Claiming Age
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: PGS -> SS Claiming Age")
    logger.info("=" * 60)
    from src.analysis.pgs_claiming import run_pgs_claiming

    claiming_results = run_pgs_claiming(df)
    all_results.update(claiming_results)

    # ------------------------------------------------------------------
    # Step 5: PGS -> Retirement Age (expected null)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: PGS -> Retirement Age")
    logger.info("=" * 60)
    from src.analysis.pgs_retirement import run_pgs_retirement

    retirement_results = run_pgs_retirement(df)
    all_results.update(retirement_results)

    # ------------------------------------------------------------------
    # Step 6: PGS -> Investment Participation (logistic)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6: PGS -> Investment Participation")
    logger.info("=" * 60)
    from src.analysis.pgs_investment import run_pgs_investment

    investment_results = run_pgs_investment(df)
    all_results.update(investment_results)

    # ------------------------------------------------------------------
    # Step 7: Mendelian Randomisation (IV/2SLS)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 7: Mendelian Randomisation")
    logger.info("=" * 60)
    from src.analysis.mr_analysis import run_mr_analysis

    mr_results = run_mr_analysis(df)
    all_results.update(mr_results)

    # ------------------------------------------------------------------
    # Step 8: MR Diagnostics
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 8: MR Diagnostics")
    logger.info("=" * 60)
    from src.analysis.mr_diagnostics import run_mr_diagnostics

    diag_results = run_mr_diagnostics(df, mr_results.get("table5"))
    all_results.update(diag_results)

    # ------------------------------------------------------------------
    # Step 8b: MR sensitivity / pleiotropy tests
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 8b: MR sensitivity / pleiotropy tests")
    logger.info("=" * 60)
    from src.analysis.mr_sensitivity import run_mr_sensitivity

    sens_results = run_mr_sensitivity(df)
    all_results.update(sens_results)

    # ------------------------------------------------------------------
    # Step 8c: Restricted-window survival check (62-70 claiming)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 8c: Restricted-window survival check")
    logger.info("=" * 60)
    from src.analysis.restricted_window_survival import run_restricted_window_claiming

    survival_results = run_restricted_window_claiming()
    all_results["ed_table7_survival"] = survival_results

    # ------------------------------------------------------------------
    # Step 9: Supplementary analyses (optional)
    # ------------------------------------------------------------------
    if not args.skip_supp:
        logger.info("=" * 60)
        logger.info("STEP 9: Supplementary analyses")
        logger.info("=" * 60)
        from src.analysis.supplementary import run_supplementary

        supp_results = run_supplementary(df)
        all_results.update(supp_results)
    else:
        logger.info("Skipping supplementary analyses (--skip-supp)")

    # ------------------------------------------------------------------
    # Step 10: Generate figures
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 10: Generating figures")
    logger.info("=" * 60)
    from src.figures import generate_all_figures

    generate_all_figures(all_results)

    # ------------------------------------------------------------------
    # Step 11: Format publication tables
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 11: Formatting tables")
    logger.info("=" * 60)
    from src.tables import format_all_tables

    format_all_tables(all_results)

    # ------------------------------------------------------------------
    # Step 12: Write results summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 12: Writing results summary")
    logger.info("=" * 60)
    write_results_summary(all_results, df)
    validate_outputs(args.skip_supp)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("COMPLETE. Total time: %.1f seconds", elapsed)
    logger.info("=" * 60)


def write_results_summary(
    all_results: dict, df, outpath: Path = OUTPUT_DIR / "results_summary.md"
) -> None:
    """Write a machine-readable results summary for manuscript-code traceability."""
    lines = [
        "# Results Summary",
        f"",
        f"Analysis sample: N = {len(df)}",
        f"SS claiming age source: {df['ss_claiming_age_source'].iloc[0]}",
        "",
    ]

    # Wealth results
    if "table2" in all_results and not all_results["table2"].empty:
        lines.append("## PGS -> Wealth (OLS)")
        t2 = all_results["table2"]
        for _, row in t2.iterrows():
            p_str = f"p < 0.001" if row["p"] < 0.001 else f"p = {row['p']:.4f}"
            lines.append(
                f"- {row['trait']} ({row['model']}): "
                f"beta = {row['beta']:.4f}, SE = {row['se']:.4f}, "
                f"{p_str}, N = {int(row['n'])}"
            )
        lines.append("")

    # Claiming results
    if "table3" in all_results and not all_results["table3"].empty:
        lines.append("## PGS -> SS Claiming Age")
        t3 = all_results["table3"]
        primary = t3[t3["model"] == "With education"]
        for _, row in primary.iterrows():
            p_str = f"p < 0.001" if row["p"] < 0.001 else f"p = {row['p']:.4f}"
            lines.append(
                f"- {row['trait']}: "
                f"beta = {row['beta']:.3f} years/SD, "
                f"{p_str}, N = {int(row['n'])}"
            )
        lines.append("")

    # Investment results
    if "table4" in all_results and not all_results["table4"].empty:
        lines.append("## PGS -> Investment Participation (Logistic)")
        t4 = all_results["table4"]
        for _, row in t4.iterrows():
            p_str = f"p < 0.001" if row["p"] < 0.001 else f"p = {row['p']:.4f}"
            lines.append(
                f"- {row['trait']} -> {row['outcome']}: "
                f"OR = {row['or']:.3f}, {p_str}, N = {int(row['n'])}"
            )
        lines.append("")

    # MR results
    if "table5" in all_results and not all_results["table5"].empty:
        lines.append("## Mendelian Randomisation (IV/2SLS)")
        t5 = all_results["table5"]
        for _, row in t5.iterrows():
            f_str = f"F = {row['first_stage_f']:.1f}"
            iv_coef = row.get("iv_coef", float("nan"))
            if pd.notna(iv_coef):
                lines.append(
                    f"- {row['trait']} -> {row['outcome']}: "
                    f"IV = {iv_coef:.4f}, {f_str}"
                )
            else:
                rf_coef = row.get("rf_coef", float("nan"))
                if pd.notna(rf_coef):
                    lines.append(
                        f"- {row['trait']} -> {row['outcome']}: "
                        f"RF = {rf_coef:.4f}, {f_str} (IV not estimated)"
                    )
        lines.append("")

    # First-stage diagnostics
    if "table6" in all_results and not all_results["table6"].empty:
        lines.append("## First-Stage Diagnostics")
        t6 = all_results["table6"]
        for _, row in t6.iterrows():
            lines.append(
                f"- {row['trait']} -> {row['endogenous']}: "
                f"F = {row['f_stat']:.1f}, partial R2 = {row['partial_r2']:.4f}, "
                f"N = {int(row['n'])} "
                f"[{'STRONG' if row['strong_instrument'] else 'WEAK'}]"
            )
        lines.append("")

    # Retirement (null)
    if "ed_table2" in all_results and not all_results["ed_table2"].empty:
        lines.append("## PGS -> Retirement Age (expected null)")
        ed2 = all_results["ed_table2"]
        primary = ed2[ed2["model"] == "With education"]
        for _, row in primary.iterrows():
            lines.append(
                f"- {row['trait']}: "
                f"beta = {row['beta']:.3f}, p = {row['p']:.4f}, N = {int(row['n'])}"
            )
        lines.append("")

    # Curated output inventory
    lines.append("## Manuscript-Facing Outputs")
    manuscript_keys = [
        "table1_sample_characteristics",
        "table2_pgs_wealth",
        "table3_pgs_claiming",
        "table4_pgs_investment",
        "table5_mr_estimates",
        "table6_first_stage",
        "ed_table2_pgs_retirement",
        "ed_table3_robustness",
        "supp_ancestry_replication",
        "supp_mr_sensitivity",
        "supp_cognitive_exclusion",
        "supp_mr_pleiotropy",
        "ed_table7_survival",
        "figure1_depression_mr",
        "figure2_claiming_vs_retirement",
        "figure3_pgs_wealth",
        "figure4_summary_heatmap",
    ]
    for key in manuscript_keys:
        path = OUTPUT_MAP[key]
        if path.suffix:
            if path.exists():
                lines.append(f"- {path.relative_to(OUTPUT_DIR)}")
        else:
            for ext in [".pdf", ".png"]:
                candidate = path.with_suffix(ext)
                if candidate.exists():
                    lines.append(f"- {candidate.relative_to(OUTPUT_DIR)}")

    lines.append("")
    lines.append("## Auxiliary Outputs")
    for extra in [
        TABLES_DIR / "table1_quintile_breakdown.csv",
        TABLES_DIR / "table1_sample_characteristics_formatted.csv",
        TABLES_DIR / "table2_pgs_wealth_formatted.csv",
        TABLES_DIR / "table3_pgs_claiming_formatted.csv",
        TABLES_DIR / "table4_pgs_investment_formatted.csv",
        TABLES_DIR / "table5_mr_estimates_formatted.csv",
        TABLES_DIR / "table6_first_stage_formatted.csv",
    ]:
        if extra.exists():
            lines.append(f"- {extra.relative_to(OUTPUT_DIR)}")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines))
    logger.info("Saved: %s", outpath)


if __name__ == "__main__":
    main()
