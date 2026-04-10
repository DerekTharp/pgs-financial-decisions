"""
Central configuration for project 26.41.

The repository is designed to run from repo-local staged data paths by default,
with environment-variable overrides for restricted or externally stored inputs.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PUBLIC_DATA_DIR = DATA_DIR / "raw" / "public"
RESTRICTED_DATA_DIR = DATA_DIR / "raw" / "restricted"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"

# ---------------------------------------------------------------------------
# Data paths (environment variable overrides for restricted data)
# ---------------------------------------------------------------------------
RAND_HRS_PARQUET = Path(os.environ.get(
    "RAND_HRS_PARQUET",
    str(PUBLIC_DATA_DIR / "rand_hrs_core.parquet"),
))

RAND_HRS_FULL_DTA = Path(os.environ.get(
    "RAND_HRS_FULL_DTA",
    str(PUBLIC_DATA_DIR / "randhrs1992_2022v1.dta"),
))

PGS_DIR = Path(os.environ.get(
    "HRS_PGS_DTA",
    str(RESTRICTED_DATA_DIR),
))

# Per-ancestry PGS file names
PGS_FILES = {
    "EUR": PGS_DIR / "PGENSCOREE_R.dta",
    "AFR": PGS_DIR / "PGENSCOREA_R.dta",
    "HIS": PGS_DIR / "PGENSCOREH_R.dta",
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# HRS wave numbers available in the core extract
# ---------------------------------------------------------------------------
WAVES = list(range(5, 17))  # waves 5 (2000) through 16 (2022)

# ---------------------------------------------------------------------------
# PGS variable names by ancestry prefix
# ---------------------------------------------------------------------------
ANCESTRY_PREFIX = {"EUR": "E5", "AFR": "A5", "HIS": "H5"}

# Trait name -> suffix shared across ancestries
PGS_SUFFIXES = {
    "education":    "EDU3_SSGAC18",
    "cognition":    "GENCOG2_CHARGE18",
    "depression":   "DEPSYMP_SSGAC16",
    "neuroticism":  "NEUROTICISM_SSGAC16",
    "wellbeing":    "WELLBEING_SSGAC16",
}

def pgs_varname(trait: str, ancestry: str = "EUR") -> str:
    """Return the PGS column name for a given trait and ancestry."""
    prefix = ANCESTRY_PREFIX[ancestry]
    suffix = PGS_SUFFIXES[trait]
    return f"{prefix}_{suffix}"

# Convenience dict: trait -> EUR column name
PGS_VARS = {trait: pgs_varname(trait, "EUR") for trait in PGS_SUFFIXES}

# Trait display labels for tables and figures
PGS_LABELS = {
    "education":   "Education",
    "cognition":   "Cognition",
    "depression":  "Depression",
    "neuroticism": "Neuroticism",
    "wellbeing":   "Well-being",
}

# ---------------------------------------------------------------------------
# Alternative PGS for multi-instrument MR sensitivity analyses
# ---------------------------------------------------------------------------
# Multiple PGS for the same trait from different discovery GWAS enable
# overidentification tests and MR-Egger.
ALT_PGS_SUFFIXES = {
    "depression": [
        ("MDD2_PGC18", "MDD (PGC 2018)"),
        ("MDD_PGC13", "MDD (PGC 2013)"),
    ],
    "education": [
        ("EDU2_SSGAC16", "EA2 (SSGAC 2016)"),
    ],
    "cognition": [
        ("GENCOG_CHARGE15", "Cognition (CHARGE 2015)"),
    ],
}

# Auxiliary PGS for benchmarking (height is socially patterned; this is
# a comparison, not a formal negative control)
AUX_PGS_SUFFIX = "HEIGHT2_GIANT18"

# ---------------------------------------------------------------------------
# Principal component variable names (same across ancestries)
# ---------------------------------------------------------------------------
PC_VARS = ["PC1_5A", "PC1_5B", "PC1_5C", "PC1_5D", "PC1_5E",
           "PC6_10A", "PC6_10B", "PC6_10C", "PC6_10D", "PC6_10E"]

# ---------------------------------------------------------------------------
# Financial outcome variables (RAND HRS naming convention)
# ---------------------------------------------------------------------------
# Wealth: household total assets including second residence
WEALTH_VARS = [f"h{w}atotb" for w in WAVES]

# Stock holdings (dollar amount; participation = amount > 0)
STOCK_VARS = [f"h{w}astck" for w in WAVES]

# IRA holdings (dollar amount; participation = amount > 0)
IRA_VARS = [f"h{w}aira" for w in WAVES]

# SS retirement income (>0 indicates claiming)
SS_INCOME_VARS = [f"r{w}isret" for w in WAVES]

# ---------------------------------------------------------------------------
# SS claiming age derivation
# ---------------------------------------------------------------------------
# Uses `rassageb` (RAND cross-wave "age r start receiving Social Security").
# Pipeline raises if this variable is absent.
SS_CLAIMING_AGE_VAR = "rassageb"
# Claiming age window: 62-70. Retirement benefits cannot be claimed
# before 62 (earlier starts are SSDI/survivor). Delayed retirement credits
# stop accruing at 70, so that is the upper bound of the optimisation
# decision.
SS_CLAIMING_AGE_MIN = 62
SS_CLAIMING_AGE_MAX = 70
AGE_VARS = [f"r{w}agey_e" for w in WAVES]

# ---------------------------------------------------------------------------
# Retirement variables
# ---------------------------------------------------------------------------
RETYR_VARS = [f"r{w}retyr" for w in WAVES]
SAYRET_VARS = [f"r{w}sayret" for w in WAVES]

# ---------------------------------------------------------------------------
# Trait variables for MR first stage
# ---------------------------------------------------------------------------
CESD_VARS = [f"r{w}cesd" for w in WAVES]
COGTOT_VARS = [f"r{w}cogtot" for w in range(5, 14)]  # waves 5-13 only

# ---------------------------------------------------------------------------
# Demographic / control variables
# ---------------------------------------------------------------------------
DEMO_VARS = ["rabyear", "ragender", "raedyrs"]

# Self-rated health (for robustness)
SHLT_VARS = [f"r{w}shlt" for w in WAVES]

# Marital status (for robustness)
MSTAT_VARS = [f"r{w}mstat" for w in WAVES]

# Household total income (for robustness)
HITOT_VARS = [f"h{w}itot" for w in WAVES]

# Individual earnings
IEARN_VARS = [f"r{w}iearn" for w in WAVES]

# ---------------------------------------------------------------------------
# Covariate sets
# ---------------------------------------------------------------------------
COVARIATES_BASIC = ["rabyear", "ragender"] + PC_VARS
COVARIATES_EDUCATION = COVARIATES_BASIC + ["raedyrs"]
COVARIATES_ROBUSTNESS = COVARIATES_EDUCATION + ["shlt_latest", "mstat_latest", "hitot_latest"]

# MR models: do NOT include traits influenced by the PGS instrument.
# Control only for pre-treatment variables (birth year, gender, PCs).
COVARIATES_MR = ["rabyear", "ragender"] + PC_VARS

# ---------------------------------------------------------------------------
# Sample size expectations
# ---------------------------------------------------------------------------
EXPECTED_N = {"EUR": 12090, "AFR": 3100, "HIS": 2381}
MERGE_TOLERANCE = 0.01  # allow 1% deviation

# ---------------------------------------------------------------------------
# Statistical thresholds
# ---------------------------------------------------------------------------
FIRST_STAGE_F_THRESHOLD = 10  # Stock-Yogo threshold for weak instruments
ALPHA = 0.05
RANDOM_SEED = 20260403

# ---------------------------------------------------------------------------
# Manuscript display-item -> output file mapping
# ---------------------------------------------------------------------------
OUTPUT_MAP = {
    "table1_sample_characteristics":    TABLES_DIR / "table1_sample_characteristics.csv",
    "table2_pgs_wealth":                TABLES_DIR / "table2_pgs_wealth.csv",
    "table3_pgs_claiming":              TABLES_DIR / "table3_pgs_claiming.csv",
    "table4_pgs_investment":            TABLES_DIR / "table4_pgs_investment.csv",
    "table5_mr_estimates":              TABLES_DIR / "table5_mr_estimates.csv",
    "table6_first_stage":               TABLES_DIR / "table6_first_stage.csv",
    "ed_table2_pgs_retirement":         TABLES_DIR / "ed_table2_pgs_retirement.csv",
    "ed_table3_robustness":             TABLES_DIR / "ed_table3_robustness.csv",
    "supp_ancestry_replication":        TABLES_DIR / "supp_ancestry_replication.csv",
    "supp_mr_sensitivity":              TABLES_DIR / "supp_mr_sensitivity.csv",
    "supp_cognitive_exclusion":         TABLES_DIR / "supp_cognitive_exclusion.csv",
    "supp_mr_pleiotropy":               TABLES_DIR / "supp_mr_pleiotropy.csv",
    "ed_table7_survival":               TABLES_DIR / "ed_table7_restricted_window_survival.csv",
    "figure1_depression_mr":             FIGURES_DIR / "figure1_depression_mr",
    "figure2_claiming_vs_retirement":   FIGURES_DIR / "figure2_claiming_vs_retirement",
    "figure3_pgs_wealth":               FIGURES_DIR / "figure3_pgs_wealth",
    "figure4_summary_heatmap":          FIGURES_DIR / "figure4_summary_heatmap",
}
