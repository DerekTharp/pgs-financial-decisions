# 26.41 PGS -> Financial Decisions + SS Claiming

Replication-oriented analysis pipeline for the Health and Retirement Study
(HRS) project on polygenic scores, financial outcomes, and Social Security
claiming.

## Status

The repository now prioritises internal consistency and portability.

- The code runs from staged repo-local paths by default.
- Environment variables can override those paths for larger or restricted data.
- Social Security claiming age is derived from `rassageb` (RAND harmonised
  cross-wave variable "age r start receiving Social Security"). The pipeline
  requires this variable and will error if it is missing from the staged
  parquet. Rebuilding the parquet from the full RAND `.dta` will include it.
- MR is estimated for education, cognition, and depression. Neuroticism
  remains in the reduced-form PGS analyses, but is not estimated as an MR trait
  because the current repository does not stage a neuroticism phenotype.
- Multi-instrument sensitivity analyses are run using alternative PGS from
  independent discovery GWAS: overidentification tests, PGS-level Egger,
  multivariable MR, Steiger directionality, and a height PGS benchmark.
  Results are in `output/tables/supp_mr_pleiotropy.csv`.
- A restricted-window event-history analysis (62-70 claiming window, Cox PH
  and discrete-time hazard) is run as a supplementary check on the claiming
  results. Output: `output/tables/ed_table7_restricted_window_survival.csv`.

## Runtime

- Tested with Python `3.14.3`
- Dependencies pinned in `requirements.txt`

Install:

```bash
python3 -m pip install -r requirements.txt
```

## Data Staging

Repo-local defaults:

- Public RAND parquet: `data/raw/public/rand_hrs_core.parquet`
- Public RAND raw `.dta`: `data/raw/public/randhrs1992_2022v1.dta`
- Restricted PGS directory: `data/raw/restricted/`

Environment-variable overrides:

- `RAND_HRS_PARQUET`
- `RAND_HRS_FULL_DTA`
- `HRS_PGS_DTA`

Recommended restricted-data setup:

```bash
export RAND_HRS_FULL_DTA="/absolute/path/to/randhrs1992_2022v1.dta"
export HRS_PGS_DTA="/absolute/path/to/pgs/stata"
```

If `data/raw/public/rand_hrs_core.parquet` is missing but `RAND_HRS_FULL_DTA`
is available, the pipeline will build the staged parquet automatically from the
raw RAND file using only the required columns.

## Run

Main analysis plus supplementary outputs:

```bash
python3 run_all.py
```

Main analysis only:

```bash
python3 run_all.py --skip-supp
```

## Outputs

Authoritative outputs are written to:

- `output/tables/`
- `output/figures/`
- `output/results_summary.md`

The results summary includes:

- the active claiming-age derivation source
- core coefficient summaries
- a curated inventory of manuscript-facing and auxiliary outputs

## Current Display Mapping

- Figure 1: `figure1_depression_mr` (OLS vs IV for depression → wealth and claiming)
- Figure 2: `figure2_claiming_vs_retirement` (PGS → claiming age vs retirement age)
- Figure 3: `figure3_pgs_wealth` (PGS → wealth with/without education control)
- Figure 4: `figure4_summary_heatmap` (all PGS × all outcomes)
- Table 1: sample characteristics (`table1_sample_characteristics.csv`)
- Table 2: PGS → SS claiming age (`table3_pgs_claiming.csv`)
- Table 3: MR estimates (`table5_mr_estimates.csv`)
- Table 4: first-stage diagnostics (`table6_first_stage.csv`)
- Table 5: PGS → wealth (`table2_pgs_wealth.csv`)
- Table 6: PGS → investment participation (`table4_pgs_investment.csv`)

**Extended Data:**
- ED Table 1: PGS → retirement age (`ed_table2_pgs_retirement.csv`)
- ED Table 2: robustness checks (`ed_table3_robustness.csv`)
- ED Table 3: multi-instrument pleiotropy tests (`supp_mr_pleiotropy.csv`)
- ED Table 4: PGS → wealth OLS (`table2_pgs_wealth.csv`)
- ED Table 5: PGS → investment (`table4_pgs_investment.csv`)
- ED Table 6: ancestry replication, cognitive exclusion, OLS vs IV
- ED Table 7: restricted-window event-history (`ed_table7_restricted_window_survival.csv`)

**Note on table numbering:** Manuscript display-item numbers differ from CSV
filenames. The pipeline generates tables with internal keys (e.g.,
`table3_pgs_claiming.csv`); the manuscript renumbers for narrative flow (e.g.,
manuscript Table 2 = `table3_pgs_claiming.csv`). The mapping is documented in
the display items listing above and in `manuscript/outline.md`.

## Article Display Mapping

The accompanying article renumbers outputs for narrative flow. This mapping
connects article display items to the generated output files:

- Fig. 1: `output/figures/figure1_depression_mr.pdf`
- Fig. 2: `output/figures/figure2_claiming_vs_retirement.pdf`
- Fig. 3: `output/figures/figure4_summary_heatmap.pdf`
- Table 1: `output/tables/table3_pgs_claiming.csv`
- Table 2: `output/tables/table5_mr_estimates.csv`
- Table 3: `output/tables/table6_first_stage.csv`
- SI Table S1: `output/tables/table1_sample_characteristics_formatted.csv`
- SI Table S2: `output/tables/ed_table2_pgs_retirement.csv`
- SI Table S3: `output/tables/table2_pgs_wealth_formatted.csv`
- SI Table S4: attenuation summary from `output/tables/table2_pgs_wealth.csv`
- SI Table S5: `output/tables/table4_pgs_investment_formatted.csv`
- SI Table S6: `output/tables/ed_table3_robustness.csv`
- SI Table S7: `output/tables/supp_mr_pleiotropy.csv`
- SI Table S8a: `output/tables/supp_mr_sensitivity.csv`
- SI Table S8b: `output/tables/supp_ancestry_replication.csv`
- SI Table S8c: `output/tables/supp_cognitive_exclusion.csv`
- SI Table S9: `output/tables/ed_table7_restricted_window_survival.csv`
- SI Fig. S1: `output/figures/figure3_pgs_wealth.pdf`

## Restricted Data Note

The PGS files are restricted and should not be committed. This repository can
reproduce the analysis only for users who already have authorised access to the
restricted HRS genetic data.

## Verification

Current smoke check:

```bash
python3 run_all.py
python3 -m compileall run_all.py src
```
