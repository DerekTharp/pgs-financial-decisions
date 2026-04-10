"""
Restricted-window event-history analysis for Social Security claiming.

Extended Data Table 7. Supplementary reduced-form timing check:

- event = first retirement-benefit claim observed between ages 62 and 70
- censor follow-up at age 70
- exclude respondents with evidence of a Social Security start before age 62
- fit both Cox PH and discrete-time hazard models
- compare hazard directions with the paper's current OLS claiming regressions

Produces: ed_table7_restricted_window_survival.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from src.config import (
    AGE_VARS,
    OUTPUT_MAP,
    PC_VARS,
    RAND_HRS_PARQUET,
    SS_CLAIMING_AGE_MIN,
    SS_CLAIMING_AGE_VAR,
)
from src.data.merge import load_analysis_sample


WINDOW_MAX = 70
CONTINUOUS_COVARIATES = ["rabyear", "raedyrs"] + PC_VARS
RESULTS_PATH = OUTPUT_MAP.get(
    "ed_table7_survival",
    Path(__file__).with_name("restricted_window_survival_results.csv"),
)

TRAITS = [
    ("pgs_education", "Education", "education"),
    ("pgs_cognition", "Cognition", "cognition"),
    ("pgs_depression", "Depression", "depression"),
    ("pgs_neuroticism", "Neuroticism", "neuroticism"),
    ("pgs_wellbeing", "Well-being", "wellbeing"),
]


def _zscore_in_place(df: pd.DataFrame, columns: list[str]) -> None:
    """Z-score continuous nuisance covariates for numerical stability."""
    for col in columns:
        values = df[col]
        sd = values.std(ddof=0)
        if pd.notna(sd) and sd > 0:
            df[col] = (values - values.mean()) / sd


def _load_current_ols_claiming() -> pd.DataFrame | None:
    """Load the paper's current reduced-form claiming results if available."""
    table_path = OUTPUT_MAP["table3_pgs_claiming"]
    if not table_path.exists():
        return None

    ols = pd.read_csv(table_path)
    ols = ols[ols["model"] == "With education"].copy()
    ols["trait_key"] = ols["trait"].str.lower().str.replace("-", "", regex=False)
    return ols.set_index("trait_key")


def _expected_hr_direction(beta: float) -> str:
    """
    Map OLS years-of-claiming coefficients to expected hazard direction.

    Positive beta => later claiming => lower annual hazard => HR < 1
    Negative beta => earlier claiming => higher annual hazard => HR > 1
    """
    if beta > 0:
        return "HR < 1 expected"
    if beta < 0:
        return "HR > 1 expected"
    return "HR ~= 1 expected"


def build_restricted_window_sample(ancestry: str = "EUR") -> tuple[pd.DataFrame, int]:
    """
    Build the restricted 62-70 claiming window sample.

    Event:
        First observed retirement-benefit claim in the 62-70 window.
    Exit age:
        Claiming age if observed in-window, otherwise min(last observed age, 70).
    Exclusion:
        Any respondent with evidence of a Social Security start before age 62.
    """
    df = load_analysis_sample(ancestry)
    rand = pd.read_parquet(RAND_HRS_PARQUET)
    rand_merged = rand.set_index("hhidpn").loc[df.index]

    age_cols = [c for c in AGE_VARS if c in rand_merged.columns]
    if not age_cols:
        raise ValueError("No age columns available for survival construction.")

    last_age = rand_merged[age_cols].max(axis=1)
    raw_claim = rand_merged[SS_CLAIMING_AGE_VAR]
    analysis_claim = df["ss_claiming_age"].copy()

    exclude_sub62 = raw_claim.notna() & (raw_claim < SS_CLAIMING_AGE_MIN)
    event_claim_age = analysis_claim.where(analysis_claim <= WINDOW_MAX)
    event = event_claim_age.notna().astype(int)

    exit_age = np.minimum(last_age, WINDOW_MAX)
    exit_age[event == 1] = event_claim_age[event == 1]

    surv = pd.DataFrame(
        {
            "event": event,
            "claiming_age_62_70": event_claim_age,
            "exit_age": exit_age,
            "raw_claim": raw_claim,
            "rabyear": df["rabyear"],
            "female": df["female"],
            "raedyrs": df["raedyrs"],
            "hhid": df["hhid"] if "hhid" in df.columns else pd.Series(df.index, index=df.index) // 1000,
        },
        index=df.index,
    )

    for trait_col, _, _ in TRAITS:
        surv[trait_col] = df[trait_col]
    for pc in PC_VARS:
        surv[pc] = df[pc]

    keep_cols = [
        "event",
        "exit_age",
        "rabyear",
        "female",
        "raedyrs",
        "hhid",
    ] + [t[0] for t in TRAITS] + PC_VARS

    n_dropped_sub62 = int(exclude_sub62.sum())
    surv = surv[~exclude_sub62].copy()
    surv = surv[surv["exit_age"] >= SS_CLAIMING_AGE_MIN].copy()
    surv = surv.dropna(subset=keep_cols)

    return surv, n_dropped_sub62


def fit_trait_specific_cox(surv: pd.DataFrame, trait_col: str) -> dict[str, float | int]:
    """Fit a household-clustered Cox model for one trait in the 62-70 window."""
    cols = ["event", "exit_age", "hhid", trait_col, "rabyear", "female", "raedyrs"] + PC_VARS
    sample = surv[cols].copy()
    sample["duration_since_62"] = sample["exit_age"] - SS_CLAIMING_AGE_MIN + 1e-3
    sample = sample.drop(columns=["exit_age"])

    _zscore_in_place(sample, CONTINUOUS_COVARIATES)

    cph = CoxPHFitter()
    cph.fit(
        sample,
        duration_col="duration_since_62",
        event_col="event",
        cluster_col="hhid",
        robust=True,
    )

    hr = float(np.exp(cph.params_[trait_col]))
    ci = np.exp(cph.confidence_intervals_.loc[trait_col])
    p = float(cph.summary.loc[trait_col, "p"])
    ph_p = float(
        proportional_hazard_test(cph, sample, time_transform="rank")
        .summary.loc[trait_col, "p"]
    )

    return {
        "hr": hr,
        "ci_lower": float(ci.iloc[0]),
        "ci_upper": float(ci.iloc[1]),
        "p": p,
        "ph_p": ph_p,
    }


def build_person_age_panel(surv: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the restricted sample to one row per annual age interval.

    Each interval is [age, age + 1). Claims at age 70.x are counted as events in
    the age-70 interval.
    """
    rows: list[dict[str, float | int]] = []
    base_cols = ["rabyear", "female", "raedyrs", "hhid"] + [t[0] for t in TRAITS] + PC_VARS

    for hhidpn, row in surv.iterrows():
        has_event = bool(row["event"])
        claim_age = float(row["claiming_age_62_70"]) if has_event else np.nan
        exit_age = float(row["exit_age"])

        if has_event:
            event_bin = int(np.floor(claim_age))
            max_bin = event_bin
        else:
            max_bin = int(np.ceil(exit_age) - 1)

        if max_bin < SS_CLAIMING_AGE_MIN:
            continue

        base = {col: row[col] for col in base_cols}
        for age_bin in range(SS_CLAIMING_AGE_MIN, max_bin + 1):
            rec = base.copy()
            rec["hhidpn"] = hhidpn
            rec["age_bin"] = age_bin
            rec["event_interval"] = int(has_event and age_bin == event_bin)
            rows.append(rec)

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ValueError("Person-age panel is empty.")
    return panel


def fit_trait_specific_discrete_time(
    panel: pd.DataFrame,
    trait_col: str,
) -> dict[str, float | int]:
    """Fit a clustered complementary log-log discrete-time hazard model."""
    sample = panel.copy()
    _zscore_in_place(sample, CONTINUOUS_COVARIATES)

    formula = (
        "event_interval ~ "
        f"{trait_col} + rabyear + female + raedyrs + "
        + " + ".join(PC_VARS)
        + " + C(age_bin)"
    )

    result = smf.glm(
        formula=formula,
        data=sample,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["hhid"]},
    )

    hr = float(np.exp(result.params[trait_col]))
    ci = np.exp(result.conf_int().loc[trait_col])
    p = float(result.pvalues[trait_col])

    return {
        "hr": hr,
        "ci_lower": float(ci.iloc[0]),
        "ci_upper": float(ci.iloc[1]),
        "p": p,
    }


def run_restricted_window_claiming(ancestry: str = "EUR") -> pd.DataFrame:
    """Run the exploratory restricted-window survival analysis."""
    surv, n_dropped_sub62 = build_restricted_window_sample(ancestry)
    panel = build_person_age_panel(surv)
    ols = _load_current_ols_claiming()

    print("=" * 88)
    print("EXPLORATORY: Restricted 62-70 Claiming Window Survival Check")
    print("=" * 88)
    print()
    print("Definition: event = first observed retirement-benefit claim in ages 62-70")
    print("Censoring: follow-up truncated at age 70")
    print("Exclusion: respondents with raw Social Security start before age 62")
    print()
    print(f"Restricted-window sample: N = {len(surv)}")
    print(f"Households: N = {surv['hhid'].nunique()}")
    print(f"Events in 62-70 window: {surv['event'].sum()} ({surv['event'].mean() * 100:.1f}%)")
    print(
        f"Censored by age 70: {(surv['event'] == 0).sum()} "
        f"({(surv['event'] == 0).mean() * 100:.1f}%)"
    )
    print(
        "Observed through age 70 without a claim in-window: "
        f"{((surv['event'] == 0) & (surv['exit_age'] == WINDOW_MAX)).sum()}"
    )
    print(f"Dropped due to raw claim age < 62: {n_dropped_sub62}")
    print(
        f"In-window claim ages: {surv.loc[surv['event'] == 1, 'claiming_age_62_70'].min():.1f} "
        f"to {surv.loc[surv['event'] == 1, 'claiming_age_62_70'].max():.1f}"
    )
    print()
    print("Interpretation: HR > 1 implies earlier claiming; HR < 1 implies later claiming")
    print()

    print(
        f"{'Trait':<15} {'Model':<14} {'HR':>8} {'95% CI':>22} {'p':>10} "
        f"{'Diag':>10} {'OLS check':>18}"
    )
    print("-" * 110)

    rows: list[dict[str, float | str | int]] = []
    for trait_col, label, key in TRAITS:
        cox = fit_trait_specific_cox(surv, trait_col)
        dth = fit_trait_specific_discrete_time(panel, trait_col)

        ols_check = "n/a"
        ols_beta = np.nan
        if ols is not None and key in ols.index:
            ols_beta = float(ols.loc[key, "beta"])
            expected = _expected_hr_direction(ols_beta)
            observed = "HR > 1" if cox["hr"] > 1 else "HR < 1"
            ols_check = f"{observed}; {expected}"

        print(
            f"{label:<15} {'Cox':<14} {cox['hr']:>8.4f} "
            f"[{cox['ci_lower']:.4f}, {cox['ci_upper']:.4f}] "
            f"{cox['p']:>10.4f} {cox['ph_p']:>10.4f} {ols_check:>18}"
        )
        print(
            f"{'':<15} {'Discrete-time':<14} {dth['hr']:>8.4f} "
            f"[{dth['ci_lower']:.4f}, {dth['ci_upper']:.4f}] "
            f"{dth['p']:>10.4f} {'n/a':>10} {ols_check:>18}"
        )

        rows.append(
            {
                "trait": label,
                "trait_key": key,
                "ols_beta_with_education": ols_beta,
                "ols_direction_check": ols_check,
                "cox_hr": cox["hr"],
                "cox_ci_lower": cox["ci_lower"],
                "cox_ci_upper": cox["ci_upper"],
                "cox_p": cox["p"],
                "cox_ph_p": cox["ph_p"],
                "discrete_time_hr": dth["hr"],
                "discrete_time_ci_lower": dth["ci_lower"],
                "discrete_time_ci_upper": dth["ci_upper"],
                "discrete_time_p": dth["p"],
                "n_people": int(len(surv)),
                "n_households": int(surv["hhid"].nunique()),
                "n_events": int(surv["event"].sum()),
                "person_age_rows": int(len(panel)),
                "panel_people": int(panel["hhidpn"].nunique()),
                "panel_events": int(panel["event_interval"].sum()),
                "window_max": WINDOW_MAX,
                "dropped_raw_claim_lt62": n_dropped_sub62,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_PATH, index=False)

    print()
    print(f"Saved exploratory results to {RESULTS_PATH}")
    if ols is not None:
        print()
        print("=== Current paper: reduced-form OLS on observed claimers only ===")
        for _, row in ols.reset_index(drop=True).iterrows():
            print(
                f"{row['trait']}: beta = {row['beta']:+.3f}, "
                f"p = {row['p']:.4f}, N = {int(row['n'])}"
            )

    return out


if __name__ == "__main__":
    run_restricted_window_claiming()
