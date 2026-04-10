# Results Summary

Analysis sample: N = 12090
SS claiming age source: rassageb

## PGS -> Wealth (OLS)
- Education (Without education): beta = 0.5614, SE = 0.0469, p < 0.001, N = 12090
- Education (With education): beta = 0.2744, SE = 0.0472, p < 0.001, N = 12060
- Cognition (Without education): beta = 0.4803, SE = 0.0488, p < 0.001, N = 12090
- Cognition (With education): beta = 0.2329, SE = 0.0483, p < 0.001, N = 12060
- Depression (Without education): beta = -0.2667, SE = 0.0463, p < 0.001, N = 12090
- Depression (With education): beta = -0.1946, SE = 0.0453, p < 0.001, N = 12060
- Neuroticism (Without education): beta = -0.3097, SE = 0.0560, p < 0.001, N = 12090
- Neuroticism (With education): beta = -0.2252, SE = 0.0547, p < 0.001, N = 12060
- Well-being (Without education): beta = 0.1913, SE = 0.0464, p < 0.001, N = 12090
- Well-being (With education): beta = 0.1615, SE = 0.0456, p < 0.001, N = 12060

## PGS -> SS Claiming Age
- Education: beta = 0.098 years/SD, p < 0.001, N = 6700
- Cognition: beta = 0.094 years/SD, p < 0.001, N = 6700
- Depression: beta = -0.055 years/SD, p = 0.0174, N = 6700
- Neuroticism: beta = -0.082 years/SD, p = 0.0022, N = 6700
- Well-being: beta = 0.035 years/SD, p = 0.1130, N = 6700

## PGS -> Investment Participation (Logistic)
- Education -> Stock (current): OR = 1.173, p < 0.001, N = 12060
- Education -> IRA (current): OR = 1.183, p < 0.001, N = 12060
- Education -> Stock (ever): OR = 1.173, p < 0.001, N = 12060
- Education -> IRA (ever): OR = 1.182, p < 0.001, N = 12060
- Cognition -> Stock (current): OR = 1.148, p < 0.001, N = 12060
- Cognition -> IRA (current): OR = 1.142, p < 0.001, N = 12060
- Cognition -> Stock (ever): OR = 1.106, p < 0.001, N = 12060
- Cognition -> IRA (ever): OR = 1.118, p < 0.001, N = 12060
- Depression -> Stock (current): OR = 0.906, p < 0.001, N = 12060
- Depression -> IRA (current): OR = 0.936, p = 0.0014, N = 12060
- Depression -> Stock (ever): OR = 0.892, p < 0.001, N = 12060
- Depression -> IRA (ever): OR = 0.929, p < 0.001, N = 12060
- Neuroticism -> Stock (current): OR = 0.939, p = 0.0163, N = 12060
- Neuroticism -> IRA (current): OR = 0.952, p = 0.0420, N = 12060
- Neuroticism -> Stock (ever): OR = 0.922, p < 0.001, N = 12060
- Neuroticism -> IRA (ever): OR = 0.938, p = 0.0117, N = 12060
- Well-being -> Stock (current): OR = 1.005, p = 0.8211, N = 12060
- Well-being -> IRA (current): OR = 1.049, p = 0.0193, N = 12060
- Well-being -> Stock (ever): OR = 1.052, p = 0.0153, N = 12060
- Well-being -> IRA (ever): OR = 1.033, p = 0.1325, N = 12060

## Mendelian Randomisation (IV/2SLS)
- Education -> IHS(wealth): IV = 0.7983, F = 987.8
- Education -> SS claiming age: IV = 0.2870, F = 463.0
- Education -> Stock participation: IV = 0.0822, F = 987.8
- Education -> IRA participation: IV = 0.1020, F = 987.8
- Cognition -> IHS(wealth): IV = 0.4245, F = 964.1
- Cognition -> SS claiming age: IV = 0.1674, F = 595.3
- Cognition -> Stock participation: IV = 0.0459, F = 964.1
- Cognition -> IRA participation: IV = 0.0534, F = 964.1
- Depression -> IHS(wealth): IV = -1.7271, F = 130.7
- Depression -> SS claiming age: IV = -0.6946, F = 46.4
- Depression -> Stock participation: IV = -0.1624, F = 130.7
- Depression -> IRA participation: IV = -0.1496, F = 130.7

## First-Stage Diagnostics
- Education -> raedyrs: F = 1016.8, partial R2 = 0.0755, N = 12060 [STRONG]
- Cognition -> cogtot_avg: F = 974.5, partial R2 = 0.0752, N = 11689 [STRONG]
- Depression -> cesd_avg: F = 129.5, partial R2 = 0.0108, N = 12090 [STRONG]

## PGS -> Retirement Age (expected null)
- Education: beta = 0.087, p = 0.2395, N = 9625
- Cognition: beta = 0.116, p = 0.1301, N = 9625
- Depression: beta = 0.030, p = 0.6778, N = 9625
- Neuroticism: beta = -0.035, p = 0.6838, N = 9625
- Well-being: beta = -0.082, p = 0.2522, N = 9625

## Manuscript-Facing Outputs
- tables/table1_sample_characteristics.csv
- tables/table2_pgs_wealth.csv
- tables/table3_pgs_claiming.csv
- tables/table4_pgs_investment.csv
- tables/table5_mr_estimates.csv
- tables/table6_first_stage.csv
- tables/ed_table2_pgs_retirement.csv
- tables/ed_table3_robustness.csv
- tables/supp_ancestry_replication.csv
- tables/supp_mr_sensitivity.csv
- tables/supp_cognitive_exclusion.csv
- tables/supp_mr_pleiotropy.csv
- tables/ed_table7_restricted_window_survival.csv
- figures/figure1_depression_mr.pdf
- figures/figure1_depression_mr.png
- figures/figure2_claiming_vs_retirement.pdf
- figures/figure2_claiming_vs_retirement.png
- figures/figure3_pgs_wealth.pdf
- figures/figure3_pgs_wealth.png
- figures/figure4_summary_heatmap.pdf
- figures/figure4_summary_heatmap.png

## Auxiliary Outputs
- tables/table1_quintile_breakdown.csv
- tables/table1_sample_characteristics_formatted.csv
- tables/table2_pgs_wealth_formatted.csv
- tables/table3_pgs_claiming_formatted.csv
- tables/table4_pgs_investment_formatted.csv
- tables/table5_mr_estimates_formatted.csv
- tables/table6_first_stage_formatted.csv