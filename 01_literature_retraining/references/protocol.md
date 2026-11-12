# Canonical protocol

## Public model names and data roles

| Name | Fit data | Experimental input use | Experimental strength use |
|---|---|---|---|
| Lit-only | Literature Train+Validation only | None | None |
| Lit-exp | Frozen Lit-only model | Nine mixtures are prediction inputs | Metrics only, after prediction |
| SW-Lit | Similarity-gated literature Train+Validation only | All 33 Round-4 input designs define similarity | Metrics only, after prediction |

These are source-learning and transfer-audit workflows. None is a target-label calibration model.

## Literature cleaning

The raw workbook parser retains all valid strength ages and converts each strength cell into one long-format record. It uses effective MgO and effective SiO2, excludes rows containing aggregate/sand/gravel, excludes non-SHMP admixtures, maps missing/not-reported CO2 to 0.04%, and derives Mg/Si molar ratio, water/binder ratio, and allowed-admixture percentage. The canonical cleaned benchmark contains 518 records from 170 mixtures and 29 studies.

## Features

1. effective MgO (kg/m³)
2. effective SiO2 (kg/m³)
3. water (kg/m³)
4. SHMP (kg/m³)
5. PCE (kg/m³)
6. DKP (kg/m³)
7. curing age (d)
8. curing temperature (°C)
9. curing humidity (%)
10. CO2 concentration (%)
11. Mg/Si molar ratio
12. water/binder ratio
13. allowed-admixture percentage

## Lit-only

- Split unit: individual long-format strength record.
- Shuffle seed: 111.
- Fractions: 70% Train, 15% Validation, 15% Test.
- Canonical counts: 362/78/78.
- The same study or mixture may appear in more than one partition.
- Candidates: Ridge, SVR, Random Forest, Extra Trees, XGBoost.
- Search: SSA, population 4, iterations 2.
- Selection: lowest Validation RMSE; Test is not used for selection.
- Selected benchmark family: XGBoost.
- Production refit: Train+Validation only, n=440.
- Canonical Test result: R²=0.9013374756, RMSE=5.0175236830 MPa, MAE=3.4316449579 MPa.

This measures record-level interpolation. It is intentionally different from study-grouped or leave-one-study-out validation.

## Lit-exp

Use the frozen Lit-only production model and the following nine Round-4 design points. Each point contributes 3, 7, and 28 d observations, giving 27 external records.

| Image label | Mg/Si | w/b | Expected rank |
|---:|---:|---:|---:|
| 1 | 0.50 | 0.40 | 1 |
| 2 | 1.75 | 0.50 | 17 |
| 3 | 3.00 | 0.55 | 33 |
| 4 | 3.00 | 0.40 | 11 |
| 5 | 0.50 | 0.55 | 23 |
| 6 | 1.25 | 0.40 | 4 |
| 7 | 2.25 | 0.55 | 30 |
| 8 | 2.25 | 0.40 | 8 |
| 9 | 1.25 | 0.55 | 26 |

Selection is by nearest `(Mg/Si, w/b)` coordinate with a uniqueness check, not by physical spreadsheet row. The aligned-nine frozen reference in the companion archive gives pooled R²=0.2155723464, RMSE=10.8400769107 MPa, MAE=8.5123764996 MPa. Recomputed values from this runner must be checked against that reference before claiming exact reproduction.

## SW-Lit

- Use all 33 Round-4 mixtures' input covariates, never their strengths, to calculate literature-to-target similarity.
- Similarity features: Mg/Si, w/b, allowed-admixture percentage, curing temperature, curing humidity, and CO2.
- Retain the top 40% of literature records by target-input similarity as a hard binary gate.
- Normalize nonzero sample weights so the mean weight over all literature records is one.
- Reuse the XGBoost family and hyperparameters selected by Lit-only.
- Fit 30 source-only models with seeds 8111 through 8140 and average predictions equally.
- Refit each model on the same 440 Train+Validation rows; Test remains excluded from fitting.
- Evaluate the same nine external mixtures only after all predictions are frozen.
- The aligned-nine frozen reference in the companion archive gives pooled R²=0.3741462335, RMSE=9.6826080239 MPa, MAE=7.6040572465 MPa. Recomputed values from this runner must be checked against that reference before claiming exact reproduction.

## Reproducibility checks

`RUN_MANIFEST.json` records source paths and SHA-256 hashes, counts, selected ranks, fixed seeds, data-role restrictions, and benchmark comparisons. Small floating-point differences can occur across XGBoost/library versions; any material deviation must be reported rather than silently relabeled as a reproduction.
