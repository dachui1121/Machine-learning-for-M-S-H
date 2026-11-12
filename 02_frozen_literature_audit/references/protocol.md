# Frozen literature protocol and expected results

**Public release note:** The `assets/` files described below are distributed separately through the journal submission system to authorized reviewers. Extract the private assets package into the public repository root before running this stage.

`assets/frozen_inputs/` contains the cleaned 518-row literature table, 99-row experimental table, literature split assignments, input-only similarity weights, and selected source-model settings. Experimental evaluation uses the fixed nine mixture ranks `1,17,33,11,23,4,30,8,26` (27 records) and their complementary 24 mixtures (72 records).

The 13 features, in model order, are effective MgO, effective SiO2, water, SHMP, PCE, DKP, curing age, curing temperature, curing humidity, CO2, Mg/Si, water/binder ratio, and allowed-admixture percentage. The literature strength records are randomly split by record with seed 111 into Train 362, Validation 78, Test 78; study and mixture IDs may cross partitions. The original SSA search evaluated Ridge, SVR, Random Forest, Extra Trees, and XGBoost with population 4 and iterations 2, selecting by Validation RMSE. This stage refits the selected XGBoost settings; it does not rerun SSA.

Selected XGBoost settings: `n_estimators=183`, `max_depth=4`, `learning_rate=0.0844778775221203`, `subsample=0.8630703185618583`, `colsample_bytree=0.8206494728698313`, `reg_lambda=0.01824223377323712`, and `min_child_weight=6.010297182456568`. The estimator uses median imputation, square-error objective, and `n_jobs=1`.

M1 refits on the 440 Train+Validation rows with seed 8111. SW-Lit-exp uses the same 440 rows and parameters, plus the frozen target-X-only similarity gate computed from all 33 experimental mixture inputs. The top 40% literature records have nonzero weights; 177 of the 440 fit rows are nonzero. Thirty weighted XGBoost fits use seeds 8111–8140, and their predictions are averaged equally. Experimental strength labels are read only for post-fit external metrics.

Expected pooled R² after refit:

| Public model | Fixed 9 (n=27) | Complementary 24 (n=72) |
|---|---:|---:|
| M1 | 0.215572 | 0.387791 |
| SW-Lit-exp | 0.374146 | 0.503953 |

RMSE, MAE, floor-1-MPa MAPE, bias, per-age values, software versions, and SHA-256 input hashes are written by `scripts/run.py`. The run manifest checks pooled R² on the complementary 24 mixtures against these fixed references.
