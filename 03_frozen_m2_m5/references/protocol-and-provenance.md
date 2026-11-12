# Protocol and provenance for the M2–M5 package

**Public release note:** The `assets/` files described below are distributed separately through the journal submission system to authorized reviewers. Extract the private assets package into the public repository root before running the frozen audit. The historical source code remains in the public repository.

## Exact new-to-old mapping and source runs

| Current label | Source run | Run identity | Algorithm(s) | Search role | 24-mix pooled R² |
|---|---|---|---|---|---:|
| M2 | 2026-10-19 `XGBoost_27条训练评估_20261019` | fixed local XGBoost candidate, not old M0 winner | XGBoost | Candidate's parameters came from `validation` SSA; 5-fold grouped OOF on nine mixes; all 27 records refitted with seed 1009 | 0.6280687746476659 |
| M3 | 2026-09-28 `Dachui-newly_20260928_publication_all_models` | `M3_equal_merge_SSA` | XGBoost | Independent `publication` SSA 8×6, target `experiment_mix_id`-grouped five-fold OOF | 0.7485938933673911 |
| M4 | 2026-10-16 `Dachui-only6_M5_daily` | `M5_unweighted_source_pred_target_calibrator_SSA` | source XGBoost → target SVR | Independent `daily` SSA 4×2 for each stage; source `study_id` grouped and target `experiment_mix_id` grouped, both five folds | 0.8341040524865244 |
| M5 | 2026-11-01 focused rebuild | `weighted_XGBoost_source_SVR_target_calibrator` | similarity-weighted source XGBoost → target SVR | Source uses the 2026-09-28 weighted-XGBoost candidate parameters, refit on all 518 rows; target SVR uses 8×6 SSA and `experiment_mix_id`-grouped five-fold OOF | 0.8807288467902785 |

M4 and M5 are **direct target-strength calibrators using source prediction as an auxiliary feature**. Current M5 uses the same target-feature architecture as M4, while its source was trained on all eligible literature with 0–2 input-similarity weights. M4 and current M5 were not selected under equal SSA budgets, so their observed R² difference is not an isolated weighting effect. The former M5 was a weighted SVR → SVR run (R² 0.8821908389454742); it is retained only in `original_*` provenance records and is not the current M5 artifact.

## Data and feature contract

- Fourth round only: 33 mixes × 3 ages = 99 records. Training ranks in acquisition order: `1,17,33,11,23,4,30,8,26`; the other 24 rank IDs form the common panel for **these four** archives.
- M2 training is 27 local rows, with 22 target features. In-sample R² was 0.7431, grouped OOF R² 0.2472, and holdout R² 0.6281. It should not be described as the locally selected best model.
- M3 fits one direct model on 518 source rows plus 27 target rows at weight 1 each. Inner OOF validation groups target rows by `experiment_mix_id`; source rows remain in the training side.
- M4 trains its unweighted source XGBoost on 518 literature rows and its target SVR on 27 local rows, where source predictions are extra features. Its source and target selected parameters are in `assets/records/M4_model_selection.csv`.
- M5 refits the archived weighted-XGBoost **candidate parameters** on all 518 literature rows with the original continuous mix-level similarity weights in `[0,2]`, calculated from all 33 experimental **inputs**. Its target SVR is newly tuned on the same 27 local rows. Parameters and evaluation metrics are in `assets/records/M5_run_manifest.json` and `M5_metrics.csv`; all 99 row predictions are in `M5_all_row_predictions.csv`.
- Every two-stage target frame contains materials, categorical ages, nonlinear ratio terms, `source_pred`, and source-prediction × age interactions (26 columns). It is **not** a mandatory additive correction to the source prediction.

Source-model input columns, in exact order: `effective_MgO`, `effective_SiO2`, `water`, `SHMP`, `PCE`, `DKP`, `curing_age`, `curing_temperature`, `curing_humidity`, `CO2`, `Mg_Si_ratio`, `water_binder_ratio`, `allowed_admixture_pct`. Input values should be in the same units and cleaning convention as `assets/data/experiment_cleaned_all_ages.csv`; `curing_age` must be 3, 7, or 28 d for the calibration models. The exact 22/26 target columns and order are defined in `scripts/frozen_common.py`.

## What is bundled

- `assets/data/experiment_cleaned_all_ages.csv`: 99 cleaned fourth-round rows, with labels for the frozen audit.
- `assets/data/literature_input_selected_fixed.csv`: 518 literature rows and original continuous similarity weights. It is a clean-data snapshot, not the raw workbook.
- `assets/data/round4_geometric_nested_order.csv`: frozen nine acquisition ranks and full nested sequence.
- `assets/models/`: M2 reconstructed local model, M3 historical direct model, M4 historical source/calibrator, and newly fitted M5 weighted XGBoost source/SVR calibrator.
- `assets/records/`: current M2–M5 metrics and row-level predictions plus historical `original_*` records for provenance. The historical 0.882 SVR → SVR record is not a current-model reference.
- `originals/newly/` and `originals/only6/`: source-code/config/SSA snapshots that implement historical cleaning, candidate selection, target features, weighting and fitting. They are included for methodological inspection and require the original raw workbooks for a full new training run. `originals/local/evaluate_xgboost_27.py` is the historical one-off local candidate script and contains absolute paths from its original machine; use `scripts/rebuild_local_m2.py` for a portable local rebuild.

## Numerical and scientific limits

`scripts/verify_archive.py` matches all 72 predictions per current model to its stored rows and recomputes pooled R²/RMSE/MAE. `scripts/rebuild_m5.py --output DIR` refits the current M5 source and retunes its target SVR, then checks all 72 holdout predictions against the packaged reference. These checks confirm artifacts and feature ordering; they do not establish prospective performance or a causal weighting effect. The historical serialized models came from Python 3.11, scikit-learn 1.2.2, and XGBoost 3.0.0; newer versions may issue compatibility warnings or fail.

The historical weighted-XGBoost candidate reported literature grouped-OOF R² = 0.163930. Rechecking its saved parameters with the bundled data and code gave R² = 0.254195. Current M5 therefore uses the archived **parameter vector** as a fixed starting choice, but is a new fit rather than an exact replay of the historical candidate's OOF predictions. Both scores and the row-level recheck are retained in the private records.

The all-33-X similarity feature makes M5 transductive with respect to evaluation covariates. No evaluation **strength** was used to form weights or train its source and calibrator. The 24-mix panel is common to these four selected records, but M2's later candidate check occurred after earlier panel results had been inspected; it must not be presented as an untouched prospective benchmark.
