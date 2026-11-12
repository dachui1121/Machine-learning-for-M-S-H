# M–S–H cement strength transfer: literature, similarity, and local calibration

Public code for the three-stage analysis of M–S–H cement compressive strength. The analysis uses 518 cleaned literature strength records and 33 fourth-round experimental mixtures measured at 3, 7, and 28 days. **No data records, trained models, or row-level predictions are included in this public repository.** The authors provide the separate private assets package through the journal submission system to authorized reviewers.

## Repository map

| Directory | Purpose | Main command |
|---|---|---|
| `01_literature_retraining/` | Retrain `Lit-only`; audit it and the source-only similarity model (`Lit-exp`, `SW-Lit`) on the aligned nine mixtures | `scripts/run_pipeline.py` |
| `02_frozen_literature_audit/` | Refit the selected literature settings and evaluate the fixed nine and complementary 24 mixtures | `scripts/run.py` |
| `03_frozen_m2_m5/` | Audit M2–M5 on the common 24-mixture panel; rebuild current M5 | `scripts/verify_archive.py`, `scripts/rebuild_m5.py` |

## Experimental split and workflow

The nine calibration mixture ranks, in acquisition order, are **1, 17, 33, 11, 23, 4, 30, 8, 26**. Each mixture has strength measurements at 3, 7, and 28 days, giving 27 calibration records. The remaining 24 mixtures provide 72 evaluation records.

- `Lit-only` trains a literature model using a 70/15/15 random split of strength records (seed 111; 362/78/78 records). `Lit-exp` evaluates its experimental predictions on the fixed nine mixtures.
- `SW-Lit` weights literature samples using the input designs of all 33 experimental mixtures, without experimental strength labels for source-model fitting or weighting.
- M2–M5 use the same nine labeled mixtures for local fitting or calibration and the complementary 24 for evaluation. M5 combines a similarity-weighted literature XGBoost model with an SVR target calibrator.

Detailed data roles, feature definitions, model-selection settings, and interpretation of the comparisons are documented in the stage [1](01_literature_retraining/references/protocol.md), [2](02_frozen_literature_audit/references/protocol.md), and [3](03_frozen_m2_m5/references/protocol-and-provenance.md) protocols.

## Reproduce the audits

First obtain the private assets package from the journal submission system. The one-command runner below extracts it into this repository's root, restoring `02_frozen_literature_audit/assets/` and `03_frozen_m2_m5/assets/` at their expected paths. The public code alone cannot reproduce numerical results without these files.

Use Python 3.11 in an isolated environment. The frozen models include scikit-learn 1.2.2 and 1.9.0 serializations, so use the versions in `requirements-publication.txt` and run the archive check before using model files. Only load pickle/joblib files from a source you trust.

### One-command reviewer run

After downloading the private ZIP, run this from the repository root with Python 3.11:

```bash
python reviewer_run.py --assets-zip path/to/MSH-private-reviewer-assets.zip
```

The script extracts the private files, creates `.venv`, installs the locked dependencies, runs all three stages, and saves logs and a summary under `results/reviewer_run/`. Package installation requires internet access. On Windows, use `py -3.11` in place of `python` if needed. The script refuses to overwrite existing private assets or results.

### Manual commands

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements-publication.txt

python 01_literature_retraining/scripts/run_pipeline.py \
  --literature 02_frozen_literature_audit/assets/frozen_inputs/literature_hard_cleaned_all_ages.csv \
  --experiment 02_frozen_literature_audit/assets/frozen_inputs/experiment_cleaned_all_ages.csv \
  --output results/stage1 --mode all

python 02_frozen_literature_audit/scripts/run.py --output results/stage2
python 03_frozen_m2_m5/scripts/verify_archive.py
python 03_frozen_m2_m5/scripts/rebuild_m5.py --output results/stage3_m5_rebuild
```

Run these commands from the repository root after extracting the private assets. Stage 1 can also accept the original workbooks, if available; the private cleaned CSVs permit reproduction without them. The stage 1 and 2 runners refuse to overwrite existing output directories. `results/` and `assets/` are ignored by Git.

### Packaged reference values

| Audit | Panel | Pooled R² |
|---|---:|---:|
| `Lit-only` | literature Test, 78 records | 0.901337 |
| `Lit-exp` | aligned nine, 27 records | 0.215572 |
| `SW-Lit` | aligned nine, 27 records | 0.374146 |
| `M1` literature-only, stage 2 | complementary 24, 72 records | 0.387791 |
| `SW-Lit-exp`, stage 2 | complementary 24, 72 records | 0.503953 |
| M2 local XGBoost | complementary 24 | 0.628069 |
| M3 equal literature–experiment merge | complementary 24 | 0.748594 |
| M4 unweighted source + target calibrator | complementary 24 | 0.834104 |
| M5 similarity-weighted XGBoost source + SVR target calibrator | complementary 24 | 0.880729 |

Stage 1 and 2 have distinct literature-model labels; their protocol files specify the corresponding fit settings. The table reports reference values for the packaged models and data.

## Data and software availability

The private assets package supplies the processed data, fitted models, model-selection records, and row-level predictions required by these commands. See the stage protocols for model provenance and detailed reproducibility conditions.

See `PUBLICATION_CHECKLIST.md` for author, citation, license, and upstream software attribution details to complete before publication.
