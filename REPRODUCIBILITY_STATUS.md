# Reproducibility check for this package

The stage 1 and 2 checks below were completed for the original release. On 2026-11-01, the updated stage 3 package was checked with Python 3.11.3 and `requirements-publication.txt` in an isolated environment after overlaying the paired private assets. Authorized reviewers can repeat the checks after extracting the private assets package into the repository root:

| Check | Result |
|---|---|
| Stage 1 `--mode all` with the private cleaned literature and experiment CSVs | PASS: `Lit-only`, `Lit-exp`, and `SW-Lit` R² matched reference values within `1e-6`; selected ranks were `1,17,33,11,23,4,30,8,26` |
| Stage 2 frozen literature refit | PASS: aligned-nine and complementary-24 metrics matched the archived results |
| Stage 3 `verify_archive.py`, updated M5 | PASS: all 72 predictions per model matched current packaged reference rows; maximum absolute difference across M2–M5 was below `4e-14` MPa. Current M5 R² = 0.8807288468 |
| Stage 3 `rebuild_m5.py`, updated M5 | PASS: refitted weighted XGBoost source and retuned SVR matched all 72 packaged M5 predictions within `2e-14` MPa |

The stage 3 audit emits version warnings when loading models serialized with scikit-learn 1.2.2 into the 1.9.0 audit environment. The script's compatibility bridge was used, and every prediction was checked against its packaged value. The current M5 rebuild reruns its target-only SSA search; it does not reproduce the historical weighted-XGBoost candidate's original literature OOF predictions, which differ from the saved-parameter recheck (see stage 3 provenance). These checks do not prove prospective generalization.

The public repository also provides `reviewer_run.py`, which performs the private-asset overlay, isolated environment setup, and all three audit commands from one entry point.
