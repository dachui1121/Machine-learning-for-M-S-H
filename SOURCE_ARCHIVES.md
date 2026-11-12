# Source archive provenance

This repository was assembled from three author-supplied ZIP packages. SHA-256 values identify the source files before repository assembly:

| Source archive | SHA-256 | Repository directory |
|---|---|---|
| `纯文献内部+文献-9组+相似度-9组.zip` | `4701da5cc0d170c38d4e0d90500532129c0617e6e5105ccc6f7c343d1c9dea4e` | `01_literature_retraining/` |
| `dachui-M1-sw-lit-exp.zip` | `7703d53cc295a988c91b33c29e83076d02d18ebf8986cac5f2a2190c1361831a` | `02_frozen_literature_audit/` |
| `dachui-NC-M2-M5.zip` | `ed9b6f6c19b4d988b0f5410aa3af74eb18bf1c3cdcb96eb390a1ae5e0146369b` | `03_frozen_m2_m5/` |

Repository edits: stage 1 was aligned to ranks `1,17,33,11,23,4,30,8,26`; its reference checks and documentation were updated, and standardized experiment CSV input was enabled. Agent-specific `SKILL.md` and `agents/openai.yaml` files were removed from the publication copy. The historical `originals/local/evaluate_xgboost_27.py` one-off was omitted because it contained private machine paths; the portable M2 rebuild remains available. On 2026-11-01, current M5 was replaced with a new similarity-weighted XGBoost source + SVR calibrator fit. Its models, metrics, and prediction reference are in the paired private assets ZIP; `03_frozen_m2_m5/scripts/rebuild_m5.py` reproduces it. The earlier SVR + SVR M5 survives only in historical `original_*` records. Stage 2 and M2–M4 assets remain unchanged.
