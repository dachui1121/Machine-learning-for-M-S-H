"""Rebuild current M5: similarity-weighted literature XGBoost -> target SVR.

Run after restoring the private assets: python scripts/rebuild_m5.py --output DIR
The original 2026-09-28 source-candidate parameters are fixed; the target SVR
is retuned by the 2026-11-01 grouped SSA protocol. The 24 validation strengths
are read only after both fitted models are complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from frozen_common import CALIBRATOR_FEATURES, SOURCE_FEATURES, TRAIN_RANKS, target_features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "originals" / "newly" / "vendor" / "ssa-mix-design" / "scripts"))
from grouped_model_selection import fit_model, make_model, select_grouped_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    output.mkdir(parents=True)

    assets = ROOT / "assets"
    records = assets / "records"
    manifest = json.loads((records / "M5_run_manifest.json").read_text(encoding="utf-8"))
    lit = pd.read_csv(assets / "data" / "literature_input_selected_fixed.csv")
    exp = pd.read_csv(assets / "data" / "experiment_cleaned_all_ages.csv")
    assert len(lit) == 518 and len(exp) == 99
    weights = lit["domain_similarity_weight"].to_numpy(float)
    assert np.isfinite(weights).all() and weights.min() >= 0 and weights.max() <= 2
    train = exp.loc[exp["rank"].isin(TRAIN_RANKS)].copy().reset_index(drop=True)
    holdout = exp.loc[~exp["rank"].isin(TRAIN_RANKS)].copy().reset_index(drop=True)
    assert len(train) == 27 and train["experiment_mix_id"].nunique() == 9
    assert len(holdout) == 72 and holdout["experiment_mix_id"].nunique() == 24
    assert set(train["experiment_mix_id"]).isdisjoint(holdout["experiment_mix_id"])

    source = make_model("XGBoost", manifest["source_params"], 601)
    fit_model(source, lit[list(SOURCE_FEATURES)].to_numpy(float), lit["strength"].to_numpy(float), weights)
    joblib.dump(source, output / "M5_weighted_source_XGBoost.joblib")

    train_cal = train.copy()
    train_cal[list(CALIBRATOR_FEATURES)] = target_features(train, source)
    selection = select_grouped_model(
        train_cal, list(CALIBRATOR_FEATURES), "strength", "experiment_mix_id",
        output / "target_SVR_SSA", candidates=["SVR"], n_splits=5,
        fast=False, random_state=6011, population=8, iterations=6,
        artifact_prefix="target_SVR",
    )
    calibrator = selection["model"]
    joblib.dump(calibrator, output / "M5_target_calibrator_SVR.joblib")
    if selection["best_info"]["best_source_model_params"] != manifest["target_params"]:
        raise RuntimeError("Target SSA parameters differ from the packaged M5 run")

    prediction = np.asarray(calibrator.predict(target_features(holdout, source)), float)
    reference = pd.read_csv(records / "M5_all_row_predictions.csv")
    reference = reference.loc[reference["subset"] == "holdout24"]
    indexed = reference.set_index(["experiment_mix_id", "curing_age_d"])
    keys = list(zip(holdout["experiment_mix_id"], holdout["curing_age"].astype(int)))
    expected = np.array([indexed.loc[key, "predicted_MPa"] for key in keys], float)
    delta = float(np.max(np.abs(prediction - expected)))
    y = holdout["strength"].to_numpy(float)
    metrics = {
        "model": "M5_weighted_XGBoost_source_SVR_target_calibrator",
        "n_train_mixes": 9, "n_holdout_mixes": 24, "n_holdout_records": 72,
        "R2": float(r2_score(y, prediction)),
        "RMSE_MPa": float(np.sqrt(mean_squared_error(y, prediction))),
        "MAE_MPa": float(mean_absolute_error(y, prediction)),
        "max_reference_prediction_delta_MPa": delta,
    }
    if abs(metrics["R2"] - 0.8807288467902785) > 1e-6 or delta > 1e-5:
        raise RuntimeError(f"M5 rebuild does not match packaged reference: {metrics}")
    (output / "rebuild_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
