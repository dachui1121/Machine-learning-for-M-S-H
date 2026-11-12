"""Audit the current M2–M5 artifacts against their 24-mix prediction records."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from frozen_common import SOURCE_FEATURES, TRAIN_RANKS, target_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets" / "data" / "experiment_cleaned_all_ages.csv"
MODELS = ROOT / "assets" / "models"
RECORDS = ROOT / "assets" / "records"
EXPECTED_R2 = {"M2": 0.6280687746476659, "M3": 0.7485938933673911,
               "M4": 0.8341040524865244, "M5": 0.8807288467902785}


def load_model(path: Path):
    """Bridge an old SimpleImputer attribute for archive auditing on sklearn 1.9."""
    model = joblib.load(path)
    for _, step in getattr(model, "steps", []):
        if isinstance(step, SimpleImputer) and not hasattr(step, "_fill_dtype"):
            step._fill_dtype = step._fit_dtype
    return model


def archived_predictions(label: str) -> pd.Series:
    if label == "M2":
        frame = pd.read_csv(RECORDS / "M2_local_predictions.csv")
        frame = frame[frame["scope"] == "holdout_24"]
        return frame.set_index(["experiment_mix_id", "curing_age_d"])["predicted_MPa"]
    if label == "M4":
        frame = pd.read_csv(RECORDS / "M4_holdout24_predictions.csv")
        return frame.set_index(["experiment_mix_id", "curing_age"])["predicted"]
    if label == "M5":
        frame = pd.read_csv(RECORDS / "M5_all_row_predictions.csv")
        frame = frame[frame["subset"] == "holdout24"]
        return frame.set_index(["experiment_mix_id", "curing_age_d"])["predicted_MPa"]
    frame = pd.read_csv(RECORDS / "original_holdout24_predictions.csv")
    frame = frame[frame["method"] == "M3_equal_merge_SSA"]
    return frame.set_index(["experiment_mix_id", "curing_age"])["predicted"]


def predict(label: str, holdout: pd.DataFrame) -> np.ndarray:
    if label == "M2":
        model = load_model(MODELS / "M2_local_XGBoost_reconstructed.joblib")
        return np.asarray(model.predict(target_features(holdout)), dtype=float)
    if label == "M3":
        model = load_model(MODELS / "M3_equal_merge_XGBoost.pkl")
        return np.asarray(model.predict(holdout[list(SOURCE_FEATURES)].to_numpy(float)), dtype=float)
    if label == "M4":
        source = load_model(MODELS / "M4_unweighted_source_XGBoost.joblib")
        calibrator = load_model(MODELS / "M4_target_calibrator_SVR.joblib")
    else:
        source = load_model(MODELS / "M5_weighted_source_XGBoost.joblib")
        calibrator = load_model(MODELS / "M5_target_calibrator_SVR.joblib")
        if source.named_steps["model"].__class__.__name__ != "XGBRegressor":
            raise TypeError("M5 source artifact is not XGBoost")
        if calibrator.named_steps["model"].__class__.__name__ != "SVR":
            raise TypeError("M5 target artifact is not SVR")
    return np.asarray(calibrator.predict(target_features(holdout, source)), dtype=float)


def main() -> None:
    data = pd.read_csv(DATA)
    train = data[data["rank"].isin(TRAIN_RANKS)]
    holdout = data[~data["rank"].isin(TRAIN_RANKS)].reset_index(drop=True)
    if (train["experiment_mix_id"].nunique(), len(train), holdout["experiment_mix_id"].nunique(), len(holdout)) != (9, 27, 24, 72):
        raise ValueError("The frozen 9/24 experiment split is not intact")
    keys = list(zip(holdout["experiment_mix_id"], holdout["curing_age"].astype(int)))
    y = holdout["strength"].to_numpy(float)
    results = []
    for label in ("M2", "M3", "M4", "M5"):
        prediction = predict(label, holdout)
        archived = archived_predictions(label)
        expected = np.array([archived.loc[key] for key in keys], dtype=float)
        max_delta = float(np.max(np.abs(prediction - expected)))
        r2 = float(r2_score(y, prediction))
        result = {
            "model": label, "n_mixes": 24, "n_records": 72,
            "R2": r2, "RMSE_MPa": float(np.sqrt(mean_squared_error(y, prediction))),
            "MAE_MPa": float(mean_absolute_error(y, prediction)),
            "max_archived_prediction_delta_MPa": max_delta,
        }
        results.append(result)
        if abs(r2 - EXPECTED_R2[label]) > 1e-6 or max_delta > 1e-5:
            raise RuntimeError(f"{label} differs from its packaged reference: {result}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
