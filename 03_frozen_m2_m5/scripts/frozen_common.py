"""Feature contract for the four archived DACHUI models."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRAIN_RANKS = (1, 17, 33, 11, 23, 4, 30, 8, 26)
SOURCE_FEATURES = (
    "effective_MgO", "effective_SiO2", "water", "SHMP", "PCE", "DKP",
    "curing_age", "curing_temperature", "curing_humidity", "CO2",
    "Mg_Si_ratio", "water_binder_ratio", "allowed_admixture_pct",
)
TARGET_FEATURES = (
    "Mg_Si_ratio", "water_binder_ratio", "effective_MgO", "effective_SiO2",
    "water", "SHMP", "allowed_admixture_pct", "curing_temperature",
    "curing_humidity", "CO2", "age_3d", "age_7d", "age_28d",
    "Mg_Si_ratio_sq", "water_binder_ratio_sq", "MgSi_x_wb",
    "MgSi_x_age3", "MgSi_x_age7", "MgSi_x_age28",
    "wb_x_age3", "wb_x_age7", "wb_x_age28",
)
CALIBRATOR_FEATURES = TARGET_FEATURES + (
    "source_pred", "source_pred_x_age3", "source_pred_x_age7",
    "source_pred_x_age28",
)


def target_features(data: pd.DataFrame, source_model=None) -> np.ndarray:
    """Recreate the original _target_feature_frame column order exactly."""
    frame = data.copy().reset_index(drop=True)
    age = pd.to_numeric(frame["curing_age"], errors="coerce").astype(float)
    for day in (3, 7, 28):
        frame[f"age_{day}d"] = (age == float(day)).astype(float)
    mgsi = pd.to_numeric(frame["Mg_Si_ratio"], errors="coerce").astype(float)
    wb = pd.to_numeric(frame["water_binder_ratio"], errors="coerce").astype(float)
    frame["Mg_Si_ratio_sq"] = mgsi**2
    frame["water_binder_ratio_sq"] = wb**2
    frame["MgSi_x_wb"] = mgsi * wb
    for day in (3, 7, 28):
        flag = frame[f"age_{day}d"]
        frame[f"MgSi_x_age{day}"] = mgsi * flag
        frame[f"wb_x_age{day}"] = wb * flag
    if source_model is None:
        return frame[list(TARGET_FEATURES)].to_numpy(float)
    pred = np.asarray(source_model.predict(frame[list(SOURCE_FEATURES)].to_numpy(float)), dtype=float)
    frame["source_pred"] = pred
    for day in (3, 7, 28):
        frame[f"source_pred_x_age{day}"] = pred * frame[f"age_{day}d"].to_numpy(float)
    return frame[list(CALIBRATOR_FEATURES)].to_numpy(float)
