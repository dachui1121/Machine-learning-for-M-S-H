"""Rebuild the archived 27-row local XGBoost candidate, then save its fitted model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from frozen_common import TRAIN_RANKS, target_features


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "originals" / "newly" / "vendor" / "ssa-mix-design" / "scripts"
sys.path.insert(0, str(VENDOR))
from grouped_model_selection import make_model  # noqa: E402


def main() -> None:
    data = pd.read_csv(ROOT / "assets" / "data" / "experiment_cleaned_all_ages.csv")
    train = data[data["rank"].isin(TRAIN_RANKS)].reset_index(drop=True)
    holdout = data[~data["rank"].isin(TRAIN_RANKS)].reset_index(drop=True)
    if (train["experiment_mix_id"].nunique(), len(train), holdout["experiment_mix_id"].nunique(), len(holdout)) != (9, 27, 24, 72):
        raise ValueError("The frozen 9/24 experiment split is not intact")
    record = json.loads((ROOT / "assets" / "records" / "M2_local_XGBoost_params.json").read_text(encoding="utf-8"))
    model = make_model("XGBoost", record["ssa_params"], 1009)
    model.fit(target_features(train), train["strength"].to_numpy(float))
    predicted = np.asarray(model.predict(target_features(holdout)), dtype=float)
    reference = pd.read_csv(ROOT / "assets" / "records" / "M2_local_predictions.csv")
    reference = reference[reference["scope"] == "holdout_24"]
    expected = reference.set_index(["experiment_mix_id", "curing_age_d"])["predicted_MPa"]
    keys = list(zip(holdout["experiment_mix_id"], holdout["curing_age"].astype(int)))
    delta = float(np.max(np.abs(predicted - np.array([expected.loc[key] for key in keys]))))
    if delta > 1e-5:
        raise RuntimeError(f"Local M2 reconstruction differs from its archive by {delta:.6g} MPa; do not freeze it")
    output = ROOT / "assets" / "models" / "M2_local_XGBoost_reconstructed.joblib"
    joblib.dump(model, output)
    print(f"Saved {output}; max archived-prediction difference = {delta:.3g} MPa")


if __name__ == "__main__":
    main()
