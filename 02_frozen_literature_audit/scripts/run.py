"""Refit frozen 10.9 M1 and SW-Lit-exp; audit fixed nine and remaining 24 mixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from ensemble_models import MeanPredictionEnsemble


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "assets" / "frozen_inputs"
FIXED9 = [1, 17, 33, 11, 23, 4, 30, 8, 26]
FEATURES = [
    "effective_MgO", "effective_SiO2", "water", "SHMP", "PCE", "DKP",
    "curing_age", "curing_temperature", "curing_humidity", "CO2",
    "Mg_Si_ratio", "water_binder_ratio", "allowed_admixture_pct",
]
PARAMS = {
    "n_estimators": 183, "max_depth": 4, "learning_rate": 0.0844778775221203,
    "subsample": 0.8630703185618583, "colsample_bytree": 0.8206494728698313,
    "reg_lambda": 0.01824223377323712, "min_child_weight": 6.010297182456568,
}
REFERENCE_24_R2 = {'M1': 0.3877912412603818, 'SW-Lit-exp': 0.5039528876763074}


def model(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(objective="reg:squarederror", random_state=seed,
                               n_jobs=1, verbosity=0, **PARAMS)),
    ])


def score(frame: pd.DataFrame) -> dict[str, float | int]:
    actual = frame["actual_strength_MPa"].to_numpy(float)
    predicted = frame["predicted_strength_MPa"].to_numpy(float)
    return {
        "n": len(frame),
        "R2": float(r2_score(actual, predicted)),
        "RMSE_MPa": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE_MPa": float(mean_absolute_error(actual, predicted)),
        "MAPE_pct_denominator_floor_1MPa": float(np.mean(np.abs(actual - predicted) / np.maximum(np.abs(actual), 1)) * 100),
        "bias_pred_minus_actual_MPa": float(np.mean(predicted - actual)),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New output directory; existing paths are not overwritten")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    paths = {
        "literature": INPUTS / "literature_hard_cleaned_all_ages.csv",
        "weights": INPUTS / "literature_input_selected_fixed.csv",
        "split": INPUTS / "literature_random_row_split_assignments.csv",
        "experiment": INPUTS / "experiment_cleaned_all_ages.csv",
    }
    lit, weighted, assignments, experiment = (pd.read_csv(path) for path in paths.values())
    archived = json.loads((INPUTS / "M1_M2_matched_protocol_manifest.json").read_text(encoding="utf-8"))
    if archived["M1"]["selected"]["best_source_model_params"] != PARAMS:
        raise ValueError("Frozen XGBoost settings differ from the 10.9 archive")
    if len(lit) != 518 or len(weighted) != 518 or len(experiment) != 99:
        raise ValueError("Frozen data row counts changed")
    if lit["reproduction_row_id"].tolist() != weighted["reproduction_row_id"].tolist():
        raise ValueError("Literature and similarity weights are misaligned")
    if experiment["rank"].nunique() != 33 or set(FIXED9) - set(experiment["rank"].astype(int)):
        raise ValueError("Fourth-round mixture ranks changed")
    if weighted["similarity_uses_strength"].astype(str).str.lower().eq("true").any():
        raise ValueError("Experimental strength must not enter similarity calculation")

    split = lit["reproduction_row_id"].map(assignments.set_index("reproduction_row_id")["split"])
    if split.value_counts().to_dict() != {"train": 362, "validation": 78, "test": 78}:
        raise ValueError("Frozen 70/15/15 record split changed")
    train = np.flatnonzero(split.eq("train").to_numpy())
    validation = np.flatnonzero(split.eq("validation").to_numpy())
    test = np.flatnonzero(split.eq("test").to_numpy())
    train_validation = np.r_[train, validation]
    tv_mask = split.isin(["train", "validation"]).to_numpy()
    X = lit[FEATURES].to_numpy(float)
    y = lit["strength"].to_numpy(float)
    weights = weighted["similarity_gate_weight"].to_numpy(float)
    if np.count_nonzero(weights[tv_mask]) != 177:
        raise ValueError("Frozen 40% similarity gate changed")

    # M1's original production model uses Train rows followed by Validation rows.
    m1 = model(8111)
    m1.fit(X[train_validation], y[train_validation])
    # SW-Lit-exp uses the same hyperparameters and the original natural row order.
    members = []
    for seed in range(8111, 8141):
        fitted = model(seed)
        fitted.fit(X[tv_mask], y[tv_mask], model__sample_weight=weights[tv_mask])
        members.append(fitted)
    sw_lit_exp = MeanPredictionEnsemble(members, seeds=range(8111, 8141), model_name="XGBoost")

    output.mkdir(parents=True)
    joblib.dump(m1, output / "M1.pkl")
    joblib.dump(sw_lit_exp, output / "SW-Lit-exp.pkl")
    source_test = {}
    for name, fitted in (("M1", m1), ("SW-Lit-exp", sw_lit_exp)):
        source_test[name] = score(pd.DataFrame({
            "actual_strength_MPa": y[test], "predicted_strength_MPa": fitted.predict(X[test]),
        }))

    predictions = []
    metrics = []
    for name, fitted in (("M1", m1), ("SW-Lit-exp", sw_lit_exp)):
        frame = experiment[["rank", "experiment_mix_id", "strength", *FEATURES]].copy()
        frame["rank"] = frame["rank"].astype(int)
        frame.rename(columns={"strength": "actual_strength_MPa"}, inplace=True)
        frame["model"] = name
        frame["predicted_strength_MPa"] = fitted.predict(frame[FEATURES].to_numpy(float))
        frame["residual_pred_minus_actual_MPa"] = frame["predicted_strength_MPa"] - frame["actual_strength_MPa"]
        frame["fixed9_role"] = np.where(frame["rank"].isin(FIXED9), "fixed9", "holdout24")
        predictions.append(frame)
        for role, part in (("fixed9", frame[frame["rank"].isin(FIXED9)]),
                           ("holdout24", frame[~frame["rank"].isin(FIXED9)])):
            for age, subset in (("pooled", part), *((f"{d}d", part[part["curing_age"].eq(d)]) for d in (3, 7, 28))):
                metrics.append({"model": name, "partition": "requested_9", "evaluation_scope": role,
                                "age_scope": age, **score(subset)})
    pd.concat(predictions, ignore_index=True).to_csv(output / "all33_predictions.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(metrics)
    summary.to_csv(output / "all_metrics.csv", index=False, encoding="utf-8-sig")
    mapping = experiment.loc[experiment["rank"].isin(FIXED9),
                             ["rank", "experiment_mix_id", "Mg_Si_ratio", "water_binder_ratio"]].drop_duplicates().copy()
    mapping["user_order"] = mapping["rank"].astype(int).map({rank: i + 1 for i, rank in enumerate(FIXED9)})
    mapping.sort_values("user_order").to_csv(output / "fixed9_mix_mapping.csv", index=False, encoding="utf-8-sig")

    holdout = summary[(summary.partition == "requested_9") &
                      (summary.evaluation_scope == "holdout24") & (summary.age_scope == "pooled")]
    deltas = {row.model: float(row.R2 - REFERENCE_24_R2[row.model]) for row in holdout.itertuples()}
    manifest = {
        "skill": "dachui-m1-sw-lit-exp", "legacy_name_mapping": {"M2": "SW-Lit-exp"},
        "fixed9_ranks_in_user_order": FIXED9,
        "source_split": {"unit": "literature_strength_record", "seed": 111,
                         "train": 362, "validation": 78, "test": 78, "production_fit": 440},
        "models": {"M1": {"algorithm": "XGBoost", "seed": 8111, "params": PARAMS},
                   "SW-Lit-exp": {"algorithm": "30-seed similarity-weighted XGBoost mean ensemble",
                                  "seeds": [8111, 8140], "params": PARAMS,
                                  "gate": "frozen top-40% target-X-only gate, all 33 experiment mixtures",
                                  "nonzero_train_validation_rows": 177}},
        "experimental_strength_used_for_training": False,
        "source_test_metrics": source_test,
        "holdout24_R2_reference": REFERENCE_24_R2,
        "holdout24_R2_delta": deltas,
        "holdout24_reproduced_within_1e_6": all(abs(d) < 1e-6 for d in deltas.values()),
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
    }
    (output / "RUN_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary[summary.age_scope.eq("pooled")].to_string(index=False))
    print("Holdout 24 R2 delta:", deltas)


if __name__ == "__main__":
    main()
