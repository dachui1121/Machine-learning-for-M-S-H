"""Deterministic record-level literature holdout and validation-only SSA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from grouped_model_selection import (
    _available_candidates, _metric_dict, _profile, _write_csv,
    bounds_for_model, make_model, params_from_vector,
)
from ssa_algorithm import ssa_optimize


def _hash_rows(rows: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(list(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fit_source_model(model, X, y, sample_weight):
    if sample_weight is None:
        return model.fit(X, y)
    if hasattr(model, "steps"):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": sample_weight})
    return model.fit(X, y, sample_weight=sample_weight)


def _mape_pct(y, prediction) -> float:
    actual = np.asarray(y, float)
    predicted = np.asarray(prediction, float)
    return float(np.mean(np.abs(actual - predicted) / np.maximum(np.abs(actual), 1.0)) * 100.0)


def make_random_row_assignments(
    record_ids: Sequence[str], *, train_fraction: float, validation_fraction: float,
    test_fraction: float, random_state: int,
) -> pd.DataFrame:
    ids = [str(value) for value in record_ids]
    ratios = np.array([train_fraction, validation_fraction, test_fraction], dtype=float)
    if len(ids) < 3 or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("Need at least three unique nonempty literature_record_id values")
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0, atol=1e-9):
        raise ValueError("Split fractions must be positive and sum to one")
    raw = len(ids) * ratios
    counts = np.floor(raw).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() > len(ids):
        options = [i for i in range(3) if counts[i] > 1]
        i = max(options, key=lambda j: (counts[j] - raw[j], counts[j]))
        counts[i] -= 1
    while counts.sum() < len(ids):
        i = max(range(3), key=lambda j: (raw[j] - counts[j], -j))
        counts[i] += 1
    ordered = np.asarray(sorted(ids), dtype=object)
    np.random.default_rng(int(random_state)).shuffle(ordered)
    labels = np.repeat(["train", "validation", "test"], counts)
    result = pd.DataFrame({"literature_record_id": ordered, "split": labels, "shuffle_position": np.arange(len(ids))})
    return result.sort_values("literature_record_id", kind="stable").reset_index(drop=True)


def split_assignment_manifest(df: pd.DataFrame, assignments: pd.DataFrame, *, dataset_sha256: str, random_state: int) -> dict[str, Any]:
    if not df["literature_record_id"].is_unique or not assignments["literature_record_id"].is_unique:
        raise ValueError("Duplicate literature record ID")
    if set(df.literature_record_id.astype(str)) != set(assignments.literature_record_id.astype(str)):
        raise ValueError("Assignment does not cover cleaned literature records exactly once")
    merged = df[["literature_record_id", "study_id", "mix_id"]].merge(assignments, on="literature_record_id", validate="one_to_one")
    counts = assignments.split.value_counts().to_dict()
    overlap = {}
    for col in ("study_id", "mix_id"):
        sets = {role: set(merged.loc[merged.split.eq(role), col].astype(str)) for role in ("train", "validation", "test")}
        overlap[col] = {f"{a}_{b}": len(sets[a] & sets[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    canonical = assignments.sort_values("literature_record_id").to_dict("records")
    return {
        "split_unit": "literature_strength_record", "grouping": "none", "shuffle": True,
        "random_seed": int(random_state), "dataset_sha256": dataset_sha256,
        "assignment_sha256": _hash_rows(canonical),
        "requested_fractions": {"train": .70, "validation": .15, "test": .15},
        "actual_counts": {role: int(counts.get(role, 0)) for role in ("train", "validation", "test")},
        "actual_fractions": {role: float(counts.get(role, 0) / len(assignments)) for role in ("train", "validation", "test")},
        "overlap_counts": overlap,
        "interpretation": "Record-level interpolation; records from the same study or mix may cross partitions.",
    }


def select_random_holdout_model(
    df: pd.DataFrame, features: Sequence[str], target: str, record_id_col: str,
    assignments: pd.DataFrame, output_dir, *, sample_weight_col: str | None = None,
    candidates=None, random_state: int = 111, population: int | None = None,
    iterations: int | None = None, artifact_prefix: str = "source",
    reproduction_seed_schedule: bool = False,
    production_refit_scope: str = "all",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not df[record_id_col].is_unique or not assignments.literature_record_id.is_unique:
        raise ValueError("Duplicate literature record IDs")
    if set(df[record_id_col].astype(str)) != set(assignments.literature_record_id.astype(str)):
        raise ValueError("Assignments must cover every record exactly once")
    if set(assignments.split) != {"train", "validation", "test"}:
        raise ValueError("Train, validation, and test must be nonempty")
    data = df.merge(assignments[["literature_record_id", "split"]], left_on=record_id_col, right_on="literature_record_id", validate="one_to_one", sort=False)
    X = data[list(features)].to_numpy(float)
    y = data[target].to_numpy(float)
    weights = None if sample_weight_col is None else data[sample_weight_col].to_numpy(float)
    if weights is not None and (not np.all(np.isfinite(weights)) or np.any(weights < 0)):
        raise ValueError("Sample weights must be finite and nonnegative")
    index = {role: np.flatnonzero(data.split.eq(role).to_numpy()) for role in ("train", "validation", "test")}
    tr, va, te = (index[role] for role in ("train", "validation", "test"))
    cand = _available_candidates(candidates)
    if not cand:
        raise ValueError("No available model candidates")
    pop, iters = _profile(False, population, iterations)
    summaries = []
    for ci, name in enumerate(cand):
        lb, ub = bounds_for_model(name)

        def fitness(vector):
            try:
                params = params_from_vector(name, np.asarray(vector, float))
                fitness_seed = random_state + 100 * ci + (1 if reproduction_seed_schedule else 0)
                model = make_model(name, params, fitness_seed)
                _fit_source_model(model, X[tr], y[tr], None if weights is None else weights[tr])
                return _metric_dict(y[va], model.predict(X[va]))["RMSE"]
            except Exception:
                return 1e12

        np.random.seed(random_state + ci)
        vector, fitness_rmse, curve = ssa_optimize(fitness, lb, ub, pop=pop, max_iter=iters, verbose=False, model_name=name)
        params = params_from_vector(name, np.asarray(vector, float))
        validation_seed = random_state + (1000 + 100 * ci if reproduction_seed_schedule else 100 * ci)
        model = make_model(name, params, validation_seed)
        _fit_source_model(model, X[tr], y[tr], None if weights is None else weights[tr])
        val = _metric_dict(y[va], model.predict(X[va]))
        summaries.append({"model": name, "params": params, "validation_RMSE": val["RMSE"],
                          "validation_MAE": val["MAE"], "validation_R2": val["R2"],
                          "ssa_fitness_validation_RMSE": float(fitness_rmse),
                          "ssa_curve": [float(value) for value in curve], "population": pop, "iterations": iters})
    best = min(summaries, key=lambda row: (row["validation_RMSE"], row["validation_MAE"], -row["validation_R2"]))
    name, params = best["model"], best["params"]
    selection_model = make_model(name, params, random_state + 7000)
    _fit_source_model(selection_model, X[tr], y[tr], None if weights is None else weights[tr])
    train_pred = np.asarray(selection_model.predict(X[tr]), float)
    validation_pred = np.asarray(selection_model.predict(X[va]), float)
    train_metrics = _metric_dict(y[tr], train_pred)
    validation_metrics = _metric_dict(y[va], validation_pred)
    fit_idx = np.r_[tr, va]
    test_model = make_model(name, params, random_state + 8000)
    _fit_source_model(test_model, X[fit_idx], y[fit_idx], None if weights is None else weights[fit_idx])
    test_pred = np.asarray(test_model.predict(X[te]), float)
    test_metrics = {**_metric_dict(y[te], test_pred), "MAPE_pct": _mape_pct(y[te], test_pred)}
    if production_refit_scope == "train_validation":
        production_model = test_model
        production_fit_rows = int(len(fit_idx))
    elif production_refit_scope == "all":
        production_model = make_model(name, params, random_state + 8000)
        _fit_source_model(production_model, X, y, weights)
        production_fit_rows = int(len(X))
    else:
        raise ValueError("production_refit_scope must be 'train_validation' or 'all'")
    production_path = out / f"{artifact_prefix}_production_model.pkl"
    test_path = out / f"{artifact_prefix}_test_model.pkl"
    joblib.dump(production_model, production_path)
    joblib.dump(test_model, test_path)
    info = {
        "best_source_model_name": name, "best_source_model_params": params,
        "best_source_model_path": str(production_path), "test_model_path": str(test_path),
        "selection_metric": "validation_RMSE",
        "selection_validation_RMSE": best["validation_RMSE"],
        "train_RMSE": train_metrics["RMSE"], "train_MAE": train_metrics["MAE"], "train_R2": train_metrics["R2"],
        "validation_RMSE": validation_metrics["RMSE"], "validation_MAE": validation_metrics["MAE"], "validation_R2": validation_metrics["R2"],
        "test_RMSE": test_metrics["RMSE"], "test_MAE": test_metrics["MAE"], "test_R2": test_metrics["R2"],
        "test_used_for_selection": False, "ssa_population": pop, "ssa_iterations": iters,
        "random_state": int(random_state), "split_unit": "literature_strength_record", "grouping": "none",
        "features": list(features), "sample_weight_column": sample_weight_col,
        "reproduction_seed_schedule": bool(reproduction_seed_schedule),
        "production_refit_scope": production_refit_scope,
        "production_fit_rows": production_fit_rows,
        "source_rows": production_fit_rows,
    }
    rows = []
    metric_rows = []
    for role, idx, predictions in (("train", tr, train_pred), ("validation", va, validation_pred), ("test", te, test_pred)):
        met = _metric_dict(y[idx], predictions)
        metric_rows.append({"split": role, "n_rows": len(idx), **met, "MAPE_pct": _mape_pct(y[idx], predictions)})
        rows.extend({"literature_record_id": str(data.iloc[j][record_id_col]), "split": role,
                     "actual": float(y[j]), "predicted": float(prediction), "model": name}
                    for j, prediction in zip(idx, predictions))
    _write_csv(out / f"{artifact_prefix}_selected_model_predictions.csv", rows)
    _write_csv(out / f"{artifact_prefix}_metrics.csv", metric_rows)
    _write_csv(out / f"{artifact_prefix}_candidate_validation_summary.csv", summaries)
    (out / f"{artifact_prefix}_best_model.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model": production_model, "production_model": production_model, "test_model": test_model,
            "best_info": info, "assignments": assignments, "candidate_summary": summaries,
            "test_metrics": test_metrics, "metrics": metric_rows, "predictions": rows}
