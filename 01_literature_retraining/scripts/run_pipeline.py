#!/usr/bin/env python3
"""Public runner for Lit-only, Lit-exp, and SW-Lit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
VENDOR_DIR = SKILL_DIR / "vendor" / "ssa-mix-design" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(VENDOR_DIR))

from _processing_core import (  # noqa: E402
    FEATURES,
    _fit_fixed_seed_source_ensemble,
    clean_experiment,
    hard_clean_literature,
    input_only_literature_similarity,
    round4_mix_design,
    rows_to_df,
    select_literature_fraction_by_input_geometry,
    validate_round4_experiment_integrity,
)
from random_row_model_selection import (  # noqa: E402
    make_random_row_assignments,
    select_random_holdout_model,
    split_assignment_manifest,
)


PUBLIC_NAMES = ["Lit-only", "Lit-exp", "SW-Lit"]
CANDIDATES = ["Ridge", "SVR", "RandomForest", "ExtraTrees", "XGBoost"]
SPLIT_SEED = 111
ENSEMBLE_SEEDS = list(range(8111, 8141))
EXPECTED_RANKS = [1, 17, 33, 11, 23, 4, 30, 8, 26]
IMAGE_POINTS = [
    (1, 0.50, 0.40),
    (2, 1.75, 0.50),
    (3, 3.00, 0.55),
    (4, 3.00, 0.40),
    (5, 0.50, 0.55),
    (6, 1.25, 0.40),
    (7, 2.25, 0.55),
    (8, 2.25, 0.40),
    (9, 1.25, 0.55),
]
BENCHMARKS = {
    "Lit-only_test_R2": 0.901337475596655,
    "Lit-exp_pooled_R2": 0.21557234640699097,
    "SW-Lit_pooled_R2": 0.37414623352367704,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Lit-only, Lit-exp, and SW-Lit with the frozen publication protocol."
    )
    parser.add_argument("--literature", required=True, type=Path, help="Raw literature XLSX or cleaned long CSV")
    parser.add_argument("--experiment", type=Path, help="Round-4 experiment XLSX or standardized cleaned long CSV; required except for lit-only")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument(
        "--mode", choices=["all", "lit-only", "lit-exp", "sw-lit"], default="all",
        help="lit-exp and sw-lit automatically run the Lit-only prerequisite",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_config() -> dict[str, Any]:
    return json.loads((SKILL_DIR / "protocol_config.json").read_text(encoding="utf-8"))


def load_literature(path: Path, cfg: dict[str, Any], output: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        cleaning_summary = {"input_type": "standardized_cleaned_csv", "long_records": int(len(frame))}
    else:
        cleaned = hard_clean_literature(path, cfg)
        frame = rows_to_df(cleaned.rows)
        cleaning_summary = {"input_type": "raw_workbook", **cleaned.summary}
        audit_dir = output / "cleaning_audits"
        audit_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in cleaned.audits.items():
            pd.DataFrame(rows).to_csv(audit_dir / f"literature_{name}.csv", index=False, encoding="utf-8-sig")
    missing = sorted(set(FEATURES + ["strength", "study_id", "mix_id"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Literature input is missing required columns: {missing}")
    frame = frame.reset_index(drop=True).copy()
    if frame["strength"].isna().any():
        raise ValueError("Cleaned literature contains missing strength values")
    frame["reproduction_row_id"] = [f"row_{idx:04d}" for idx in range(len(frame))]
    frame["literature_record_id"] = frame["reproduction_row_id"]
    frame.to_csv(output / "literature_cleaned.csv", index=False, encoding="utf-8-sig")
    cleaning_summary.update({
        "long_records": int(len(frame)),
        "mixtures": int(frame["mix_id"].astype(str).nunique()),
        "studies": int(frame["study_id"].astype(str).nunique()),
    })
    write_json(output / "literature_cleaning_summary.json", cleaning_summary)
    return frame


def load_experiment(path: Path, cfg: dict[str, Any], output: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        audits = {}
    else:
        cleaned = clean_experiment(path, cfg)
        frame = rows_to_df(cleaned.rows)
        audits = cleaned.audits
    integrity = validate_round4_experiment_integrity(frame, cfg)
    frame.to_csv(output / "round4_experiment_cleaned.csv", index=False, encoding="utf-8-sig")
    audit_dir = output / "cleaning_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in audits.items():
        pd.DataFrame(rows).to_csv(audit_dir / f"experiment_{name}.csv", index=False, encoding="utf-8-sig")
    write_json(output / "round4_integrity.json", integrity)
    return frame, integrity


def make_assignments(literature: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    assignments = make_random_row_assignments(
        literature["literature_record_id"].astype(str).tolist(),
        train_fraction=0.70, validation_fraction=0.15, test_fraction=0.15,
        random_state=SPLIT_SEED,
    )
    assignments.to_csv(output / "Lit-only_split_assignments.csv", index=False, encoding="utf-8-sig")
    manifest = split_assignment_manifest(
        literature, assignments, dataset_sha256=sha256(output / "literature_cleaned.csv"),
        random_state=SPLIT_SEED,
    )
    write_json(output / "Lit-only_split_manifest.json", manifest)
    return assignments, manifest


def run_lit_only(
    literature: pd.DataFrame, assignments: pd.DataFrame, output: Path,
) -> dict[str, Any]:
    stage = output / "Lit-only"
    result = select_random_holdout_model(
        literature, FEATURES, "strength", "reproduction_row_id", assignments, stage,
        candidates=CANDIDATES, random_state=SPLIT_SEED, population=4, iterations=2,
        artifact_prefix="Lit-only", reproduction_seed_schedule=True,
        production_refit_scope="train_validation",
    )
    info = dict(result["best_info"])
    info.update({
        "public_name": "Lit-only",
        "interpretation": "record-level interpolation; studies and mixtures may cross partitions",
        "experimental_strength_used": False,
    })
    write_json(stage / "Lit-only_best_model.json", info)
    return {**result, "best_info": info}


def select_nine(experiment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    design = round4_mix_design(experiment)
    chosen: list[str] = []
    audit: list[dict[str, Any]] = []
    for label, target_mgsi, target_wb in IMAGE_POINTS:
        available = design[~design["experiment_mix_id"].astype(str).isin(chosen)].copy()
        distance = np.sqrt(
            ((available["Mg_Si_ratio"].astype(float) - target_mgsi) / 2.5) ** 2
            + ((available["water_binder_ratio"].astype(float) - target_wb) / 0.15) ** 2
        )
        row = available.loc[distance.idxmin()]
        mix_id = str(row["experiment_mix_id"])
        chosen.append(mix_id)
        audit.append({
            "image_label": int(label),
            "target_Mg_Si_ratio": target_mgsi,
            "target_water_binder_ratio": target_wb,
            "matched_rank": int(row["rank_int"]),
            "matched_experiment_mix_id": mix_id,
            "matched_Mg_Si_ratio": float(row["Mg_Si_ratio"]),
            "matched_water_binder_ratio": float(row["water_binder_ratio"]),
            "normalized_distance": float(distance.loc[row.name]),
        })
    audit_frame = pd.DataFrame(audit)
    ranks = audit_frame["matched_rank"].astype(int).tolist()
    if ranks != EXPECTED_RANKS:
        raise ValueError(f"Nine-point mapping changed: expected {EXPECTED_RANKS}, found {ranks}")
    label_map = dict(zip(audit_frame["matched_experiment_mix_id"], audit_frame["image_label"]))
    selected = experiment[experiment["experiment_mix_id"].astype(str).isin(chosen)].copy()
    selected["image_label"] = selected["experiment_mix_id"].astype(str).map(label_map).astype(int)
    selected = selected.sort_values(["image_label", "curing_age"]).reset_index(drop=True)
    if len(selected) != 27:
        raise ValueError(f"Expected 27 external records, found {len(selected)}")
    return audit_frame, selected


def metric_rows(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, part in [("pooled_3_7_28d", frame)] + [
        (f"{age}d", frame[np.isclose(frame["curing_age"].astype(float), float(age))])
        for age in (3, 7, 28)
    ]:
        actual = part["actual_strength_MPa"].to_numpy(float)
        predicted = part["predicted_strength_MPa"].to_numpy(float)
        rows.append({
            "workflow": name,
            "evaluation_scope": scope,
            "n": int(len(part)),
            "R2": float(r2_score(actual, predicted)),
            "RMSE_MPa": float(math.sqrt(mean_squared_error(actual, predicted))),
            "MAE_MPa": float(mean_absolute_error(actual, predicted)),
            "Bias_pred_minus_actual_MPa": float(np.mean(predicted - actual)),
        })
    return pd.DataFrame(rows)


def evaluate_external(name: str, model: Any, selected: pd.DataFrame, output: Path) -> pd.DataFrame:
    predictions = selected[
        ["image_label", "rank", "experiment_mix_id", *FEATURES, "strength"]
    ].copy().rename(columns={"strength": "actual_strength_MPa"})
    predictions["predicted_strength_MPa"] = np.asarray(model.predict(selected[FEATURES].to_numpy(float)), float)
    predictions["error_pred_minus_actual_MPa"] = (
        predictions["predicted_strength_MPa"] - predictions["actual_strength_MPa"]
    )
    predictions.to_csv(output / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = metric_rows(name, predictions)
    metrics.to_csv(output / f"{name}_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics


def run_sw_lit(
    literature: pd.DataFrame, assignments: pd.DataFrame, experiment: pd.DataFrame,
    selected: pd.DataFrame, lit_only: dict[str, Any], cfg: dict[str, Any], output: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    stage = output / "SW-Lit"
    design = round4_mix_design(experiment)
    weighted = input_only_literature_similarity(literature, design, cfg)
    fraction, fraction_audit, gated = select_literature_fraction_by_input_geometry(weighted, design, cfg)
    result = _fit_fixed_seed_source_ensemble(
        gated, assignments, stage,
        weight_col="similarity_gate_weight", model_name="XGBoost",
        model_params=lit_only["best_info"]["best_source_model_params"],
        seeds=ENSEMBLE_SEEDS, artifact_prefix="SW-Lit",
    )
    gated.to_csv(stage / "SW-Lit_similarity_weights.csv", index=False, encoding="utf-8-sig")
    fraction_audit.to_csv(stage / "SW-Lit_gate_audit.csv", index=False, encoding="utf-8-sig")
    info = dict(result["best_info"])
    info.update({
        "public_name": "SW-Lit",
        "selection_metric": "Lit-only-selected XGBoost hyperparameters; fixed target-X-only similarity gate",
        "similarity_fraction": float(fraction),
        "target_Y_used_for_similarity_training_selection_or_ensemble": False,
        "target_X_scope": "all 33 Round-4 mixtures",
        "external_metrics_role": "post-hoc audit only",
    })
    write_json(stage / "SW-Lit_best_model.json", info)
    metrics = evaluate_external("SW-Lit", result["model"], selected, stage)
    return info, metrics


def benchmark_check(key: str, observed: float, enabled: bool) -> dict[str, Any]:
    expected = BENCHMARKS[key]
    delta = abs(observed - expected)
    return {
        "expected": expected,
        "observed": observed,
        "absolute_delta": delta,
        "tolerance": 1e-6,
        "status": "PASS" if enabled and delta <= 1e-6 else ("FAIL" if enabled else "NOT_APPLICABLE"),
    }


def main() -> None:
    args = parse_args()
    literature_path = args.literature.resolve()
    experiment_path = args.experiment.resolve() if args.experiment else None
    output = args.output.resolve()
    if not literature_path.is_file():
        raise FileNotFoundError(literature_path)
    if args.mode != "lit-only" and (experiment_path is None or not experiment_path.is_file()):
        raise ValueError("--experiment is required for all, lit-exp, and sw-lit modes")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    cfg = load_config()
    literature = load_literature(literature_path, cfg, output)
    assignments, split_manifest = make_assignments(literature, output)
    lit_only = run_lit_only(literature, assignments, output)

    experiment = None
    integrity = None
    selection_audit = None
    selected = None
    lit_exp_metrics = None
    sw_info = None
    sw_metrics = None
    if args.mode != "lit-only":
        assert experiment_path is not None
        experiment, integrity = load_experiment(experiment_path, cfg, output)
        selection_audit, selected = select_nine(experiment)
        selection_audit.to_csv(output / "nine_mixture_selection_audit.csv", index=False, encoding="utf-8-sig")
        selected.to_csv(output / "nine_mixture_3_7_28d_dataset.csv", index=False, encoding="utf-8-sig")
    if args.mode in {"all", "lit-exp"}:
        assert selected is not None
        lit_exp_dir = output / "Lit-exp"
        lit_exp_dir.mkdir(parents=True)
        lit_exp_metrics = evaluate_external("Lit-exp", lit_only["model"], selected, lit_exp_dir)
        write_json(lit_exp_dir / "Lit-exp_protocol.json", {
            "public_name": "Lit-exp",
            "model": "frozen Lit-only production model",
            "fit_scope": "literature Train+Validation only (440 benchmark rows)",
            "experimental_strength_used_for_training_or_tuning": False,
            "external_strength_role": "post-hoc metrics only",
        })
    if args.mode in {"all", "sw-lit"}:
        assert experiment is not None and selected is not None
        sw_info, sw_metrics = run_sw_lit(
            literature, assignments, experiment, selected, lit_only, cfg, output
        )

    benchmark_data = (
        len(literature) == 518
        and literature["mix_id"].astype(str).nunique() == 170
        and literature["study_id"].astype(str).nunique() == 29
        and split_manifest["actual_counts"] == {"train": 362, "validation": 78, "test": 78}
    )
    checks = {
        "Lit-only": benchmark_check("Lit-only_test_R2", float(lit_only["best_info"]["test_R2"]), benchmark_data)
    }
    if lit_exp_metrics is not None:
        observed = float(lit_exp_metrics.loc[lit_exp_metrics["evaluation_scope"].eq("pooled_3_7_28d"), "R2"].iloc[0])
        checks["Lit-exp"] = benchmark_check("Lit-exp_pooled_R2", observed, benchmark_data)
    if sw_metrics is not None:
        observed = float(sw_metrics.loc[sw_metrics["evaluation_scope"].eq("pooled_3_7_28d"), "R2"].iloc[0])
        checks["SW-Lit"] = benchmark_check("SW-Lit_pooled_R2", observed, benchmark_data)

    manifest = {
        "skill": "lit-only-sw-lit",
        "public_workflows": PUBLIC_NAMES,
        "requested_mode": args.mode,
        "inputs": {
            "literature": {"path": str(literature_path), "sha256": sha256(literature_path)},
            "experiment": None if experiment_path is None else {"path": str(experiment_path), "sha256": sha256(experiment_path)},
        },
        "fixed_protocol": {
            "split": "record-random 70/15/15",
            "split_seed": SPLIT_SEED,
            "split_counts": split_manifest["actual_counts"],
            "production_scope": "Train+Validation only",
            "nine_expected_ranks": EXPECTED_RANKS,
            "sw_lit_gate_fraction": 0.40,
            "sw_lit_ensemble_seeds": ENSEMBLE_SEEDS,
            "target_Y_used_before_external_scoring": False,
        },
        "round4_integrity": integrity,
        "selected_ranks": None if selection_audit is None else selection_audit["matched_rank"].astype(int).tolist(),
        "benchmark_dataset_shape_matched": benchmark_data,
        "benchmark_checks": checks,
    }
    write_json(output / "RUN_MANIFEST.json", manifest)
    print(json.dumps({"output": str(output), "benchmark_checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
