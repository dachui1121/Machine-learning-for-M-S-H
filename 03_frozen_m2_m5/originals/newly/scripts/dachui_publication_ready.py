#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-ready DACHUI V7 story-replay pipeline.

Responsibilities:
- M1: raw-literature source model; literature-internal grouped validation is a validation stage of M1, followed by zero-shot prediction on 9 real experiments.
- M1: raw-literature zero-shot transfer baseline.
- Target-space definition: all 33 Round-4 INPUT covariates, never target strengths.
- M2: independent similarity-weighted literature-only SSA using the fixed 0-2 input-similarity weights.
- M0/M3/M4/M5/M6/M7/M8: few-shot strategies; 9 real experiments are the manuscript operating point, while 3/5/7/9/11/13 remains supplementary sensitivity functionality.
- Primary operating point: 9 labeled target mixtures; the other 24 mixtures are not
  used to train/calibrate the selected 9-mix model.
- Strict cross-budget holdout: 20 mixtures outside the full 13-point sensitivity envelope.

Round-4 strengths are NEVER used to calculate literature similarity weights. The 33
Round-4 input designs define the target covariate space before any strength is released.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import itertools

import joblib
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from scipy.spatial.distance import mahalanobis
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
SSA_CANDIDATES = [
    Path(os.environ["DACHUI_SSA_DIR"]) if os.environ.get("DACHUI_SSA_DIR") else None,
    SKILL_DIR / "vendor" / "ssa-mix-design" / "scripts",
    SKILL_DIR.parent / "ssa-mix-design" / "scripts",
    Path("/Users/dachui/.codex/skills/ssa-mix-design/scripts"),
]
SSA_DIR = next(
    (
        p
        for p in SSA_CANDIDATES
        if p is not None
        and p.exists()
        and (p / "grouped_model_selection.py").exists()
        and "select_target_grouped_model" in (p / "grouped_model_selection.py").read_text(encoding="utf-8")
    ),
    None,
)
if SSA_DIR is None:
    raise RuntimeError(
        "Cannot find a V7-compatible ssa-mix-design interface containing "
        "select_target_grouped_model. Keep the bundled vendor copy or set "
        "DACHUI_SSA_DIR to a compatible scripts directory."
    )
sys.path.insert(0, str(SSA_DIR))

from grouped_model_selection import select_grouped_holdout_model, select_grouped_model, select_target_grouped_model  # noqa: E402

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None


NR_TOKENS = {"", "nr", "n/r", "na", "n/a", "none", "null", "-", "—", "not reported", "未报告"}
MG_O_MOLAR_MASS = 40.304
SI_O2_MOLAR_MASS = 60.0843


@dataclass
class CleanData:
    rows: list[dict[str, Any]]
    audits: dict[str, list[dict[str, Any]]]
    summary: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_number(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        f = float(value)
        return default if not math.isfinite(f) else f
    text = str(value).strip().lower().replace(",", "").replace("％", "%")
    if text in NR_TOKENS:
        return default
    import re

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group()) if match else default


def is_nr(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in NR_TOKENS


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: safe_cell(row.get(k)) for k in keys})


def safe_cell(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def standard_fields(row: dict[str, Any]) -> None:
    mgo = row.get("effective_MgO")
    sio2 = row.get("effective_SiO2")
    water = row.get("water")
    shmp = row.get("SHMP") or 0.0
    pce = row.get("PCE") or 0.0
    dkp = row.get("DKP") or 0.0
    binder = None if mgo is None or sio2 is None else mgo + sio2
    row["water_binder_ratio"] = None if binder in (None, 0) or water is None else water / binder
    row["Mg_Si_ratio"] = None if mgo is None or sio2 in (None, 0) else (mgo / MG_O_MOLAR_MASS) / (sio2 / SI_O2_MOLAR_MASS)
    row["allowed_admixture_pct"] = None if binder in (None, 0) else 100.0 * (shmp + pce + dkp) / binder


def stable_param_hash(values: Sequence[Any]) -> str:
    payload = "|".join("" if v is None else f"{v}" for v in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def detect_strength_columns(headers: Sequence[str]) -> list[tuple[int, float, str]]:
    import re

    cols = []
    for idx, header in enumerate(headers):
        match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*d\s+strength\s*\(MPa\)\s*$", str(header), flags=re.I)
        if match:
            cols.append((idx, float(match.group(1)), str(header)))
    return cols


def hard_clean_literature(path: Path, cfg: dict[str, Any]) -> CleanData:
    lcfg = cfg["literature"]
    wb = load_workbook(path, read_only=False, data_only=True)
    ws = wb[lcfg["sheet"]]
    header_row = int(lcfg["header_row_index"]) + 1
    data_start = int(lcfg["data_start_row_index"]) + 1
    headers = [str(ws.cell(header_row, c).value).strip() if ws.cell(header_row, c).value is not None else "" for c in range(1, ws.max_column + 1)]
    index = {h: i + 1 for i, h in enumerate(headers)}
    cols = lcfg["columns"]
    strength_cols = detect_strength_columns(headers)
    aggregate_headers = [
        h for h in headers
        if any(token in h.lower() for token in ["aggregate", "sand", "gravel"])
        or any(token in h for token in ["粗骨料", "细骨料", "砂", "石子"])
    ]
    aggregate_headers = [h for h in aggregate_headers if h in index]
    allowed_admixtures = set(lcfg.get("allowed_admixtures", [cols["shmp"]]))
    non_shmp_admixture_headers = [
        h for h in headers[33:54]
        if h in index and h not in allowed_admixtures
    ]

    rows: list[dict[str, Any]] = []
    column_log: list[dict[str, Any]] = [
        {"standard_field": "effective_MgO", "source_column": cols["effective_mgo"], "rule": "effective MgO only; nominal MgO forbidden"},
        {"standard_field": "effective_SiO2", "source_column": cols["effective_sio2"], "rule": "effective SiO2/effective silica fume only; nominal SF forbidden"},
        {"standard_field": "SHMP", "source_column": cols["shmp"], "rule": "allowed admixture"},
        {"standard_field": "PCE", "source_column": cols["pce"], "rule": "excluded; SHMP is the only allowed admixture"},
        {"standard_field": "DKP", "source_column": cols["dkp"], "rule": "excluded; SHMP is the only allowed admixture"},
        {"standard_field": "curing_age", "source_column": "all '* d strength (MPa)' columns", "rule": "retain all valid ages"},
    ]
    unit_log = [
        {"dataset": "literature", "field": "effective_MgO", "unit": "kg/m3", "source": cols["effective_mgo"]},
        {"dataset": "literature", "field": "effective_SiO2", "unit": "kg/m3", "source": cols["effective_sio2"]},
        {"dataset": "literature", "field": "SHMP/PCE/DKP", "unit": "kg/m3", "source": "SHMP/NaHMP, PCE, DKP"},
        {"dataset": "literature", "field": "CO2", "unit": "percent", "source": cols["co2"], "note": "NR/N/R/blank/not reported/未报告 -> 0.04, not 0.0004"},
    ]
    co2_log: list[dict[str, Any]] = []
    aggregate_log: list[dict[str, Any]] = []
    removal_log: list[dict[str, Any]] = []

    raw_mix_rows = 0
    kept_mix_ids = set()
    for excel_row in range(data_start, ws.max_row + 1):
        raw_mix_rows += 1
        raw = {h: ws.cell(excel_row, index[h]).value for h in headers if h in index}
        study_id = str(raw.get(cols["study_id"]))
        sample_id = str(raw.get(cols["sample_id"]))
        mix_id = f"L|{study_id}|{sample_id}"
        strengths = [(age, h, as_number(raw.get(h))) for _, age, h in strength_cols]
        has_strength = any(v is not None for _, _, v in strengths)

        positive_aggs = []
        for h in aggregate_headers:
            v = as_number(raw.get(h), 0.0)
            if v is not None and v > 0:
                positive_aggs.append(f"{h}={v}")
        if positive_aggs:
            aggregate_log.append({"source_row": excel_row, "study_id": study_id, "mix_id": mix_id, "positive_aggregate_fields": "; ".join(positive_aggs), "has_valid_strength": has_strength})
            removal_log.append({"source_row": excel_row, "study_id": study_id, "mix_id": mix_id, "delete_reason": "aggregate_or_sand_or_gravel_positive", "problem_fields": "; ".join(positive_aggs), "has_valid_strength": has_strength})
            continue

        positive_other_admixtures = []
        for h in non_shmp_admixture_headers:
            v = as_number(raw.get(h), 0.0)
            if v is not None and v > 0:
                positive_other_admixtures.append(f"{h}={v}")
        if positive_other_admixtures:
            removal_log.append({"source_row": excel_row, "study_id": study_id, "mix_id": mix_id, "delete_reason": "non_SHMP_admixture_positive", "problem_fields": "; ".join(positive_other_admixtures), "has_valid_strength": has_strength})
            continue

        effective_mgo = as_number(raw.get(cols["effective_mgo"]))
        effective_sio2 = as_number(raw.get(cols["effective_sio2"]))
        water = as_number(raw.get(cols["water"]))
        missing = []
        if effective_mgo is None:
            missing.append(cols["effective_mgo"])
        if effective_sio2 is None:
            missing.append(cols["effective_sio2"])
        if water is None:
            missing.append(cols["water"])
        if missing or (effective_mgo is not None and effective_mgo <= 0) or (effective_sio2 is not None and effective_sio2 <= 0):
            reason = "missing_core_field" if missing else "nonpositive_effective_MgO_or_effective_SiO2"
            removal_log.append({"source_row": excel_row, "study_id": study_id, "mix_id": mix_id, "delete_reason": reason, "problem_fields": "; ".join(missing), "effective_MgO": effective_mgo, "effective_SiO2": effective_sio2, "water": water, "has_valid_strength": has_strength})
            continue

        co2_raw = raw.get(cols["co2"])
        co2 = as_number(co2_raw, cfg["co2"]["nr_default_percent"])
        if is_nr(co2_raw):
            co2_log.append({"source_row": excel_row, "study_id": study_id, "mix_id": mix_id, "raw_value": co2_raw, "replacement_percent": cfg["co2"]["nr_default_percent"]})

        base = {
            "domain": "literature",
            "source_row": excel_row,
            "study_id": study_id,
            "mix_id": mix_id,
            "sample_id": sample_id,
            "effective_MgO": effective_mgo,
            "effective_SiO2": effective_sio2,
            "water": water,
            "SHMP": as_number(raw.get(cols["shmp"]), 0.0) or 0.0,
            "PCE": 0.0,
            "DKP": 0.0,
            "curing_temperature": as_number(raw.get(cols["temperature"])),
            "curing_humidity": as_number(raw.get(cols["humidity"])),
            "CO2": co2,
        }
        standard_fields(base)
        kept_mix_ids.add(mix_id)
        for age, strength_col, strength in strengths:
            if strength is None:
                continue
            item = dict(base)
            item["curing_age"] = float(age)
            item["strength"] = strength
            item["strength_source_column"] = strength_col
            rows.append(item)

    wb.close()
    return CleanData(
        rows=rows,
        audits={
            "column_mapping_log": column_log,
            "unit_mapping_log": unit_log,
            "co2_replacement_log": co2_log,
            "aggregate_removal_log": aggregate_log,
            "row_removal_audit": removal_log,
        },
        summary={"raw_mix_rows": raw_mix_rows, "kept_mix_rows": len(kept_mix_ids), "long_records": len(rows), "strength_ages": sorted({r["curing_age"] for r in rows})},
    )


def is_experiment_header(row: Sequence[Any]) -> bool:
    texts = [str(v).strip() for v in row if v is not None]
    return "MgO (kg/m3)" in texts and any("实测" in t and "强度" in t for t in texts)



def initial_space_filling_order(exp_pool: pd.DataFrame, cfg: dict[str, Any], n_initial: int = 3) -> tuple[list[str], list[dict[str, Any]]]:
    """Choose a feature-only space-filling initialization without using strengths."""
    vcfg = cfg.get("validation", {})
    features = list(vcfg.get("pilot_design_features", ["water_binder_ratio", "Mg_Si_ratio"]))
    mixes = exp_pool.sort_values("experiment_mix_id").drop_duplicates("experiment_mix_id").copy()
    if len(mixes) < n_initial:
        raise ValueError(f"Acquisition pool must contain at least {n_initial} independent experiment mixes.")
    raw = mixes[features].to_numpy(float)
    lo, hi = np.nanmin(raw, axis=0), np.nanmax(raw, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    x = (raw - lo) / span

    targets = [np.zeros(len(features)), np.ones(len(features)), np.full(len(features), 0.5)]
    selected: list[int] = []
    for target in targets:
        if len(selected) >= n_initial:
            break
        distances = np.linalg.norm(x - target, axis=1)
        for idx in np.argsort(distances, kind="stable"):
            if int(idx) not in selected:
                selected.append(int(idx))
                break
    while len(selected) < n_initial:
        remaining = [i for i in range(len(mixes)) if i not in selected]
        min_dist = np.array([np.min(np.linalg.norm(x[i] - x[selected], axis=1)) for i in remaining])
        selected.append(int(remaining[int(np.argmax(min_dist))]))

    ordered = mixes.iloc[selected].reset_index(drop=True)
    audit = []
    for order, (_, row) in enumerate(ordered.iterrows(), start=1):
        audit.append({
            "acquisition_step": order,
            "experiment_mix_id": row["experiment_mix_id"],
            "selection_mode": "initial_space_filling",
            "uses_candidate_strength": False,
            **{f: row[f] for f in features},
        })
    return ordered["experiment_mix_id"].astype(str).tolist(), audit


def nested_spatial_pilot_order(exp: pd.DataFrame, cfg: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Backward-compatible feature-only ordering; the publication pipeline uses adaptive_experiment_order."""
    fourth = exp[exp["round"].astype(str).str.contains("第四轮")].copy()
    ids, audit = initial_space_filling_order(fourth, cfg, n_initial=min(3, fourth["experiment_mix_id"].nunique()))
    mixes = fourth.sort_values("experiment_mix_id").drop_duplicates("experiment_mix_id")
    features = list(cfg.get("validation", {}).get("pilot_design_features", ["water_binder_ratio", "Mg_Si_ratio"]))
    raw = mixes[features].to_numpy(float)
    lo, hi = np.nanmin(raw, axis=0), np.nanmax(raw, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    x = (raw - lo) / span
    id_to_idx = {str(mid): i for i, mid in enumerate(mixes["experiment_mix_id"].astype(str))}
    selected = [id_to_idx[i] for i in ids]
    while len(selected) < len(mixes):
        remaining = [i for i in range(len(mixes)) if i not in selected]
        min_dist = np.array([np.min(np.linalg.norm(x[i] - x[selected], axis=1)) for i in remaining])
        idx = int(remaining[int(np.argmax(min_dist))])
        selected.append(idx)
        row = mixes.iloc[idx]
        audit.append({
            "acquisition_step": len(selected),
            "experiment_mix_id": row["experiment_mix_id"],
            "selection_mode": "legacy_feature_only_maximin",
            "uses_candidate_strength": False,
            **{f: row[f] for f in features},
        })
    return mixes.iloc[selected]["experiment_mix_id"].astype(str).tolist(), audit

def experiment_mix_id(round_name: str, scenario: Any, rank: Any, row: dict[str, Any]) -> str:
    if scenario not in (None, "", "未命名场景"):
        return f"E|{round_name}|{scenario}|{rank}"
    h = stable_param_hash([row.get("effective_MgO"), row.get("effective_SiO2"), row.get("water"), row.get("SHMP"), row.get("PCE"), row.get("DKP"), row.get("curing_temperature"), row.get("curing_humidity"), row.get("CO2"), row.get("Mg_Si_ratio"), row.get("water_binder_ratio")])
    return f"E|{round_name}|paramhash:{h}|rank:{rank}"


def clean_experiment(path: Path, cfg: dict[str, Any]) -> CleanData:
    ecfg = cfg["experiment"]
    wb = load_workbook(path, read_only=False, data_only=True)
    ws = wb[ecfg["main_sheet"]]
    table = [[ws.cell(r, c).value for c in range(1, ws.max_column + 1)] for r in range(1, ws.max_row + 1)]
    current_round = "unknown"
    headers: list[str] | None = None
    mixes: list[dict[str, Any]] = []
    column_log = [
        {"standard_field": "effective_MgO", "source_column": "MgO (kg/m3)", "rule": "direct mapping; no activity factor"},
        {"standard_field": "effective_SiO2", "source_column": "SF/硅灰 (kg/m3)", "rule": "direct mapping; no activity factor"},
        {"standard_field": "SHMP", "source_column": "SP (kg/m3)", "rule": "SP means SHMP"},
        {"standard_field": "PCE", "source_column": "none", "rule": "default 0 unless independent PCE column exists"},
        {"standard_field": "DKP", "source_column": "none", "rule": "default 0 unless independent DKP column exists"},
    ]
    unit_log = [
        {"dataset": "experiment", "field": "MgO->effective_MgO", "unit": "kg/m3"},
        {"dataset": "experiment", "field": "SF->effective_SiO2", "unit": "kg/m3"},
        {"dataset": "experiment", "field": "SP->SHMP", "unit": "kg/m3"},
        {"dataset": "experiment", "field": "CO2", "unit": "percent", "note": "0.04 means 0.04%, not 0.0004"},
    ]
    repeated_log: list[dict[str, Any]] = []

    for source_row, raw in enumerate(table, start=1):
        first = raw[0] if raw else None
        if isinstance(first, str) and "轮实验" in first:
            current_round = first.strip()
            headers = None
            continue
        if is_experiment_header(raw):
            headers = [str(v).strip() if v is not None else "" for v in raw]
            continue
        if not headers:
            continue
        record = {headers[i]: raw[i] for i in range(min(len(headers), len(raw))) if headers[i]}
        # V7 Round4-only: earlier rounds are excluded at the experiment parser entry.
        if bool(ecfg.get("round4_only", False)) and "第四轮" not in str(current_round):
            continue
        mgo = as_number(record.get("MgO (kg/m3)"))
        sf = as_number(record.get("SF/硅灰 (kg/m3)"))
        water = as_number(record.get("Water/水 (kg/m3)"))
        if mgo is None or sf is None or water is None:
            continue
        row = {
            "domain": "experiment",
            "source_row": source_row,
            "round": current_round,
            "scenario": record.get("场景") if record.get("场景") not in ("", None) else None,
            "rank": record.get("场景内排名"),
            "effective_MgO": mgo,
            "effective_SiO2": sf,
            "water": water,
            "SHMP": as_number(record.get("SP (kg/m3)"), 0.0) or 0.0,
            "PCE": 0.0,
            "DKP": 0.0,
            "curing_temperature": as_number(record.get("养护温度_C")),
            "curing_humidity": as_number(record.get("湿度条件（%）")),
            "CO2": as_number(record.get("CO2浓度_%"), cfg["co2"]["nr_default_percent"]),
            "strength_3": as_number(record.get("实测3d强度（MPa）")),
            "strength_7": as_number(record.get("实测7d强度（MPa）")),
            "strength_28": as_number(record.get("实测28d强度（MPa）")),
        }
        standard_fields(row)
        row["experiment_mix_id"] = experiment_mix_id(current_round, row["scenario"], row["rank"], row)
        mixes.append(row)

    rows: list[dict[str, Any]] = []
    by_round_scenario_rank = {(m["round"], str(m["scenario"]), str(m["rank"])): m for m in mixes}
    for mix in mixes:
        for age in (3, 7, 28):
            if mix.get(f"strength_{age}") is None:
                continue
            item = {k: v for k, v in mix.items() if not k.startswith("strength_")}
            item["curing_age"] = float(age)
            item["strength"] = float(mix[f"strength_{age}"])
            rows.append(item)

    if (not bool(ecfg.get("round4_only", False))) and ecfg.get("third_round_28d_sheet") in wb.sheetnames:
        ws3 = wb[ecfg["third_round_28d_sheet"]]
        for r in range(2, ws3.max_row + 1):
            scenario = ws3.cell(r, 1).value
            rank = ws3.cell(r, 2).value
            reps = [as_number(ws3.cell(r, c).value) for c in (14, 15, 16)]
            reps = [v for v in reps if v is not None]
            if not reps:
                direct = as_number(ws3.cell(r, 13).value)
                reps = [] if direct is None else [direct]
            if not reps:
                continue
            key = ("第三轮实验（7.4）", str(scenario), str(rank))
            if key not in by_round_scenario_rank:
                continue
            mix = by_round_scenario_rank[key]
            item = {k: v for k, v in mix.items() if not k.startswith("strength_")}
            item["curing_age"] = 28.0
            item["strength"] = float(np.mean(reps))
            item["replicate_values"] = json.dumps(reps, ensure_ascii=False)
            item["replicate_std"] = float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0
            item["replicate_cv"] = item["replicate_std"] / item["strength"] if item["strength"] else None
            rows.append(item)
            repeated_log.append({"source_sheet": ecfg["third_round_28d_sheet"], "source_row": r, "experiment_mix_id": item["experiment_mix_id"], "curing_age": 28, "replicate_values": item["replicate_values"], "mean": item["strength"], "std": item["replicate_std"], "cv": item["replicate_cv"]})
    wb.close()
    return CleanData(
        rows=rows,
        audits={"column_mapping_log": column_log, "unit_mapping_log": unit_log, "replicate_audit": repeated_log},
        summary={"mix_count": len({r["experiment_mix_id"] for r in rows}), "long_records": len(rows), "strength_ages": sorted({r["curing_age"] for r in rows})},
    )


def rows_to_df(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


FEATURES = [
    "effective_MgO",
    "effective_SiO2",
    "water",
    "SHMP",
    "PCE",
    "DKP",
    "curing_age",
    "curing_temperature",
    "curing_humidity",
    "CO2",
    "Mg_Si_ratio",
    "water_binder_ratio",
    "allowed_admixture_pct",
]


# Literature screening deliberately excludes strength and curing_age.  The primary
# Round-4 design domain is Mg/Si x water/binder; curing environment and admixture
# dosage can be retained as secondary input constraints without using any output.
DEFAULT_INPUT_SCREEN_FEATURES = [
    "Mg_Si_ratio",
    "water_binder_ratio",
    "allowed_admixture_pct",
    "curing_temperature",
    "curing_humidity",
    "CO2",
]


def validate_round4_experiment_integrity(exp: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the experiment table is the complete Round-4 target dataset.

    V7 deliberately excludes Rounds 1-3 before any target-domain modeling.
    Validation is mix-group based so all ages of a released mix travel together.
    """
    ecfg = cfg.get("experiment", {})
    required_label = str(ecfg.get("required_round_label", "第四轮"))
    expected_mixes = int(ecfg.get("expected_round4_mix_count", 33))
    required_ages = [int(x) for x in ecfg.get("required_round4_ages_days", [3, 7, 28])]
    expected_per_age = int(ecfg.get("expected_records_per_age", expected_mixes))
    expected_total = int(ecfg.get("expected_total_round4_long_records", expected_mixes * len(required_ages)))

    if exp.empty:
        raise ValueError("Round4-only experiment table is empty.")
    if not exp["round"].astype(str).str.contains(required_label, regex=False).all():
        bad = sorted(exp.loc[~exp["round"].astype(str).str.contains(required_label, regex=False), "round"].astype(str).unique().tolist())
        raise ValueError(f"Round4-only violation: non-Round4 rows remain: {bad}")
    n_mix = int(exp["experiment_mix_id"].astype(str).nunique())
    if n_mix != expected_mixes:
        raise ValueError(f"Expected exactly {expected_mixes} Round-4 mixes, found {n_mix}.")
    dup = exp.duplicated(["experiment_mix_id", "curing_age"], keep=False)
    if bool(dup.any()):
        bad = exp.loc[dup, ["experiment_mix_id", "curing_age"]].astype(str).drop_duplicates().head(20).to_dict("records")
        raise ValueError(f"Duplicate Round-4 mix-age records found: {bad}")

    per_age: dict[int, int] = {}
    for age in required_ages:
        n = int(exp.loc[np.isclose(exp["curing_age"].astype(float), float(age)), "experiment_mix_id"].astype(str).nunique())
        per_age[age] = n
        if bool(ecfg.get("require_complete_age_coverage", True)) and n != expected_per_age:
            raise ValueError(f"Round-4 age {age} d must contain {expected_per_age} independent mixes; found {n}.")
    if len(exp) != expected_total:
        raise ValueError(f"Expected exactly {expected_total} Round-4 long records, found {len(exp)}.")

    ranks = pd.to_numeric(exp["rank"], errors="coerce").dropna().astype(int).unique().tolist()
    ranks = sorted(set(ranks))
    if ranks != list(range(1, expected_mixes + 1)):
        raise ValueError(f"Round-4 ranks must be exactly 1..{expected_mixes}; found {ranks}.")

    return {
        "round4_only": True,
        "round_label": required_label,
        "independent_mix_count": n_mix,
        "long_record_count": int(len(exp)),
        "required_ages_days": required_ages,
        "mix_count_by_age": {str(k): int(v) for k, v in per_age.items()},
        "rank_coverage": [1, expected_mixes],
        "all_target_cv_grouped_by": "experiment_mix_id",
        "earlier_rounds_used": False,
    }


def round4_mix_design(exp: pd.DataFrame) -> pd.DataFrame:
    """Return one input row per Round-4 mix, without consulting strength values."""
    fourth = exp[exp["round"].astype(str).str.contains("第四轮")].copy()
    if fourth.empty:
        raise ValueError("No 第四轮 experiment records were found.")
    design = fourth.sort_values(["rank", "experiment_mix_id"]).drop_duplicates("experiment_mix_id").copy()
    design["rank_int"] = pd.to_numeric(design["rank"], errors="coerce").astype("Int64")
    if design["rank_int"].isna().any():
        raise ValueError("Round-4 mixes require numeric rank values for geometric nested sampling.")
    design["rank_int"] = design["rank_int"].astype(int)
    if design["rank_int"].duplicated().any():
        raise ValueError("Round-4 rank values must be unique at mix level.")
    return design.sort_values("rank_int").reset_index(drop=True)


def _ordinal_grid_coordinates(design: pd.DataFrame, features: Sequence[str]) -> dict[str, np.ndarray]:
    """Map each design level to equally-spaced ordinal coordinates in [0,1].

    This treats the three w/b design levels as lower/middle/upper grid rows.  It
    matches the intended experimental-design geometry even though 0.40, 0.50,
    0.55 are not numerically equally spaced.
    """
    coords_by_feature: dict[str, dict[float, float]] = {}
    rounded_values: dict[str, pd.Series] = {}
    for feature in features:
        # Derived ratios can differ at ~1e-15 from floating-point arithmetic even
        # when they represent the same experimental design level.  Collapse those
        # numerical artifacts before constructing ordinal grid coordinates.
        vals = pd.to_numeric(design[feature], errors="raise").astype(float).round(6)
        rounded_values[feature] = vals
        levels = sorted(vals.unique())
        if len(levels) < 2:
            coords_by_feature[feature] = {float(levels[0]): 0.5}
        else:
            coords_by_feature[feature] = {float(v): i / (len(levels) - 1) for i, v in enumerate(levels)}
    out: dict[str, np.ndarray] = {}
    for idx, row in design.iterrows():
        out[str(row["experiment_mix_id"])] = np.asarray(
            [coords_by_feature[f][float(rounded_values[f].loc[idx])] for f in features], dtype=float
        )
    return out


def _space_coverage_metrics(all_coords: np.ndarray, selected_coords: np.ndarray) -> dict[str, float]:
    d = np.sqrt(((all_coords[:, None, :] - selected_coords[None, :, :]) ** 2).sum(axis=2))
    nearest = d.min(axis=1)
    if len(selected_coords) >= 2:
        sd = np.sqrt(((selected_coords[:, None, :] - selected_coords[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(sd, np.inf)
        selected_nn = sd.min(axis=1)
        min_sep = float(selected_nn.min())
        mean_sep = float(selected_nn.mean())
    else:
        min_sep = mean_sep = float("nan")
    return {
        "covering_radius": float(nearest.max()),
        "mean_nearest_design_distance": float(nearest.mean()),
        "min_selected_separation": min_sep,
        "mean_selected_nearest_separation": mean_sep,
    }


def round4_geometric_nested_order(
    fourth_design: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Predefine the 3/5/7/9/11/13 Round-4 feedback sequence using inputs only.

    Agreed hierarchy for the 11x3 grid:
      3: ranks 1,17,33 (opposite corners + center)
      5: +11,23 (the other two corners)
      7: +4,30 (approximately 1/4 and 3/4 on one diagonal)
      9: +8,26 (approximately 1/4 and 3/4 on the other diagonal)
    The 11- and 13-point additions are then chosen as centrally-symmetric pairs
    that minimize the maximum uncovered design distance, followed by mean coverage
    distance, while preferring better selected-point separation.  No strength is read.
    """
    vcfg = cfg.get("validation", {})
    features = list(vcfg.get("pilot_design_features", ["water_binder_ratio", "Mg_Si_ratio"]))
    budgets = [int(x) for x in vcfg.get("pilot_budgets", [3, 5, 7, 9, 11, 13])]
    if budgets != sorted(budgets) or any(b % 2 == 0 for b in budgets):
        raise ValueError("pilot_budgets must be an increasing odd sequence for paired geometric expansion.")
    if max(budgets) > len(fourth_design):
        raise ValueError("Maximum pilot budget exceeds the number of Round-4 mixes.")

    design = fourth_design.copy()
    rank_to_id = {int(r["rank_int"]): str(r["experiment_mix_id"]) for _, r in design.iterrows()}
    agreed_rank_sequence = [1, 17, 33, 11, 23, 4, 30, 8, 26]
    missing = [r for r in agreed_rank_sequence if r not in rank_to_id]
    if missing:
        raise ValueError(f"Round-4 grid is missing agreed geometric ranks: {missing}")

    coord_map = _ordinal_grid_coordinates(design, features)
    all_ids = design["experiment_mix_id"].astype(str).tolist()
    all_coords = np.vstack([coord_map[mid] for mid in all_ids])
    center = np.full(len(features), 0.5, dtype=float)

    selected_ids = [rank_to_id[r] for r in agreed_rank_sequence]
    selected_ranks = agreed_rank_sequence.copy()
    selection_reason = {
        1: "opposite_corner_A",
        17: "grid_center",
        33: "opposite_corner_B",
        11: "remaining_corner_C",
        23: "remaining_corner_D",
        4: "main_diagonal_quarter_region",
        30: "main_diagonal_three_quarter_region",
        8: "cross_diagonal_quarter_region",
        26: "cross_diagonal_three_quarter_region",
    }

    # From 9 points onward, add two points at a time. Prefer point-reflection
    # symmetry around the ordinal-grid center to preserve two-sided inward coverage.
    for target_budget in [b for b in budgets if b > len(selected_ids)]:
        while len(selected_ids) < target_budget:
            remaining = [mid for mid in all_ids if mid not in selected_ids]
            symmetric_pairs: list[tuple[str, str]] = []
            for i, a in enumerate(remaining):
                ca = coord_map[a]
                target = 2.0 * center - ca
                for b in remaining[i + 1:]:
                    if np.allclose(coord_map[b], target, atol=1e-12):
                        symmetric_pairs.append((a, b))
            candidate_pairs = symmetric_pairs or list(itertools.combinations(remaining, 2))
            scored = []
            for a, b in candidate_pairs:
                trial = selected_ids + [a, b]
                m = _space_coverage_metrics(all_coords, np.vstack([coord_map[mid] for mid in trial]))
                # Lexicographic objective: minimize worst uncovered radius, then
                # average uncovered distance; if tied, maximize point separation.
                key = (
                    m["covering_radius"],
                    m["mean_nearest_design_distance"],
                    -m["min_selected_separation"],
                    -m["mean_selected_nearest_separation"],
                    min(int(design.loc[design["experiment_mix_id"].astype(str) == a, "rank_int"].iloc[0]),
                        int(design.loc[design["experiment_mix_id"].astype(str) == b, "rank_int"].iloc[0])),
                )
                scored.append((key, a, b, m))
            _, a, b, _ = min(scored, key=lambda x: x[0])
            pair = sorted([a, b], key=lambda mid: int(design.loc[design["experiment_mix_id"].astype(str) == mid, "rank_int"].iloc[0]))
            for mid in pair:
                selected_ids.append(mid)
                rank = int(design.loc[design["experiment_mix_id"].astype(str) == mid, "rank_int"].iloc[0])
                selected_ranks.append(rank)
                selection_reason[rank] = f"geometry_optimized_pair_for_budget_{target_budget}"

    audit: list[dict[str, Any]] = []
    for step, mid in enumerate(selected_ids, start=1):
        row = design.loc[design["experiment_mix_id"].astype(str) == mid].iloc[0]
        rank = int(row["rank_int"])
        audit.append({
            "acquisition_step": step,
            "rank": rank,
            "experiment_mix_id": mid,
            "selection_reason": selection_reason.get(rank, "geometry_optimized"),
            "uses_strength": False,
            **{f: float(row[f]) for f in features},
        })

    budget_geometry: list[dict[str, Any]] = []
    for budget in budgets:
        ids = selected_ids[:budget]
        ranks = selected_ranks[:budget]
        m = _space_coverage_metrics(all_coords, np.vstack([coord_map[mid] for mid in ids]))
        budget_geometry.append({
            "pilot_budget": budget,
            "selected_ranks": ",".join(map(str, ranks)),
            "selected_mix_ids": json.dumps(ids, ensure_ascii=False),
            **m,
        })
    return selected_ids, audit, budget_geometry


def input_only_literature_similarity(
    lit: pd.DataFrame, fourth_design: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Compute literature similarity from Round-4 inputs only; never read strength.

    Similarity is calculated at literature-mix level and then propagated to all
    age rows of that mix, preserving the long-table age data for later modeling.
    """
    sim_cfg = cfg.get("similarity", {})
    features = list(sim_cfg.get("input_features", DEFAULT_INPUT_SCREEN_FEATURES))
    primary = [f for f in sim_cfg.get("primary_geometry_features", ["Mg_Si_ratio", "water_binder_ratio"]) if f in features]
    if not primary:
        raise ValueError("At least one primary_geometry_feature is required.")
    missing_cols = [f for f in features if f not in lit.columns or f not in fourth_design.columns]
    if missing_cols:
        raise ValueError(f"Input-only similarity features are missing: {missing_cols}")

    lit_mix = lit.sort_values(["mix_id", "curing_age"]).drop_duplicates("mix_id").copy()
    target_mix = fourth_design.drop_duplicates("experiment_mix_id").copy()
    combined = pd.concat([lit_mix[features], target_mix[features]], ignore_index=True)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(combined)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    L = Xs[:len(lit_mix)]
    T = Xs[len(lit_mix):]

    primary_total = float(sim_cfg.get("primary_geometry_weight", 0.80))
    primary_total = min(max(primary_total, 0.0), 1.0)
    weights = np.zeros(len(features), dtype=float)
    p_idx = [features.index(f) for f in primary]
    s_idx = [i for i in range(len(features)) if i not in p_idx]
    for i in p_idx:
        weights[i] = primary_total / len(p_idx)
    if s_idx:
        for i in s_idx:
            weights[i] = (1.0 - primary_total) / len(s_idx)
    else:
        weights[p_idx] = 1.0 / len(p_idx)

    def wd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.sqrt((((A[:, None, :] - B[None, :, :]) ** 2) * weights[None, None, :]).sum(axis=2))

    dist_lt = wd(L, T)
    k = max(1, min(int(sim_cfg.get("input_domain_neighbors", 5)), len(T)))
    local_dist = np.sort(dist_lt, axis=1)[:, :k].mean(axis=1)
    if len(T) >= 2:
        dtt = wd(T, T)
        np.fill_diagonal(dtt, np.inf)
        target_spacing = float(np.median(np.min(dtt, axis=1)))
    else:
        target_spacing = 1.0
    target_spacing = max(target_spacing, 1e-6)
    geometry_similarity = np.exp(-local_dist / target_spacing)
    geometry_similarity = geometry_similarity / max(float(np.max(geometry_similarity)), 1e-12)

    # M6 keeps ALL physically eligible literature mixes.  The final training
    # weight is assigned by INPUT-similarity RANK, linearly across [0, 2].
    # This avoids the collapse caused by directly rescaling the exponential
    # similarity values: most similar = 2.0, least similar = 0.0, and all
    # intermediate mixes are evenly graded according to similarity order.
    weight_min = float(sim_cfg.get("similarity_weight_min", 0.0))
    weight_max = float(sim_cfg.get("similarity_weight_max", 2.0))
    if weight_max < weight_min:
        raise ValueError("similarity_weight_max must be >= similarity_weight_min")
    n_mix = len(geometry_similarity)
    if n_mix <= 1:
        geometry_weight = np.full_like(geometry_similarity, weight_max, dtype=float)
    else:
        order = np.argsort(geometry_similarity, kind="mergesort")  # low -> high
        rank_scaled = np.empty(n_mix, dtype=float)
        rank_scaled[order] = np.linspace(0.0, 1.0, n_mix)
        geometry_weight = weight_min + (weight_max - weight_min) * rank_scaled

    # Data-quality penalty remains input-only; CO2 missing is intentionally NOT
    # penalized because NR/blank CO2 has already been fixed to 0.04% by rule.
    quality_fields = ["curing_temperature", "curing_humidity"]
    miss_penalty = float(sim_cfg.get("optional_missing_penalty", 0.15))
    quality = np.ones(len(lit_mix), dtype=float)
    for i, (_, row) in enumerate(lit_mix.iterrows()):
        n_missing = sum(pd.isna(row.get(f)) for f in quality_fields)
        quality[i] = max(0.5, 1.0 - miss_penalty * n_missing)

    mix_score = lit_mix[["mix_id", "study_id"]].copy()
    mix_score["input_geometry_distance"] = local_dist
    mix_score["input_geometry_similarity_weight"] = geometry_weight
    mix_score["data_quality_weight"] = quality
    score_map = mix_score.set_index("mix_id")

    out = lit.copy()
    out["input_geometry_distance"] = out["mix_id"].map(score_map["input_geometry_distance"])
    out["input_geometry_similarity_weight"] = out["mix_id"].map(score_map["input_geometry_similarity_weight"])
    out["data_quality_weight"] = out["mix_id"].map(score_map["data_quality_weight"])
    counts = out.groupby("study_id")["mix_id"].transform("count").astype(float)
    exponent = float(sim_cfg.get("study_balance_weight_exponent", 0.5))
    study_balance = np.power(np.maximum(counts, 1.0), -exponent)
    study_balance = study_balance / max(float(study_balance.max()), 1e-12)
    out["study_balance_weight"] = study_balance
    # The M6 training weight is intentionally the pure input-similarity weight.
    # data_quality_weight and study_balance_weight are retained as audit fields
    # but are NOT multiplied into M6, otherwise the most similar mix would no
    # longer be guaranteed to receive exactly 2.0 and the least exactly 0.0.
    out["domain_similarity_weight"] = out["input_geometry_similarity_weight"].astype(float)
    out["similarity_uses_strength"] = False
    out.attrs["input_similarity_features"] = features
    out.attrs["input_feature_weights"] = {f: float(weights[i]) for i, f in enumerate(features)}
    out.attrs["target_spacing"] = target_spacing
    return out


def select_literature_fraction_by_input_geometry(
    weighted_lit: pd.DataFrame, fourth_design: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper for the full-soft-weight M6 literature pool.

    No top-fraction hard screening is performed. Every physically eligible
    literature mix remains in M6 and receives a continuous input-similarity
    weight. The returned fraction is therefore always 1.0. No experiment or
    literature strength is used in this step.
    """
    selected = weighted_lit.copy()
    mix_level = selected.sort_values("input_geometry_similarity_weight", ascending=False).drop_duplicates("mix_id")
    table = pd.DataFrame([{
        "fraction": 1.0,
        "selected_literature_mixes": int(selected["mix_id"].nunique()),
        "selected_long_records": int(len(selected)),
        "selected_studies": int(selected["study_id"].nunique()),
        "mean_selected_input_similarity": float(mix_level["input_geometry_similarity_weight"].mean()),
        "uses_experiment_strength": False,
        "hard_fraction_screening": False,
        "selected_fixed_pool": True,
        "selection_rule": "all_physically_eligible_literature_soft_weighted",
    }])
    return 1.0, table, selected

def input_distribution_comparison(lit: pd.DataFrame, target_design: pd.DataFrame, features: Sequence[str]) -> list[dict[str, Any]]:
    """Descriptive distribution shift for INPUT features only."""
    rows: list[dict[str, Any]] = []
    for f in features:
        lv = pd.to_numeric(lit[f], errors="coerce").dropna().to_numpy(float)
        tv = pd.to_numeric(target_design[f], errors="coerce").dropna().to_numpy(float)
        if len(lv) < 2 or len(tv) < 2:
            continue
        pooled = float(np.std(np.concatenate([lv, tv]), ddof=1))
        ks = ks_2samp(lv, tv)
        wd = float(wasserstein_distance(lv, tv))
        rows.append({
            "feature": f,
            "literature_n": len(lv),
            "round4_design_n": len(tv),
            "literature_mean": float(np.mean(lv)),
            "round4_design_mean": float(np.mean(tv)),
            "KS_statistic": float(ks.statistic),
            "KS_pvalue": float(ks.pvalue),
            "Wasserstein_raw": wd,
            "Wasserstein_standardized": wd / pooled if pooled > 0 else None,
            "uses_strength": False,
        })
    return rows

def distribution_comparison(lit: pd.DataFrame, exp: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for f in FEATURES:
        lv = pd.to_numeric(lit[f], errors="coerce").dropna().to_numpy(float)
        ev = pd.to_numeric(exp[f], errors="coerce").dropna().to_numpy(float)
        if len(lv) < 2 or len(ev) < 2:
            continue
        pooled_std = float(np.std(np.concatenate([lv, ev]), ddof=1))
        ks = ks_2samp(lv, ev)
        raw_w = float(wasserstein_distance(lv, ev))
        rows.append({
            "feature": f,
            "literature_n": len(lv),
            "experiment_n": len(ev),
            "literature_mean": float(np.mean(lv)),
            "experiment_mean": float(np.mean(ev)),
            "literature_median": float(np.median(lv)),
            "experiment_median": float(np.median(ev)),
            "literature_std": float(np.std(lv, ddof=1)),
            "experiment_std": float(np.std(ev, ddof=1)),
            "literature_q25": float(np.quantile(lv, 0.25)),
            "experiment_q25": float(np.quantile(ev, 0.25)),
            "literature_q75": float(np.quantile(lv, 0.75)),
            "experiment_q75": float(np.quantile(ev, 0.75)),
            "SMD": float((np.mean(lv) - np.mean(ev)) / pooled_std) if pooled_std > 0 else None,
            "KS_statistic": float(ks.statistic),
            "KS_pvalue": float(ks.pvalue),
            "Wasserstein_raw": raw_w,
            "Wasserstein_standardized": raw_w / pooled_std if pooled_std > 0 else None,
            "literature_cover_experiment_range": float(np.mean((lv >= np.min(ev)) & (lv <= np.max(ev)))),
            "experiment_cover_literature_range": float(np.mean((ev >= np.min(lv)) & (ev <= np.max(lv)))),
        })
    return sorted(rows, key=lambda r: -abs(r["Wasserstein_standardized"] or 0))


def correlation_tables(df: pd.DataFrame, domain: str, repeats: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    corr_rows, boot_rows = [], []
    for f in FEATURES:
        x = pd.to_numeric(df[f], errors="coerce")
        y = pd.to_numeric(df["strength"], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 5 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
            corr_rows.append({"domain": domain, "feature": f, "rho": None, "pvalue": None, "n": int(mask.sum()), "note": "insufficient_or_constant"})
            continue
        rho = spearmanr(x[mask], y[mask])
        corr_rows.append({"domain": domain, "feature": f, "rho": float(rho.statistic), "pvalue": float(rho.pvalue), "n": int(mask.sum()), "note": ""})
        vals = np.column_stack([x[mask].to_numpy(float), y[mask].to_numpy(float)])
        estimates = []
        for _ in range(repeats):
            idx = rng.integers(0, len(vals), len(vals))
            s = vals[idx]
            if len(np.unique(s[:, 0])) >= 2 and len(np.unique(s[:, 1])) >= 2:
                estimates.append(float(spearmanr(s[:, 0], s[:, 1]).statistic))
        boot_rows.append({"domain": domain, "feature": f, "rho": float(rho.statistic), "ci_2_5": float(np.quantile(estimates, 0.025)) if estimates else None, "ci_97_5": float(np.quantile(estimates, 0.975)) if estimates else None, "n": int(mask.sum())})
    return corr_rows, boot_rows


def _residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values) & np.all(np.isfinite(controls), axis=1)
    out = np.full(len(values), np.nan)
    if mask.sum() < controls.shape[1] + 3:
        return out
    reg = LinearRegression().fit(controls[mask], values[mask])
    out[mask] = values[mask] - reg.predict(controls[mask])
    return out


def partial_correlation_table(lit: pd.DataFrame, exp_train: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for domain, df in [("literature", lit), ("experiment_train", exp_train)]:
        y = pd.to_numeric(df["strength"], errors="coerce").to_numpy(float)
        control_sets = {
            "control_age": ["curing_age"],
            "control_age_water_binder": ["curing_age", "water_binder_ratio"],
        }
        for feature in FEATURES:
            x = pd.to_numeric(df[feature], errors="coerce").to_numpy(float)
            for label, controls in control_sets.items():
                C = df[controls].apply(pd.to_numeric, errors="coerce").to_numpy(float)
                xr = _residualize(x, C)
                yr = _residualize(y, C)
                mask = np.isfinite(xr) & np.isfinite(yr)
                if mask.sum() < 6 or len(np.unique(xr[mask])) < 2 or len(np.unique(yr[mask])) < 2:
                    rows.append({"domain": domain, "feature": feature, "control": label, "rho": None, "pvalue": None, "n": int(mask.sum()), "note": "insufficient_or_constant"})
                    continue
                rho = spearmanr(xr[mask], yr[mask])
                rows.append({"domain": domain, "feature": feature, "control": label, "rho": float(rho.statistic), "pvalue": float(rho.pvalue), "n": int(mask.sum()), "note": ""})
    return rows


def domain_classifier_table(lit: pd.DataFrame, exp_train: pd.DataFrame, oof_path: Path, seed: int) -> list[dict[str, Any]]:
    combined = pd.concat([lit.assign(domain_label=0), exp_train.assign(domain_label=1)], ignore_index=True)
    X = SimpleImputer(strategy="median").fit_transform(combined[FEATURES])
    y = combined["domain_label"].to_numpy(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    n_splits = max(2, min(5, n_pos, n_neg))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    classifiers: dict[str, Any] = {
        "LogisticRegression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=1),
    }
    if XGBClassifier is not None:
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    rows = []
    prob_rows = []
    for name, clf in classifiers.items():
        try:
            probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
            auc = float(roc_auc_score(y, probs))
            fpr, tpr, _ = roc_curve(y, probs)
            rows.append({"model": name, "ROC_AUC": auc, "n_splits": n_splits, "n_literature": n_neg, "n_experiment_train": n_pos})
            for idx, p in enumerate(probs):
                prob_rows.append({
                    "model": name,
                    "row_index": idx,
                    "domain": "experiment" if y[idx] == 1 else "literature",
                    "probability_experiment_domain": float(p),
                    "fpr_tpr_curve_points": "" if idx else json.dumps({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, ensure_ascii=False),
                })
        except Exception as exc:
            rows.append({"model": name, "ROC_AUC": None, "n_splits": n_splits, "error": str(exc)})
    write_csv(oof_path, prob_rows)
    return sorted(rows, key=lambda r: -float(r["ROC_AUC"] if r.get("ROC_AUC") is not None else -1))



# Legacy output-aware similarity and strength-response screening removed in v3.0.

def budget_performance_table(
    pred_rows: list[dict[str, Any]], budgets: Sequence[int], method: str = "M6_fixed_selected_weighted_residual",
    primary_ages: Sequence[int] = (7, 28), mape_floor_mpa: float = 1.0,
) -> list[dict[str, Any]]:
    """Descriptive budget curve only; never used to choose the budget when fed outer-test predictions."""
    df = pd.DataFrame(pred_rows)
    if df.empty:
        return []
    df = df[(df["method"] == method) & df["curing_age"].isin(primary_ages)].copy()
    rows = []
    for budget in budgets:
        item = {"pilot_budget": int(budget), "evaluation_only": True}
        age_vals = []
        for age in primary_ages:
            part = df[(df["pilot_budget"] == budget) & (df["curing_age"] == age)]
            if part.empty:
                continue
            y, p = part["actual"].to_numpy(float), part["predicted"].to_numpy(float)
            m = metrics(y, p)
            mape = float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), mape_floor_mpa)) * 100.0)
            item.update({f"age_{age}_n": len(part), f"age_{age}_RMSE": m["RMSE"], f"age_{age}_MAE": m["MAE"], f"age_{age}_R2": m["R2"], f"age_{age}_MAPE_pct": mape})
            age_vals.append((m["RMSE"], m["MAE"], m["R2"], mape))
        if len(age_vals) == len(primary_ages):
            a = np.asarray(age_vals, dtype=float)
            item.update({"mean_age_RMSE": float(a[:,0].mean()), "mean_age_MAE": float(a[:,1].mean()), "mean_age_R2": float(a[:,2].mean()), "mean_age_MAPE_pct": float(a[:,3].mean())})
        rows.append(item)
    return rows

def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    slope, intercept = (None, None)
    if len(y) >= 2 and len(np.unique(pred)) >= 2:
        slope, intercept = np.polyfit(pred, y, 1)
    return {
        "RMSE": float(math.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)) if len(y) >= 2 else None,
        "mean_bias_pred_minus_true": float(np.mean(pred - y)),
        "calibration_slope_actual_vs_pred": None if slope is None else float(slope),
        "calibration_intercept_actual_vs_pred": None if intercept is None else float(intercept),
    }


def budget_selection_table(
    pred_rows: list[dict[str, Any]],
    budgets: Sequence[int],
    method: str = "M6_fixed_selected_weighted_residual",
    primary_ages: Sequence[int] = (7, 28),
    mape_floor_mpa: float = 1.0,
) -> tuple[list[dict[str, Any]], int, str]:
    """Choose the smallest sufficient pilot budget from fixed-model predictions.

    Age-specific metrics are calculated after pooling all held-out predictions
    for that age. The two primary ages then receive equal weight. MAPE uses a
    1 MPa denominator floor so near-zero strengths cannot dominate selection.
    """
    df = pd.DataFrame(pred_rows)
    df = df[(df["method"] == method) & df["curing_age"].isin(primary_ages)].copy()
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        item: dict[str, Any] = {"pilot_budget": int(budget), "selection_method": method}
        age_metrics = []
        for age in primary_ages:
            part = df[(df["pilot_budget"] == budget) & (df["curing_age"] == age)]
            if part.empty:
                continue
            y = part["actual"].to_numpy(float)
            p = part["predicted"].to_numpy(float)
            m = metrics(y, p)
            mape = float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), mape_floor_mpa)) * 100.0)
            item.update({
                f"age_{int(age)}_n": int(len(part)),
                f"age_{int(age)}_RMSE": m["RMSE"],
                f"age_{int(age)}_MAE": m["MAE"],
                f"age_{int(age)}_R2": m["R2"],
                f"age_{int(age)}_MAPE_pct": mape,
            })
            age_metrics.append((m["RMSE"], m["MAE"], m["R2"], mape))
        if len(age_metrics) != len(primary_ages):
            item.update({"eligible": False, "selection_note": "missing_primary_age"})
        else:
            a = np.asarray(age_metrics, dtype=float)
            item.update({
                "mean_age_RMSE": float(a[:, 0].mean()),
                "mean_age_MAE": float(a[:, 1].mean()),
                "mean_age_R2": float(a[:, 2].mean()),
                "mean_age_MAPE_pct": float(a[:, 3].mean()),
            })
        rows.append(item)

    valid = [r for r in rows if "mean_age_RMSE" in r]
    if not valid:
        raise ValueError("No complete 7 d/28 d fixed-model predictions are available for automatic budget selection.")
    best = {
        "RMSE": min(r["mean_age_RMSE"] for r in valid),
        "MAE": min(r["mean_age_MAE"] for r in valid),
        "R2": max(r["mean_age_R2"] for r in valid),
        "MAPE": min(r["mean_age_MAPE_pct"] for r in valid),
    }
    eligible = []
    for r in valid:
        ok = (
            r["mean_age_RMSE"] <= best["RMSE"] * 1.02
            and r["mean_age_MAE"] <= best["MAE"] * 1.03
            and r["mean_age_MAPE_pct"] <= best["MAPE"] * 1.05
            and r["mean_age_R2"] >= best["R2"] - 0.02
        )
        r["eligible"] = bool(ok)
        r["RMSE_within_2pct"] = bool(r["mean_age_RMSE"] <= best["RMSE"] * 1.02)
        r["MAE_within_3pct"] = bool(r["mean_age_MAE"] <= best["MAE"] * 1.03)
        r["MAPE_within_5pct"] = bool(r["mean_age_MAPE_pct"] <= best["MAPE"] * 1.05)
        r["R2_within_0_02"] = bool(r["mean_age_R2"] >= best["R2"] - 0.02)
        if ok:
            eligible.append(r)
    if eligible:
        chosen = min(int(r["pilot_budget"]) for r in eligible)
        reason = "smallest budget satisfying RMSE/MAE/MAPE/R2 equivalence thresholds"
    else:
        keys = [("mean_age_RMSE", False), ("mean_age_MAE", False), ("mean_age_MAPE_pct", False), ("mean_age_R2", True)]
        weights = [0.35, 0.20, 0.20, 0.25]
        for r in valid:
            score = 0.0
            for (key, higher_is_better), weight in zip(keys, weights):
                vals = [x[key] for x in valid]
                span = max(vals) - min(vals)
                regret = 0.0 if span == 0 else ((max(vals) - r[key]) if higher_is_better else (r[key] - min(vals))) / span
                score += weight * regret
            r["normalized_regret_score"] = float(score)
        min_score = min(r["normalized_regret_score"] for r in valid)
        near = [r for r in valid if r["normalized_regret_score"] <= min_score + 0.02]
        chosen = min(int(r["pilot_budget"]) for r in near)
        reason = "fallback minimum normalized multi-metric regret; smallest budget within 0.02 of best score"
    for r in rows:
        r["selected_for_transfer_ssa"] = int(r["pilot_budget"]) == chosen
    return rows, chosen, reason


def summary_with_ci(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluation-split summary; M7 repeats are averaged within each split first."""
    if not rows:
        return []
    df = pd.DataFrame(rows)
    metric_cols = [c for c in ["RMSE", "MAE", "R2"] if c in df]
    reduced = []
    for (split_id, method), part in df.groupby(["split_id", "method"], dropna=False):
        item = {"split_id": split_id, "method": method}
        for c in metric_cols:
            vals = pd.to_numeric(part[c], errors="coerce").dropna()
            item[c] = float(vals.mean()) if len(vals) else None
        reduced.append(item)
    rdf = pd.DataFrame(reduced)
    winners = Counter()
    for split_id, part in rdf.groupby("split_id"):
        valid = part.dropna(subset=["RMSE"])
        if len(valid):
            winners[str(valid.loc[valid["RMSE"].idxmin(), "method"])] += 1
    out = []
    for method, part in rdf.groupby("method"):
        rmses = pd.to_numeric(part["RMSE"], errors="coerce").dropna().to_numpy(float)
        maes = pd.to_numeric(part["MAE"], errors="coerce").dropna().to_numpy(float)
        r2s = pd.to_numeric(part["R2"], errors="coerce").dropna().to_numpy(float)
        row = {
            "method": method,
            "evaluation_splits": int(len(part)),
            "RMSE_mean": float(rmses.mean()) if len(rmses) else None,
            "RMSE_std": float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0,
            "MAE_mean": float(maes.mean()) if len(maes) else None,
            "R2_fold_mean": float(r2s.mean()) if len(r2s) else None,
            "winner_count": int(winners.get(str(method), 0)),
        }
        if len(rmses) > 1:
            se = rmses.std(ddof=1) / math.sqrt(len(rmses))
            row["RMSE_ci95_low"] = float(rmses.mean() - 1.96 * se)
            row["RMSE_ci95_high"] = float(rmses.mean() + 1.96 * se)
        elif len(rmses):
            row["RMSE_ci95_low"] = row["RMSE_ci95_high"] = float(rmses.mean())
        out.append(row)
    return sorted(out, key=lambda r: float(r["RMSE_mean"] if r["RMSE_mean"] is not None else float("inf")))


def pooled_prediction_summary(pred_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Primary pooled held-out metrics; R2 is computed after pooling predictions, never averaged across folds."""
    if not pred_rows:
        return []
    df = pd.DataFrame(pred_rows)
    rows = []
    for method, part in df.groupby("method"):
        if str(method).startswith("M7_") and "m7_random_repeat" in part:
            keys = [c for c in ["outer_fold", "experiment_mix_id", "curing_age", "actual"] if c in part]
            part = part.groupby(keys, as_index=False)["predicted"].mean()
        y = part["actual"].to_numpy(float)
        p = part["predicted"].to_numpy(float)
        rows.append({"method": method, "pooled_n": int(len(part)), **metrics(y, p)})
    return sorted(rows, key=lambda r: r["RMSE"])
def fourth_round_representative(lit: pd.DataFrame, exp: pd.DataFrame) -> list[dict[str, Any]]:
    fourth = exp[exp["round"].astype(str).str.contains("第四轮")]
    previous = exp[~exp["round"].astype(str).str.contains("第四轮")]
    rows = []
    for label, target in [("fourth_vs_previous", previous), ("fourth_vs_literature", lit)]:
        for f in FEATURES + ["strength"]:
            if f not in target or f not in fourth:
                continue
            a = pd.to_numeric(fourth[f], errors="coerce").dropna().to_numpy(float)
            b = pd.to_numeric(target[f], errors="coerce").dropna().to_numpy(float)
            if len(a) < 2 or len(b) < 2:
                continue
            pooled = np.std(np.concatenate([a, b]), ddof=1)
            rows.append({"comparison": label, "feature": f, "fourth_min": float(a.min()), "fourth_max": float(a.max()), "other_min": float(b.min()), "other_max": float(b.max()), "standardized_wasserstein": float(wasserstein_distance(a, b) / pooled) if pooled > 0 else None, "fourth_n": len(a), "other_n": len(b)})
    return rows


def applicability_domain(exp: pd.DataFrame, train: pd.DataFrame) -> list[dict[str, Any]]:
    X_train = SimpleImputer(strategy="median").fit_transform(train[FEATURES])
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    cov = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * 1e-6
    inv_cov = np.linalg.pinv(cov)
    rows = []
    for _, r in exp.iterrows():
        x = scaler.transform(SimpleImputer(strategy="median").fit(train[FEATURES]).transform(pd.DataFrame([r[FEATURES]])))[0]
        nn = float(np.min(np.linalg.norm(Xs - x, axis=1)))
        md = float(mahalanobis(x, np.mean(Xs, axis=0), inv_cov))
        status = "in-domain" if nn <= 3 and md <= 6 else "borderline" if nn <= 5 and md <= 10 else "out-of-domain"
        rows.append({"experiment_mix_id": r["experiment_mix_id"], "round": r["round"], "curing_age": r["curing_age"], "nearest_neighbor_distance": nn, "mahalanobis_distance": md, "applicability_status": status})
    return rows


def create_excel(path: Path, tables: dict[str, Sequence[dict[str, Any]]]) -> None:
    wb = Workbook()
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    for name, rows in tables.items():
        ws = wb.create_sheet(name[:31])
        if not rows:
            ws.append(["empty"])
            continue
        keys = list(rows[0].keys())
        ws.append(keys)
        for r in rows:
            ws.append([safe_cell(r.get(k)) for k in keys])
    wb.save(path)


def placeholder_figures(out: Path, names: Sequence[str]) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    for name in names:
        plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, name.replace("_", " "), ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{name}.png", dpi=120)
        plt.close()


def publication_figures(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    def load(name: str) -> pd.DataFrame:
        p = out / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    age = load("age_strength_count.csv")
    if not age.empty:
        pivot = age.pivot_table(index="curing_age", columns="domain", values="records", fill_value=0)
        ax = pivot.plot(kind="bar", figsize=(8, 4))
        ax.set_xlabel("Curing age (d)")
        ax.set_ylabel("Records")
        ax.set_title("All-age strength counts")
        plt.tight_layout()
        plt.savefig(fig_dir / "age_count.png", dpi=180)
        plt.close()

    dist = load("distribution_comparison.csv")
    if not dist.empty:
        d = dist.head(12).iloc[::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(d["feature"], d["Wasserstein_standardized"].astype(float))
        plt.xlabel("Standardized Wasserstein distance")
        plt.title("Literature vs experiment feature shift")
        plt.tight_layout()
        plt.savefig(fig_dir / "feature_distribution.png", dpi=180)
        plt.close()

    corr = load("spearman_correlation.csv")
    if not corr.empty and "rho" in corr:
        mat = corr.pivot_table(index="feature", columns="domain", values="rho")
        plt.figure(figsize=(6, 7))
        plt.imshow(mat.fillna(0), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Spearman rho")
        plt.xticks(range(len(mat.columns)), mat.columns, rotation=30, ha="right")
        plt.yticks(range(len(mat.index)), mat.index)
        plt.title("Feature-strength correlations")
        plt.tight_layout()
        plt.savefig(fig_dir / "correlation_heatmap.png", dpi=180)
        plt.close()

    mi = load("mutual_information.csv")
    if not mi.empty:
        col = "feature_weight" if "feature_weight" in mi else mi.select_dtypes("number").columns[-1]
        d = mi.sort_values(col).tail(12)
        plt.figure(figsize=(8, 5))
        plt.barh(d["feature"], d[col].astype(float))
        plt.xlabel(col)
        plt.title("Mutual-information feature weights")
        plt.tight_layout()
        plt.savefig(fig_dir / "mutual_information.png", dpi=180)
        plt.close()

    pred = load("model_prediction_rows.csv")
    if not pred.empty:
        best = load("model_ablation_summary_with_ci.csv")
        method = best.iloc[0]["method"] if not best.empty else pred["method"].iloc[0]
        d = pred[pred["method"] == method]
        plt.figure(figsize=(5, 5))
        plt.scatter(d["actual"], d["predicted"], s=16, alpha=0.75)
        lo = float(min(d["actual"].min(), d["predicted"].min()))
        hi = float(max(d["actual"].max(), d["predicted"].max()))
        plt.plot([lo, hi], [lo, hi], "k--", lw=1)
        plt.xlabel("Actual strength (MPa)")
        plt.ylabel("Predicted strength (MPa)")
        plt.title(f"Actual vs predicted: {method}")
        plt.tight_layout()
        plt.savefig(fig_dir / "actual_vs_predicted.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.scatter(d["predicted"], d["predicted"] - d["actual"], s=16, alpha=0.75)
        plt.axhline(0, color="k", ls="--", lw=1)
        plt.xlabel("Predicted strength (MPa)")
        plt.ylabel("Residual pred - actual (MPa)")
        plt.title(f"Residuals: {method}")
        plt.tight_layout()
        plt.savefig(fig_dir / "residuals.png", dpi=180)
        plt.close()

    summ = load("model_ablation_summary_with_ci.csv")
    if not summ.empty:
        d = summ.sort_values("RMSE_mean", ascending=True)
        plt.figure(figsize=(9, 5))
        plt.barh(d["method"], d["RMSE_mean"].astype(float), xerr=d["RMSE_std"].astype(float) if "RMSE_std" in d else None)
        plt.xlabel("RMSE (MPa)")
        plt.title("Round-4 common-holdout model ablation")
        plt.tight_layout()
        plt.savefig(fig_dir / "model_ablation_comparison.png", dpi=180)
        plt.savefig(fig_dir / "repeated_validation_ci.png", dpi=180)
        plt.close()

    m7 = load("m7_random_control_distribution.csv")
    if not m7.empty:
        plt.figure(figsize=(6, 4))
        plt.hist(m7["RMSE"].astype(float), bins=min(20, max(5, len(m7) // 3)), alpha=0.8)
        plt.xlabel("M7 RMSE (MPa)")
        plt.ylabel("Count")
        plt.title("M7 random-control distribution")
        plt.tight_layout()
        plt.savefig(fig_dir / "m7_random_distribution.png", dpi=180)
        plt.close()

    for name, csv_name, xcol, ycol, title in [
        ("cluster_similarity", "cluster_similarity.csv", "cluster", "cluster_similarity", "Cluster similarity"),
        ("fourth_round_representative", "fourth_round_representative_analysis.csv", "feature", "standardized_wasserstein", "Round-4 representativeness"),
        ("applicability_domain", "prediction_applicability_domain.csv", "nearest_neighbor_distance", "mahalanobis_distance", "Applicability domain"),
    ]:
        d = load(csv_name)
        if d.empty or xcol not in d or ycol not in d:
            continue
        plt.figure(figsize=(7, 4))
        if d[xcol].dtype == object:
            dd = d.dropna(subset=[ycol]).head(15)
            plt.barh(dd[xcol].astype(str), dd[ycol].astype(float))
        else:
            plt.scatter(d[xcol], d[ycol], s=16, alpha=0.75)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{name}.png", dpi=180)
        plt.close()

    # PCA and ROC are represented from available tables if full OOF probability
    # rows are not present yet.
    for name in ["domain_classifier_roc", "pca"]:
        p = fig_dir / f"{name}.png"
        if not p.exists():
            plt.figure(figsize=(4, 3))
            plt.text(0.5, 0.5, f"{name}\nsee CSV outputs", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(p, dpi=180)
            plt.close()



def _ssa_settings(cfg: dict[str, Any], profile: str) -> tuple[list[str], int, int]:
    engine = cfg.get("modeling_engine", {})
    candidates = list(engine.get("source_ssa_candidates", engine.get("ssa_candidates", ["Ridge", "SVR", "RandomForest", "ExtraTrees", "XGBoost"])))
    profiles = engine.get("ssa_profiles", {})
    default = {"diagnostic": [1, 1], "daily": [4, 2], "validation": [6, 4], "publication": [8, 6]}
    pop, iters = profiles.get(profile, default.get(profile, [6, 4]))
    return candidates, int(pop), int(iters)


def _budget_ssa_settings(cfg: dict[str, Any], profile: str, budget: int) -> tuple[list[str], int, int]:
    """V5: light learning-curve budgets, deeper search for 9/11/13 without changing validation logic."""
    engine = cfg.get("modeling_engine", {})
    focused = set(int(x) for x in cfg.get("validation", {}).get("focused_deep_budgets", [9, 11, 13]))
    stage = "focused" if int(budget) in focused else "light"
    candidates = list(engine.get("focused_budget_candidates" if stage == "focused" else "light_budget_candidates",
                                 engine.get("target_ssa_candidates", ["Ridge", "Huber", "SVR", "ExtraTrees"])))
    profs = engine.get("v5_budget_profiles", {})
    default = {
        "diagnostic": {"light": [1, 1], "focused": [2, 1]},
        "daily": {"light": [2, 1], "focused": [4, 2]},
        "validation": {"light": [3, 1], "focused": [6, 4]},
        "publication": {"light": [4, 2], "focused": [8, 6]},
    }
    pp = profs.get(profile, default.get(profile, default["daily"]))
    pop, iters = pp.get(stage, default.get(profile, default["daily"])[stage])
    return candidates, int(pop), int(iters)


def _target_feature_frame(df: pd.DataFrame, source_model=None) -> tuple[pd.DataFrame, list[str]]:
    """Compact target-domain physics features with categorical age and nonlinear ratio interactions.

    The source model is optional. When present its prediction is an auxiliary feature,
    not a mandatory additive baseline; this is the V5 M6 calibration architecture.
    """
    d = df.copy().reset_index(drop=True)
    age = pd.to_numeric(d["curing_age"], errors="coerce").astype(float)
    d["age_3d"] = (age == 3.0).astype(float)
    d["age_7d"] = (age == 7.0).astype(float)
    d["age_28d"] = (age == 28.0).astype(float)
    d["Mg_Si_ratio_sq"] = pd.to_numeric(d["Mg_Si_ratio"], errors="coerce").astype(float) ** 2
    d["water_binder_ratio_sq"] = pd.to_numeric(d["water_binder_ratio"], errors="coerce").astype(float) ** 2
    d["MgSi_x_wb"] = pd.to_numeric(d["Mg_Si_ratio"], errors="coerce").astype(float) * pd.to_numeric(d["water_binder_ratio"], errors="coerce").astype(float)
    for suffix, col in [("3", "age_3d"), ("7", "age_7d"), ("28", "age_28d")]:
        d[f"MgSi_x_age{suffix}"] = pd.to_numeric(d["Mg_Si_ratio"], errors="coerce").astype(float) * d[col]
        d[f"wb_x_age{suffix}"] = pd.to_numeric(d["water_binder_ratio"], errors="coerce").astype(float) * d[col]

    base_features = [
        "Mg_Si_ratio", "water_binder_ratio", "effective_MgO", "effective_SiO2", "water", "SHMP",
        "allowed_admixture_pct", "curing_temperature", "curing_humidity", "CO2",
        "age_3d", "age_7d", "age_28d", "Mg_Si_ratio_sq", "water_binder_ratio_sq", "MgSi_x_wb",
        "MgSi_x_age3", "MgSi_x_age7", "MgSi_x_age28", "wb_x_age3", "wb_x_age7", "wb_x_age28",
    ]
    if source_model is not None:
        src = np.asarray(source_model.predict(d[FEATURES].to_numpy(float)), dtype=float)
        d["source_pred"] = src
        d["source_pred_x_age3"] = src * d["age_3d"].to_numpy(float)
        d["source_pred_x_age7"] = src * d["age_7d"].to_numpy(float)
        d["source_pred_x_age28"] = src * d["age_28d"].to_numpy(float)
        base_features += ["source_pred", "source_pred_x_age3", "source_pred_x_age7", "source_pred_x_age28"]
    return d, base_features


def _best_oof_prediction_map(result: dict[str, Any]) -> dict[int, float]:
    info = result.get("best_info", {})
    best = info.get("best_model_name", info.get("best_source_model_name"))
    rows = [r for r in result.get("oof_predictions", []) if r.get("model") == best]
    return {int(r["row_index"]): float(r["predicted"]) for r in rows}


def _fit_target_calibrator_ssa(
    source_model, exp_train: pd.DataFrame, out: Path, cfg: dict[str, Any], profile: str, budget: int, seed: int, artifact_prefix: str,
) -> tuple[dict[str, Any], list[str]]:
    candidates, pop, iters = _budget_ssa_settings(cfg, profile, budget)
    d, cal_features = _target_feature_frame(exp_train, source_model=source_model)
    ng = int(d["experiment_mix_id"].nunique())
    res = select_grouped_model(
        d, cal_features, "strength", "experiment_mix_id", out,
        candidates=candidates, n_splits=min(5, ng), fast=False, random_state=seed,
        population=pop, iterations=iters, artifact_prefix=artifact_prefix,
    )
    info = dict(res["best_info"])
    info["selection_protocol"] = "target-only grouped OOF; source prediction is an auxiliary feature; categorical age; no common holdout"
    info["target_calibrator_features"] = cal_features
    res["best_info"] = info
    return res, cal_features


def _predict_target_calibrator(source_model, calibrator_model, cal_features: Sequence[str], df: pd.DataFrame) -> np.ndarray:
    d, _ = _target_feature_frame(df, source_model=source_model)
    return np.asarray(calibrator_model.predict(d[list(cal_features)].to_numpy(float)), dtype=float)


def _select_adaptive_blend(
    m0_result: dict[str, Any], m6_result: dict[str, Any], exp_train: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Choose alpha in y=alpha*M0+(1-alpha)*M6 using released-experiment OOF only."""
    p0 = _best_oof_prediction_map(m0_result); p6 = _best_oof_prediction_map(m6_result)
    d = exp_train.copy().reset_index(drop=True)
    valid = [i for i in range(len(d)) if i in p0 and i in p6]
    if not valid:
        return {"pooled": 0.5}, []
    grid = [float(x) for x in cfg.get("validation", {}).get("adaptive_blend_alpha_grid", [i / 10 for i in range(11)])]
    primary = [int(x) for x in cfg.get("validation", {}).get("primary_reported_age_days", [7, 28])]
    rows: list[dict[str, Any]] = []
    alphas: dict[str, float] = {}

    def choose(indices: list[int], label: str) -> float:
        if len(indices) < 2:
            return float(alphas.get("pooled", 0.5))
        y = d.loc[indices, "strength"].to_numpy(float)
        a = np.asarray([p0[i] for i in indices], float)
        b = np.asarray([p6[i] for i in indices], float)
        candidates_local = []
        for alpha in grid:
            pr = alpha * a + (1.0 - alpha) * b
            mm = metrics(y, pr)
            mape = float(np.mean(np.abs(y - pr) / np.maximum(np.abs(y), 1.0)) * 100.0)
            candidates_local.append((mm["RMSE"], mm["MAE"], -float(mm["R2"] if mm["R2"] is not None else -1e9), alpha, mape))
            rows.append({"age_scope": label, "alpha_M0": alpha, "RMSE": mm["RMSE"], "MAE": mm["MAE"], "R2": mm["R2"], "MAPE_pct": mape})
        return float(min(candidates_local)[3])

    pooled_idx = valid
    alphas["pooled"] = choose(pooled_idx, "pooled")
    for age in primary:
        idx = [i for i in valid if int(round(float(d.loc[i, "curing_age"]))) == age]
        alphas[str(age)] = choose(idx, f"{age}d") if len(idx) >= 2 else alphas["pooled"]
    return alphas, rows


def _predict_adaptive_blend(m0_pred: np.ndarray, m6_pred: np.ndarray, ages: np.ndarray, alphas: dict[str, float]) -> np.ndarray:
    out = np.empty_like(np.asarray(m0_pred, float))
    pooled = float(alphas.get("pooled", 0.5))
    for i, age in enumerate(np.asarray(ages, float)):
        alpha = float(alphas.get(str(int(round(age))), pooled))
        out[i] = alpha * float(m0_pred[i]) + (1.0 - alpha) * float(m6_pred[i])
    return out


def _info_manifest_row(method: str, stage: str, budget: int | str, info: dict[str, Any], *, repeat: int | None = None) -> dict[str, Any]:
    return {
        "method": method,
        "stage": stage,
        "pilot_budget": budget,
        "repeat": repeat,
        "best_algorithm": info.get("best_model_name", info.get("best_source_model_name")),
        "best_hyperparameters": json.dumps(info.get("best_model_params", info.get("best_source_model_params", {})), ensure_ascii=False),
        "inner_pooled_OOF_R2": info.get("pooled_target_OOF_R2", info.get("pooled_grouped_cv_R2", info.get("mean_grouped_cv_R2"))),
        "inner_pooled_OOF_RMSE": info.get("pooled_target_OOF_RMSE", info.get("pooled_grouped_cv_RMSE", info.get("mean_grouped_cv_RMSE"))),
        "inner_pooled_OOF_MAE": info.get("pooled_target_OOF_MAE", info.get("pooled_grouped_cv_MAE", info.get("mean_grouped_cv_MAE"))),
        "target_weight": info.get("target_weight"),
        "selection_protocol": info.get("selection_protocol", info.get("selection_metric")),
        "ssa_population": info.get("ssa_population"),
        "ssa_iterations": info.get("ssa_iterations"),
        "n_splits": info.get("n_splits"),
        "train_R2": info.get("train_R2"),
        "train_RMSE": info.get("train_RMSE"),
        "train_MAE": info.get("train_MAE"),
        "validation_R2": info.get("validation_R2"),
        "validation_RMSE": info.get("validation_RMSE"),
        "validation_MAE": info.get("validation_MAE"),
        "test_R2": info.get("test_R2"),
        "test_RMSE": info.get("test_RMSE"),
        "test_MAE": info.get("test_MAE"),
        "uses_external_data": info.get("uses_external_data"),
        "test_used_for_selection": info.get("test_used_for_selection"),
    }


def _permuted_similarity_weight_literature(weighted_lit: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Negative control for M6: preserve the exact 0-2 weight distribution but break its mapping to input similarity."""
    out = weighted_lit.copy()
    mix = out[["mix_id", "domain_similarity_weight"]].drop_duplicates("mix_id").sort_values("mix_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    shuffled = mix["domain_similarity_weight"].to_numpy(float).copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(mix["mix_id"].astype(str), shuffled))
    out["sample_weight"] = out["mix_id"].astype(str).map(mapping).astype(float)
    out["permuted_weight_seed"] = int(seed)
    return out


def _fit_independent_source_stages(
    lit: pd.DataFrame,
    weighted_lit: pd.DataFrame,
    ssa_out: Path,
    cfg: dict[str, Any],
    profile: str,
    m7_repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Every M gets its OWN source SSA. No model is borrowed across M definitions."""
    candidates, pop, iters = _ssa_settings(cfg, profile)
    nsplit = min(5, int(lit["study_id"].nunique()))
    if nsplit < 2:
        raise ValueError("Literature source requires at least two study_id groups.")
    manifest: list[dict[str, Any]] = []
    stages: dict[str, Any] = {}

    # M1: literature-only baseline: independent unweighted source SSA.
    r1 = select_grouped_model(
        lit, FEATURES, "strength", "study_id", ssa_out / "M1_source_only_SSA",
        candidates=candidates, n_splits=nsplit, fast=False, random_state=101,
        population=pop, iterations=iters, artifact_prefix="M1_source",
    )
    stages["M1"] = r1
    manifest.append(_info_manifest_row("M1_source_only_SSA", "source", "all", r1["best_info"]))

    # M1 internal validation: this is a validation stage of the M1 raw-literature
    # strategy, not a separate manuscript model. It is intentionally never added
    # to the real-experiment evaluation suite.
    holdout_cfg = cfg.get("literature_internal_validation", {})
    if bool(holdout_cfg.get("enabled", True)):
        r1b = select_grouped_holdout_model(
            lit, FEATURES, "strength", "study_id", ssa_out / "M1_literature_internal_holdout_SSA",
            train_fraction=float(holdout_cfg.get("train_fraction", 0.70)),
            validation_fraction=float(holdout_cfg.get("validation_fraction", 0.15)),
            test_fraction=float(holdout_cfg.get("test_fraction", 0.15)),
            candidates=candidates, fast=False,
            random_state=int(holdout_cfg.get("random_seed", 111)),
            population=pop, iterations=iters, artifact_prefix="M1_literature_internal",
        )
        stages["M1_internal"] = r1b
        manifest.append(_info_manifest_row(
            "M1_literature_internal_holdout_SSA", "literature_internal_validation", "all", r1b["best_info"]
        ))

    # M2: similarity-weighted literature-only model. It is a standalone manuscript
    # model and therefore owns its own SSA; it does NOT borrow the M6 source model
    # even though both use the same fixed 0-2 input-similarity weights.
    m2df = weighted_lit.copy()
    m2df["sample_weight"] = m2df["domain_similarity_weight"].astype(float)
    r2 = select_grouped_model(
        m2df, FEATURES, "strength", "study_id", ssa_out / "M2_similarity_weighted_literature_only_SSA",
        sample_weight_col="sample_weight", candidates=candidates, n_splits=nsplit, fast=False, random_state=151,
        population=pop, iterations=iters, artifact_prefix="M2_source",
    )
    stages["M2"] = r2
    manifest.append(_info_manifest_row(
        "M2_similarity_weighted_literature_only_SSA", "source_soft_weighted", "all", r2["best_info"]
    ))

    # M5: raw-literature source + residual-calibration strategy owns its source SSA; it does not borrow M1.
    r5 = select_grouped_model(
        lit, FEATURES, "strength", "study_id", ssa_out / "M5_own_source_SSA",
        candidates=candidates, n_splits=nsplit, fast=False, random_state=401,
        population=pop, iterations=iters, artifact_prefix="M5_source",
    )
    stages["M5"] = r5
    manifest.append(_info_manifest_row("M5_source_residual_SSA", "source", "all", r5["best_info"]))

    # M6: all physically eligible literature + correct 0-2 similarity weights.
    m6df = weighted_lit.copy()
    m6df["sample_weight"] = m6df["domain_similarity_weight"].astype(float)
    r6 = select_grouped_model(
        m6df, FEATURES, "strength", "study_id", ssa_out / "M6_similarity_weighted_source_SSA",
        sample_weight_col="sample_weight", candidates=candidates, n_splits=nsplit, fast=False, random_state=601,
        population=pop, iterations=iters, artifact_prefix="M6_source",
    )
    stages["M6"] = r6
    manifest.append(_info_manifest_row("M6_source_pred_target_calibrator_SSA", "source", "all", r6["best_info"]))

    # M7: proper negative control for full-soft M6. Keep every literature mix and the
    # exact same 0-2 weight distribution, but randomly permute WHICH mix receives which weight.
    m7_list = []
    for rep in range(1, max(1, int(m7_repeats)) + 1):
        rdf = _permuted_similarity_weight_literature(weighted_lit, seed=7000 + rep)
        rr = select_grouped_model(
            rdf, FEATURES, "strength", "study_id", ssa_out / f"M7_permuted_weight_source_rep_{rep:02d}_SSA",
            sample_weight_col="sample_weight", candidates=candidates, n_splits=nsplit, fast=False,
            random_state=701 + rep, population=pop, iterations=iters, artifact_prefix=f"M7_rep_{rep:02d}_source",
        )
        m7_list.append({"repeat": rep, "result": rr, "weighted_lit": rdf})
        manifest.append(_info_manifest_row("M7_permuted_weight_control_SSA", "source", "all", rr["best_info"], repeat=rep))
    stages["M7"] = m7_list
    return stages, manifest


def _fit_residual_ssa(
    base_model,
    exp_train: pd.DataFrame,
    out: Path,
    cfg: dict[str, Any],
    profile: str,
    seed: int,
    artifact_prefix: str,
    budget: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    if budget is None:
        candidates, pop, iters = _ssa_settings(cfg, profile)
    else:
        candidates, pop, iters = _budget_ssa_settings(cfg, profile, budget)
    d = exp_train.copy()
    base_pred = np.asarray(base_model.predict(d[FEATURES].to_numpy(float)), dtype=float)
    d["base_pred"] = base_pred
    d["residual_target"] = d["strength"].to_numpy(float) - base_pred
    residual_features = FEATURES + ["base_pred"]
    res = select_grouped_model(
        d, residual_features, "residual_target", "experiment_mix_id", out,
        candidates=candidates, n_splits=min(5, int(d["experiment_mix_id"].nunique())), fast=False,
        random_state=seed, population=pop, iterations=iters, artifact_prefix=artifact_prefix,
    )
    info = dict(res["best_info"])
    info["residual_features"] = residual_features
    return res["model"], info


def _predict_base_residual(base_model, residual_model, df: pd.DataFrame) -> np.ndarray:
    base = np.asarray(base_model.predict(df[FEATURES].to_numpy(float)), dtype=float)
    x = df[FEATURES].copy()
    x["base_pred"] = base
    return base + np.asarray(residual_model.predict(x[FEATURES + ["base_pred"]].to_numpy(float)), dtype=float)


def _fit_budget_independent_ssa_suite(
    lit: pd.DataFrame,
    weighted_lit: pd.DataFrame,
    exp_train: pd.DataFrame,
    source_stages: dict[str, Any],
    ssa_out: Path,
    cfg: dict[str, Any],
    profile: str,
    budget: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """V5: every data strategy is independently optimized for this budget.

    M0 gets compact categorical-age physics features.
    M6 uses its own similarity-weighted literature source model only to generate
    source_pred; a target-only grouped-SSA calibrator then learns final strength.
    M8 is an age-specific convex M0/M6 blend selected from released-experiment OOF only.
    """
    candidates, pop, iters = _budget_ssa_settings(cfg, profile, budget)
    ngroups = int(exp_train["experiment_mix_id"].nunique())
    if ngroups < 2:
        raise ValueError(f"Budget {budget} has fewer than two experiment_mix_id groups.")
    ns = min(5, ngroups)
    manifest: list[dict[str, Any]] = []
    suite: dict[str, Any] = {}

    # M0: experiment-only with V5 compact categorical-age physics features.
    m0df, m0_features = _target_feature_frame(exp_train, source_model=None)
    m0 = select_target_grouped_model(
        None, m0df, m0_features, "strength", "experiment_mix_id",
        ssa_out / f"budget_{budget:02d}" / "M0_experiment_only_SSA",
        candidates=candidates, n_splits=ns, fast=False, random_state=1000 + budget,
        population=pop, iterations=iters, artifact_prefix=f"M0_budget_{budget:02d}",
    )
    suite["M0_experiment_only_SSA"] = {"kind": "target_plain", "model": m0["model"], "features": m0_features}
    m0_info = dict(m0["best_info"]); m0_info["target_features"] = m0_features
    manifest.append(_info_manifest_row("M0_experiment_only_SSA", "target_model", budget, m0_info))

    # M1: its own source-only model, unchanged with budget.
    suite["M1_source_only_SSA"] = {"kind": "plain", "model": source_stages["M1"]["model"]}

    # M2: similarity-weighted literature-only source model, unchanged with feedback
    # budget and independent from M6's source SSA.
    suite["M2_similarity_weighted_literature_only_SSA"] = {"kind": "plain", "model": source_stages["M2"]["model"]}

    # M3: equal-weight direct literature-experiment merge; independent target-grouped SSA per budget.
    m3 = select_target_grouped_model(
        lit, exp_train, FEATURES, "strength", "experiment_mix_id",
        ssa_out / f"budget_{budget:02d}" / "M3_equal_merge_SSA",
        target_weight=1.0, candidates=candidates, n_splits=ns, fast=False, random_state=2000 + budget,
        population=pop, iterations=iters, artifact_prefix=f"M3_budget_{budget:02d}",
    )
    suite["M3_equal_merge_SSA"] = {"kind": "plain", "model": m3["model"]}
    manifest.append(_info_manifest_row("M3_equal_merge_SSA", "target_guided_direct", budget, m3["best_info"]))

    # M4: real-experiment sample weight lambda is optimized inside its own SSA.
    m4 = select_target_grouped_model(
        lit, exp_train, FEATURES, "strength", "experiment_mix_id",
        ssa_out / f"budget_{budget:02d}" / "M4_target_weighted_merge_SSA",
        target_weight_search=(1.0, float(cfg.get("validation", {}).get("m4_target_weight_max", cfg.get("validation", {}).get("m3_target_weight_max", 50.0)))),
        candidates=candidates, n_splits=ns, fast=False, random_state=3000 + budget,
        population=pop, iterations=iters, artifact_prefix=f"M4_budget_{budget:02d}",
    )
    suite["M4_target_weighted_merge_SSA"] = {"kind": "plain", "model": m4["model"]}
    manifest.append(_info_manifest_row("M4_target_weighted_merge_SSA", "target_guided_weighted_direct", budget, m4["best_info"]))

    # M5: raw-literature source + target residual calibration; source and residual are both M5-owned.
    m5res, m5info = _fit_residual_ssa(
        source_stages["M5"]["model"], exp_train,
        ssa_out / f"budget_{budget:02d}" / "M5_residual_calibrator_SSA",
        cfg, profile, 5000 + budget, f"M5_residual_budget_{budget:02d}", budget=budget,
    )
    suite["M5_source_residual_SSA"] = {"kind": "residual", "base": source_stages["M5"]["model"], "residual": m5res}
    manifest.append(_info_manifest_row("M5_source_residual_SSA", "residual_calibrator", budget, m5info))

    # M6 V5: literature source prediction is an auxiliary feature, not an additive baseline.
    m6, m6_features = _fit_target_calibrator_ssa(
        source_stages["M6"]["model"], exp_train,
        ssa_out / f"budget_{budget:02d}" / "M6_source_pred_target_calibrator_SSA",
        cfg, profile, budget, 6000 + budget, f"M6_calibrator_budget_{budget:02d}",
    )
    suite["M6_source_pred_target_calibrator_SSA"] = {
        "kind": "target_calibrator", "base": source_stages["M6"]["model"],
        "calibrator": m6["model"], "features": m6_features,
    }
    manifest.append(_info_manifest_row("M6_source_pred_target_calibrator_SSA", "target_calibrator", budget, m6["best_info"]))

    # M7: permuted similarity weights negative control, retained as residual ablation.
    m7_models = []
    for item in source_stages["M7"]:
        rep = int(item["repeat"])
        rmodel, rinfo = _fit_residual_ssa(
            item["result"]["model"], exp_train,
            ssa_out / f"budget_{budget:02d}" / f"M7_residual_rep_{rep:02d}_SSA",
            cfg, profile, 7000 + budget * 10 + rep, f"M7_residual_budget_{budget:02d}_rep_{rep:02d}", budget=budget,
        )
        m7_models.append({"repeat": rep, "base": item["result"]["model"], "residual": rmodel})
        manifest.append(_info_manifest_row("M7_permuted_weight_control_SSA", "residual_calibrator", budget, rinfo, repeat=rep))
    suite["M7_permuted_weight_control_SSA"] = {"kind": "ensemble_residual", "models": m7_models}

    # M8: target-only OOF-selected age-specific M0/M6 convex blend.
    alphas, alpha_rows = _select_adaptive_blend(m0, m6, exp_train, cfg)
    suite["M8_adaptive_M0_M6_blend_SSA"] = {
        "kind": "adaptive_blend",
        "m0_model": m0["model"], "m0_features": m0_features,
        "m6_base": source_stages["M6"]["model"], "m6_calibrator": m6["model"], "m6_features": m6_features,
        "alphas": alphas,
    }
    for k, alpha in alphas.items():
        manifest.append({
            "method": "M8_adaptive_M0_M6_blend_SSA", "stage": "blend_alpha", "pilot_budget": budget,
            "repeat": None, "best_algorithm": "convex_blend", "best_hyperparameters": json.dumps({"age_scope": k, "alpha_M0": alpha}, ensure_ascii=False),
            "inner_pooled_OOF_R2": None, "inner_pooled_OOF_RMSE": None, "inner_pooled_OOF_MAE": None,
            "target_weight": alpha, "selection_protocol": "released-experiment grouped OOF only; no common holdout",
            "ssa_population": None, "ssa_iterations": None, "n_splits": ns,
        })
    suite["M8_adaptive_M0_M6_blend_SSA"]["alpha_search_rows"] = alpha_rows
    return suite, manifest


def _predict_suite(suite: dict[str, Any], eval_df: pd.DataFrame, budget: int, scope: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pred_rows: list[dict[str, Any]] = []
    repeat_metric_rows: list[dict[str, Any]] = []
    for method, spec in suite.items():
        if spec["kind"] == "plain":
            pred = np.asarray(spec["model"].predict(eval_df[FEATURES].to_numpy(float)), dtype=float)
        elif spec["kind"] == "target_plain":
            d, _ = _target_feature_frame(eval_df, source_model=None)
            pred = np.asarray(spec["model"].predict(d[spec["features"]].to_numpy(float)), dtype=float)
        elif spec["kind"] == "residual":
            pred = _predict_base_residual(spec["base"], spec["residual"], eval_df)
        elif spec["kind"] == "target_calibrator":
            pred = _predict_target_calibrator(spec["base"], spec["calibrator"], spec["features"], eval_df)
        elif spec["kind"] == "adaptive_blend":
            d0, _ = _target_feature_frame(eval_df, source_model=None)
            p0 = np.asarray(spec["m0_model"].predict(d0[spec["m0_features"]].to_numpy(float)), dtype=float)
            p6 = _predict_target_calibrator(spec["m6_base"], spec["m6_calibrator"], spec["m6_features"], eval_df)
            pred = _predict_adaptive_blend(p0, p6, eval_df["curing_age"].to_numpy(float), spec["alphas"])
        elif spec["kind"] == "ensemble_residual":
            repeat_preds = []
            for m in spec["models"]:
                rp = _predict_base_residual(m["base"], m["residual"], eval_df)
                repeat_preds.append(rp)
                rm = metrics(eval_df["strength"].to_numpy(float), rp)
                repeat_metric_rows.append({"pilot_budget": budget, "evaluation_scope": scope, "method": method, "m7_random_repeat": m["repeat"], **rm})
            pred = np.mean(np.vstack(repeat_preds), axis=0)
        else:
            raise KeyError(spec["kind"])
        phash = hashlib.sha1(np.round(pred, 8).tobytes()).hexdigest()
        for i, (_, row) in enumerate(eval_df.iterrows()):
            pred_rows.append({
                "pilot_budget": int(budget), "evaluation_scope": scope, "method": method,
                "experiment_mix_id": row["experiment_mix_id"], "round": row["round"], "curing_age": row["curing_age"],
                "actual": row["strength"], "predicted": float(pred[i]), "prediction_array_hash": phash,
            })
    return pred_rows, repeat_metric_rows


def _prediction_metrics_by_method(pred_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not pred_rows:
        return []
    df = pd.DataFrame(pred_rows)
    out=[]
    for (budget, scope, method), part in df.groupby(["pilot_budget", "evaluation_scope", "method"], dropna=False):
        y=part["actual"].to_numpy(float); p=part["predicted"].to_numpy(float)
        out.append({"pilot_budget":int(budget),"evaluation_scope":scope,"method":method,"age_scope":"pooled",**metrics(y,p),"n":len(part)})
        for age, ap in part.groupby("curing_age"):
            out.append({"pilot_budget":int(budget),"evaluation_scope":scope,"method":method,"age_scope":f"{int(age)}d",**metrics(ap["actual"].to_numpy(float),ap["predicted"].to_numpy(float)),"n":len(ap)})
    return out


def _negative_transfer_table(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df=pd.DataFrame([r for r in metric_rows if r.get("age_scope")=="pooled" and r.get("evaluation_scope")=="strict_common_holdout_20_mixes"])
    if df.empty:
        return []
    out=[]
    for budget, part in df.groupby("pilot_budget"):
        m0=part[part["method"]=="M0_experiment_only_SSA"]
        m1=part[part["method"]=="M1_source_only_SSA"]
        m0r=m0.iloc[0] if len(m0) else None; m1r=m1.iloc[0] if len(m1) else None
        for _,r in part.iterrows():
            item=r.to_dict()
            if m0r is not None:
                item["RMSE_gain_vs_M0"] = float(m0r["RMSE"] - r["RMSE"])
                item["R2_gain_vs_M0"] = float(r["R2"] - m0r["R2"])
                item["negative_transfer_vs_M0_RMSE"] = bool(r["RMSE"] > m0r["RMSE"])
            if m1r is not None:
                item["RMSE_gain_vs_M1"] = float(m1r["RMSE"] - r["RMSE"])
                item["negative_transfer_vs_M1_RMSE"] = bool(r["RMSE"] > m1r["RMSE"])
            out.append(item)
    return out

def mark_deprecated(output_parent: Path) -> None:
    old = output_parent / "DACHUI_fast_results"
    if not old.exists():
        return
    dep = output_parent / "deprecated_fixed_model_results"
    dep.mkdir(exist_ok=True)
    for name in ["m0_m7_grouped_cv_summary.csv", "external_round_validation.csv", "analysis_report.md", "MSC_domain_adaptation_report.xlsx"]:
        src = old / name
        if src.exists():
            dst = dep / name
            if not dst.exists():
                shutil.copy2(src, dst)
    (dep / "DEPRECATED_NOTICE.txt").write_text("These files were produced by the old fixed-model/Ridge fast path and must not be used as publication conclusions.\n", encoding="utf-8")



def run_pipeline(args: argparse.Namespace) -> int:
    cfg = read_json(Path(args.config))
    pilot_budgets = [int(x) for x in cfg["validation"].get("pilot_budgets", [3, 5, 7, 9, 11, 13])]
    if pilot_budgets != [3, 5, 7, 9, 11, 13]:
        raise ValueError("This Round-4 geometric protocol requires pilot_budgets = [3,5,7,9,11,13].")
    profile_defaults = {"diagnostic": 1, "daily": 1, "validation": 2, "publication": 3}
    m7_random_repeats = int(args.m7_random_repeats if args.m7_random_repeats is not None else profile_defaults[args.profile])
    mode_label = args.profile
    requested_transfer_budget = str(args.transfer_budget).strip().lower()
    operating_point_budget = int(cfg.get("validation", {}).get("operating_point_budget", 9))
    if operating_point_budget not in pilot_budgets:
        raise ValueError(f"operating_point_budget={operating_point_budget} must be one of {pilot_budgets}.")
    primary_method = str(cfg.get("validation", {}).get("primary_selection_method", "M8_adaptive_M0_M6_blend_SSA"))
    secondary_method = str(cfg.get("validation", {}).get("report_secondary_method", "M6_source_pred_target_calibrator_SSA"))
    residual_candidates = cfg["residual_models"] if args.profile == "publication" else ["LinearRegression", "Ridge", "RandomForestShallow"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    mark_deprecated(out.parent)

    # ------------------------------------------------------------------
    # 1) Clean data and define the target domain from all 33 Round-4 INPUTS.
    #    No Round-4 strength is permitted to influence literature screening.
    # ------------------------------------------------------------------
    lit_clean = hard_clean_literature(Path(args.literature), cfg)
    exp_clean = clean_experiment(Path(args.experiment), cfg)
    lit = rows_to_df(lit_clean.rows)
    exp = rows_to_df(exp_clean.rows)
    round4_integrity = validate_round4_experiment_integrity(exp, cfg)
    (out / "round4_experiment_integrity.json").write_text(json.dumps(round4_integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out / "literature_hard_cleaned_all_ages.csv", lit_clean.rows)
    # In V7 this file is intentionally Round-4-only despite the legacy filename.
    write_csv(out / "experiment_cleaned_all_ages.csv", exp.to_dict("records"))
    for name, rows in lit_clean.audits.items():
        write_csv(out / f"{name}.csv", rows)
    write_csv(out / "experiment_column_mapping_log.csv", exp_clean.audits["column_mapping_log"])
    write_csv(out / "experiment_unit_mapping_log.csv", exp_clean.audits["unit_mapping_log"])
    write_csv(out / "replicate_audit.csv", exp_clean.audits["replicate_audit"])

    fourth_design = round4_mix_design(exp)
    fourth = exp[exp["round"].astype(str).str.contains("第四轮")].copy()
    if fourth_design["experiment_mix_id"].nunique() != 33:
        raise ValueError(f"Expected exactly 33 Round-4 input designs, found {fourth_design['experiment_mix_id'].nunique()}.")
    design_cols = ["rank_int", "experiment_mix_id"] + list(cfg["validation"].get("pilot_design_features", ["water_binder_ratio", "Mg_Si_ratio"]))
    for c in DEFAULT_INPUT_SCREEN_FEATURES:
        if c not in design_cols and c in fourth_design.columns:
            design_cols.append(c)
    fourth_design[design_cols].to_csv(out / "round4_input_design_33.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 2) Fixed geometric nested feedback sequence. Strengths are invisible.
    # ------------------------------------------------------------------
    geometric_order, geometric_audit, geometry_metrics = round4_geometric_nested_order(fourth_design, cfg)
    write_csv(out / "round4_geometric_nested_order.csv", geometric_audit)
    write_csv(out / "round4_geometric_budget_coverage.csv", geometry_metrics)
    max_budget = max(pilot_budgets)
    all_mix_ids = set(fourth_design["experiment_mix_id"].astype(str))
    common_holdout_ids = sorted(all_mix_ids - set(geometric_order[:max_budget]))
    operating_holdout_ids = sorted(all_mix_ids - set(geometric_order[:operating_point_budget]))
    expected_holdout = int(cfg.get("validation", {}).get("strict_common_holdout_expected_mix_count", cfg.get("validation", {}).get("common_holdout_expected_mix_count", 20)))
    expected_operating_holdout = int(cfg.get("validation", {}).get("operating_point_holdout_expected_mix_count", 24))
    if len(common_holdout_ids) != expected_holdout:
        raise ValueError(f"Expected {expected_holdout} strict common-holdout mixes, found {len(common_holdout_ids)}.")
    if len(operating_holdout_ids) != expected_operating_holdout:
        raise ValueError(f"Expected {expected_operating_holdout} operating-point holdout mixes, found {len(operating_holdout_ids)}.")
    rank_by_id = {str(r["experiment_mix_id"]): int(r["rank_int"]) for _, r in fourth_design.iterrows()}
    write_csv(out / "round4_common_holdout_mix_ids.csv", [
        {"experiment_mix_id": mid, "rank": rank_by_id[mid], "role": "strict_common_holdout_outside_full_13_point_sensitivity_envelope"}
        for mid in sorted(common_holdout_ids, key=lambda x: rank_by_id[x])
    ])
    write_csv(out / "round4_operating_point_holdout_24_mix_ids.csv", [
        {
            "experiment_mix_id": mid, "rank": rank_by_id[mid],
            "role": "not_used_by_selected_9_mix_model",
            "strictly_untouched_across_3_to_13_sensitivity": mid in set(common_holdout_ids),
        }
        for mid in sorted(operating_holdout_ids, key=lambda x: rank_by_id[x])
    ])

    # ------------------------------------------------------------------
    # 3) Screen literature ONCE from Round-4 input geometry only.  The pool
    #    remains fixed for every 3/5/7/9/11/13 experiment budget.
    # ------------------------------------------------------------------
    weighted_lit = input_only_literature_similarity(lit, fourth_design, cfg)
    chosen_fraction, fraction_table, selected_lit = select_literature_fraction_by_input_geometry(weighted_lit, fourth_design, cfg)
    weighted_lit.to_csv(out / "literature_input_similarity_weights.csv", index=False, encoding="utf-8-sig")
    fraction_table.to_csv(out / "literature_input_similarity_fraction_table.csv", index=False, encoding="utf-8-sig")
    selected_lit.to_csv(out / "literature_input_selected_fixed.csv", index=False, encoding="utf-8-sig")
    (out / "literature_screening_rule.json").write_text(json.dumps({
        "round4_input_mixes": 33,
        "uses_round4_strength": False,
        "uses_literature_strength_for_similarity": False,
        "similarity_features": weighted_lit.attrs.get("input_similarity_features", []),
        "input_feature_weights": weighted_lit.attrs.get("input_feature_weights", {}),
        "selected_fraction_of_literature_mixes": chosen_fraction,
        "selected_literature_mixes": int(selected_lit["mix_id"].nunique()),
        "selected_long_records": int(len(selected_lit)),
        "selected_studies": int(selected_lit["study_id"].nunique()),
        "pool_fixed_across_all_budgets": True,
        "selection_type": "soft_weighting_not_hard_screening",
        "weight_range": [float(cfg["similarity"].get("similarity_weight_min", 0.0)), float(cfg["similarity"].get("similarity_weight_max", 2.0))],
        "all_33_target_X_used": True,
        "target_Y_used": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Independent source stages. Every M owns its own SSA result; no cross-M locked model reuse.
    # M6 uses all physically eligible literature with the fixed 0-2 similarity weights.
    source_stages, source_ssa_manifest = _fit_independent_source_stages(
        lit, selected_lit, out / "ssa_calls", cfg, args.profile, m7_random_repeats
    )
    write_csv(out / "independent_ssa_model_selection_manifest_source.csv", source_ssa_manifest)

    # Story stages 2-4: zero-shot failure of raw literature (M1), input-only
    # input-distribution difference, then zero-shot reassessment after input-similarity soft weighting (M2).
    development9_ids = set(geometric_order[:operating_point_budget])
    development9 = fourth[fourth["experiment_mix_id"].astype(str).isin(development9_ids)].copy()
    zero_shot_suite = {
        "M1_source_only_SSA": {"kind": "plain", "model": source_stages["M1"]["model"]},
        "M2_similarity_weighted_literature_only_SSA": {"kind": "plain", "model": source_stages["M2"]["model"]},
    }
    zero_shot_rows, _ = _predict_suite(zero_shot_suite, development9, 0, "zero_shot_development_9_mixes")
    zero_shot_metrics = _prediction_metrics_by_method(zero_shot_rows)
    write_csv(out / "story_M1_M2_zero_shot_dev9_predictions.csv", zero_shot_rows)
    write_csv(out / "story_M1_M2_zero_shot_dev9_metrics.csv", zero_shot_metrics)
    domain_shift_rows = input_distribution_comparison(
        lit, fourth_design, cfg["similarity"].get("input_features", DEFAULT_INPUT_SCREEN_FEATURES)
    )
    write_csv(out / "story_input_only_domain_shift_all33X.csv", domain_shift_rows)
    m1_vs_m2 = []
    zdf = pd.DataFrame([r for r in zero_shot_metrics if r.get("age_scope") == "pooled"])
    if not zdf.empty:
        m1r = zdf[zdf["method"] == "M1_source_only_SSA"]
        m2r = zdf[zdf["method"] == "M2_similarity_weighted_literature_only_SSA"]
        if len(m1r) and len(m2r):
            a, b = m1r.iloc[0], m2r.iloc[0]
            m1_vs_m2.append({
                "comparison": "raw_literature_M1_vs_similarity_weighted_literature_M2_on_same_9_real_experiment_mixes",
                "M1_R2": float(a["R2"]), "M1_RMSE": float(a["RMSE"]), "M1_MAE": float(a["MAE"]),
                "M2_R2": float(b["R2"]), "M2_RMSE": float(b["RMSE"]), "M2_MAE": float(b["MAE"]),
                "RMSE_improvement_M1_to_M2": float(a["RMSE"] - b["RMSE"]),
                "uses_target_X_33": True, "uses_target_Y_for_similarity": False,
            })
    write_csv(out / "story_M1_to_M2_improvement.csv", m1_vs_m2)

    # Basic all-age counts remain descriptive only.
    age_rows = []
    for domain, df in [("literature_hard_cleaned", lit), ("literature_input_selected", selected_lit), ("round4_experiment", fourth)]:
        for age, n in df.groupby("curing_age").size().items():
            age_rows.append({"domain": domain, "curing_age": age, "records": int(n)})
    write_csv(out / "age_strength_count.csv", age_rows)

    # ------------------------------------------------------------------
    # 4) Release real Round-4 strengths only according to the fixed nested
    #    3/5/7/9/11/13 geometric order. For EACH budget, every M whose
    #    target-training data change runs its OWN SSA using only released target
    #    mixes for inner grouped model selection. The common holdout is prediction-only.
    # ------------------------------------------------------------------
    common_pred_rows: list[dict[str, Any]] = []
    operating_pred_rows: list[dict[str, Any]] = []
    operating_repeat_rows: list[dict[str, Any]] = []
    unseen_pred_rows: list[dict[str, Any]] = []
    m7_repeat_rows: list[dict[str, Any]] = []
    all_ssa_manifest: list[dict[str, Any]] = list(source_ssa_manifest)
    train_by_budget: dict[int, pd.DataFrame] = {}
    suite_by_budget: dict[int, dict[str, Any]] = {}

    common_eval = fourth[fourth["experiment_mix_id"].astype(str).isin(set(common_holdout_ids))].copy()
    operating_eval = fourth[fourth["experiment_mix_id"].astype(str).isin(set(operating_holdout_ids))].copy()
    for budget in pilot_budgets:
        train_ids = set(geometric_order[:budget])
        train = fourth[fourth["experiment_mix_id"].astype(str).isin(train_ids)].copy()
        train_by_budget[budget] = train
        all_unseen = fourth[~fourth["experiment_mix_id"].astype(str).isin(train_ids)].copy()

        suite, budget_manifest = _fit_budget_independent_ssa_suite(
            lit, selected_lit, train, source_stages, out / "ssa_calls", cfg, args.profile, budget
        )
        suite_by_budget[budget] = suite
        all_ssa_manifest.extend(budget_manifest)

        cp, cr = _predict_suite(suite, common_eval, budget, "strict_common_holdout_20_mixes")
        up, ur = _predict_suite(suite, all_unseen, budget, "all_currently_unseen_mixes")
        common_pred_rows.extend(cp); unseen_pred_rows.extend(up)
        m7_repeat_rows.extend(cr); m7_repeat_rows.extend(ur)
        if budget == operating_point_budget:
            op, opr = _predict_suite(suite, operating_eval, budget, "operating_point_holdout_24_mixes")
            operating_pred_rows.extend(op)
            operating_repeat_rows.extend(opr)

    common_metric_rows = _prediction_metrics_by_method(common_pred_rows)
    operating_metric_rows = _prediction_metrics_by_method(operating_pred_rows)
    unseen_metric_rows = _prediction_metrics_by_method(unseen_pred_rows)
    write_csv(out / "budget_common_holdout_predictions.csv", common_pred_rows)
    write_csv(out / "strict_common_holdout_20_predictions.csv", common_pred_rows)
    write_csv(out / "operating_point_9_holdout24_predictions.csv", operating_pred_rows)
    write_csv(out / "operating_point_9_holdout24_metrics.csv", operating_metric_rows)
    write_csv(out / "budget_common_holdout_fold_metrics.csv", common_metric_rows)
    write_csv(out / "budget_all_unseen_predictions.csv", unseen_pred_rows)
    write_csv(out / "budget_all_unseen_fold_metrics.csv", unseen_metric_rows)
    write_csv(out / "independent_ssa_model_selection_manifest.csv", all_ssa_manifest)
    write_csv(out / "m7_random_control_distribution.csv", m7_repeat_rows)
    write_csv(out / "m7_operating_point_9_holdout24_distribution.csv", operating_repeat_rows)

    # V5 primary curve is the OOF-selected adaptive M0/M6 blend; M6 calibrator is also saved separately.
    common_curve = budget_performance_table(common_pred_rows, pilot_budgets, method=primary_method)
    unseen_curve = budget_performance_table(unseen_pred_rows, pilot_budgets, method=primary_method)
    m6_common_curve = budget_performance_table(common_pred_rows, pilot_budgets, method=secondary_method)
    m6_unseen_curve = budget_performance_table(unseen_pred_rows, pilot_budgets, method=secondary_method)
    write_csv(out / "budget_common_holdout_metrics.csv", common_curve)
    write_csv(out / "budget_all_unseen_metrics.csv", unseen_curve)
    write_csv(out / "budget_common_holdout_metrics_M6_calibrator.csv", m6_common_curve)
    write_csv(out / "budget_all_unseen_metrics_M6_calibrator.csv", m6_unseen_curve)
    write_csv(out / "all_models_all_budgets_common_holdout_metrics.csv", common_metric_rows)
    write_csv(out / "all_models_all_budgets_unseen_metrics.csv", unseen_metric_rows)
    negative_transfer = _negative_transfer_table(common_metric_rows)
    write_csv(out / "negative_transfer_audit.csv", negative_transfer)

    # Retrospective budget comparison remains clearly labeled non-independent.
    budget_rows, auto_transfer_budget, budget_reason = budget_selection_table(
        common_pred_rows, pilot_budgets, method=primary_method
    )
    if requested_transfer_budget in {"operating", "primary", "configured"}:
        transfer_budget = operating_point_budget
        selection_mode = "preconfigured_operating_point"
        budget_reason = (
            f"primary operating point fixed at {operating_point_budget} mixtures; "
            "the 3/5/7/9/11/13 curve is sensitivity/discussion material rather than the primary article question"
        )
    elif requested_transfer_budget == "auto":
        transfer_budget = auto_transfer_budget
        selection_mode = "retrospective_strict_common_holdout_comparison_nonprimary"
        budget_reason = "explicit auto sensitivity request; not the default primary protocol"
    else:
        transfer_budget = int(requested_transfer_budget)
        if transfer_budget not in pilot_budgets:
            raise ValueError(f"--transfer-budget must be operating, auto, or one of {pilot_budgets}")
        selection_mode = "manual_override"
        budget_reason = f"manual override; configured primary operating point is {operating_point_budget}"
    for row in budget_rows:
        row["selected_for_deployment"] = int(row["pilot_budget"]) == transfer_budget
        row["selection_mode"] = selection_mode
        row["selection_reason"] = budget_reason
        row["independent_final_test"] = False
    write_csv(out / "budget_selection_metrics.csv", budget_rows)
    (out / "budget_selection_decision.json").write_text(json.dumps({
        "selected_budget": transfer_budget,
        "automatic_recommendation": auto_transfer_budget,
        "selection_mode": selection_mode,
        "reason": budget_reason,
        "comparison_set": "strict 20-mix common holdout outside the full 13-point sensitivity envelope",
        "common_holdout_mix_count": len(common_holdout_ids),
        "primary_operating_point_budget": operating_point_budget,
        "primary_operating_point_holdout_mix_count": len(operating_holdout_ids),
        "primary_24_role": "unseen by selected 9-mix model; four of these mixes enter only 11/13 supplementary sensitivity",
        "strict_20_role": "untouched by every 3/5/7/9/11/13 target-training model",
        "independent_final_test_claimed": False,
        "model_selection_never_uses_common_holdout": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 5) Selected-budget ablation is read from the already independently tuned
    #    all-model budget run. No model is re-used from a different M strategy.
    # ------------------------------------------------------------------
    strict_pred_rows = [r for r in common_pred_rows if int(r["pilot_budget"]) == transfer_budget]
    strict_rows = []
    for r in common_metric_rows:
        if int(r["pilot_budget"]) == transfer_budget and r.get("age_scope") == "pooled":
            strict_rows.append({"split_id": transfer_budget, **r})
    if transfer_budget == operating_point_budget:
        pred_rows = list(operating_pred_rows)
        primary_metric_rows = [r for r in operating_metric_rows if r.get("age_scope") == "pooled"]
    else:
        pred_rows = strict_pred_rows
        primary_metric_rows = [r for r in common_metric_rows if int(r["pilot_budget"]) == transfer_budget and r.get("age_scope") == "pooled"]
    write_csv(out / "model_ablation_common_holdout.csv", strict_rows)
    write_csv(out / "model_ablation_operating_point_holdout24.csv", primary_metric_rows)
    write_csv(out / "model_ablation_summary_with_ci.csv", summary_with_ci(strict_rows))
    write_csv(out / "model_ablation_pooled_metrics.csv", pooled_prediction_summary(pred_rows))
    write_csv(out / "model_prediction_rows.csv", pred_rows)
    write_csv(out / "calibration_model_results.csv", [r for r in all_ssa_manifest if r.get("stage") in {"residual_calibrator", "target_calibrator", "blend_alpha"}])

    # V5 deployment artifacts: selected-budget M8 contains its own M0, M6 source+calibrator, and OOF-selected alphas.
    transfer_train = train_by_budget[transfer_budget]
    m8_deploy = suite_by_budget[transfer_budget]["M8_adaptive_M0_M6_blend_SSA"]
    joblib.dump(m8_deploy["m0_model"], out / "deployment_M8_M0_model.pkl")
    joblib.dump(m8_deploy["m6_base"], out / "deployment_M8_M6_source_model.pkl")
    joblib.dump(m8_deploy["m6_calibrator"], out / "deployment_M8_M6_target_calibrator.pkl")
    (out / "deployment_M8_blend_alphas.json").write_text(json.dumps(m8_deploy["alphas"], ensure_ascii=False, indent=2), encoding="utf-8")
    m8_stage_rows = [r for r in all_ssa_manifest if r.get("method") in {"M0_experiment_only_SSA", "M6_source_pred_target_calibrator_SSA", "M8_adaptive_M0_M6_blend_SSA"} and (r.get("pilot_budget") in {"all", transfer_budget})]
    deployment_info = {
        "role": "post_validation_deployment_only",
        "method": "M8_adaptive_M0_M6_blend_SSA",
        "selected_budget": int(transfer_budget),
        "independent_test_performance_claimed": False,
        "cross_M_model_reuse": False,
        "source_and_target_calibrator_own_SSA": True,
        "blend_alpha_selected_from_released_experiment_OOF_only": True,
        "blend_alphas": m8_deploy["alphas"],
        "selection_manifest": m8_stage_rows,
    }
    (out / "deployment_transfer_model.json").write_text(json.dumps(deployment_info, ensure_ascii=False, indent=2), encoding="utf-8")
    freeze_manifest = {
        "protocol_version": "DACHUI_V7_story_replay",
        "selected_operating_point_budget": int(operating_point_budget),
        "deployed_budget": int(transfer_budget),
        "all_33_target_X_used_for_similarity": True,
        "target_Y_used_for_similarity": False,
        "soft_similarity_weight_range": [0.0, 2.0],
        "primary_9_training_mix_ids": [str(x) for x in geometric_order[:operating_point_budget]],
        "primary_24_evaluation_mix_ids": [str(x) for x in sorted(operating_holdout_ids, key=lambda x: rank_by_id[x])],
        "strict_20_common_holdout_mix_ids": [str(x) for x in sorted(common_holdout_ids, key=lambda x: rank_by_id[x])],
        "primary_24_not_used_by_selected_9_model": True,
        "primary_24_strictly_untouched_across_all_3_to_13_sensitivity": False,
        "strict_20_untouched_across_all_3_to_13_sensitivity": True,
        "sensitivity_budgets": pilot_budgets,
        "sensitivity_role": "discussion_or_supplementary_material",
        "source_stage_methods": [r.get("method") for r in source_ssa_manifest],
        "deployment_selection_mode": selection_mode,
    }
    (out / "MODEL_FREEZE_AND_DATA_ROLE_MANIFEST.json").write_text(json.dumps(freeze_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Descriptive input-domain audits.  No strength-response similarity file exists.
    write_csv(out / "round4_vs_literature_input_distribution.csv", input_distribution_comparison(lit, fourth_design, cfg["similarity"].get("input_features", DEFAULT_INPUT_SCREEN_FEATURES)))
    write_csv(out / "prediction_applicability_domain.csv", applicability_domain(common_eval, transfer_train))

    report = [
        f"# DACHUI Round-4 Geometric Feedback {mode_label} report", "",
        "Primary story: literature-internal learnability -> zero-shot source-target failure -> input-only target-space soft weighting -> few-shot target adaptation -> operating-point evaluation.", "",
        f"- Hard-cleaned literature long records: {len(lit)}",
        f"- Physically eligible literature retained for M6: {int(selected_lit["mix_id"].nunique())} mixes / {len(selected_lit)} long records (100%; no similarity hard cut)",
        "- All 33 Round-4 MIX INPUTS define the target domain before any strength is released.",
        "- Literature similarity uses input geometry only. No 3 d/7 d/28 d experiment strength and no literature strength is used for screening.",
        "- CO2 NR/N/R/blank/not reported/未报告 remains fixed at 0.04%; no missing-CO2 penalty is applied.",
        "- Fixed nested rank sequence is 1,17,33 -> +11,23 -> +4,30 -> +8,26 -> geometry-optimized +13,21 -> +15,19.",
        "- Budgets are 3/5/7/9/11/13 and are strictly nested. Experimental points are selected only from design geometry, never from strength or model error.",
        f"- Primary operating point is {operating_point_budget} labeled mixes; its remaining {len(operating_holdout_ids)} mixes are not used to train/calibrate the selected operating-point model.",
        f"- Strict cross-budget common holdout contains {len(common_holdout_ids)} mixes outside the full 13-point sensitivity envelope and is identical for every budget.",
        "- A secondary all-currently-unseen curve is also saved, but its test set shrinks as the feedback budget grows.",
        f"- The 3/5/7/9/11/13 curve is retained as sensitivity/discussion material; retrospective strict-holdout recommendation: {auto_transfer_budget}.",
        f"- Primary/deployment budget: {transfer_budget} ({selection_mode}); configured article operating point is {operating_point_budget}.",
        "- The 24-mix operating-point panel is unseen by the selected 9-mix model, but four of those mixes are used by the 11/13 supplementary sensitivity runs; therefore the strict all-sensitivity untouched panel is the 20-mix common holdout.",
        "- The literature pool is never re-screened after real strengths are released; this cleanly isolates the effect of 3/5/7/9/11/13 strength feedback.",
        "- Every formal M strategy owns its own SSA result; no best algorithm or hyperparameters are borrowed across M models.",
        "- M0/M2/M3 are target-grouped SSA-tuned separately at every budget. M4/M7 retain independent source+residual ablations; M6 V5 uses its own source prediction as an auxiliary feature in a target-only categorical-age calibrator.",
        "- Inner model selection uses released experiment_mix_id groups only; the common holdout never enters SSA, lambda selection, or calibrator selection.",
        "- Primary model-selection metrics are pooled unique OOF RMSE/MAE/R2; fold means are stability descriptors only.",
        "- M8 adaptively blends M0 and M6 with age-specific alpha chosen only from released-experiment grouped OOF predictions. M7 remains a permuted-similarity-weight negative control.",
    ]
    (out / "analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    create_excel(out / "DACHUI_final_report.xlsx", {
        "summary": [{
            "literature_records": len(lit),
            "input_screened_records": len(selected_lit),
            "round4_input_mixes": 33,
            "operating_point_holdout_mixes": len(operating_holdout_ids),
            "strict_common_holdout_mixes": len(common_holdout_ids),
            "common_holdout_mixes": len(common_holdout_ids),
            "selected_budget": transfer_budget,
            "configured_operating_point_budget": operating_point_budget,
            "literature_fraction": chosen_fraction,
        }],
        "geometry": geometry_metrics,
        "literature_fraction": fraction_table.to_dict("records"),
        "operating_point_24_metrics": operating_metric_rows,
        "budget_common": common_curve,
        "budget_all_unseen": unseen_curve,
        "model_ablation": strict_rows,
        "model_pooled": pooled_prediction_summary(pred_rows),
    })

    # Keep only figures that can be generated from the new files; old nested/active
    # figure names are no longer part of the formal protocol.
    try:
        publication_figures(out)
    except Exception as exc:
        write_csv(out / "figure_generation_warning.csv", [{"warning": str(exc)}])

    summary = {
        "scheme": "round4_input_screened_geometric_feedback",
        "mode": mode_label,
        "round4_input_mixes": 33,
        "pilot_budgets": pilot_budgets,
        "pilot_rank_sequence": [int(r["rank"]) for r in geometric_audit[:max(pilot_budgets)]],
        "pilot_strategy": "fixed hierarchical geometry; no strength-driven acquisition",
        "literature_screening": "all physically eligible literature retained; input similarity assigns 0-2 weights",
        "similarity_uses_experiment_strength": False,
        "similarity_uses_literature_strength": False,
        "strength_response_similarity_removed": True,
        "literature_pool_fixed_across_budgets": True,
        "operating_point_budget": operating_point_budget,
        "operating_point_holdout_mix_count": len(operating_holdout_ids),
        "operating_point_24_not_used_by_selected_model": True,
        "operating_point_24_strictly_untouched_across_all_sensitivity": False,
        "common_holdout_mix_count": len(common_holdout_ids),
        "strict_common_holdout_untouched_across_all_sensitivity": True,
        "all_budgets_share_common_holdout": True,
        "automatic_transfer_budget": auto_transfer_budget,
        "transfer_ssa_budget": transfer_budget,
        "budget_selection_mode": selection_mode,
        "independent_final_test_claimed": False,
        "CO2_NR_default_percent": cfg["co2"]["nr_default_percent"],
        "deployment_model_role": deployment_info.get("role"),
        "all_models_independent_ssa": True,
        "cross_M_model_reuse": False,
        "target_grouped_inner_validation": True,
        "primary_inner_metric": "pooled_unique_OOF_RMSE",
        "primary_selection_method": primary_method,
        "secondary_transfer_method": secondary_method,
        "outputs_complete": True,
    }
    (out / f"{mode_label}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description="DACHUI lightweight-to-publication pipeline.")
    parser.add_argument("--literature", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(SKILL_DIR / "config.json"))
    parser.add_argument("--profile", choices=["diagnostic", "daily", "validation", "publication"], default="daily")
    parser.add_argument("--transfer-budget", default="operating", help="operating (default, config operating point), auto (retrospective sensitivity only), or a configured pilot budget")
    parser.add_argument("--m7-random-repeats", type=int, default=None)
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
