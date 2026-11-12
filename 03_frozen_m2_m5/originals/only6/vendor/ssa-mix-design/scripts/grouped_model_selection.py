from __future__ import annotations
import csv, json, math
from pathlib import Path
from typing import Any, Sequence
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
from ssa_algorithm import ssa_optimize


def _canon(name: str) -> str:
    return {
        "RF": "RandomForest", "RandomForest": "RandomForest",
        "ET": "ExtraTrees", "ExtraTrees": "ExtraTrees",
        "XGB": "XGBoost", "XGBoost": "XGBoost",
    }.get(name, name)


def params_from_vector(model_name: str, x: np.ndarray) -> dict[str, Any]:
    model_name = _canon(model_name)
    if model_name == "Ridge":
        return {"alpha": float(10 ** x[0])}
    if model_name == "Huber":
        return {"alpha": float(10 ** x[0]), "epsilon": float(x[1]), "max_iter": 1000}
    if model_name == "SVR":
        return {"C": float(10 ** x[0]), "epsilon": float(10 ** x[1]), "gamma": float(10 ** x[2])}
    if model_name in {"RandomForest", "ExtraTrees"}:
        return {
            "n_estimators": int(round(x[0])),
            "max_depth": int(round(x[1])),
            "min_samples_leaf": int(round(x[2])),
            "max_features": float(x[3]),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": int(round(x[0])), "max_depth": int(round(x[1])),
            "learning_rate": float(10 ** x[2]), "subsample": float(x[3]),
            "colsample_bytree": float(x[4]), "reg_lambda": float(10 ** x[5]),
            "min_child_weight": float(max(1.0, x[6])),
        }
    raise KeyError(model_name)


def bounds_for_model(model_name: str):
    model_name = _canon(model_name)
    if model_name == "Ridge":
        return [-3.0], [3.0]
    if model_name == "Huber":
        return [-6.0, 1.1], [1.0, 2.5]
    if model_name == "SVR":
        return [-1.0, -3.0, -4.0], [3.0, 1.0, 0.0]
    if model_name in {"RandomForest", "ExtraTrees"}:
        return [60, 2, 1, 0.45], [240, 10, 6, 1.0]
    if model_name == "XGBoost":
        return [60, 2, -2.5, 0.70, 0.70, -2.0, 1.0], [260, 6, -0.7, 1.0, 1.0, 1.2, 8.0]
    raise KeyError(model_name)


def make_model(model_name: str, params: dict[str, Any], seed: int):
    name = _canon(model_name)
    if name == "Ridge":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(**params))])
    if name == "Huber":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", HuberRegressor(**params))])
    if name == "SVR":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", SVR(**params))])
    if name == "RandomForest":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(random_state=seed, n_jobs=1, **params))])
    if name == "ExtraTrees":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(random_state=seed, n_jobs=1, **params))])
    if name == "XGBoost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed")
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=1, verbosity=0, **params))])
    raise KeyError(model_name)


def fit_model(model, X, y, sample_weight=None):
    if sample_weight is None:
        return model.fit(X, y)
    try:
        return model.fit(X, y, sample_weight=sample_weight)
    except (TypeError, ValueError):
        if isinstance(model, Pipeline):
            last = model.steps[-1][0]
            try:
                return model.fit(X, y, **{f"{last}__sample_weight": sample_weight})
            except (TypeError, ValueError):
                return model.fit(X, y)
        return model.fit(X, y)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f, fieldnames=keys or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _metric_dict(y, pred):
    y=np.asarray(y,float); pred=np.asarray(pred,float)
    err=pred-y
    return {
        "R2": float(r2_score(y,pred)) if len(y)>1 else float("nan"),
        "RMSE": float(math.sqrt(mean_squared_error(y,pred))),
        "MAE": float(mean_absolute_error(y,pred)),
        "Bias": float(np.mean(err)),
        "MaxAE": float(np.max(np.abs(err))),
        "Within_5MPa_percent": float(np.mean(np.abs(err)<=5.0)*100.0),
    }


def _profile(fast: bool, population: int | None = None, iterations: int | None = None):
    if population is not None and iterations is not None:
        return int(population), int(iterations)
    return (3,1) if fast else (6,4)


def _available_candidates(candidates):
    cand=[_canon(c) for c in list(candidates or ["Ridge","SVR","RandomForest","ExtraTrees","XGBoost"])]
    if XGBRegressor is None:
        cand=[c for c in cand if c!="XGBoost"]
    return cand


def select_grouped_model(
    df, features: Sequence[str], target: str, group_col: str, output_dir,
    sample_weight_col: str | None = None, candidates=None, n_splits: int = 5,
    fast: bool = False, random_state: int = 42, population: int | None = None,
    iterations: int | None = None, artifact_prefix: str = "source_model",
):
    """Independent source/target-only SSA with grouped OOF selection.

    Primary model-family comparison uses pooled unique OOF metrics. Fold means are
    exported only as stability descriptors. Preprocessing stays inside sklearn pipelines.
    """
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    X=df[list(features)].to_numpy(float); y=df[target].to_numpy(float); groups=df[group_col].astype(str).to_numpy()
    uniq=np.unique(groups); ns=min(int(n_splits),len(uniq))
    if ns<2:
        raise ValueError(f"Need at least 2 independent groups, got {len(uniq)}")
    cv=GroupKFold(n_splits=ns)
    weights=None if sample_weight_col is None else df[sample_weight_col].to_numpy(float)
    cand=_available_candidates(candidates)
    pop,iters=_profile(fast,population,iterations)
    summaries=[]; all_fold_rows=[]; all_oof_rows=[]

    for ci,name in enumerate(cand):
        lb,ub=bounds_for_model(name)
        def fitness(vec):
            p=params_from_vector(name,np.asarray(vec,float)); ys=[]; ps=[]
            for fi,(tr,te) in enumerate(cv.split(X,y,groups),start=1):
                try:
                    m=make_model(name,p,random_state+100*ci+fi)
                    sw=None if weights is None else weights[tr]
                    fit_model(m,X[tr],y[tr],sw)
                    pred=np.asarray(m.predict(X[te]),float)
                    ys.extend(y[te].tolist()); ps.extend(pred.tolist())
                except Exception:
                    return 1e12
            return _metric_dict(ys,ps)["RMSE"]
        np.random.seed(random_state+ci)
        vec,best_fit,curve=ssa_optimize(fitness,lb,ub,pop=pop,max_iter=iters,verbose=False,model_name=name)
        params=params_from_vector(name,np.asarray(vec,float))
        ys=[]; ps=[]; local_rows=[]
        for fi,(tr,te) in enumerate(cv.split(X,y,groups),start=1):
            m=make_model(name,params,random_state+1000+100*ci+fi)
            sw=None if weights is None else weights[tr]
            fit_model(m,X[tr],y[tr],sw)
            pred=np.asarray(m.predict(X[te]),float)
            fm=_metric_dict(y[te],pred)
            row={"fold":fi,"model":name,**fm,"n_train":len(tr),"n_test":len(te),"n_train_groups":len(np.unique(groups[tr])),"n_test_groups":len(np.unique(groups[te]))}
            local_rows.append(row); all_fold_rows.append(row)
            for idx,pv in zip(te,pred):
                all_oof_rows.append({"row_index":int(idx),"group":str(groups[idx]),"model":name,"actual":float(y[idx]),"predicted":float(pv),"fold":fi})
            ys.extend(y[te].tolist()); ps.extend(pred.tolist())
        pooled=_metric_dict(ys,ps)
        summaries.append({
            "model":name,
            "pooled_OOF_R2":pooled["R2"],"pooled_OOF_RMSE":pooled["RMSE"],"pooled_OOF_MAE":pooled["MAE"],
            "pooled_OOF_Bias":pooled["Bias"],"pooled_OOF_MaxAE":pooled["MaxAE"],"pooled_OOF_Within_5MPa_percent":pooled["Within_5MPa_percent"],
            "fold_R2_mean":float(np.nanmean([r["R2"] for r in local_rows])),
            "fold_R2_SD":float(np.nanstd([r["R2"] for r in local_rows],ddof=1)) if len(local_rows)>1 else 0.0,
            "fold_RMSE_mean":float(np.mean([r["RMSE"] for r in local_rows])),
            "fold_RMSE_SD":float(np.std([r["RMSE"] for r in local_rows],ddof=1)) if len(local_rows)>1 else 0.0,
            "ssa_fitness_pooled_RMSE":float(best_fit),"best_params":json.dumps(params,ensure_ascii=False),
            "ssa_curve":json.dumps([float(v) for v in curve],ensure_ascii=False),"pop":pop,"iterations":iters,"n_splits":ns,
        })
    best=min(summaries,key=lambda r:(r["pooled_OOF_RMSE"],r["pooled_OOF_MAE"],-r["pooled_OOF_R2"]))
    best_name=best["model"]; best_params=json.loads(best["best_params"])
    model=make_model(best_name,best_params,random_state)
    fit_model(model,X,y,weights)
    model_path=out/f"{artifact_prefix}_best_model.pkl"; joblib.dump(model,model_path)
    feat_path=out/f"{artifact_prefix}_features.json"; feat_path.write_text(json.dumps(list(features),ensure_ascii=False,indent=2),encoding="utf-8")
    _write_csv(out/f"{artifact_prefix}_fold_metrics.csv",all_fold_rows)
    _write_csv(out/f"{artifact_prefix}_oof_predictions.csv",all_oof_rows)
    _write_csv(out/f"{artifact_prefix}_selection_summary.csv",summaries)
    best_info={
        "best_source_model_name":best_name,"best_source_model_params":best_params,
        "best_source_model_path":str(model_path),"best_source_preprocessor_path":None,"best_source_features_path":str(feat_path),
        "selection_metric":"pooled_unique_OOF_RMSE_on_grouped_validation",
        "pooled_grouped_cv_RMSE":best["pooled_OOF_RMSE"],"pooled_grouped_cv_MAE":best["pooled_OOF_MAE"],"pooled_grouped_cv_R2":best["pooled_OOF_R2"],
        # backward-compatible aliases
        "mean_grouped_cv_RMSE":best["pooled_OOF_RMSE"],"mean_grouped_cv_MAE":best["pooled_OOF_MAE"],"mean_grouped_cv_R2":best["pooled_OOF_R2"],
        "fold_R2_mean":best["fold_R2_mean"],"fold_R2_SD":best["fold_R2_SD"],
        "ssa_population":pop,"ssa_iterations":iters,"group_column":group_col,"n_splits":ns,
    }
    (out/f"{artifact_prefix}_best_model.json").write_text(json.dumps(best_info,ensure_ascii=False,indent=2),encoding="utf-8")
    # compatibility with old callers expecting this filename
    if artifact_prefix == "source_model":
        (out/"best_source_model.json").write_text(json.dumps(best_info,ensure_ascii=False,indent=2),encoding="utf-8")
        joblib.dump(model,out/"best_source_model.pkl")
        (out/"best_source_features.json").write_text(json.dumps(list(features),ensure_ascii=False,indent=2),encoding="utf-8")
        _write_csv(out/"source_model_cv_results.csv",all_fold_rows)
        _write_csv(out/"source_model_selection_summary.csv",summaries)
    return {"model":model,"best_info":best_info,"cv_results":all_fold_rows,"oof_predictions":all_oof_rows,"summary":summaries}



def _group_holdout_split_indices(groups, train_fraction: float, validation_fraction: float, test_fraction: float, random_state: int):
    """Deterministically split whole groups into train/validation/test partitions.

    Fractions are targets on the number of independent groups (study_id), not exact row
    counts. This prevents rows from the same study leaking across partitions.
    """
    groups=np.asarray(groups).astype(str)
    ratios=np.asarray([train_fraction, validation_fraction, test_fraction], dtype=float)
    if np.any(ratios <= 0) or not np.isclose(float(ratios.sum()), 1.0, atol=1e-9):
        raise ValueError("train/validation/test fractions must all be > 0 and sum to 1")
    uniq=np.unique(groups)
    if len(uniq) < 3:
        raise ValueError(f"Need at least 3 independent groups for train/validation/test, got {len(uniq)}")
    rng=np.random.default_rng(int(random_state))
    shuffled=uniq.copy(); rng.shuffle(shuffled)

    # Largest-remainder allocation, with at least one independent study in every split.
    raw=ratios * len(shuffled)
    counts=np.floor(raw).astype(int)
    counts=np.maximum(counts, 1)
    while counts.sum() > len(shuffled):
        candidates=[i for i in range(3) if counts[i] > 1]
        if not candidates:
            break
        i=max(candidates, key=lambda j: (counts[j]-raw[j], counts[j]))
        counts[i]-=1
    while counts.sum() < len(shuffled):
        i=max(range(3), key=lambda j: (raw[j]-counts[j], ratios[j]))
        counts[i]+=1

    ntr,nva,nte=map(int,counts)
    train_groups=set(shuffled[:ntr])
    validation_groups=set(shuffled[ntr:ntr+nva])
    test_groups=set(shuffled[ntr+nva:ntr+nva+nte])
    tr=np.flatnonzero(np.isin(groups, list(train_groups)))
    va=np.flatnonzero(np.isin(groups, list(validation_groups)))
    te=np.flatnonzero(np.isin(groups, list(test_groups)))
    return tr,va,te,train_groups,validation_groups,test_groups


def select_grouped_holdout_model(
    df, features: Sequence[str], target: str, group_col: str, output_dir,
    *, train_fraction: float = 0.70, validation_fraction: float = 0.15,
    test_fraction: float = 0.15, candidates=None, fast: bool = False,
    random_state: int = 42, population: int | None = None, iterations: int | None = None,
    artifact_prefix: str = "literature_holdout",
):
    """Literature-only train/validation/test benchmark with strict study-level separation.

    SSA hyperparameters and model family are selected ONLY from the validation split.
    The test split is untouched until model selection is finished. The selected model is
    then refit on train+validation and evaluated exactly once on the test split. No
    external/experimental data are accepted by this function.
    """
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    X=df[list(features)].to_numpy(float); y=df[target].to_numpy(float)
    groups=df[group_col].astype(str).to_numpy()
    tr,va,te,trg,vag,teg=_group_holdout_split_indices(
        groups, train_fraction, validation_fraction, test_fraction, random_state
    )
    cand=_available_candidates(candidates); pop,iters=_profile(fast,population,iterations)
    summaries=[]; validation_rows=[]

    for ci,name in enumerate(cand):
        lb,ub=bounds_for_model(name)
        def fitness(vec):
            try:
                params=params_from_vector(name,np.asarray(vec,float))
                m=make_model(name,params,random_state+100*ci+1)
                fit_model(m,X[tr],y[tr])
                pred=np.asarray(m.predict(X[va]),float)
                return _metric_dict(y[va],pred)["RMSE"]
            except Exception:
                return 1e12
        np.random.seed(random_state+ci)
        vec,best_fit,curve=ssa_optimize(fitness,lb,ub,pop=pop,max_iter=iters,verbose=False,model_name=name)
        params=params_from_vector(name,np.asarray(vec,float))
        m=make_model(name,params,random_state+1000+100*ci)
        fit_model(m,X[tr],y[tr])
        pred=np.asarray(m.predict(X[va]),float)
        vm=_metric_dict(y[va],pred)
        summaries.append({
            "model":name,"validation_R2":vm["R2"],"validation_RMSE":vm["RMSE"],
            "validation_MAE":vm["MAE"],"validation_Bias":vm["Bias"],
            "validation_MaxAE":vm["MaxAE"],"validation_Within_5MPa_percent":vm["Within_5MPa_percent"],
            "ssa_fitness_validation_RMSE":float(best_fit),
            "best_params":json.dumps(params,ensure_ascii=False),
            "ssa_curve":json.dumps([float(v) for v in curve],ensure_ascii=False),
            "pop":pop,"iterations":iters,
        })
        for idx,pv in zip(va,pred):
            validation_rows.append({"row_index":int(idx),"group":str(groups[idx]),"model":name,
                                    "actual":float(y[idx]),"predicted":float(pv),"split":"validation"})

    best=min(summaries,key=lambda r:(r["validation_RMSE"],r["validation_MAE"],-r["validation_R2"]))
    best_name=best["model"]; best_params=json.loads(best["best_params"])

    # Train-only fit used to report training/validation behavior of the selected configuration.
    selection_model=make_model(best_name,best_params,random_state+7000)
    fit_model(selection_model,X[tr],y[tr])
    train_pred=np.asarray(selection_model.predict(X[tr]),float)
    validation_pred=np.asarray(selection_model.predict(X[va]),float)
    train_metrics=_metric_dict(y[tr],train_pred)
    validation_metrics=_metric_dict(y[va],validation_pred)

    # Final evaluated model: refit selected configuration on train + validation; test remains untouched.
    fit_idx=np.r_[tr,va]
    model=make_model(best_name,best_params,random_state+8000)
    fit_model(model,X[fit_idx],y[fit_idx])
    test_pred=np.asarray(model.predict(X[te]),float)
    test_metrics=_metric_dict(y[te],test_pred)

    model_path=out/f"{artifact_prefix}_best_model.pkl"; joblib.dump(model,model_path)
    feat_path=out/f"{artifact_prefix}_features.json"
    feat_path.write_text(json.dumps(list(features),ensure_ascii=False,indent=2),encoding="utf-8")
    _write_csv(out/f"{artifact_prefix}_candidate_validation_summary.csv",summaries)
    _write_csv(out/f"{artifact_prefix}_validation_predictions_all_models.csv",validation_rows)

    selected_rows=[]
    for split,idxs,preds in [("train",tr,train_pred),("validation",va,validation_pred),("test",te,test_pred)]:
        for idx,pv in zip(idxs,preds):
            selected_rows.append({"row_index":int(idx),"group":str(groups[idx]),"split":split,
                                  "actual":float(y[idx]),"predicted":float(pv),"model":best_name})
    _write_csv(out/f"{artifact_prefix}_selected_model_predictions.csv",selected_rows)

    split_rows=[]
    for idx,g in enumerate(groups):
        split="train" if idx in set(tr.tolist()) else ("validation" if idx in set(va.tolist()) else "test")
        row={"row_index":int(idx),"group":str(g),"split":split,"actual":float(y[idx])}
        for col in ["study_id","mix_id","sample_id","curing_age"]:
            if col in df.columns:
                row[col]=df.iloc[idx][col]
        split_rows.append(row)
    _write_csv(out/f"{artifact_prefix}_split_assignments.csv",split_rows)

    metrics_rows=[]
    for split,metrics,nrows,ngroups in [
        ("train",train_metrics,len(tr),len(trg)),
        ("validation",validation_metrics,len(va),len(vag)),
        ("test",test_metrics,len(te),len(teg)),
    ]:
        metrics_rows.append({"split":split,"n_rows":int(nrows),"n_groups":int(ngroups),**metrics})
    _write_csv(out/f"{artifact_prefix}_metrics.csv",metrics_rows)

    split_summary={
        "group_column":group_col,"random_state":int(random_state),
        "requested_group_fractions":{"train":float(train_fraction),"validation":float(validation_fraction),"test":float(test_fraction)},
        "actual_group_counts":{"train":len(trg),"validation":len(vag),"test":len(teg),"total":len(np.unique(groups))},
        "actual_row_counts":{"train":len(tr),"validation":len(va),"test":len(te),"total":len(df)},
        "actual_row_fractions":{"train":len(tr)/len(df),"validation":len(va)/len(df),"test":len(te)/len(df)},
        "group_overlap_checks":{
            "train_validation":len(trg & vag),"train_test":len(trg & teg),"validation_test":len(vag & teg)
        },
        "selection_uses_test":False,"uses_external_data":False,
    }
    (out/f"{artifact_prefix}_split_summary.json").write_text(json.dumps(split_summary,ensure_ascii=False,indent=2),encoding="utf-8")

    best_info={
        "best_source_model_name":best_name,"best_source_model_params":best_params,
        "best_source_model_path":str(model_path),"best_source_features_path":str(feat_path),
        "selection_metric":"validation_RMSE_on_fixed_study_group_holdout; test_never_used_for_selection",
        "train_RMSE":train_metrics["RMSE"],"train_MAE":train_metrics["MAE"],"train_R2":train_metrics["R2"],
        "validation_RMSE":validation_metrics["RMSE"],"validation_MAE":validation_metrics["MAE"],"validation_R2":validation_metrics["R2"],
        "test_RMSE":test_metrics["RMSE"],"test_MAE":test_metrics["MAE"],"test_R2":test_metrics["R2"],
        "ssa_population":pop,"ssa_iterations":iters,"group_column":group_col,
        "train_fraction":float(train_fraction),"validation_fraction":float(validation_fraction),"test_fraction":float(test_fraction),
        "random_state":int(random_state),"uses_external_data":False,"test_used_for_selection":False,
    }
    (out/f"{artifact_prefix}_best_model.json").write_text(json.dumps(best_info,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"model":model,"best_info":best_info,"split_summary":split_summary,
            "metrics":metrics_rows,"summary":summaries}

def select_target_grouped_model(
    source_df: pd.DataFrame | None,
    target_df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    target_group_col: str,
    output_dir,
    *,
    source_weight_col: str | None = None,
    target_weight: float = 1.0,
    target_weight_search: tuple[float, float] | None = None,
    candidates=None,
    n_splits: int = 5,
    fast: bool = False,
    random_state: int = 42,
    population: int | None = None,
    iterations: int | None = None,
    artifact_prefix: str = "target_guided_model",
):
    """SSA selected ONLY by held-out target mixes; source rows are training-only in every fold.

    This is the DACHUI analogue of symmetric target-domain LOBO. It prevents a large
    literature source table from dominating model selection merely by being numerous.
    """
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    source=pd.DataFrame(columns=list(features)+[target]) if source_df is None else source_df.copy()
    target_df=target_df.copy()
    Xt=target_df[list(features)].to_numpy(float); yt=target_df[target].to_numpy(float)
    groups=target_df[target_group_col].astype(str).to_numpy(); uniq=np.unique(groups)
    ns=min(int(n_splits),len(uniq))
    if ns<2:
        raise ValueError(f"Need at least 2 target groups, got {len(uniq)}")
    Xs=source[list(features)].to_numpy(float) if len(source) else np.empty((0,len(features)))
    ys=source[target].to_numpy(float) if len(source) else np.empty((0,),float)
    sw_source=None if source_weight_col is None or not len(source) else source[source_weight_col].to_numpy(float)
    cv=GroupKFold(n_splits=ns); cand=_available_candidates(candidates); pop,iters=_profile(fast,population,iterations)
    summaries=[]; all_fold_rows=[]; all_oof_rows=[]

    def fold_train_arrays(tr, active_target_weight: float):
        Xtrain=np.vstack([Xs,Xt[tr]]) if len(source) else Xt[tr]
        ytrain=np.r_[ys,yt[tr]] if len(source) else yt[tr]
        if len(source):
            srcw=np.ones(len(source),float) if sw_source is None else sw_source
            weights=np.r_[srcw,np.full(len(tr),float(active_target_weight))]
        else:
            weights=np.full(len(tr),float(active_target_weight)) if active_target_weight != 1.0 else None
        return Xtrain,ytrain,weights

    for ci,name in enumerate(cand):
        base_lb,base_ub=bounds_for_model(name)
        if target_weight_search is not None:
            tw_lo,tw_hi=map(float,target_weight_search)
            if not (tw_lo > 0 and tw_hi >= tw_lo):
                raise ValueError("target_weight_search requires 0 < low <= high")
            lb=list(base_lb)+[math.log10(tw_lo)]; ub=list(base_ub)+[math.log10(tw_hi)]
        else:
            lb,ub=list(base_lb),list(base_ub)
        def unpack(vec):
            arr=np.asarray(vec,float)
            params=params_from_vector(name,arr[:len(base_lb)])
            active_tw=float(10**arr[-1]) if target_weight_search is not None else float(target_weight)
            return params,active_tw
        def fitness(vec):
            params,active_tw=unpack(vec); yall=[]; pall=[]
            for fi,(tr,te) in enumerate(cv.split(Xt,yt,groups),start=1):
                try:
                    Xtr,ytr,sw=fold_train_arrays(tr,active_tw)
                    m=make_model(name,params,random_state+100*ci+fi); fit_model(m,Xtr,ytr,sw)
                    pred=np.asarray(m.predict(Xt[te]),float)
                    yall.extend(yt[te].tolist()); pall.extend(pred.tolist())
                except Exception:
                    return 1e12
            return _metric_dict(yall,pall)["RMSE"]
        np.random.seed(random_state+ci)
        vec,best_fit,curve=ssa_optimize(fitness,lb,ub,pop=pop,max_iter=iters,verbose=False,model_name=name)
        params,active_tw=unpack(vec); yall=[]; pall=[]; local=[]
        for fi,(tr,te) in enumerate(cv.split(Xt,yt,groups),start=1):
            Xtr,ytr,sw=fold_train_arrays(tr,active_tw)
            m=make_model(name,params,random_state+1000+100*ci+fi); fit_model(m,Xtr,ytr,sw)
            pred=np.asarray(m.predict(Xt[te]),float); fm=_metric_dict(yt[te],pred)
            row={"fold":fi,"model":name,**fm,"n_source_train":int(len(source)),"n_target_train":int(len(tr)),"n_target_test":int(len(te)),"n_target_train_groups":len(np.unique(groups[tr])),"n_target_test_groups":len(np.unique(groups[te]))}
            local.append(row); all_fold_rows.append(row)
            for idx,pv in zip(te,pred):
                all_oof_rows.append({"row_index":int(idx),"target_group":str(groups[idx]),"model":name,"actual":float(yt[idx]),"predicted":float(pv),"fold":fi})
            yall.extend(yt[te].tolist()); pall.extend(pred.tolist())
        pooled=_metric_dict(yall,pall)
        summaries.append({
            "model":name,"pooled_target_OOF_R2":pooled["R2"],"pooled_target_OOF_RMSE":pooled["RMSE"],"pooled_target_OOF_MAE":pooled["MAE"],
            "pooled_target_OOF_Bias":pooled["Bias"],"pooled_target_OOF_MaxAE":pooled["MaxAE"],"pooled_target_OOF_Within_5MPa_percent":pooled["Within_5MPa_percent"],
            "fold_target_R2_mean":float(np.nanmean([r["R2"] for r in local])),"fold_target_RMSE_mean":float(np.mean([r["RMSE"] for r in local])),
            "ssa_fitness_pooled_target_RMSE":float(best_fit),"best_params":json.dumps(params,ensure_ascii=False),"ssa_curve":json.dumps([float(v) for v in curve],ensure_ascii=False),
            "target_weight":float(active_tw),"target_weight_search":None if target_weight_search is None else list(target_weight_search),"pop":pop,"iterations":iters,"n_splits":ns,"source_rows":int(len(source)),"target_rows":int(len(target_df)),
        })
    best=min(summaries,key=lambda r:(r["pooled_target_OOF_RMSE"],r["pooled_target_OOF_MAE"],-r["pooled_target_OOF_R2"]))
    best_name=best["model"]; best_params=json.loads(best["best_params"])
    Xfull=np.vstack([Xs,Xt]) if len(source) else Xt; yfull=np.r_[ys,yt] if len(source) else yt
    best_tw=float(best["target_weight"])
    if len(source):
        srcw=np.ones(len(source),float) if sw_source is None else sw_source
        fullw=np.r_[srcw,np.full(len(target_df),best_tw)]
    else:
        fullw=np.full(len(target_df),best_tw) if best_tw != 1.0 else None
    model=make_model(best_name,best_params,random_state); fit_model(model,Xfull,yfull,fullw)
    model_path=out/f"{artifact_prefix}_best_model.pkl"; joblib.dump(model,model_path)
    _write_csv(out/f"{artifact_prefix}_fold_metrics.csv",all_fold_rows); _write_csv(out/f"{artifact_prefix}_oof_predictions.csv",all_oof_rows); _write_csv(out/f"{artifact_prefix}_selection_summary.csv",summaries)
    info={
        "best_model_name":best_name,"best_model_params":best_params,"best_model_path":str(model_path),
        "selection_protocol":"target_grouped_OOF; source always train-only; validation target-only",
        "selection_metric":"pooled_target_OOF_RMSE","pooled_target_OOF_RMSE":best["pooled_target_OOF_RMSE"],"pooled_target_OOF_MAE":best["pooled_target_OOF_MAE"],"pooled_target_OOF_R2":best["pooled_target_OOF_R2"],
        "target_weight":best_tw,"target_weight_search":None if target_weight_search is None else list(target_weight_search),"target_group_column":target_group_col,"n_splits":ns,"source_rows":int(len(source)),"target_rows":int(len(target_df)),
        "ssa_population":pop,"ssa_iterations":iters,
    }
    (out/f"{artifact_prefix}_best_model.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"model":model,"best_info":info,"cv_results":all_fold_rows,"oof_predictions":all_oof_rows,"summary":summaries}
