"""Planning-input-only surrogates with grouped nested CV and frozen simulation validation."""
from __future__ import annotations
import json, time, hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.inspection import permutation_importance
from randomization_core import exact_risk

HERE=Path(__file__).absolute().parent
FEATURES=["method","N","log_N","N_inv","n_factors","block_size","max_block_size",
    "block_remainder","total_strata","log_strata","min_prevalence","max_prevalence",
    "covariate_correlation","expected_min_stratum_size","preferred_probability","mti"]
TARGETS={"binary":"high_risk","multiclass":"severity","regression":"imbalance_probability"}
SCORING={"binary":"neg_brier_score","multiclass":"neg_log_loss","regression":"neg_mean_absolute_error"}

def preprocessing():
    return ColumnTransformer([("method",OneHotEncoder(handle_unknown="ignore",sparse_output=False),["method"]),
                              ("numeric",StandardScaler(),FEATURES[1:])])

def candidates(task):
    if task=="regression":
        options={
          "Ridge":(Ridge(),{"model__alpha":[1.,100.]}),
          "RandomForest":(RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=1),{"model__min_samples_leaf":[1,5]}),
          "GradientBoosting":(GradientBoostingRegressor(n_estimators=100,random_state=42,loss="squared_error"),{"model__max_depth":[2,3]})}
    else:
        options={
          "LogisticRegression":(LogisticRegression(max_iter=3000,random_state=42),{"model__C":[.1,10.]}),
          "RandomForest":(RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=1),{"model__min_samples_leaf":[1,5]}),
          "GradientBoosting":(GradientBoostingClassifier(n_estimators=80,random_state=42),{"model__max_depth":[2,3]})}
    return {name:(Pipeline([("features",preprocessing()),("model",model)]),grid) for name,(model,grid) in options.items()}

def aligned_probabilities(model,x,n_classes):
    raw=model.predict_proba(x)
    out=np.zeros((len(x),n_classes))
    out[:,model.classes_.astype(int)]=raw
    return out

def multiclass_scorer(model,x,y):
    return -log_loss(y,aligned_probabilities(model,x,3),labels=[0,1,2])

def predictions(model,x,task):
    if task=="regression": return np.clip(model.predict(x),0.,1.)
    return aligned_probabilities(model,x,2 if task=="binary" else 3)

def metrics(y,p,task):
    if task=="regression":
        return {"R2":float(r2_score(y,p)),"MAE":float(mean_absolute_error(y,p)),
                "RMSE":float(np.sqrt(mean_squared_error(y,p)))}
    pred=np.argmax(p,axis=1)
    out={"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(np.mean([np.mean(pred[np.asarray(y)==label]==label) for label in np.unique(y)])),
         "macro_F1":float(f1_score(y,pred,average="macro",zero_division=0)),"log_loss":float(log_loss(y,p,labels=list(range(p.shape[1]))))}
    if task=="binary":
        out.update(AUC=float(roc_auc_score(y,p[:,1])) if len(np.unique(y))>1 else None,
                   Brier=float(brier_score_loss(y,p[:,1])))
    return out

def prediction_frame(df,p,task,model,stage):
    out=df[["scenario_id","N","method","profile","n_factors","block_size","iterations",
             "imbalance_probability","high_risk","severity","binary_label_uncertain"]].copy()
    out["task"]=task;out["model"]=model;out["stage"]=stage
    if task=="regression": out["prediction"]=p
    else:
        for i in range(p.shape[1]): out["probability_"+str(i)]=p[:,i]
        out["prediction"]=np.argmax(p,axis=1)
    return out

def run():
    start=time.time()
    train=pd.read_csv(HERE/"results/training_scenarios.csv")
    valid=pd.read_csv(HERE/"results/validation_scenarios.csv")
    assert set(train.N).isdisjoint(set(valid.N))
    assert set(train.scenario_id).isdisjoint(set(valid.scenario_id))
    x=train[FEATURES];xval=valid[FEATURES];groups=train.N.to_numpy()
    outer=GroupKFold(n_splits=5)
    fold_rows=[];summary=[];predframes=[];selection={};validation_report={}
    all_outer=list(outer.split(x,groups=groups))
    foldmap=[]
    for fold,(tr,te) in enumerate(all_outer,1):
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        foldmap.extend({"scenario_id":train.iloc[i].scenario_id,"N":int(groups[i]),"outer_fold":fold} for i in te)
    pd.DataFrame(foldmap).to_csv(HERE/"results/GroupedCV_FoldAssignments.csv",index=False)
    for task,target in TARGETS.items():
        y=train[target].to_numpy();yv=valid[target].to_numpy()
        task_scores={}
        for name,(pipe,grid) in candidates(task).items():
            oof=np.zeros(len(train)) if task=="regression" else np.zeros((len(train),2 if task=="binary" else 3))
            for fold,(tr,te) in enumerate(all_outer,1):
                search=GridSearchCV(clone(pipe),grid,scoring=(multiclass_scorer if task=="multiclass" else SCORING[task]),cv=GroupKFold(3),n_jobs=2,error_score="raise")
                search.fit(x.iloc[tr],y[tr],groups=groups[tr])
                p=predictions(search.best_estimator_,x.iloc[te],task);oof[te]=p
                fold_rows.append({"task":task,"model":name,"outer_fold":fold,
                    "training_N":",".join(map(str,sorted(set(groups[tr])))),
                    "held_out_N":",".join(map(str,sorted(set(groups[te])))),
                    "chosen_parameters":json.dumps(search.best_params_,sort_keys=True),
                    **metrics(y[te],p,task)})
            score=metrics(y,oof,task)
            selection_loss=score["MAE"] if task=="regression" else (score["Brier"] if task=="binary" else score["log_loss"])
            task_scores[name]=selection_loss
            summary.append({"task":task,"model":name,"stage":"grouped_nested_oof",**score})
            predframes.append(prediction_frame(train,oof,task,name,"grouped_nested_oof"))
            final=GridSearchCV(clone(pipe),grid,scoring=(multiclass_scorer if task=="multiclass" else SCORING[task]),cv=GroupKFold(3),n_jobs=2,error_score="raise")
            final.fit(x,y,groups=groups)
            joblib.dump(final.best_estimator_,HERE/"models"/(task+"_"+name+".joblib"))
            pv=predictions(final.best_estimator_,xval,task)
            summary.append({"task":task,"model":name,"stage":"independent_simulation_validation",**metrics(yv,pv,task)})
            predframes.append(prediction_frame(valid,pv,task,name,"independent_simulation_validation"))
            print(f"{task} {name}: nested selection loss {selection_loss:.4f}; {time.time()-start:.1f}s",flush=True)
        selected=min(task_scores,key=task_scores.get)
        selection[task]={"model":selected,"criterion":SCORING[task],"selection_source":"pooled grouped outer-CV predictions; no validation-based selection"}
        model=joblib.load(HERE/"models"/(task+"_"+selected+".joblib"))
        pv=predictions(model,xval,task)
        subsets={"all":np.ones(len(valid),dtype=bool),"N_interpolation":valid.N<1000,
            "N_extrapolation_1200":valid.N==1200,"new_block_size_6":valid.block_size==6,
            "balanced_profile":valid.profile=="balanced","unseen_covariate_profiles":valid.profile!="balanced"}
        if task=="binary": subsets["labels_clear_at_95pct_MC_interval"]=~valid.binary_label_uncertain.astype(bool)
        validation_report[task]={}
        for label,mask in subsets.items():
            mask=np.asarray(mask)
            validation_report[task][label]={"scenarios":int(mask.sum()),**metrics(yv[mask],pv[mask],task)}
        if task=="regression":
            imp=permutation_importance(model,xval,yv,scoring="neg_mean_absolute_error",n_repeats=5,random_state=42,n_jobs=2)
            pd.DataFrame({"feature":FEATURES,"mean_MAE_increase":imp.importances_mean,
                          "sd_MAE_increase":imp.importances_std}).sort_values("mean_MAE_increase",ascending=False).to_csv(HERE/"results/PlanningFeature_PermutationImportance.csv",index=False)
    # A fixed five-neighbour planning-input baseline, using exactly the same outer splits.
    knn=Pipeline([("features",preprocessing()),("model",KNeighborsRegressor(n_neighbors=5,weights="distance"))])
    yo=train.imbalance_probability.to_numpy();po=np.zeros(len(train))
    for tr,te in all_outer:
        knn.fit(x.iloc[tr],yo[tr]);po[te]=np.clip(knn.predict(x.iloc[te]),0,1)
    knn.fit(x,yo);pv=np.clip(knn.predict(xval),0,1)
    joblib.dump(knn,HERE/"models/risk_KNN_baseline.joblib")
    summary.extend([{"task":"regression","model":"KNN_baseline","stage":"grouped_nested_oof",**metrics(yo,po,"regression")},
                    {"task":"regression","model":"KNN_baseline","stage":"independent_simulation_validation",**metrics(valid.imbalance_probability,pv,"regression")}])
    predframes.append(prediction_frame(valid,pv,"regression","KNN_baseline","independent_simulation_validation"))
    analytic=valid[["scenario_id","method","N","block_size","imbalance_probability"]].copy()
    analytic["exact_prediction"]=[exact_risk(r.method,int(r.N),int(r.block_size) or 4,.10) for r in analytic.itertuples()]
    covered=analytic.exact_prediction.notna()
    exact_metrics={"scenarios":int(covered.sum()),"coverage_fraction":float(covered.mean()),
         **metrics(analytic.loc[covered,"imbalance_probability"],analytic.loc[covered,"exact_prediction"],"regression")}
    sel=joblib.load(HERE/"models"/("regression_"+selection["regression"]["model"]+".joblib"))
    analytic["ML_prediction"]=predictions(sel,xval,"regression")
    exact_metrics["ML_same_subset"]=metrics(analytic.loc[covered,"imbalance_probability"],analytic.loc[covered,"ML_prediction"],"regression")
    analytic.to_csv(HERE/"results/ExactBaseline_Validation.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(HERE/"results/GroupedNestedCV_PerFold.csv",index=False)
    pd.DataFrame(summary).to_csv(HERE/"results/ModelPerformance.csv",index=False)
    pd.concat(predframes,ignore_index=True).to_csv(HERE/"results/ModelPredictions.csv",index=False)
    report={"model_source_sha256":hashlib.sha256((HERE/"fit_models.py").read_bytes()).hexdigest(),"training_csv_sha256":hashlib.sha256((HERE/"results/training_scenarios.csv").read_bytes()).hexdigest(),"validation_csv_sha256":hashlib.sha256((HERE/"results/validation_scenarios.csv").read_bytes()).hexdigest(),"feature_columns":FEATURES,"selected_models":selection,"validation":validation_report,
        "exact_probability_baseline":exact_metrics,"training_scenarios":len(train),"validation_scenarios":len(valid),
        "training_N":sorted(set(map(int,train.N))),"validation_N":sorted(set(map(int,valid.N))),
        "training_ambiguous_binary_labels":int(train.binary_label_uncertain.sum()),
        "validation_ambiguous_binary_labels":int(valid.binary_label_uncertain.sum()),
        "risk_definition":"P(abs(N_A-N_B)/N > 0.10)","high_risk_definition":"estimated risk > 0.30",
        "severity_definition":"low <=0.10; moderate >0.10 to <=0.50; high >0.50",
        "uncertainty_note":"Model point predictions are surrogates, without prediction intervals. Simulation validation is not clinical validation. Scenario metrics weight cells equally.",
        "elapsed_seconds":time.time()-start}
    (HERE/"results/model_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("Model fitting and frozen validation complete.",flush=True)

if __name__=="__main__": run()
