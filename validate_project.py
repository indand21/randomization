"""Cross-check saved scientific outputs, fitted pipelines and deliverables."""
from __future__ import annotations
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import nbformat
from randomization_core import wilson_interval,stable_seed
from run_simulations import source_fingerprint
from fit_models import FEATURES,predictions
HERE=Path(__file__).absolute().parent

def check(notebook=False,write=True):
    checks=[]
    def require(condition,label):
        if not bool(condition):raise AssertionError(label)
        checks.append(label)
    train=pd.read_csv(HERE/"results/training_scenarios.csv")
    valid=pd.read_csv(HERE/"results/validation_scenarios.csv")
    context=pd.read_csv(HERE/"results/adaptive_and_capping_scenarios.csv")
    inference=pd.read_csv(HERE/"results/inference_substudy.csv")
    require(len(train)==2250 and len(valid)==1560 and len(context)==75 and len(inference)==126,"Expected design and analysis row counts")
    require(set(train.N).isdisjoint(set(valid.N)),"Independent validation uses wholly unseen sample sizes")
    require(set(train.scenario_id).isdisjoint(set(valid.scenario_id)),"Training and validation scenario identifiers are disjoint")
    require(int(train.iterations.sum()+valid.iterations.sum()+context.iterations.sum()+315000)==6660000,"Distinct simulated trial total is 6,660,000")
    seeds=[]
    for phase,df in [("training",train),("validation",valid),("context",context)]:
        require(df.scenario_id.is_unique,phase+" scenario IDs are unique")
        manifest=json.loads((HERE/"results"/(phase+"_manifest.json")).read_text())
        require(manifest["source_sha256"]==source_fingerprint(),phase+" source fingerprint matches current simulation code")
        archive=HERE/"raw"/(phase+"_allocation_counts.npz" if phase!="context" else "context_allocation_counts.npz")
        with np.load(archive) as raw:
            require(set(raw.files)==set(df.scenario_id),phase+" raw count archive covers every scenario")
            for row in df.itertuples():
                a=raw[row.scenario_id];n=int(row.N);k=int(row.iterations)
                require(len(a)==k and np.all(a<=n),phase+" valid allocation counts "+row.scenario_id)
                if phase!="context":
                    seeds.append(stable_seed(42,phase,"allocation",row.method,n,int(row.n_factors),row.profile,int(row.block_size),k))
                for delta in [.02,.05,.10,.15,.20]:
                    tag=f"{delta:.2f}";events=int(np.sum(abs(2*a.astype(int)-n)/n>delta))
                    stored=df.loc[df.scenario_id==row.scenario_id].iloc[0]
                    require(events==stored["events_"+tag] and abs(events/k-stored["risk_"+tag])<1e-12,"Recomputed risk "+row.scenario_id+" "+tag)
                    lo,hi=wilson_interval(events,k)
                    require(abs(lo-stored["ci_low_"+tag])<1e-12 and abs(hi-stored["ci_high_"+tag])<1e-12,"Recomputed Wilson interval "+row.scenario_id+" "+tag)
    require(len(seeds)==len(set(seeds)),"No allocation-seed collision across training and validation")
    analytic=pd.read_csv(HERE/"results/AnalyticalChecks.csv")
    require(len(analytic)==75 and analytic.within_six_mc_standard_errors.all(),"All 75 analytical checks pass")
    folds=pd.read_csv(HERE/"results/GroupedCV_FoldAssignments.csv")
    require(len(folds)==len(train) and folds.scenario_id.is_unique,"Each training cell receives exactly one outer assessment fold")
    require(folds.groupby("N").outer_fold.nunique().max()==1,"No sample size appears in multiple outer assessment folds")
    require(folds.outer_fold.nunique()==5,"Five grouped outer folds recorded")
    require(not any(x in FEATURES for x in ["imbalance_probability","mean_log_ratio","sd_log_ratio","high_risk","severity","mean_correct_guess"]),"Predictor schema excludes outcomes and post-allocation summaries")
    report=json.loads((HERE/"results/model_report.json").read_text())
    require(report["feature_columns"]==FEATURES,"Saved predictor schema matches model code")
    require(report["model_source_sha256"]==hashlib.sha256((HERE/"fit_models.py").read_bytes()).hexdigest(),"Model source fingerprint matches fitted analysis")
    for phase in ["training","validation"]:
        require(report[phase+"_csv_sha256"]==hashlib.sha256((HERE/"results"/(phase+"_scenarios.csv")).read_bytes()).hexdigest(),"Fitted model input fingerprint "+phase)
    pred=pd.read_csv(HERE/"results/ModelPredictions.csv")
    for task in ["regression","binary","multiclass"]:
        for name in ["Ridge","RandomForest","GradientBoosting"] if task=="regression" else ["LogisticRegression","RandomForest","GradientBoosting"]:
            stored=pred[(pred.task==task)&(pred.model==name)&(pred.stage=="independent_simulation_validation")].set_index("scenario_id").reindex(valid.scenario_id)
            model=joblib.load(HERE/"models"/(task+"_"+name+".joblib"))
            actual=predictions(model,valid[FEATURES],task)
            col="prediction" if task=="regression" else ["probability_"+str(i) for i in range(actual.shape[1])]
            expected=stored[col].to_numpy()
            require(np.allclose(actual,expected,atol=1e-12),"Saved "+task+" "+name+" predictions match fitted pipeline and validation rows")
            require(np.allclose(stored.imbalance_probability,valid.imbalance_probability),"Target alignment "+task+" "+name)
    require(inference.valid_trials.between(1,5000).all(),"Inference denominators are recorded and nonzero")
    require(np.allclose(inference.coverage_95+inference.null_rejection_probability,1),"Null coverage and rejection probabilities agree")
    require(np.allclose(inference.nonestimable_probability,1-inference.valid_trials/5000),"Nonestimable-fit reporting agrees with denominators")
    for _,row in inference.iterrows():
        for metric in ["null_rejection_probability","power_effect_0.4","coverage_95"]:
            require(0<=row[metric]<=1,"Inference probability bounds")
    for name in ["Figure1_ImbalanceRisk","Figure2_BlocksAndTradeoffs","Figure3_StratificationRisk","Figure4_IndependentValidation","Figure5_Inference"]:
        require((HERE/"figures"/(name+".png")).stat().st_size>20000,"Figure PNG present "+name)
        require((HERE/"figures"/(name+".pdf")).stat().st_size>10000,"Vector figure present "+name)
    if notebook:
        nb=nbformat.read(HERE/"Randomization_Revised.ipynb",as_version=4)
        cells=[c for c in nb.cells if c.cell_type=="code"]
        require(all(c.execution_count is not None for c in cells),"Notebook executed top to bottom")
        require(not any(o.output_type=="error" for c in cells for o in c.get("outputs",[])),"Notebook contains no execution errors")
    # Compact report: the per-cell checks are counted, not dumped into a huge log.
    hashes={}
    for pattern in ["*.py","*.ipynb","README.md",".gitignore",".gitattributes","requirements.txt","project_config.json","environment.json","results/*.csv","results/*.json","raw/*.npz","models/*.joblib","figures/*.png","figures/*.pdf"]:
        for path in HERE.glob(pattern):
            if path.name in ["validation_report.json","artifact_hashes.json"]:continue
            hashes[str(path.relative_to(HERE)).replace("\\","/")]=hashlib.sha256(path.read_bytes()).hexdigest()
    output={"status":"passed","checks_passed":len(checks),"simulated_trials":6660000,
        "scenario_cells":3948,"analytic_checks":75,"notebook_checked":notebook,
        "python":platform.python_version(),"simulation_source_sha256":source_fingerprint()}
    if write:
        (HERE/"qa").mkdir(exist_ok=True)
        (HERE/"qa/validation_report.json").write_text(json.dumps(output,indent=2))
        (HERE/"qa/artifact_hashes.json").write_text(json.dumps(hashes,indent=2))
    return output

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--notebook",action="store_true")
    args=ap.parse_args();print(json.dumps(check(notebook=args.notebook),indent=2))
