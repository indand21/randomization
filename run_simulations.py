"""Reproducible Monte Carlo grid and inference substudy. No original files are changed."""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from randomization_core import (BALANCE_METHODS, BLOCK_METHODS, METHOD_IDS,
    stable_seed, make_covariates, simulate_allocations, wilson_interval,
    design_features, exact_risk)

HERE = Path(__file__).absolute().parent
RESULTS = HERE/"results"
RAW = HERE/"raw"
TRAIN_N = [10,20,30,40,50,75,100,150,200,250,300,400,500,750,1000]
VALIDATION_N = [15,60,125,350,650,1200]
THRESHOLDS = [.02,.05,.10,.15,.20]
CONFIG = {"revision":"2026-10-12","master_seed":42,"train_N":TRAIN_N,
    "validation_N":VALIDATION_N,"n_factors":[1,2,3,4,5],
    "training_profiles":["balanced","skewed","correlated"],
    "validation_profiles":["balanced","moderate_skew","high_correlation","rare"],
    "training_block_sizes":[4,8],"validation_block_sizes":[4,6,8],
    "training_iterations":1000,"primary_iterations":5000,
    "validation_iterations":2000,"inference_iterations":5000,
    "imbalance_thresholds":THRESHOLDS,"binary_risk_threshold":.30,
    "stratification_max_risk":.10,"inference_N":[30,100,250],
    "inference_effect":.4,"inference_profiles":["balanced","skewed","correlated"],
    "inference_n_factors":2,"inference_block_size":4,
    "validation_description":"independent simulation with unseen sample sizes and unseen joint covariate and block settings; no clinical external validation"}
def source_fingerprint():
    h=hashlib.sha256()
    for name in ["randomization_core.py","run_simulations.py"]:
        h.update((HERE/name).read_bytes())
    h.update(json.dumps(CONFIG,sort_keys=True).encode())
    return h.hexdigest()

def summary_row(raw,method,n,v,profile,b,phase,k,pa=.6,pb=.4):
    features=design_features(method,n,v,profile,b)
    sid=hashlib.sha256(json.dumps([phase,method,n,v,profile,b,k,pa,pb]).encode()).hexdigest()[:20]
    n_a=raw[:,0].astype(np.int64)
    differences=np.abs(2*n_a-n)
    log_ratio=np.log((n_a+.5)/(n-n_a+.5))
    covvalid=raw[:,3][np.isfinite(raw[:,3])]
    row={**features,"scenario_id":sid,"phase":phase,"profile":profile,
         "iterations":k,"pA":pa,"pB":pb,"mean_log_ratio":float(log_ratio.mean()),
         "sd_log_ratio":float(log_ratio.std(ddof=1)),
         "mean_correct_guess":float(raw[:,1].mean()),
         "mcse_correct_guess":float(raw[:,1].std(ddof=1)/np.sqrt(k)),
         "mean_deterministic_fraction":float(raw[:,2].mean()),
         "mean_max_standardized_covariate_difference":float(covvalid.mean()) if len(covvalid) else np.nan,
         "mcse_covariate_difference":float(covvalid.std(ddof=1)/np.sqrt(len(covvalid))) if len(covvalid)>1 else np.nan,
         "covariate_metric_valid_trials":len(covvalid),
         "mean_success_fraction":float(raw[:,4].mean()/n),
         "mean_allocation_to_A":float(n_a.mean()/n),
         "mean_empty_strata":float(raw[:,5].mean()),
         "mean_max_prefix_difference":float(raw[:,6].mean()),
         "empty_arm_probability":float(raw[:,7].mean())}
    for threshold in THRESHOLDS:
        tag=f"{threshold:.2f}"
        events=int(np.sum(differences/n>threshold))
        lo,hi=wilson_interval(events,k)
        p=events/k
        row.update({"events_"+tag:events,"risk_"+tag:p,
                    "ci_low_"+tag:float(lo),"ci_high_"+tag:float(hi),
                    "mcse_"+tag:float(np.sqrt(p*(1-p)/k))})
    row["imbalance_probability"]=row["risk_0.10"]
    row["high_risk"]=int(row["risk_0.10"]>.30)
    row["severity"]=0 if row["risk_0.10"]<=.10 else (1 if row["risk_0.10"]<=.50 else 2)
    row["binary_label_uncertain"]=bool(row["ci_low_0.10"]<=.30<=row["ci_high_0.10"])
    row["is_primary"]=bool(phase=="training" and v==2 and profile=="balanced" and b in (0,4))
    return row,n_a.astype(np.uint16)

def run_grid(phase,reuse=False):
    path=RESULTS/(phase+"_scenarios.csv")
    stamp=RESULTS/(phase+"_manifest.json")
    fingerprint=source_fingerprint()
    if reuse and path.exists() and stamp.exists():
        old=json.loads(stamp.read_text())
        if old.get("source_sha256")==fingerprint and (RAW/(phase+"_allocation_counts.npz")).exists():
            print("Reusing verified "+phase+" simulation",flush=True)
            return pd.read_csv(path)
    ns=TRAIN_N if phase=="training" else VALIDATION_N
    profiles=CONFIG["training_profiles"] if phase=="training" else CONFIG["validation_profiles"]
    blocks=CONFIG["training_block_sizes"] if phase=="training" else CONFIG["validation_block_sizes"]
    rows=[];counts={}
    start=time.time()
    total=len(ns)*5*len(profiles)*(4+3*len(blocks))
    for n in ns:
        for v in range(1,6):
            for profile in profiles:
                k=5000 if phase=="training" and v==2 and profile=="balanced" else (1000 if phase=="training" else 2000)
                cov,p,_=make_covariates(n,v,profile,k,stable_seed(42,phase,"covariates",n,v,profile,k))
                for method in BALANCE_METHODS:
                    for b in (blocks if method in BLOCK_METHODS else [0]):
                        raw,_=simulate_allocations(cov,p,METHOD_IDS[method],b or 4,
                            stable_seed(42,phase,"allocation",method,n,v,profile,b,k))
                        row,n_a=summary_row(raw,method,n,v,profile,b,phase,k)
                        rows.append(row);counts[row["scenario_id"]]=n_a
                        if len(rows)%50==0 or len(rows)==total:
                            print(f"{phase}: {len(rows)}/{total} scenarios; {time.time()-start:.1f}s",flush=True)
        # Checkpoint only summaries; the completed phase always includes raw counts.
        pd.DataFrame(rows).to_csv(RESULTS/(phase+"_checkpoint.csv"),index=False)
    df=pd.DataFrame(rows)
    df.to_csv(path,index=False)
    np.savez_compressed(RAW/(phase+"_allocation_counts.npz"),**counts)
    stamp.write_text(json.dumps({"source_sha256":fingerprint,"scenario_count":len(df),
        "simulated_trials":int(df.iterations.sum()),"elapsed_seconds":time.time()-start,
        "config":CONFIG},indent=2),encoding="utf-8")
    return df

def run_context(reuse=False):
    path=RESULTS/"adaptive_and_capping_scenarios.csv"
    stamp=RESULTS/"context_manifest.json"
    if reuse and path.exists() and stamp.exists() and json.loads(stamp.read_text()).get("source_sha256")==source_fingerprint():
        return pd.read_csv(path)
    rows=[];counts={}
    for n in TRAIN_N:
        k=5000;v=2;profile="balanced"
        cov,p,_=make_covariates(n,v,profile,k,stable_seed(42,"context","covariates",n))
        for method in ["thompson","bayesian","capped_varying_block"]:
            settings=[(.6,.4),(.5,.5)] if method in ("thompson","bayesian") else [(.6,.4)]
            for pa,pb in settings:
                raw,_=simulate_allocations(cov,p,METHOD_IDS[method],4,
                    stable_seed(42,"context",method,n,pa,pb),pa,pb)
                row,n_a=summary_row(raw,method,n,v,profile,4 if method in BLOCK_METHODS else 0,"context",k,pa,pb)
                rows.append(row);counts[row["scenario_id"]]=n_a
        print(f"context: N={n} complete",flush=True)
    df=pd.DataFrame(rows);df.to_csv(path,index=False)
    np.savez_compressed(RAW/"context_allocation_counts.npz",**counts)
    stamp.write_text(json.dumps({"source_sha256":source_fingerprint(),"scenario_count":len(df),
        "simulated_trials":int(df.iterations.sum())},indent=2),encoding="utf-8")
    return df

def inference_statistics(cov,allocation,y,n,adjusted):
    k,_,v=cov.shape
    parts=[np.ones((k,n,1)),allocation[:,:,None].astype(float)]
    if adjusted: parts.append(cov.astype(float))
    x=np.concatenate(parts,axis=2)
    xtx=np.einsum("rni,rnj->rij",x,x,optimize=True)
    xty=np.einsum("rni,rn->ri",x,y,optimize=True)
    ranks=np.linalg.matrix_rank(xtx,tol=1e-9)
    p=x.shape[2]
    inv=np.linalg.pinv(xtx,rcond=1e-12)
    beta=np.einsum("rij,rj->ri",inv,xty)
    sse=np.sum(y*y,axis=1)-np.sum(beta*xty,axis=1)
    se=np.sqrt(np.maximum(sse,0.)/(n-p)*np.maximum(inv[:,1,1],0.))
    counts=allocation.sum(axis=1)
    valid=(ranks==p)&(counts>=2)&(counts<=n-2)&np.isfinite(se)&(se>1e-12)
    estimate=beta[:,1][valid];se=se[valid]
    critical=student_t.ppf(.975,n-p)
    reject_null=np.abs(estimate/se)>critical
    reject_alt=np.abs((estimate+.4)/se)>critical
    coverage=np.abs(estimate)<=critical*se
    row={"analysis":"covariate_adjusted" if adjusted else "unadjusted",
         "valid_trials":int(valid.sum()),"nonestimable_probability":float(1-valid.mean()),
         "null_rejection_probability":float(reject_null.mean()),
         "power_effect_0.4":float(reject_alt.mean()),
         "coverage_95":float(coverage.mean()),"bias":float(estimate.mean()),
         "empirical_sd":float(estimate.std(ddof=1)),"mean_standard_error":float(se.mean())}
    for name,flag in [("null_rejection",reject_null),("power",reject_alt),("coverage",coverage)]:
        lo,hi=wilson_interval(int(flag.sum()),len(flag))
        row[name+"_ci_low"]=float(lo);row[name+"_ci_high"]=float(hi)
    return row,np.column_stack([estimate,se])

def run_inference(reuse=False):
    path=RESULTS/"inference_substudy.csv";stamp=RESULTS/"inference_manifest.json"
    if reuse and path.exists() and stamp.exists() and json.loads(stamp.read_text()).get("source_sha256")==source_fingerprint():
        return pd.read_csv(path)
    rows=[];raw_estimates={}
    start=time.time()
    for n in CONFIG["inference_N"]:
        for profile in CONFIG["inference_profiles"]:
            k=5000;v=2
            cov,p,_=make_covariates(n,v,profile,k,stable_seed(42,"inference","covariates",n,profile))
            rng=np.random.default_rng(stable_seed(42,"inference","outcomes",n,profile))
            y=np.sum((cov-p)*np.array([.8,.4]),axis=2)+rng.normal(size=(k,n))
            for method in BALANCE_METHODS:
                _,alloc=simulate_allocations(cov,p,METHOD_IDS[method],4,
                    stable_seed(42,"inference","allocation",n,profile,method),.6,.4,True)
                for adjusted in [False,True]:
                    row,est=inference_statistics(cov,alloc,y,n,adjusted)
                    row.update({"method":method,"N":n,"profile":profile,"n_factors":v,
                                "iterations":k,"effect_size":.4})
                    rows.append(row);raw_estimates[f"{method}_{n}_{profile}_{adjusted}"]=est
            print(f"inference: N={n}, {profile}; {time.time()-start:.1f}s",flush=True)
    df=pd.DataFrame(rows);df.to_csv(path,index=False)
    np.savez_compressed(RAW/"inference_estimates.npz",**raw_estimates)
    stamp.write_text(json.dumps({"source_sha256":source_fingerprint(),"scenario_count":63,
        "analyses":126,"simulated_trials":315000,"elapsed_seconds":time.time()-start},indent=2),encoding="utf-8")
    return df

def derived_tables(train,context):
    main=train[train.is_primary.astype(bool)].copy()
    context_main=context[(context.pA==.6)&(context.pB==.4)&context.method.isin(["thompson","bayesian"])]
    primary=pd.concat([main,context_main],ignore_index=True)
    primary.to_csv(RESULTS/"Table1_PrimarySimulation.csv",index=False)
    caps=[]
    for n in TRAIN_N:
        for method in ["fixed_block","varying_block","capped_varying_block"]:
            source=main if method!="capped_varying_block" else context
            row=source[(source.N==n)&(source.method==method)].iloc[0]
            exact=exact_risk(method,n,4,.10)
            caps.append({"N":n,"method":method,"estimated_risk":row["risk_0.10"],
                         "exact_risk":exact,"ci_low":row["ci_low_0.10"],"ci_high":row["ci_high_0.10"]})
    pd.DataFrame(caps).to_csv(RESULTS/"Table2_TerminalBlockSensitivity.csv",index=False)
    strata=[]
    for _,row in train[train.method=="stratified"].iterrows():
        for delta in THRESHOLDS:
            tag=f"{delta:.2f}"
            lo,hi=row["ci_low_"+tag],row["ci_high_"+tag]
            status="meets_upper_bound" if hi<=.10 else ("exceeds_lower_bound" if lo>.10 else "uncertain")
            strata.append({"scenario_id":row.scenario_id,"N":row.N,"n_factors":row.n_factors,
                "total_strata":row.total_strata,"profile":row.profile,"block_size":row.block_size,
                "allocation_difference_threshold":delta,"maximum_acceptable_risk":.10,
                "estimated_risk":row["risk_"+tag],"ci_low":lo,"ci_high":hi,"status":status,
                "mean_empty_strata":row.mean_empty_strata,"iterations":row.iterations})
    st=pd.DataFrame(strata);st.to_csv(RESULTS/"Table3_StratificationThresholds.csv",index=False)
    limits=[]
    for keys,g in st.groupby(["N","profile","block_size","allocation_difference_threshold"]):
        ok=g[g.status=="meets_upper_bound"]
        vmax=int(ok.n_factors.max()) if len(ok) else 0
        limits.append(dict(zip(["N","profile","block_size","allocation_difference_threshold"],keys),
            max_supported_factors=vmax,total_strata=2**vmax if vmax else 0,
            decision_rule="upper Wilson 95% bound <= 0.10; largest tested configuration"))
    pd.DataFrame(limits).to_csv(RESULTS/"Table4_StratificationLimits.csv",index=False)
    pareto=[]
    criteria=["risk_0.10","mean_correct_guess","mean_max_standardized_covariate_difference"]
    for n,g in main.groupby("N"):
        for _,row in g.iterrows():
            others=g[g.method!=row.method]
            dominates=(others[criteria].le(row[criteria]).all(axis=1)&others[criteria].lt(row[criteria]).any(axis=1)).any()
            pareto.append({"N":n,"method":row.method,"risk":row["risk_0.10"],
                "ci_low":row["ci_low_0.10"],"ci_high":row["ci_high_0.10"],
                "correct_guess":row.mean_correct_guess,
                "covariate_difference":row.mean_max_standardized_covariate_difference,
                "pareto_candidate_on_point_estimates":not bool(dominates),
                "interpretation":"descriptive trade-off; ranking uncertainty not tested"})
    pd.DataFrame(pareto).to_csv(RESULTS/"Table5_DesignTradeoffs.csv",index=False)
    exactchecks=[]
    for _,row in main.iterrows():
        exact=exact_risk(row.method,int(row.N),int(row.block_size) or 4,.10)
        if exact is not None:
            se=np.sqrt(exact*(1-exact)/row.iterations)
            exactchecks.append({"scenario_id":row.scenario_id,"N":row.N,"method":row.method,
                "simulated_risk":row["risk_0.10"],"exact_risk":exact,
                "absolute_difference":abs(row["risk_0.10"]-exact),
                "within_six_mc_standard_errors":abs(row["risk_0.10"]-exact)<=6*se+1/row.iterations})
    pd.DataFrame(exactchecks).to_csv(RESULTS/"AnalyticalChecks.csv",index=False)

def run(reuse=False,phases=None):
    RESULTS.mkdir(exist_ok=True);RAW.mkdir(exist_ok=True)
    (HERE/"project_config.json").write_text(json.dumps(CONFIG,indent=2),encoding="utf-8")
    phases=phases or ["training","validation","context","inference"]
    train=run_grid("training",reuse) if "training" in phases else pd.read_csv(RESULTS/"training_scenarios.csv")
    if "validation" in phases: run_grid("validation",reuse)
    context=run_context(reuse) if "context" in phases else pd.read_csv(RESULTS/"adaptive_and_capping_scenarios.csv")
    if "inference" in phases: run_inference(reuse)
    derived_tables(train,context)
    print("Simulation and derived tables complete.",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--reuse",action="store_true")
    args=ap.parse_args();run(args.reuse)
