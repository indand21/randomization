"""Compare specified planning scenarios; no universal method ranking."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import joblib
from randomization_core import (BALANCE_METHODS,METHOD_IDS,BLOCK_METHODS,stable_seed,
    make_covariates,simulate_allocations,design_features,exact_risk,wilson_interval)
from fit_models import FEATURES
HERE=Path(__file__).absolute().parent

def compare(n=100,n_factors=2,profile="balanced",block_size=4,iterations=5000,mode="simulate",seed=20261012):
    if not 10<=n<=1200: raise ValueError("This prototype supports N=10..1200. Extend and validate the study before wider use.")
    if n_factors not in range(1,6): raise ValueError("Use 1..5 binary covariates.")
    if block_size not in [4,6,8]: raise ValueError("Supported block bases: 4,6,8.")
    if mode=="simulate" and iterations<1000: raise ValueError("Use at least 1000 repetitions.")
    rows=[]
    if mode=="simulate":
        cov,p,_=make_covariates(n,n_factors,profile,iterations,stable_seed(seed,"tool","covariates",n,n_factors,profile))
    else:
        report=json.loads((HERE/"results/model_report.json").read_text())
        model=joblib.load(HERE/"models/risk_KNN_baseline.joblib")
    for method in BALANCE_METHODS:
        b=block_size if method in BLOCK_METHODS else 0
        exact=exact_risk(method,n,block_size,.10)
        row={"method":method,"N":n,"n_factors":n_factors,"profile":profile,"block_size":b}
        if mode=="simulate":
            raw,_=simulate_allocations(cov,p,METHOD_IDS[method],block_size,
                stable_seed(seed,"tool","allocation",method,n,n_factors,profile,block_size))
            count=int(np.sum(abs(2*raw[:,0]-n)/n>.10));lo,hi=wilson_interval(count,iterations)
            row.update(estimated_risk=count/iterations,ci_low=float(lo),ci_high=float(hi),
                correct_guess=float(raw[:,1].mean()),covariate_difference=float(np.nanmean(raw[:,3])),
                source="new Monte Carlo simulation",iterations=iterations,
                below_risk_0_10_by_upper_bound=bool((exact if exact is not None else hi)<=.10))
        else:
            risk=exact if exact is not None else float(np.clip(model.predict(pd.DataFrame([design_features(method,n,n_factors,profile,b)])[FEATURES])[0],0,1))
            row.update(estimated_risk=risk,ci_low=exact if exact is not None else np.nan,
                ci_high=exact if exact is not None else np.nan,correct_guess=np.nan,covariate_difference=np.nan,
                source="exact allocation-count probability" if exact is not None else "five-neighbour surrogate without a prediction interval",
                iterations=0,below_risk_0_10_by_upper_bound=(risk<=.10 if exact is not None else None))
        row["exact_risk"]=exact
        row["risk_for_planning"]=exact if exact is not None else row["estimated_risk"]
        row["planning_risk_source"]="exact" if exact is not None else row["source"]
        rows.append(row)
    out=pd.DataFrame(rows)
    if mode=="simulate":
        cols=["risk_for_planning","correct_guess","covariate_difference"]
        out["pareto_candidate_point_estimates"]=[not ((out[cols].le(r[cols]).all(axis=1))&(out[cols].lt(r[cols]).any(axis=1))).any() for _,r in out.iterrows()]
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N",type=int,default=100);ap.add_argument("--factors",type=int,default=2)
    ap.add_argument("--profile",choices=["balanced","skewed","correlated","moderate_skew","high_correlation","rare"],default="balanced")
    ap.add_argument("--block-size",type=int,default=4);ap.add_argument("--iterations",type=int,default=5000)
    ap.add_argument("--mode",choices=["simulate","fast"],default="simulate")
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    df=compare(args.N,args.factors,args.profile,args.block_size,args.iterations,args.mode)
    print(df.to_string(index=False))
    print("\nRisk is Pr(|N_A-N_B|/N > 0.10). Exact values take priority when available.")
    print("Full-information guessing is an upper-bound diagnostic. Point-estimate Pareto comparisons do not test ranking uncertainty.")
    print("The 0.10 risk screen concerns allocation counts only; method selection also needs the planned outcome analysis, concealment and logistics.")
    if args.output:
        df.to_csv(args.output,index=False)
if __name__=="__main__":main()
