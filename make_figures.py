"""Publication figures generated only from corrected saved results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).absolute().parent
LABELS={"simple":"Simple","fixed_block":"Fixed block","varying_block":"Varying block",
    "minimization":"Minimisation","stratified":"Stratified block","biased_coin":"Biased coin",
    "big_stick":"Big stick","thompson":"Thompson","bayesian":"Tempered Bayesian","capped_varying_block":"Capped final block"}
COLORS=dict(zip(LABELS,["#707070","#2266AA","#E58626","#239466","#9B5BA5","#A35935","#17A1B2","#C43C39","#C69929","#333333"]))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
    "axes.spines.right":False,"axes.titlesize":12,"axes.labelsize":10,"legend.fontsize":8,
    "savefig.dpi":300,"figure.dpi":120,"mathtext.fontset":"stix"})
def save(fig,name):
    fig.savefig(HERE/"figures"/(name+".png"),bbox_inches="tight")
    fig.savefig(HERE/"figures"/(name+".pdf"),bbox_inches="tight")
    plt.close(fig)

def run():
    p=pd.read_csv(HERE/"results/Table1_PrimarySimulation.csv")
    fig,ax=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for i,methods in enumerate([list(LABELS)[:7],["thompson","bayesian"]]):
        for method in methods:
            g=p[p.method==method].sort_values("N")
            ax[i].plot(g.N,g["risk_0.10"],"o-",ms=3,color=COLORS[method],label=LABELS[method])
            ax[i].fill_between(g.N.to_numpy(),g["ci_low_0.10"].to_numpy(),g["ci_high_0.10"].to_numpy(),alpha=.13,color=COLORS[method])
        ax[i].set_xscale("log");ax[i].set_ylim(-.02,1.02);ax[i].set_xlabel(r"Total sample size $N$")
        ax[i].legend(loc="best",frameon=False)
        ax[i].grid(alpha=.15)
    ax[0].set_ylabel(r"$\Pr\!\left(\frac{|N_A-N_B|}{N}>0.10\right)$")
    ax[0].set_title("A  Balance-seeking procedures")
    ax[1].set_title("B  Response-adaptive procedures\nImmediate binary outcomes: $p_A=0.60$, $p_B=0.40$")
    fig.tight_layout();save(fig,"Figure1_ImbalanceRisk")

    caps=pd.read_csv(HERE/"results/Table2_TerminalBlockSensitivity.csv")
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for method in ["fixed_block","varying_block","capped_varying_block"]:
        g=caps[(caps.method==method)&(caps.N<=100)]
        ax[0].errorbar(g.N,g.estimated_risk,yerr=[g.estimated_risk-g.ci_low,g.ci_high-g.estimated_risk],fmt="o-",ms=4,capsize=2,color=COLORS[method],label=LABELS[method])
    ax[0].set_xlabel("$N$");ax[0].set_ylabel("Imbalance risk at threshold 0.10");ax[0].legend(frameon=False)
    ax[0].set_title("A  Stopping partway through a block")
    for method in list(LABELS)[:7]:
        g=p[(p.N==100)&(p.method==method)].iloc[0]
        ax[1].scatter(g.mean_max_standardized_covariate_difference,g.mean_correct_guess,color=COLORS[method],s=50)
        ax[1].annotate(LABELS[method],(g.mean_max_standardized_covariate_difference,g.mean_correct_guess),
            xytext=(5,5 if method not in ["varying_block","big_stick"] else -12),textcoords="offset points",fontsize=8)
    ax[1].set_xlim(0,.36);ax[1].set_ylim(.48,.77)
    ax[1].set_xlabel("Mean maximum population-standardized\ncovariate difference (smaller is better)")
    ax[1].set_ylabel("Full-information correct-guess probability")
    ax[1].set_title(r"B  Balance and predictability at $N=100$")
    fig.tight_layout();save(fig,"Figure2_BlocksAndTradeoffs")

    st=pd.read_csv(HERE/"results/Table3_StratificationThresholds.csv")
    fig,ax=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
    ns=[20,50,100,200,500,1000]
    for a,delta in zip(ax,[.02,.10]):
        g=st[(st.profile=="balanced")&(st.block_size==4)&(st.allocation_difference_threshold==delta)&st.N.isin(ns)]
        matrix=g.pivot(index="n_factors",columns="N",values="estimated_risk").reindex(index=range(1,6),columns=ns)
        im=a.imshow(matrix,aspect="auto",vmin=0,vmax=1,cmap="YlOrRd")
        for i,v in enumerate(range(1,6)):
            for j,n in enumerate(ns):
                r=g[(g.n_factors==v)&(g.N==n)].iloc[0]
                a.text(j,i,f"{r.estimated_risk:.2f}"+("*" if r.status=="meets_upper_bound" else ""),ha="center",va="center",fontsize=8,color="white" if r.estimated_risk>.6 else "black")
        a.set_xticks(range(len(ns)),ns);a.set_yticks(range(5),[f"{v} ({2**v})" for v in range(1,6)])
        a.set_xlabel(r"Total $N$");a.set_title(f"Allocation difference threshold $\\delta={delta:.2f}$")
    ax[0].set_ylabel("Binary factors (joint strata)")
    fig.colorbar(im,ax=ax.ravel().tolist(),label="Estimated imbalance risk",fraction=.025,pad=.03)
    fig.subplots_adjust(wspace=.15,left=.10,right=.88,bottom=.18,top=.85)
    save(fig,"Figure3_StratificationRisk")

    report=json.loads((HERE/"results/model_report.json").read_text())
    pred=pd.read_csv(HERE/"results/ModelPredictions.csv")
    g=pred[(pred.task=="regression")&(pred.model==report["selected_models"]["regression"]["model"])&(pred.stage=="independent_simulation_validation")].copy()
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for method in list(LABELS)[:7]:
        d=g[g.method==method]
        ax[0].scatter(d.imbalance_probability,d.prediction,s=12,alpha=.4,color=COLORS[method],label=LABELS[method])
    ax[0].plot([0,1],[0,1],"k--",lw=1);ax[0].set_xlim(-.02,1.02);ax[0].set_ylim(-.02,1.02)
    ax[0].set_xlabel("Independent simulated imbalance risk");ax[0].set_ylabel("Frozen model prediction")
    ax[0].legend(frameon=False,loc="upper left");ax[0].set_title("A  Planning-only random forest")
    g["absolute_error"]=abs(g.prediction-g.imbalance_probability)
    for method in list(LABELS)[:7]:
        z=g[g.method==method].groupby("N").absolute_error.mean()
        ax[1].plot(z.index,z,"o-",ms=3,color=COLORS[method],label=LABELS[method])
    ax[1].set_xscale("log");ax[1].set_xlabel(r"Previously unseen $N$");ax[1].set_ylabel("Mean absolute risk prediction error")
    ax[1].set_title("B  Error by sample size and method")
    fig.tight_layout();save(fig,"Figure4_IndependentValidation")

    inf=pd.read_csv(HERE/"results/inference_substudy.csv")
    g=inf[(inf.profile=="balanced")&(inf.N==100)]
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    order=list(LABELS)[:7]
    for analysis,offset,color,label in [("unadjusted",-.12,"#888888","Unadjusted"),("covariate_adjusted",.12,"#2266AA","Covariate adjusted")]:
        z=g[g.analysis==analysis].set_index("method").reindex(order)
        x=np.arange(7)+offset
        for a,y,lo,hi in [(ax[0],"null_rejection_probability","null_rejection_ci_low","null_rejection_ci_high"),(ax[1],"power_effect_0.4","power_ci_low","power_ci_high")]:
            a.errorbar(x,z[y],yerr=[z[y]-z[lo],z[hi]-z[y]],fmt="o",capsize=2,color=color,label=label)
    ax[0].axhline(.05,color="black",ls="--",lw=1);ax[0].set_ylim(0,.09)
    ax[0].set_ylabel("Null rejection probability");ax[0].set_title("A  Type I error")
    ax[1].set_ylim(0,1);ax[1].set_ylabel(r"Power for treatment effect $\tau=0.4$");ax[1].set_title("B  Power")
    for a in ax:
        a.set_xticks(range(7),[LABELS[m] for m in order],rotation=35,ha="right")
        a.legend(frameon=False)
    fig.tight_layout();save(fig,"Figure5_Inference")

    manifest={"Figure1_ImbalanceRisk":["Table1_PrimarySimulation.csv"],
        "Figure2_BlocksAndTradeoffs":["Table2_TerminalBlockSensitivity.csv","Table1_PrimarySimulation.csv"],
        "Figure3_StratificationRisk":["Table3_StratificationThresholds.csv"],
        "Figure4_IndependentValidation":["ModelPredictions.csv","model_report.json"],
        "Figure5_Inference":["inference_substudy.csv"]}
    (HERE/"results/figure_sources.json").write_text(json.dumps(manifest,indent=2))
    print("Five source-backed figures saved in PNG and PDF.")
if __name__=="__main__":run()
