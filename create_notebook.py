"""Create and execute the reproducible study companion."""
from pathlib import Path
import json,sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from math_typesetting import FORMULAS, math_markup
HERE=Path(__file__).absolute().parent
def build(execute=True):
    path=HERE/"Randomization_Revised.ipynb"
    nb=nbf.read(path,as_version=4) if path.exists() else nbf.v4.new_notebook()
    M=lambda text: nbf.v4.new_markdown_cell(math_markup(text))
    C=nbf.v4.new_code_cell
    nb.cells=[
    M("# Experiment: Randomisation design comparison and prospective risk\n\nReproducible computational study companion. The question is how stopping rules, covariate distributions, balance, predictability and planned analysis interact. Predictions use planning inputs only. Independent simulation validation is distinct from clinical validation."),
    M("## 1. Configuration and reproducibility\n\nStart in the repository root. The default reviews saved outputs. Set RUN_SIMULATIONS=True to regenerate the simulation grid, models and figures. Scenario seeds, raw allocation counts and source fingerprints support numerical checks. Full reproduction commands are in README.md."),
    C("from pathlib import Path\nimport json, sys, subprocess\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nassert (ROOT / 'randomization_core.py').exists()\nRUN_SIMULATIONS = False\nif RUN_SIMULATIONS:\n    for script in ['run_simulations.py', 'fit_models.py', 'make_figures.py']:\n        subprocess.run([sys.executable, '-B', str(ROOT / script)], cwd=ROOT, check=True)\nconfig = json.loads((ROOT / 'project_config.json').read_text())\nprint('Master seed:', config['master_seed'])\nprint('Python:', sys.version.split()[0])"),
    C("from run_simulations import source_fingerprint\ncomponents=[]\nfor phase in ['training','validation','context','inference']:\n    manifest=json.loads((ROOT/'results'/f'{phase}_manifest.json').read_text())\n    assert manifest['source_sha256']==source_fingerprint()\n    components.append({'component':phase,'cells':manifest['scenario_count'],'trials':manifest['simulated_trials']})\ncomponents=pd.DataFrame(components)\ndisplay(components)\nassert components.trials.sum()==6_660_000\nprint('Distinct simulated trials:', f'{components.trials.sum():,}')"),
    M("## 2. Scientific allocation checks\n\nTests assess incomplete blocks, Monte Carlo agreement with exact risks, posterior recurrence, count conservation, big-stick bounds, adaptive null symmetry, Wilson intervals and covariate correlation."),
    C("import io, unittest\nimport test_methods\nstream=io.StringIO()\nsuite=unittest.defaultTestLoader.loadTestsFromModule(test_methods)\nresult=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)\nprint(stream.getvalue())\nassert result.wasSuccessful()"),
    M("## 3. Allocation risk and stopping rules\n\nPrimary risk is P(|N_A-N_B|/N>0.10), using a strict inequality. Conventional blocks retain an incomplete final prefix. The capped policy is a different design. Zero observed events retain a positive Wilson upper bound."),
    C("primary=pd.read_csv(ROOT/'results/Table1_PrimarySimulation.csv')\ncols=['method','risk_0.10','ci_low_0.10','ci_high_0.10','mean_correct_guess','mean_max_standardized_covariate_difference']\ndisplay(primary[(primary.N==30)&~primary.method.isin(['thompson','bayesian'])][cols].round(4))\ncaps=pd.read_csv(ROOT/'results/Table2_TerminalBlockSensitivity.csv')\ndisplay(caps[caps.N==10].round(6))"),
    C("display(Image(filename=str(ROOT/'figures/Figure1_ImbalanceRisk.png'),width=950))\ndisplay(Image(filename=str(ROOT/'figures/Figure2_BlocksAndTradeoffs.png'),width=950))"),
    M("## 4. Conditional stratification limits\n\nThe screen requires a pointwise 95% Wilson upper bound at most 0.10 for the chosen difference threshold. Five binary factors generate 32 possible joint strata. Empty strata are valid. The screen concerns allocation counts, rather than clinical safety."),
    C("limits=pd.read_csv(ROOT/'results/Table4_StratificationLimits.csv')\nz=limits[(limits.profile=='balanced')&(limits.block_size==4)&limits.N.isin([20,30,50,100,300])&limits.allocation_difference_threshold.isin([.02,.10])]\ndisplay(z.pivot(index='N',columns='allocation_difference_threshold',values='max_supported_factors'))\ndisplay(Image(filename=str(ROOT/'figures/Figure3_StratificationRisk.png'),width=950))"),
    M("## 5. Prospective prediction and independent simulation validation\n\nAll cells at a sample size stay in the same outer fold. Inner tuning is grouped and preprocessing occurs inside each pipeline. Validation uses unseen sample sizes and fresh streams. Family selection uses outer-CV scores. Simpler baselines and unfavourable results are retained."),
    C("report=json.loads((ROOT/'results/model_report.json').read_text())\nperformance=pd.read_csv(ROOT/'results/ModelPerformance.csv')\ndisplay(performance.round(4).fillna(''))\nprint('Selected candidate families:', {k:v['model'] for k,v in report['selected_models'].items()})\nprint('Exact comparison:', report['exact_probability_baseline'])\nprint('N=1200 regression:', report['validation']['regression']['N_extrapolation_1200'])"),
    C("display(Image(filename=str(ROOT/'figures/Figure4_IndependentValidation.png'),width=950))"),
    M("The selected random forest does not establish an advantage over exact probabilities or the five-neighbour baseline. At N=1200, nearly constant risks yield strongly negative $R^2$ despite small absolute error. All binary labels there are low risk, making AUC undefined. Aggregate accuracy alone is insufficient."),
    M("## 6. Outcome analysis\n\nThe Gaussian substudy compares treatment-only OLS with OLS adjusted for two covariates at N=30,100,250. Inferential rates are conditional on estimable fits. Response-adaptive inference remains outside this substudy."),
    C("inference=pd.read_csv(ROOT/'results/inference_substudy.csv')\ndisplay(inference[(inference.N==100)&(inference.profile=='balanced')][['method','analysis','null_rejection_probability','power_effect_0.4','coverage_95','valid_trials','nonestimable_probability']].round(4))\ndisplay(Image(filename=str(ROOT/'figures/Figure5_Inference.png'),width=950))"),
    M("## 7. Planning comparison tool\n\nCompare seven balance-seeking procedures for one supported scenario. Exact risk takes priority. Simulation adds uncertainty, covariate balance and full-information predictability. Pareto flags use point estimates and do not test ranking uncertainty."),
    C("from recommend import compare\nexample=compare(n=30,n_factors=2,profile='balanced',block_size=4,iterations=5000)\ndisplay(example[['method','risk_for_planning','ci_low','ci_high','correct_guess','covariate_difference','planning_risk_source']].round(4))"),
    M("## 8. Novelty, knowledge gap and limitations\n\nComparative simulation, minimisation, stratification assessment and tools such as randomizeR already exist. The contribution is a reproducible design-specific comparison exposing terminal-block effects, covariate and predictability trade-offs, one planned outcome analysis, and prospective-surrogate performance against simpler alternatives. The practical gap is how those dimensions jointly inform a specified planning decision. No new allocation algorithm, first-ever comparison or clinical validation is claimed.\n\nFurther work should examine richer covariates, independent implementations, delayed adaptive responses and endpoint-specific inference. Algorithm implementations are in randomization_core.py; all experiment settings are in project_config.json."),
    M("## 9. Numerical consistency\n\nRecompute all risks and Wilson intervals from raw counts and align saved predictions with fitted pipelines. After execution, validate_project.py --notebook checks saved results, model predictions and notebook execution."),
    C("from validate_project import check\nvalidation=check(notebook=False,write=False)\nprint(json.dumps(validation,indent=2))\nassert validation['status']=='passed'")
    ]
    # Render the canonical scientific definitions as portable LaTeX.
    nb.cells.insert(7, nbf.v4.new_markdown_cell(
        "The primary estimand and covariate diagnostic are\n\n"
        "$$\n"+FORMULAS["risk"]+"\n$$\n\n"
        "$$\n"+FORMULAS["covariate"]+"\n$$\n\n"
        "The estimated risk's Monte Carlo standard error is\n\n"
        "$$\n"+FORMULAS["mcse"]+"\n$$\n\n"
        "For response-adaptive allocation, the normalized tempered rule is\n\n"
        "$$\n"+FORMULAS["tempered"]+"\n$$"))
    outcome_idx=next(i for i,c in enumerate(nb.cells) if c.cell_type=="markdown" and c.source.startswith("## 6. Outcome analysis"))
    nb.cells.insert(outcome_idx+1,nbf.v4.new_markdown_cell(
        "The outcome model is\n\n$$\n"+FORMULAS["gaussian"]+"\n$$"))
    nb.metadata.kernelspec={"display_name":"Python 3 (recorded environment)","language":"python","name":"python3"}
    nb.metadata.language_info={"name":"python","version":sys.version.split()[0]}
    nbf.write(nb,path)
    if execute:
        km=KernelManager(kernel_name="python3")
        km.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
        client=NotebookClient(nb,timeout=600,kernel_name="python3",km=km,resources={"metadata":{"path":str(HERE)}})
        try:
            client.execute()
            nbf.write(nb,path)
        finally:
            if km.has_kernel:km.shutdown_kernel(now=True)
    print("Notebook created and executed." if execute else "Notebook created.")
if __name__=="__main__":build()
