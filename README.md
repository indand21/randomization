# Randomisation design comparison and prospective risk

A reproducible Python study comparing allocation balance, covariate balance, full-information predictability and one planned outcome analysis in simulated two-arm trials. The repository includes allocation methods, exact probability calculations, simulation outputs, fitted prediction pipelines and a bounded planning comparison tool. All data are synthetic; there are no patient records.

Start with the executed [study notebook](Randomization_Revised.ipynb), [configuration](project_config.json) and [validation report](qa/validation_report.json).

## Study design

The saved study contains **6,660,000 simulated trials across 3,948 distinct design cells**:

| Component | Design cells | Simulated trials |
| --- | ---: | ---: |
| Training | 2,250 | 2,850,000 |
| Independent simulation validation | 1,560 | 3,120,000 |
| Adaptive and terminal-block context | 75 | 375,000 |
| Gaussian inference substudy | 63 | 315,000 |

The inference substudy has 126 analysis rows because each allocation cell is assessed with treatment-only and covariate-adjusted OLS.

Seven balance-seeking procedures are compared: simple randomisation, fixed blocks, varying blocks, minimisation, stratified blocks, biased coin and big stick. Thompson and tempered Bayesian response-adaptive procedures are assessed separately. A capped-final-block policy is a sensitivity design. Conventional fixed and varying blocks retain the incomplete prefix of the final full block when recruitment stops.

The primary estimand is

$$
R_{\delta}=\Pr\left(\frac{|N_A-N_B|}{N}>\delta\right),\qquad \delta=0.10.
$$

The strict inequality matters. Risks are also evaluated at thresholds 0.02, 0.05, 0.15 and 0.20. Pointwise Wilson intervals and Monte Carlo standard errors accompany simulated risks. Zero observed events retain a positive upper uncertainty bound.

## Main results and contribution

At total sample size 10, conventional varying blocks from {2,4,6} have exact primary count-imbalance risk 0.190672. The capped last-block policy has zero risk at that threshold, reflecting a different stopping policy.

The selected random forest has independent simulation-validation MAE 0.01515 and R-squared 0.89892. The selected binary logistic model has AUC 0.98456 and balanced accuracy 0.85595. On the same 1,080 eligible validation cells, exact probabilities have MAE 0.00105 compared with random-forest MAE 0.01317. The five-neighbour baseline matches or slightly improves average surrogate error. The results therefore do not establish an advantage for a complex surrogate over simpler alternatives.

At sample size 1,200, risks are nearly constant: small absolute errors coexist with strongly negative R-squared. All high-risk labels there are negative, so AUC is undefined. Complete results, including unfavourable results, are retained in [model_report.json](results/model_report.json).

The applied contribution is the joint, reproducible evaluation of terminal-block policies, covariate distributions, count and covariate balance, predictability, a specified outcome analysis and prospective prediction against simpler baselines. It addresses how these interacting choices inform a particular planning decision. Comparative randomisation assessment already exists; this study does not claim a new allocation algorithm or clinical validation.

## Installation and quick verification

Use Python **3.9.13**, the recorded environment for the saved model artifacts. Create an isolated environment and install the pinned dependencies:

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -B -m unittest test_methods -v
python -B validate_project.py --notebook
```

The last command verifies the saved artifacts without rerunning the full experiment. It recomputes risks and Wilson intervals from raw allocation counts, checks disjoint validation designs and grouped folds, checks simulation/model fingerprints, reloads fitted pipelines and compares their predictions with the saved outputs. It also verifies that the saved notebook has execution counts and no recorded errors. To execute the notebook again, run `python -B create_notebook.py`.

Pinned versions matter for loading the saved scikit-learn pipelines. The notebook uses the invoking Python interpreter without changing installed kernel specifications. No external document converter or LaTeX compiler is required. Mathematics renders through notebook MathJax and Matplotlib.

## Reproduce the analysis

From the repository root:

```sh
python -B run_project.py
```

This runs allocation tests, reuses simulation grids only when fingerprints match, refits models, regenerates figures and the example comparison, executes the notebook and validates the outputs. Model refitting and notebook execution can take several minutes. A complete numerical rerun regenerates all 6.66 million trials and takes longer:

```sh
python -B run_project.py --fresh
```

Individual computational stages:

```sh
python -B run_simulations.py --reuse
python -B fit_models.py
python -B make_figures.py
python -B create_notebook.py
python -B validate_project.py --notebook
```

All cells at a sample size stay together in each outer assessment fold. Inner tuning is grouped and preprocessing occurs inside each pipeline. Validation uses unseen sample sizes and fresh streams. Scenario seeds and raw counts permit numerical checks. Source and input fingerprints use exact bytes; `.gitattributes` preserves their line endings across checkouts.

## Compare a proposed design

```sh
python -B recommend.py --N 30 --factors 2 --profile balanced --block-size 4 --iterations 5000
python -B recommend.py --N 60 --factors 3 --profile moderate_skew --block-size 6 --mode fast
```

Default mode performs fresh simulation of balance and predictability and uses exact count-risk values wherever available. The planning screen uses exact risk when available and the Wilson upper bound otherwise. Fast mode uses exact risks for eligible methods and a five-neighbour planning surrogate elsewhere. Surrogate risks have no predictive intervals and do not receive an uncertainty-based screen flag. Use simulation near a decision threshold.

Supported inputs: total sample size 10–1,200, one to five binary covariates, block bases 4, 6 or 8, and the six predefined profiles in the CLI. The tool compares study designs; it does not administer participant allocation sequences.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `randomization_core.py` | Allocation methods, exact risks, covariate generators and planning features |
| `run_simulations.py` | Training, independent validation, context and inference experiments |
| `fit_models.py` | Grouped nested validation, fitted pipelines, predictions and baselines |
| `recommend.py` | Conditional planning comparison |
| `test_methods.py` | Allocation, probability, interval and covariate checks |
| `make_figures.py` | Five source-backed figures in PNG and vector PDF |
| `math_typesetting.py` | Canonical scientific LaTeX and notebook markup |
| `create_notebook.py` | Creates and executes the computational companion |
| `validate_project.py` | Numerical, split, model, figure and notebook verification |
| `run_project.py` | Computational workflow entry point |
| `results/` | Scenario-level summaries, derived tables, predictions, metrics and provenance |
| `raw/` | Compressed per-trial arm counts and inference estimates |
| `models/` | Fitted preprocessing/model pipelines and neighbour baseline |
| `figures/` | Scientific figure exports |
| `qa/` | Numerical validation summary and SHA-256 artifact manifest |
| `environment.json` | Recorded Python and package versions |

`training_scenarios.csv` and `validation_scenarios.csv` contain one row per simulated design cell, indexed by `scenario_id`. Their NPZ allocation archives use the same identifiers as keys and contain the corresponding per-trial arm-A counts. Arm-B counts follow from the total sample size. The context archive uses the same convention. `inference_estimates.npz` retains the outcome-analysis estimates; `inference_substudy.csv` records the analysis method, valid-fit denominators, nonestimable fractions, rejection rates, coverage and power. `figure_sources.json` maps each figure to its source outputs.

## Scope and future work

The main analysis concerns two arms and binary covariates under specified profiles. Independent simulation validation is distinct from clinical or independent implementation validation. Full-information guessing assumes knowledge of simulator allocation state, including the current variable block; it is an upper-bound diagnostic. Pointwise intervals do not provide simultaneous coverage across the grid, and point-estimate Pareto comparisons do not test ranking uncertainty.

Stratification screens are conditional on the chosen difference threshold, risk limit, covariate profile and block size. Five binary factors generate 32 possible joint strata; empty strata are valid.

The Gaussian substudy uses one additive model with conventional OLS inference and rates conditional on estimable fits. It does not validate binary, survival or response-adaptive analyses. Priorities include independent comparison with established software, richer covariates, endpoint-specific randomisation-aware inference, delayed adaptive outcomes and representative recruitment patterns.

No software licence has been assigned to this repository.
