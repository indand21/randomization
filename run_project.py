"""Run the computational randomisation study without document dependencies."""
from pathlib import Path
import argparse, json, os, subprocess, sys, time
HERE = Path(__file__).absolute().parent

def run(fresh=False):
    start = time.time()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    (HERE / "qa").mkdir(exist_ok=True)
    def call(args):
        subprocess.run([sys.executable, "-B"] + args, cwd=HERE, env=env, check=True)
    call(["-m", "unittest", "test_methods", "-v"])
    call(["run_simulations.py"] + ([] if fresh else ["--reuse"]))
    for script in ["fit_models.py", "make_figures.py"]:
        call([script])
    call(["recommend.py", "--N", "30", "--factors", "2", "--output", "results/PlanningTool_Example_N30.csv"])
    call(["create_notebook.py"])
    call(["validate_project.py", "--notebook"])
    (HERE / "qa/pipeline_run.json").write_text(json.dumps({
        "status": "passed", "fresh_simulations": fresh,
        "elapsed_seconds": time.time() - start, "python": sys.version.split()[0]
    }, indent=2), encoding="utf-8")
    print("Computational study workflow passed.", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh", action="store_true", help="Regenerate all simulations instead of reusing matching saved grids.")
    args = ap.parse_args()
    run(fresh=args.fresh)
