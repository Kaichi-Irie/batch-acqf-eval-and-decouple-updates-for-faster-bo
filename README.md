# Reproducibility Code for "Batch Acquisition Function Evaluations and Decouple Optimizer Updates for Faster Bayesian Optimization"
Recommended Python version: 3.12

## Project Layout
- `src/`: Source modules, including the Bayesian optimization subroutines.
- `experiments/`: Entry points used to generate the figures and tables in the paper.
- `optuna/`: Patched Optuna components required for the reproducibility package.
- `results_expected/`: Reference outputs for key experiments (see below).
- `results/`: Default output directory created by `run.sh`.
- `config.yaml`: Example configuration for the BO benchmark sweep.
- `run.sh`: Convenience script that reproduces the full experimental pipeline.
- `requirements.txt`: Frozen dependency list used for the artifact evaluation.

## Setup and Run Experiments

Setup:
```sh
# Create virtual environment and install dependencies
python -m venv venv
# Activate virtual environment
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
# Or install packages manually
# pip install optunahub coco-experiment optproblems diversipy torch scipy matplotlib pandas greenlet
```

Run experiments:
```sh
# Run experiments
# Make run.sh executable (only need to do this once)
chmod +x run.sh
# Execute the script
./run.sh
```

## Expected Results (`results_expected/`)
- `bo_benchmark_results.csv`: Aggregated summary of the expected BO benchmark metrics. Table 1 and Table 2 in the paper are generated from this file.
- `convergence_plot/`: Convergence curves for the toy Rosenbrock experiments (`*.pdf`).
- `hessian_comparison/`: Visual comparisons of inverse Hessian estimates across solvers and batch sizes (`*.pdf`).
- `logs/`: Contains JSONL logs for each benchmark configuration (`f{fid}_{dim}D_seed{seed}_{method}_{n_trials}_trials.jsonl`) and iteration information (`iterinfo_*.jsonl`). While these files are not directly used to generate figures or tables in the paper, they can be used to reproduce Optuna study objects by loading them.
- `results.jsonl`: Concatenated BO benchmark records used as input to the aggregator script. CSV file is generated from this file.
