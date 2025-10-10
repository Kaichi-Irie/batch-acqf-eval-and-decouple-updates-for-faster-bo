# %%
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optunahub

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from src.batched_sampler import SAMPLERMODE, BatchedSampler

# %%
BBOB = optunahub.load_module("benchmarks/bbob")

# %%
def execute_benchmark(
    function_id,
    dimension,
    mode: SAMPLERMODE,
    n_trials,
    seed,
    summary_file="summary.jsonl",
    output_dir="results",
    skip_if_exists=True,
):
    objective = BBOB.Problem(
        function_id=function_id, dimension=dimension, instance_id=1
    )
    sampler = BatchedSampler(mode=mode, seed=seed)

    short_mode = {
        "original": "seqopt",
        "decoupled_batch_evaluation": "dbe",
        "coupled_batch_evaluation": "cbe",
    }[mode]
    log_file = (
        f"f{function_id}_{dimension}D_seed{seed}_{short_mode}_{n_trials}_trials.jsonl"
    )
    if skip_if_exists and os.path.exists(os.path.join(output_dir, log_file)):
        print(f"Skip existing: {log_file}")
        return


    storage = JournalStorage(JournalFileBackend(os.path.join(output_dir, log_file)))

    study = optuna.create_study(
        directions=objective.directions, sampler=sampler, storage=storage
    )

    study.optimize(objective, n_trials=n_trials)

    print(study.best_trial.params, study.best_trial.value)
    time_comple = study.trials[-1].datetime_complete
    time_start = study.trials[0].datetime_start
    if time_comple and time_start:
        elapsed = (time_comple - time_start).total_seconds()
        print(f"{mode} took {elapsed:f} seconds. ")
    else:
        elapsed = -1.0
        print(f"{mode} took unknown seconds. ")

    summary = {
        "function_id": function_id,
        "dimension": dimension,
        "seed": seed,
        "mode": mode,
        "elapsed_total": round(elapsed, 2),
        "elapsed_acqf_opt": round(sampler.elapsed_acqf_opt, 2),
        "n_trials": n_trials,
        "best_value": study.best_trial.value,
        "n_iter_median": None,
        "n_iter_mean": None,
    }

    summary_file_path = os.path.join(output_dir, summary_file)
    if not sampler.nit_stats_list:
        with open(summary_file_path, "a") as f:
            f.write(json.dumps(summary) + "\n")
        return

    # first 10 trials are warm-up
    assert len(sampler.nit_stats_list) == len(study.trials) - 10

    medians = []
    means = []
    for nit_stat in sampler.nit_stats_list:
        medians.append(nit_stat["q2"])
        means.append(nit_stat["mean"])
    summary["n_iter_median"] = float(np.median(medians))
    summary["n_iter_mean"] = float(np.mean(means))
    with open(summary_file_path, "a") as f:
        f.write(json.dumps(summary) + "\n")

    # save iteration info
    iteration_info = defaultdict(list)
    for nit_stat in sampler.nit_stats_list:
        for k, v in nit_stat.items():
            iteration_info[k].append(v)

    # save as JSONL
    with open(os.path.join(output_dir, "iterinfo_" + log_file), "w") as f:
        f.write(json.dumps(iteration_info) + "\n")

# %%
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--function_id",
        type=int,
        help="BBOB function ID (1-24)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        help="BBOB problem dimension (2, 3, 5, 10, 20, 40)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["original", "decoupled_batch_evaluation", "coupled_batch_evaluation"],
        help="Sampler mode",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        help="Number of trials per run",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="summary.jsonl",
        help="Summary file name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip if the log file already exists",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)

    execute_benchmark(
        function_id=args.function_id,
        dimension=args.dimension,
        mode=args.mode,
        n_trials=args.n_trials,
        seed=args.seed,
        summary_file=args.summary_file,
        output_dir=args.output_dir,
        skip_if_exists=args.skip_if_exists,
    )
