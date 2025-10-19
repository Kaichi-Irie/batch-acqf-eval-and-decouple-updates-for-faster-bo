# %%
import argparse
import itertools
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
    results_file="summary.jsonl",
    output_dir="results",
    skip_if_exists=True,
):
    def sphere(trial: optuna.trial.Trial) -> float:
        x = np.array(
            [trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(dimension)]
        )
        return np.sum(x**2)

    if function_id == 0:
        objective = sphere
        directions = ["minimize"]
    else:
        objective = BBOB.Problem(
            function_id=function_id, dimension=dimension, instance_id=1
        )
        directions = objective.directions
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

    study = optuna.create_study(directions=directions, sampler=sampler, storage=storage)

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

    results_file_path = os.path.join(output_dir, results_file)
    if not sampler.nit_stats_list:
        with open(results_file_path, "a") as f:
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
    with open(results_file_path, "a") as f:
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
        "--n_seeds",
        type=int,
        help="Number of random seeds (used seeds: 0 - n_seeds-1)",
    )
    parser.add_argument(
        "--function_id",
        type=int,
        help="List of BBOB function IDs (1-24) or 0 for original sphere function",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        help="List of BBOB problem dimensions (2, 3, 5, 10, 20, 40)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "original",
            "decoupled_batch_evaluation",
            "coupled_batch_evaluation",
            "all",
        ],
        help="Sampler mode",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        help="Number of trials per run",
    )
    parser.add_argument(
        "--results_file",
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
    if args.mode == "all":
        modes = [
            "original",
            "decoupled_batch_evaluation",
            "coupled_batch_evaluation",
        ]
    else:
        modes = [args.mode]
    seeds = range(args.n_seeds)
    for seed, mode in itertools.product(seeds, modes):
        execute_benchmark(
            function_id=args.function_id,
            dimension=args.dimension,
            mode=mode,  # type: ignore[arg-type]
            n_trials=args.n_trials,
            seed=seed,
            results_file=args.results_file,
            output_dir=args.output_dir,
            skip_if_exists=args.skip_if_exists,
        )
