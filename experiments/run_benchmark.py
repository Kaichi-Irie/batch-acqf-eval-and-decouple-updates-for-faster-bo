# %%
import json
import os
import sys
from collections import defaultdict
from itertools import product

import numpy as np
import optuna
import optunahub
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.batched_sampler import SAMPLERMODE, BatchedSampler

# %%
BBOB = optunahub.load_module("benchmarks/bbob")


# %%
def execute_benchmark(
    mode: SAMPLERMODE,
    dimension,
    function_id,
    n_trials,
    seed,
    summary_file="summary.jsonl",
    output_dir="results",
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
    storage = JournalStorage(JournalFileBackend(os.path.join(output_dir, log_file)))

    study = optuna.create_study(
        directions=objective.directions, sampler=sampler, storage=storage
    )

    study.optimize(objective, n_trials=n_trials)

    print(study.best_trial.params, study.best_trial.value)
    elapsed = (
        study.trials[-1].datetime_complete - study.trials[0].datetime_start
    ).total_seconds()  # type: ignore
    print(f"{mode} took {elapsed:.2f} seconds. ")

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

    if not sampler.nit_stats_list:
        with open(os.path.join(output_dir, summary_file), "a") as f:
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
    with open(os.path.join(output_dir, summary_file), "a") as f:
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
if __name__ == "__main__":
    seeds = range(3)  # [42, 43, 44]  # [42, 43, 44]
    n_trials = 100  # 回せるだけ回す~500
    # https://numbbo.github.io/coco/testsuites/bbob
    function_ids = [6]  # , 15]  # [1,6,10,15,20]
    dimensions = [10]  # , 20, 40]  # [5,10,20]
    modes: list[SAMPLERMODE] = [
        "original",
        "coupled_batch_evaluation",  # "CBE",
        "decoupled_batch_evaluation",  # "DBE",
    ]
    for function_id, dimension, seed, mode in product(
        function_ids, dimensions, seeds, modes
    ):
        execute_benchmark(
            function_id=function_id,
            dimension=dimension,
            mode=mode,
            n_trials=n_trials,
            seed=seed,
            summary_file="summary_tmp.jsonl",
            output_dir="results",
        )
