import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
from experiments.run_benchmark import execute_benchmark, parse
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def create_objective(
    function_id,
    dimension,
    modes,
    seeds,
    n_trials,
    results_file,
    output_dir,
    skip_if_exists,
):
    def objective(trial):
        seed = trial.suggest_categorical("seed", seeds)
        mode = trial.suggest_categorical("mode", modes)
        execute_benchmark(
            function_id=function_id,
            dimension=dimension,
            mode=mode,
            n_trials=n_trials,
            seed=seed,
            results_file=results_file,
            output_dir=output_dir,
            skip_if_exists=skip_if_exists,
        )
        return 0.0

    return objective


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
    seeds = list(range(args.n_seeds))
    sampler = optuna.samplers.BruteForceSampler()
    study_name = f"benchmark_bruteforce_f{args.function_id}_D{args.dimension}_{args.n_seeds}seeds_{args.n_trials}trials"
    storage = JournalStorage(
        JournalFileBackend(os.path.join(args.output_dir, f"{study_name}.jsonl"))
    )

    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
    )
    job_objective = create_objective(
        function_id=args.function_id,
        dimension=args.dimension,
        modes=modes,
        seeds=seeds,
        n_trials=args.n_trials,
        results_file=args.results_file,
        output_dir=args.output_dir,
        skip_if_exists=args.skip_if_exists,
    )
    study.optimize(job_objective, n_trials=len(modes) * len(seeds))
    print("All jobs are done.")
    print(
        f"Elapsed: {study.trials[-1].datetime_complete - study.trials[0].datetime_start}"  # type: ignore
    )
