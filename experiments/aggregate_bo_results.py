# %%
import argparse

import pandas as pd


def aggregate_bo_results(
    input_file: str, output_file: str, should_append: bool
) -> None:
    with open(input_file, "r") as f:
        df = pd.read_json(f, lines=True)

    if df["function_id"].nunique() > 1 or df["dimension"].nunique() > 1:
        print("Warning: Multiple function_id or dimension found in the data.")
        print(
            "The current code will aggregate only by mode without distinguishing these."
        )

    summary = (
        df.groupby(["function_id", "dimension", "mode"])
        .agg(
            best_value_mean=("best_value", "mean"),
            best_value_std=("best_value", "std"),
            best_value_median=("best_value", "median"),
            best_value_q1=("best_value", lambda x: x.quantile(0.25)),
            best_value_q3=("best_value", lambda x: x.quantile(0.75)),
            acqf_opt_time_mean=("elapsed_acqf_opt", "mean"),
            acqf_opt_time_std=("elapsed_acqf_opt", "std"),
            acqf_opt_time_median=("elapsed_acqf_opt", "median"),
            acqf_opt_time_q1=("elapsed_acqf_opt", lambda x: x.quantile(0.25)),
            acqf_opt_time_q3=("elapsed_acqf_opt", lambda x: x.quantile(0.75)),
            avg_nits_mean=("n_iter_mean", "mean"),
            avg_nits_std=("n_iter_mean", "std"),
            med_nits_median=("n_iter_median", "median"),
            med_nits_q1=("n_iter_median", lambda x: x.quantile(0.25)),
            med_nits_q3=("n_iter_median", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    summary = summary.fillna(0)

    summary["Best Value (mean ± std.)"] = summary.apply(
        lambda row: f"{row['best_value_mean']:.2f} ± {row['best_value_std']:.2f}",
        axis=1,
    )
    summary["Best Value (median [IQR])"] = summary.apply(
        lambda row: f"{row['best_value_median']:.2f} [{row['best_value_q1']:.2f}, {row['best_value_q3']:.2f}]",
        axis=1,
    )
    summary["Acq. Opt. (sec, mean ± std.)"] = summary.apply(
        lambda row: f"{row['acqf_opt_time_mean']:.2f} ± {row['acqf_opt_time_std']:.2f}",
        axis=1,
    )
    summary["Acq. Opt. (sec, median [IQR])"] = summary.apply(
        lambda row: f"{row['acqf_opt_time_median']:.2f} [{row['acqf_opt_time_q1']:.2f}, {row['acqf_opt_time_q3']:.2f}]",
        axis=1,
    )
    summary["Avg. Iters (mean ± std.)"] = summary.apply(
        lambda row: f"{row['avg_nits_mean']:.2f} ± {row['avg_nits_std']:.2f}", axis=1
    )
    summary["Avg. Iters (median [IQR])"] = summary.apply(
        lambda row: f"{row['med_nits_median']:.2f} [{row['med_nits_q1']:.2f}, {row['med_nits_q3']:.2f}]",
        axis=1,
    )
    # speedup = original_mode_time / proposed_mode_time
    summary["mean_speedup"] = summary.apply(
        lambda row: summary[
            (summary["function_id"] == row["function_id"])
            & (summary["dimension"] == row["dimension"])
            & (summary["mode"] == "original")
        ]["acqf_opt_time_mean"].values[0]
        / row["acqf_opt_time_mean"],
        axis=1,
    )

    summary["median_speedup"] = summary.apply(
        lambda row: summary[
            (summary["function_id"] == row["function_id"])
            & (summary["dimension"] == row["dimension"])
            & (summary["mode"] == "original")
        ]["acqf_opt_time_median"].values[0]
        / row["acqf_opt_time_median"],
        axis=1,
    )
    # sort modes in the order of 'original', 'coupled_batch_evaluation', 'decoupled_batch_evaluation'
    summary["mode"] = pd.Categorical(
        summary["mode"],
        categories=[
            "original",
            "coupled_batch_evaluation",
            "decoupled_batch_evaluation",
        ],
        ordered=True,
    )
    summary = summary.sort_values(by=["function_id", "dimension", "mode"])

    output_df = summary[
        [
            "function_id",
            "dimension",
            "mode",
            # "best_value_mean",
            # "best_value_std",
            # "best_value_median",
            # "best_value_q1",
            # "best_value_q3",
            # "acqf_opt_time_mean",
            # "acqf_opt_time_std",
            # "acqf_opt_time_median",
            # "acqf_opt_time_q1",
            # "acqf_opt_time_q3",
            # "avg_nits_mean",
            # "avg_nits_std",
            # "med_nits_median",
            # "med_nits_q1",
            # "med_nits_q3",
            # "mean_speedup",
            # "median_speedup",
            "Best Value (mean ± std.)",
            "Best Value (median [IQR])",
            "Acq. Opt. (sec, mean ± std.)",
            "Acq. Opt. (sec, median [IQR])",
            "Avg. Iters (mean ± std.)",
            "Avg. Iters (median [IQR])",
        ]
    ].rename(columns={"mode": "Method"})

    output_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
        mode="a" if should_append else "w",
    )
    print(output_df.to_string(index=False))


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="results/summary.jsonl", help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aggregated_bench_stats.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--append", action="store_true", help="Append to the output file if it exists"
    )
    args = parser.parse_args()
    print("Aggregating BO results...")
    aggregate_bo_results(args.input, args.output, args.append)
