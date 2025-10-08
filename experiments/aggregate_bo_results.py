# %%

import pandas as pd

# %%
with open("results/summary_tmp.jsonl", "r") as f:
    df = pd.read_json(f, lines=True)


if df["function_id"].nunique() > 1 or df["dimension"].nunique() > 1:
    print("警告: 複数のfunction_idまたはdimensionがデータに含まれています。")
    print("現在のコードはこれらを区別せず、modeのみで集計します。")

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
        avg_nits_median=("n_iter_median", "median"),
        avg_nits_q1=("n_iter_median", lambda x: x.quantile(0.25)),
        avg_nits_q3=("n_iter_median", lambda x: x.quantile(0.75)),
    )
    .reset_index()
)
summary

# %%
summary = summary.fillna(0)

summary["Best Objective Value (mean ± std. dev.)"] = summary.apply(
    lambda row: f"{row['best_value_mean']:.2f} ± {row['best_value_std']:.2f}", axis=1
)
summary["Best Objective Value (median [Q1, Q3])"] = summary.apply(
    lambda row: f"{row['best_value_median']:.2f} [{row['best_value_q1']:.2f}, {row['best_value_q3']:.2f}]",
    axis=1,
)
summary["Acq. Func. Opt. Time (s, mean ± std. dev.)"] = summary.apply(
    lambda row: f"{row['acqf_opt_time_mean']:.2f} ± {row['acqf_opt_time_std']:.2f}",
    axis=1,
)
summary["Acq. Func. Opt. Time (s, median [Q1, Q3])"] = summary.apply(
    lambda row: f"{row['acqf_opt_time_median']:.2f} [{row['acqf_opt_time_q1']:.2f}, {row['acqf_opt_time_q3']:.2f}]",
    axis=1,
)
summary["Avg. L-BFGS Iterations (mean ± std. dev.)"] = summary.apply(
    lambda row: f"{row['avg_nits_mean']:.2f} ± {row['avg_nits_std']:.2f}", axis=1
)
summary["Avg. L-BFGS Iterations (median [Q1, Q3])"] = summary.apply(
    lambda row: f"{row['avg_nits_median']:.2f} [{row['avg_nits_q1']:.2f}, {row['avg_nits_q3']:.2f}]",
    axis=1,
)
summary
# %%


output_df = summary[
    [
        "function_id",
        "dimension",
        "mode",
        "Best Objective Value (mean ± std. dev.)",
        "Best Objective Value (median [Q1, Q3])",
        "Acq. Func. Opt. Time (s, mean ± std. dev.)",
        "Acq. Func. Opt. Time (s, median [Q1, Q3])",
        "Avg. L-BFGS Iterations (mean ± std. dev.)",
        "Avg. L-BFGS Iterations (median [Q1, Q3])",
    ]
].rename(columns={"mode": "Method"})


output_df.to_csv("aggregated_bench_stats.csv", index=False, encoding="utf-8-sig")
print(output_df.to_string(index=False))

# %%
