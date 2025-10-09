#!/bin/bash
# run_optuna_study_array.sh（スケジューラの指示行は環境に合わせて）
#$ -cwd
#$ -l h_rt=06:00:00
#$ -l rt_C.small=1
#$ -j y
#$ -o logs/optuna.$JOB_ID.$TASK_ID.log
#$ -N optuna_grid
# set -eu
# SGE_TASK_ID=1  # for debugging; comment out when using SGE/UGE
source venv/bin/activate  # or: uv run python ...
read -r SEED FID DIM MODE < <(sed -n "${SGE_TASK_ID}p" params.tsv)

uv run python experiments/run_benchmark.py \
  --function_id "${FID}" \
  --dimension "${DIM}" \
  --mode "${MODE}" \
  --seed "${SEED}" \
  --n_trials 300  \
  --output_dir results_all \
  --summary_file summary_f"${FID}"_"${DIM}"D.jsonl \
  --skip_if_exists
