#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -P gaa50073
#PBS -v USE_SSH=1
#PBS -N aaai2026-exp
#PBS -m ae

cd ${PBS_O_WORKDIR}

source venv/bin/activate

echo "PBS_O_WORKDIR=${PBS_O_WORKDIR:-"(not set)"}"
echo "PBS_ARRAY_INDEX=${PBS_ARRAY_INDEX:-"(not set)"}"
read -r N_SEEDS FID DIM N_TRIALS < <(sed -n "${PBS_ARRAY_INDEX}p" params.tsv)
echo "N_SEEDS=${N_SEEDS}, FID=${FID}, DIM=${DIM}, N_TRIALS=${N_TRIALS}"

python experiments/run_benchmark.py \
  --function_id "${FID}" \
  --dimension "${DIM}" \
  --n_seeds "${N_SEEDS}" \
  --mode "all" \
  --n_trials "${N_TRIALS}" \
  --output_dir results \
  --summary_file summary_f"${FID}"_"${DIM}"D.jsonl \
  --skip_if_exists
