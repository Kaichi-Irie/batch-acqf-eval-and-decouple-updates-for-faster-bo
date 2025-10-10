#!/bin/bash
#PBS -q rt_HF
#PBS -l select=2
#PBS -l walltime=6:00:00
#PBS -P gaa50073
#PBS -J 1-3
#PBS -v USE_SSH=1

cd ${PBS_O_WORKDIR}

source venv/bin/activate

echo "PBS_O_WORKDIR=${PBS_O_WORKDIR:-"(not set)"}"
echo "PBS_ARRAY_INDEX=${PBS_ARRAY_INDEX:-"(not set)"}"
read -r SEED FID DIM MODE < <(sed -n "${PBS_ARRAY_INDEX}p" params.tsv)
echo "SEED=${SEED}, FID=${FID}, DIM=${DIM}, MODE=${MODE}"

python experiments/run_benchmark.py \
  --function_id "${FID}" \
  --dimension "${DIM}" \
  --mode "${MODE}" \
  --seed "${SEED}" \
  --n_trials 30  \
  --output_dir results \
  --summary_file summary_f"${FID}"_"${DIM}"D.jsonl \
  --skip_if_exists
