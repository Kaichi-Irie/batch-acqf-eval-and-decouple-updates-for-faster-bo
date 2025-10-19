#!/bin/bash

OUTPUTDIR="results_tmp"
source venv/bin/activate
python experiments/make_params.py \
    --config config.yaml \
    --output params.tsv

N=$(wc -l < params.tsv)
echo "Total experiments: $N"


for i in $(seq 1 $N); do
    read -r N_SEEDS FID DIM N_TRIALS < <(sed -n "${i}p" params.tsv)
    echo "N_SEEDS=${N_SEEDS}, FID=${FID}, DIM=${DIM}, N_TRIALS=${N_TRIALS}"
    python experiments/run_benchmark.py \
    --function_id "${FID}" \
    --dimension "${DIM}" \
    --n_seeds "${N_SEEDS}" \
    --mode "all" \
    --n_trials "${N_TRIALS}" \
    --output_dir "${OUTPUTDIR}" \
    --results_file results.jsonl \
    --skip_if_exists
done

# remove params.tsv
rm params.tsv

python experiments/aggregate_bo_results.py \
    --input results_tmp/results.jsonl \
    --output_dir "${OUTPUTDIR}" \
    --results_file bo_benchmark_results.csv
