#!/bin/bash

OUTPUTDIR="results"
source venv/bin/activate

# Toy optimization experiments (Section 3 and Appendix B)
python experiments/toy_problem_compare_hessian.py \
    --method "L-BFGS-B" \
    --dimension 5 \
    --batch_size 3 \
    --seed 4 \
    --output_dir "${OUTPUTDIR}/hessian_comparison"

python experiments/toy_problem_convergence_plot.py \
    --method "L-BFGS-B" \
    --dimension 5 \
    --output_dir "${OUTPUTDIR}/convergence_plot"

python experiments/toy_problem_compare_hessian.py \
    --method "BFGS" \
    --dimension 5 \
    --batch_size 3 \
    --seed 42 \
    --output_dir "${OUTPUTDIR}/hessian_comparison"

python experiments/toy_problem_compare_hessian.py \
    --method "BFGS" \
    --dimension 5 \
    --batch_size 10 \
    --seed 0 \
    --output_dir "${OUTPUTDIR}/hessian_comparison"

python experiments/toy_problem_convergence_plot.py \
    --method "BFGS" \
    --dimension 5 \
    --output_dir "${OUTPUTDIR}/convergence_plot" \


# BO benchmark experiments (Section 5 and Appendix C)
python experiments/make_params.py \
    --config config.yaml \
    --output params.tsv

N=$(wc -l < params.tsv)
echo "Total experiments: $N"


for i in $(seq 1 $N); do
    read -r n_seeds fid dim n_trials < <(sed -n "${i}p" params.tsv)
    echo "n_seeds=${n_seeds}, fid=${fid}, dim=${dim}, n_trials=${n_trials}"
    python experiments/run_benchmark.py \
    --function_id "${fid}" \
    --dimension "${dim}" \
    --n_seeds "${n_seeds}" \
    --mode "all" \
    --n_trials "${n_trials}" \
    --output_dir "${OUTPUTDIR}" \
    --results_file results.jsonl \
    --skip_if_exists
done

# remove params.tsv
rm params.tsv

python experiments/aggregate_bo_results.py \
    --input "${OUTPUTDIR}/results.jsonl" \
    --output_dir "${OUTPUTDIR}" \
    --results_file bo_benchmark_results.csv
