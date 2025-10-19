#!/bin/bash

OUTPUTDIR="results"
source venv/bin/activate
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
