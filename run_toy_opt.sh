#!/bin/bash

source venv/bin/activate
python experiments/toy_problem_compare_hessian.py \
    --method "L-BFGS-B" \
    --dimension 5 \
    --batch_size 3 \
    --seed 4 \
    --output_dir "results_tmp/hessian_comparison"
python experiments/toy_problem_convergence_plot.py \
    --method "L-BFGS-B" \
    --dimension 5 \
    --output_dir "results_tmp/convergence_plot" \
