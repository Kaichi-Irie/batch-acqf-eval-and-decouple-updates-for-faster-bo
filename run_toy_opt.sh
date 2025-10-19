#!/bin/bash

source venv/bin/activate
python experiments/toy_problem_compare_hessian.py
python experiments/toy_problem_convergence_plot.py
