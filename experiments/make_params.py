# make_params.py
from itertools import product

seeds = range(10)  # Random seed
function_ids = [1]  # BBOB function ID (1-24)
dimensions = [5, 10, 20, 40]  # BBOB problem dimension (2, 3, 5, 10, 20, 40)
modes = ["original", "decoupled_batch_evaluation", "coupled_batch_evaluation"]

with open("params.tsv", "w") as f:
    for seed, fid, dim, mode in product(seeds, function_ids, dimensions, modes):
        f.write(f"{seed}\t{fid}\t{dim}\t{mode}\n")
