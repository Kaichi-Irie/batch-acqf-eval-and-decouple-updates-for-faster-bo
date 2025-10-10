# make_params.py
from itertools import product

seeds = range(1)  # Random seed
function_ids = [1]  # BBOB function ID (1-24)
dimensions = [2]  # BBOB problem dimension (2, 3, 5, 10, 20, 40)
modes = ["original", "decoupled_batch_evaluation", "coupled_batch_evaluation"]

with open("params.tsv", "w") as f:
    for seed, fid, dim, mode in product(seeds, function_ids, dimensions, modes):
        f.write(f"{seed}\t{fid}\t{dim}\t{mode}\n")
print("Parameters saved to params.tsv")
print("Total runs:", len(seeds) * len(function_ids) * len(dimensions) * len(modes))
