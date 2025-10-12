# make_params.py
from itertools import product

N_TRIALS = 300  # Number of trials for each run
n_seeds = 2  # Random seed (used seeds: 0 - n_seeds-1)
function_ids = [6]  # BBOB function ID (1-24)
dimensions = [40]  # BBOB problem dimension (2, 3, 5, 10, 20, 40)

with open("params.tsv", "w") as f:
    for fid, dim in product(function_ids, dimensions):
        f.write(f"{n_seeds}\t{fid}\t{dim}\t{N_TRIALS}\n")
print("Parameters saved to params.tsv")
print("Total runs:", n_seeds * len(function_ids) * len(dimensions))
