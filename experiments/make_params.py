import argparse
from itertools import product

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="params.tsv")
    args = parser.parse_args()

    # read from config.yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    N_TRIALS = config["N_TRIALS"]
    n_seeds = config["n_seeds"]
    function_ids = config["function_ids"]
    dimensions = config["dimensions"]

    with open(args.output, "w") as f:
        for fid, dim in product(function_ids, dimensions):
            f.write(f"{n_seeds}\t{fid}\t{dim}\t{N_TRIALS}\n")
    print("Parameters saved to", args.output)
