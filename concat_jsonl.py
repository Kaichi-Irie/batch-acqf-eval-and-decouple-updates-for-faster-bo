import glob

jsonl_files = glob.glob("results_abci/summary_f*_*D.jsonl")
dest_file = "results_abci/summary_abci_all.jsonl"

for jsonl_file in jsonl_files:
    with open(jsonl_file, "r") as f_in:
        lines = f_in.readlines()
    with open(dest_file, "a") as f_out:
        for line in lines:
            f_out.write(line)
