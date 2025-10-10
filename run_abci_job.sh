source venv/bin/activate
python experiments/make_params.py
N=$(wc -l < params.tsv)
echo $N
qsub -J 1-$N ./abci_job_script.sh
