source venv/bin/activate
uv run python experiments/run_benchmark.py \
    --summary_file summary.jsonl \
    --output_dir results \
    --n_seeds 1 \
    --n_trials 20 \
    --function_ids 1 2 3 \
    --dimensions 5 10 20 40 \
    --resume
