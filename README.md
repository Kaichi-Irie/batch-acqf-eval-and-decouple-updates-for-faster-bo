# aaai2026-anon-exp
Recommended Python version: 3.12

## Setup and Run Experiments

Setup:
```sh
# Create virtual environment and install dependencies
python -m venv venv
# Activate virtual environment
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
# Or install packages manually
# pip install optunahub coco-experiment optproblems diversipy torch scipy matplotlib pandas greenlet
```

Run experiments:
```sh
# Run experiments
# Make run.sh executable (only need to do this once)
chmod +x run.sh
# Execute the script
./run.sh
```
