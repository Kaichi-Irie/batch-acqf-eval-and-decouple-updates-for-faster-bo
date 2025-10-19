# aaai2026-anon-exp
Recommended Python version: 3.12

## Setup virtual environment and install dependencies:
```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod +x run_rosenbrock_opt.sh
chmod +x run_bo.sh
```

## Run experiments:
```sh
# Convergence analysis and Hessian comparison on toy problem (Rosenbrock function optimization)
./run_toy_opt.sh

# BBOB benchmark experiments
./run_bo.sh
```
