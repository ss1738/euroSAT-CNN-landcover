#!/usr/bin/env bash
# Reproduce every measured finding, in order. Total GPU time roughly 2.5h on an
# RTX 5090. Each stage is independent; run any subset. Outputs (results JSON,
# models, logs) go to EA_DIR (default ~/eurosat-assured), never the repo.
set -euo pipefail
export EA_DIR="${EA_DIR:-$HOME/eurosat-assured}"
mkdir -p "$EA_DIR"
echo "Output dir: $EA_DIR"
python3 -c "import torch, torchvision" || { echo "pip install -r requirements.txt first"; exit 1; }

run() { echo "=== stage $1 (expected ~$2) ==="; python3 "experiments/eurosat_stage$1.py"; }

run 1  "6 min (incl. 90MB EuroSAT download on first run)"
run 2  "5 min"
run 3  "3 min"
run 4  "5 min"
run 5  "40 min"
run 6  "15 min"
if [ -d "$EA_DIR/data/ms/EuroSAT_MS" ]; then
  run 7 "15 min"
else
  echo "=== stage 7 SKIPPED: needs the 2GB multispectral zip unpacked to \$EA_DIR/data/ms/ (see script header) ==="
fi
run 8  "10 min"
run 9  "10 min"
run 10 "60 min (downloads CIFAR-10, 170MB)"
run 11 "5 min"
run 12 "15 min"
run 13 "20 min (paper baselines/estimators)"
echo "=== all stages complete; regenerate figures with: python3 make_figures.py (run from repo root, reads results from \$EA_DIR if copied to results/) ==="
