#!/bin/bash
set -e  # exit on first error

# ----------------------------
# 1. Run WITH distillation
# ----------------------------
echo "=== Running WITH distillation ==="
python3 main.py > distilled_run.txt

# Move / rename checkpoint parts so they're not overwritten
if ls ckpt.part* 1> /dev/null 2>&1; then
    mkdir -p checkpoints
    ts=$(date +"%Y%m%d_%H%M%S")
    distill_dir="checkpoints/distill_$ts"
    mkdir -p "$distill_dir"
    mv ckpt.part* "$distill_dir"/
    echo "[INFO] Distilled weights saved to $distill_dir"
else
    echo "[WARN] No ckpt.part* files found after distillation run!"
fi

# ----------------------------
# 2. Run WITHOUT distillation
# ----------------------------
echo "=== Running WITHOUT distillation ==="
python3 main.py --no_distill > no_distilled_run.txt

# Move / rename checkpoint parts so they're not overwritten
if ls ckpt.part* 1> /dev/null 2>&1; then
    mkdir -p checkpoints
    ts=$(date +"%Y%m%d_%H%M%S")
    nodistill_dir="checkpoints/nodistill_$ts"
    mkdir -p "$nodistill_dir"
    mv ckpt.part* "$nodistill_dir"/
    echo "[INFO] Non-distilled weights saved to $nodistill_dir"
else
    echo "[WARN] No ckpt.part* files found after non-distillation run!"
fi

echo "=== All runs completed ==="

if [ -f distilled_run.txt ] && [ -f no_distilled_run.txt ]; then
    python3 plot_val_loss.py
else
    echo "[WARN] Missing log files; skipping plot generation."
fi
