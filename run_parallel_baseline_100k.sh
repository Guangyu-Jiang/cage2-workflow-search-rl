#!/bin/bash

# Run PPO Baseline for 100k episodes

echo "🚀 Starting PPO Baseline Training (100k episodes)"
echo "=================================================="
echo ""

python train_parallel_baseline.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --episodes-per-update 50 \
    --red-agent B_lineAgent

echo ""
echo "✅ PPO Baseline complete!"
echo "Results: logs/parallel_baseline_*/"

