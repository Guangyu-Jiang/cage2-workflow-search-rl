#!/bin/bash

# Run SAC Baseline for 100k episodes

echo "🚀 Starting SAC Baseline Training (100k episodes)"
echo "=================================================="
echo ""

python train_parallel_sac.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --episodes-per-update 50 \
    --batch-size 256 \
    --red-agent B_lineAgent

echo ""
echo "✅ SAC Baseline complete!"
echo "Results: logs/parallel_sac_*/"

