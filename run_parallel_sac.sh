#!/bin/bash

# Run parallel SAC baseline training (NO workflow conditioning)
# For comparison with PPO and workflow search methods

echo "🚀 Starting Parallel SAC Baseline Training"
echo "=========================================="
echo ""
echo "Features:"
echo "  ✓ SAC (Soft Actor-Critic) algorithm"
echo "  ✓ 50 parallel workers"
echo "  ✓ Off-policy learning with replay buffer"
echo "  ✓ Dual Q-networks for stability"
echo "  ✓ NO workflow conditioning"
echo "  ✓ For L4DC baseline comparison"
echo ""

python train_parallel_sac.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --episodes-per-update 50 \
    --batch-size 256 \
    --red-agent B_lineAgent

echo ""
echo "✅ SAC baseline training complete!"

