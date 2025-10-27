#!/bin/bash

# Run parallel baseline PPO training (NO workflow conditioning)
# For fair speed comparison with workflow search

echo "🚀 Starting Parallel Baseline PPO Training"
echo "=========================================="
echo ""
echo "Features:"
echo "  ✓ 200 parallel workers (same as workflow search)"
echo "  ✓ Standard PPO (NO workflow conditioning)"
echo "  ✓ Full 145 action space"
echo "  ✓ For baseline comparison"
echo "  ✓ Expected 100-150 episodes/sec"
echo ""

python train_parallel_baseline.py \
    --n-workers 200 \
    --total-episodes 100000 \
    --episodes-per-update 200 \
    --red-agent B_lineAgent

echo ""
echo "✅ Baseline training complete!"

