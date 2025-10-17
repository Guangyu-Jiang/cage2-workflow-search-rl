#!/bin/bash

# Run async training with independent episode collection
# Each environment collects full episodes independently - no synchronous barriers!

echo "🚀 Starting ASYNC Workflow RL Training"
echo "======================================"
echo ""
echo "Key Features:"
echo "  ✓ Workers collect FULL EPISODES independently"
echo "  ✓ No synchronization at each step"
echo "  ✓ Should be significantly faster than synchronous version"
echo ""

python workflow_rl/async_train_workflow_rl.py \
    --n-envs 100 \
    --total-episodes 100000 \
    --max-episodes-per-workflow 500 \
    --episodes-per-update 100 \
    --red-agent B_lineAgent \
    --alignment-lambda 30.0 \
    --compliance-threshold 0.95

echo ""
echo "✅ Training complete!"

