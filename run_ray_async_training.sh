#!/bin/bash

# Run Ray-based async training with TRUE independent episode collection
# Each worker is a Ray actor that collects episodes without synchronization!

echo "🚀 Starting RAY ASYNC Workflow RL Training"
echo "======================================"
echo ""
echo "Key Features:"
echo "  ✓ Ray actors for distributed workers"
echo "  ✓ Workers collect episodes INDEPENDENTLY (true async!)"
echo "  ✓ No synchronous barriers at all"
echo "  ✓ Expected 150-200+ episodes/sec"
echo ""

python workflow_rl/ray_async_train_workflow_rl.py \
    --n-workers 100 \
    --total-episodes 100000 \
    --max-episodes-per-workflow 500 \
    --episodes-per-update 100 \
    --red-agent B_lineAgent \
    --alignment-lambda 30.0 \
    --compliance-threshold 0.95

echo ""
echo "✅ Training complete!"

