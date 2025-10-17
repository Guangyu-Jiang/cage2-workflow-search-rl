#!/bin/bash

# Run ProcessPoolExecutor-based async training
# TRUE async using Python's built-in concurrent.futures!

echo "🚀 Starting ProcessPoolExecutor ASYNC Training"
echo "======================================"
echo ""
echo "Key Features:"
echo "  ✓ Python's built-in ProcessPoolExecutor"
echo "  ✓ No external dependencies (no Ray needed)"
echo "  ✓ Workers collect episodes INDEPENDENTLY"
echo "  ✓ as_completed() for true async collection"
echo "  ✓ Expected 50-150+ episodes/sec with 100 workers"
echo ""

python workflow_rl/executor_async_train_workflow_rl.py \
    --n-workers 100 \
    --total-episodes 100000 \
    --max-episodes-per-workflow 500 \
    --episodes-per-update 100 \
    --red-agent B_lineAgent \
    --alignment-lambda 30.0 \
    --compliance-threshold 0.95

echo ""
echo "✅ Training complete!"

