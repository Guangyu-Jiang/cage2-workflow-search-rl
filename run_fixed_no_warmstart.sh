#!/bin/bash

# Run fixed-episodes training WITHOUT warm starting
# Each workflow trains from scratch (random initialization)

echo "🚀 Starting Fixed-Episodes Training (NO Warm Start)"
echo "=========================================="
echo ""
echo "Key Features:"
echo "  ✅ Each workflow trains from scratch"
echo "  ✅ Compliance rewards enabled"
echo "  ❌ NO policy reuse between workflows"
echo "  ❌ NO early stopping"
echo "  ✅ Fixed 2500 episodes per workflow"
echo ""
echo "Purpose:"
echo "  Test pure fixed-episode training without transfer learning"
echo ""

python workflow_rl/executor_async_fixed_episodes_no_warmstart.py \
    --n-workers 25 \
    --total-episodes 100000 \
    --fixed-episodes-per-workflow 2500 \
    --episodes-per-update 25 \
    --alignment-lambda 30.0 \
    --red-agent B_lineAgent

echo ""
echo "✅ Fixed-episodes (no warmstart) training complete!"
echo ""
echo "Results: logs/exp_fixed_no_warmstart_*/"

