#!/bin/bash

# Run fixed-episodes training (NO compliance-based training)
# Ablation study to show value of compliance-based training

echo "🚀 Starting Fixed-Episodes Training (Ablation Study)"
echo "=========================================="
echo ""
echo "Key Differences from Compliance-Based Training:"
echo "  ❌ NO alignment rewards (alignment_lambda = 0)"
echo "  ❌ NO compliance-based early stopping"
echo "  ✓ Trains for EXACTLY 1000 episodes per workflow"
echo "  ✓ Compliance still logged (for analysis)"
echo ""
echo "Purpose:"
echo "  Show value of compliance-based training by comparison"
echo ""

python workflow_rl/executor_async_fixed_episodes.py \
    --n-workers 200 \
    --total-episodes 100000 \
    --fixed-episodes-per-workflow 1000 \
    --episodes-per-update 200 \
    --red-agent B_lineAgent

echo ""
echo "✅ Fixed-episodes training complete!"
echo ""
echo "Compare results with compliance-based training:"
echo "  Fixed-episodes: logs/exp_fixed_episodes_*/"
echo "  Compliance-based: logs/exp_executor_async_*/"

