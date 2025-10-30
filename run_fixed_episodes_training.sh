#!/bin/bash

# Run fixed-episodes training (NO compliance-based training)
# Ablation study to show value of compliance-based training

echo "🚀 Starting Fixed-Episodes Training (Ablation Study)"
echo "=========================================="
echo ""
echo "Key Differences from Adaptive Termination Version:"
echo "  ✓ KEEPS alignment rewards (lambda × compliance)"
echo "  ❌ NO compliance-based early stopping (no stop at 90%)"
echo "  ❌ NO plateau detection"
echo "  ✓ Trains for EXACTLY 2500 episodes per workflow"
echo ""
echo "Purpose:"
echo "  Test value of adaptive termination vs fixed episodes"
echo ""

python workflow_rl/executor_async_fixed_episodes.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --fixed-episodes-per-workflow 2500 \
    --episodes-per-update 50 \
    --alignment-lambda 30.0 \
    --red-agent B_lineAgent

echo ""
echo "✅ Fixed-episodes training complete!"
echo ""
echo "Compare results with compliance-based training:"
echo "  Fixed-episodes: logs/exp_fixed_episodes_*/"
echo "  Compliance-based: logs/exp_executor_async_*/"

