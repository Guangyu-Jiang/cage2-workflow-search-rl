#!/bin/bash

# Run Hierarchical RL baseline (2-level hierarchy)

echo "🚀 Starting Hierarchical RL Baseline Training"
echo "=========================================="
echo ""
echo "Features:"
echo "  ✓ 2-level hierarchy (high-level + low-level policies)"
echo "  ✓ High-level: Selects unit type to fix"
echo "  ✓ Low-level: Selects specific action"
echo "  ✓ 50 parallel workers"
echo "  ✓ For L4DC baseline comparison"
echo ""

python train_hierarchical_baseline.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --episodes-per-update 50 \
    --red-agent B_lineAgent

echo ""
echo "✅ Hierarchical RL training complete!"
echo "Results: logs/hierarchical_baseline_*/"

