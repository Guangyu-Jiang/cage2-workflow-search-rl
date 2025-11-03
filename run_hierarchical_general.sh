#!/bin/bash

# Run General Hierarchical RL baseline (Options Framework)
# Domain-agnostic approach using learned options

echo "🚀 Starting General Hierarchical RL Baseline"
echo "=========================================="
echo ""
echo "Features:"
echo "  ✓ Options Framework (Sutton et al.)"
echo "  ✓ High-level: Selects from 8 learned options"
echo "  ✓ Low-level: Executes primitive actions"
echo "  ✓ NO domain-specific knowledge"
echo "  ✓ 50 parallel workers"
echo ""

python train_hierarchical_general.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --episodes-per-update 50 \
    --n-options 8 \
    --option-duration 10 \
    --red-agent B_lineAgent

echo ""
echo "✅ Hierarchical RL (general) training complete!"
echo "Results: logs/hierarchical_options_*/"

