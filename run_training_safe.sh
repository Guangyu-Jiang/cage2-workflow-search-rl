#!/bin/bash
# Optimized training script for 100K episode budget

echo "========================================"
echo "Starting Training with Optimized Settings"
echo "========================================"
echo ""
echo "Configuration:"
echo "  - 100 parallel environments (optimal)"
echo "  - 100,000 episode budget"
echo "  - 100 max episodes per workflow"
echo "  - Red agent: $1 (default: meander)"
echo ""

RED_AGENT=${1:-meander}

# Run with optimized defaults for 100K episodes
python workflow_rl/parallel_train_workflow_rl.py \
    --n-envs 100 \
    --total-episodes 100000 \
    --max-episodes 100 \
    --red-agent $RED_AGENT \
    --alignment-lambda 30.0 \
    --compliance-threshold 0.95 \
    --update-steps 100

echo ""
echo "Training complete!"
