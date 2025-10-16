#!/bin/bash
# Safe training script with optimized settings

echo "========================================"
echo "Starting Training with Safe Settings"
echo "========================================"
echo ""
echo "Configuration:"
echo "  - 100 parallel environments (stable)"
echo "  - 500 episode budget"
echo "  - Red agent: $1 (default: meander)"
echo ""

RED_AGENT=${1:-meander}

# Run with 100 environments for stability
python workflow_rl/parallel_train_workflow_rl.py \
    --n-envs 100 \
    --total-episodes 500 \
    --max-episodes 50 \
    --red-agent $RED_AGENT \
    --alignment-lambda 30.0 \
    --compliance-threshold 0.95 \
    --update-steps 100

echo ""
echo "Training complete!"
