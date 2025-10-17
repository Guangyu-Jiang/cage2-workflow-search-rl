#!/bin/bash
# Run sequential training with single environment

echo "🔄 Starting SEQUENTIAL training..."
echo "   - Single environment"
echo "   - 100 episodes per update"
echo "   - Sequential collection"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CAGE2

# Run sequential training
python workflow_rl/sequential_train_workflow_rl.py \
    --total-episodes 100000 \
    --episodes-per-update 100 \
    --max-episodes 5000 \
    --alignment-lambda 30.0 \
    --compliance-threshold 0.95 \
    --red-agent B_lineAgent
