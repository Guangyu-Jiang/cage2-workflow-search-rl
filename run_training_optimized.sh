#!/bin/bash
# Run training with optimized settings to work around performance issues

echo "🚀 Running training with performance optimizations..."
echo ""
echo "Using reduced environments (25 instead of 100) for better scaling efficiency"
echo "This should give ~20-25 episodes/sec instead of 1.1 episodes/sec"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CAGE2

python workflow_rl/parallel_train_workflow_rl.py \
    --n-envs 25 \
    --update-steps 25 \
    --total-episodes 100000 \
    --max-episodes 200 \
    --alignment-lambda 30.0 \
    --compliance-threshold 0.95 \
    --red-agent B_lineAgent

# Note: With 25 environments:
# - Better scaling efficiency (52% vs 21% with 100 envs)
# - ~90 steps/sec per env = ~22 episodes/sec total
# - 100,000 episodes in ~75 minutes instead of 25 hours
