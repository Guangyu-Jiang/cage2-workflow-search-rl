#!/bin/bash
# Run optimized training with best settings

echo "🚀 Starting FAST training with optimizations..."
echo "   - Vectorized environments (2.2x faster)"
echo "   - K_epochs=4 (stable PPO convergence)"  
echo "   - Batch logging (reduced I/O)"
echo "   - More frequent updates (every 50 steps)"
echo ""

python workflow_rl/parallel_train_workflow_rl_fast.py \
    --n-envs 200 \
    --total-episodes 100000 \
    --max-episodes 50 \
    --update-steps 50 \
    --alignment-lambda 0.01 \
    --compliance-threshold 0.95 \
    --red-agent B_lineAgent
