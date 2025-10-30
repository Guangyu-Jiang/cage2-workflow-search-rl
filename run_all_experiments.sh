#!/bin/bash

# Master script to run ALL baseline experiments for L4DC paper
# All experiments use 100,000 episodes for fair comparison

echo "🚀 Running All Baseline Experiments for L4DC Paper"
echo "===================================================="
echo ""
echo "Total experiments: 7"
echo "Estimated time: 4-5 hours"
echo ""
echo "All using:"
echo "  - 100,000 episodes"
echo "  - B_lineAgent red agent"
echo "  - Full logging and config files"
echo ""
read -p "Press Enter to start..."

# 1. Random Policy (30 min)
echo ""
echo "=================================================="
echo "1/7 - Random Policy Baseline"
echo "=================================================="
python baselines/random_policy.py --n-episodes 100000 --red-agent B_lineAgent
echo "✅ Random complete"
echo ""

# 2. Greedy Heuristic with best workflow (30 min)
echo "=================================================="
echo "2/7 - Greedy Heuristic Baseline"
echo "=================================================="
python baselines/greedy_heuristic.py \
    --n-episodes 100000 \
    --workflow defender,enterprise,op_server,op_host,user \
    --red-agent B_lineAgent
echo "✅ Greedy complete"
echo ""

# 3. PPO Baseline (15-20 min)
echo "=================================================="
echo "3/7 - PPO Baseline (No Workflow)"
echo "=================================================="
python train_parallel_baseline.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --episodes-per-update 50 \
    --red-agent B_lineAgent
echo "✅ PPO Baseline complete"
echo ""

# 4. SAC Baseline (20-25 min)
echo "=================================================="
echo "4/7 - SAC Baseline (No Workflow)"
echo "=================================================="
python train_parallel_sac.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --episodes-per-update 50 \
    --batch-size 256 \
    --red-agent B_lineAgent
echo "✅ SAC Baseline complete"
echo ""

# 5. Fixed Episodes (60-90 min)
echo "=================================================="
echo "5/7 - Fixed-Episodes Ablation"
echo "=================================================="
python workflow_rl/executor_async_fixed_episodes.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --fixed-episodes-per-workflow 2500 \
    --episodes-per-update 50 \
    --alignment-lambda 30.0 \
    --red-agent B_lineAgent
echo "✅ Fixed-Episodes complete"
echo ""

# 6. Your Method - Adaptive Termination (30-40 min)
echo "=================================================="
echo "6/7 - Adaptive Termination (Your Method)"
echo "=================================================="
python workflow_rl/executor_async_train_workflow_rl.py \
    --n-workers 200 \
    --total-episodes 100000 \
    --max-episodes-per-workflow 10000 \
    --episodes-per-update 200 \
    --alignment-lambda 30.0 \
    --compliance-threshold 0.90 \
    --red-agent B_lineAgent
echo "✅ Adaptive Termination complete"
echo ""

# 7. All Fixed Workflows (3 hours - 6 workflows × 100k each)
echo "=================================================="
echo "7/7 - All Fixed Priority Workflows"
echo "=================================================="
python baselines/fixed_priority_workflows.py \
    --n-episodes 100000 \
    --red-agent B_lineAgent
echo "✅ All Fixed Workflows complete"
echo ""

# Summary
echo ""
echo "===================================================="
echo "✅ ALL EXPERIMENTS COMPLETE!"
echo "===================================================="
echo ""
echo "Results saved to logs/:"
echo ""
ls -lth logs/ | head -15
echo ""
echo "Log files:"
echo "  - Random: logs/random_policy_*/training_log.csv"
echo "  - Greedy: logs/greedy_heuristic_*/training_log.csv"
echo "  - PPO: logs/parallel_baseline_*/training_log.csv"
echo "  - SAC: logs/parallel_sac_*/training_log.csv"
echo "  - Fixed-Episodes: logs/exp_fixed_episodes_*/training_log.csv"
echo "  - Your Method: logs/exp_executor_async_*/training_log.csv"
echo "  - Fixed Workflows: logs/fixed_workflows_*/summary.csv"
echo ""
echo "All experiments logged and ready for analysis!"

