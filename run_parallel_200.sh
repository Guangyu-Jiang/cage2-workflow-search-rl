#!/bin/bash
# Script to run training with new 200 parallel environments configuration

echo "============================================================"
echo "Starting Workflow RL Training with 200 Parallel Environments"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  - 200 parallel environments"
echo "  - 50 max episodes per workflow"
echo "  - 5 min episodes before early stopping"
echo "  - Update after each episode (100 steps) from all envs"
echo "  - 20,000 transitions per update"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CAGE2

# Set Python path
export PYTHONPATH=/home/ubuntu/CAGE2/-cyborg-cage-2:$PYTHONPATH

# Run with default settings (which are now 200 envs, 50 episodes, etc.)
python workflow_rl/parallel_train_workflow_rl.py "$@"
