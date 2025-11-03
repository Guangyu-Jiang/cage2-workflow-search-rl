#!/bin/bash

# Run fixed-episodes training (Workflow-Specific, NO Cross-Transfer)
# Can resume SAME workflow, but NO transfer between workflows

echo "🚀 Starting Fixed-Episodes Training (Workflow-Specific)"
echo "=========================================="
echo ""
echo "Key Features:"
echo "  ✅ Can resume SAME workflow if GP selects it again"
echo "  ❌ NO transfer learning from OTHER workflows"
echo "  ✅ Compliance rewards enabled"
echo "  ❌ NO early stopping"
echo "  ✅ Fixed 2500 episodes per workflow"
echo ""
echo "Purpose:"
echo "  Test fixed episodes with workflow-specific policies"
echo ""

python workflow_rl/executor_async_fixed_episodes_no_warmstart.py \
    --n-workers 25 \
    --total-episodes 100000 \
    --fixed-episodes-per-workflow 2500 \
    --episodes-per-update 25 \
    --alignment-lambda 30.0 \
    --red-agent B_lineAgent

echo ""
echo "✅ Fixed-episodes (no warmstart) training complete!"
echo ""
echo "Results: logs/exp_fixed_no_warmstart_*/"

