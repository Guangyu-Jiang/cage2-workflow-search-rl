#!/bin/bash

# Cleanup unnecessary files from CAGE2 directory

echo "🧹 Cleaning up unnecessary files..."
echo ""

# Remove temporary test and debug scripts
echo "Removing test and debug scripts..."
rm -f demonstrate_scaling_limits.py
rm -f diagnose_parallel_speed.py
rm -f test_100_envs.py
rm -f test_fast_training.py
rm -f test_optimized_parallel.py
rm -f test_ray_async.py
rm -f quick_benchmark.py
rm -f quick_speed_test.py
rm -f debug_training.py

# Remove analysis and investigation scripts
echo "Removing analysis scripts..."
rm -f analyze_cage2_units.py
rm -f analyze_comparison.py
rm -f analyze_detailed_sequence.py
rm -f analyze_optimal_policy.py
rm -f benchmark_parallel_envs.py
rm -f check_op_hosts_importance.py
rm -f confirm_unit_types.py
rm -f investigate_action_sharing.py
rm -f investigate_defender_and_hosts.py
rm -f measure_baseline_times.py
rm -f measure_training_times.py

# Remove redundant documentation
echo "Removing redundant documentation..."
rm -f SPEED_OPTIMIZATION_SUMMARY_v2.md
rm -f OPTIMIZATION_APPLIED.md
rm -f OPTIMIZATION_RESULTS.md
rm -f TIMING_ANALYSIS_RESULTS.md
rm -f WHY_PARALLEL_IS_SLOWER.md
rm -f PERFORMANCE_FIX_APPLIED.md
rm -f ASYNC_ARCHITECTURE_NOTE.md

# Remove old shell scripts that are superseded
echo "Removing old shell scripts..."
rm -f run_fast_training.sh
rm -f run_training_optimized.sh
rm -f run_sequential_training.sh
rm -f run_ray_async_training.sh
rm -f run_async_training.sh

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Files removed:"
echo "  - Temporary test scripts (test_*.py, debug_*.py)"
echo "  - Analysis/investigation scripts"
echo "  - Redundant documentation"
echo "  - Superseded shell scripts"
echo ""
echo "Kept:"
echo "  - Main training script: workflow_rl/executor_async_train_workflow_rl.py"
echo "  - Run script: run_executor_async_training.sh"
echo "  - Core documentation"
echo "  - Training logs"

