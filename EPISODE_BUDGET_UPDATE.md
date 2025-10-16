# Episode Budget Training Update

## Summary
Updated the workflow search training to use a total episode budget instead of limiting the number of workflows, and removed the minimum episode requirement for early stopping.

## Changes Made

### 1. **Replaced Workflow Count with Episode Budget**
   - **Old**: Training was limited to a fixed number of workflows (default: 20)
   - **New**: Training continues until a total episode budget is exhausted (default: 500)
   - **Benefit**: More efficient use of training resources - successful workflows that achieve compliance quickly use fewer episodes, allowing more workflows to be explored

### 2. **Removed Minimum Episode Requirement**
   - **Old**: Required at least 5 episodes before checking compliance
   - **New**: Can stop immediately when compliance threshold (95%) is achieved
   - **Benefit**: Faster convergence for easy workflows, no wasted episodes

### 3. **Dynamic Workflow Exploration**
   - The system now explores as many workflows as the budget allows
   - If workflows achieve compliance quickly, more workflows can be explored
   - If workflows struggle, fewer workflows are explored but each gets more training

## Command-Line Usage

### Default Configuration
```bash
python workflow_rl/parallel_train_workflow_rl.py
```
- Total episode budget: 500
- Max episodes per workflow: 50
- Parallel environments: 200

### Custom Budget Examples
```bash
# Larger budget for more exploration
python workflow_rl/parallel_train_workflow_rl.py --total-episodes 1000

# Smaller budget for quick experiments
python workflow_rl/parallel_train_workflow_rl.py --total-episodes 200

# Allow more episodes per difficult workflow
python workflow_rl/parallel_train_workflow_rl.py --total-episodes 1000 --max-episodes 100
```

## Implementation Details

### Episode Tracking
- `total_episodes_used`: Tracks cumulative episodes across all workflows
- `total_episode_budget`: Maximum allowed episodes (default: 500)
- Each workflow uses `episodes_used` episodes (minimum across all parallel environments)

### Early Stopping Logic
- Checks compliance after **every** episode (no minimum)
- Stops immediately when compliance >= 95% AND meaningful fixes detected (>= 10)
- If compliance is achieved, the workflow is marked successful and added to GP-UCB

### Budget Allocation
- Each workflow can use up to `max_train_episodes_per_env` episodes (default: 50)
- If remaining budget < 50, the workflow uses whatever budget remains
- Training stops when budget is exhausted

## Benefits

1. **Efficiency**: No wasted episodes on workflows that achieve compliance quickly
2. **Flexibility**: Automatically balances between exploring many workflows vs. training difficult ones
3. **Simplicity**: Single budget parameter instead of managing both workflow count and episode limits
4. **Adaptability**: System automatically adjusts to the difficulty of the task

## Example Output

```
Iteration 1
Episode Budget: 0/500 used
...
📊 Episodes used this workflow: 3
   Total episodes used: 3/500

Iteration 2
Episode Budget: 3/500 used
...
📊 Episodes used this workflow: 12
   Total episodes used: 15/500

...

Training Complete!
Total Episodes Used: 498/500
Total Workflows Explored: 23
```

## Git Commit
All changes have been committed to version control with message:
"Replace n_workflows limit with total episode budget"
