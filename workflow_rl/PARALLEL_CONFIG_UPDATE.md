# Parallel Training Configuration Update

## Changes Made (2024-10-16)

Updated default training configuration for more efficient parallel training:

### 1. **Parallel Environments: 25 → 200**
   - Increased from 25 to 200 parallel environments
   - Dramatically increases data collection efficiency
   - Each update now uses 200 trajectories (20,000 transitions)

### 2. **Max Updates per Workflow: 400 → 50**
   - Reduced from 400 to 50 episodes per environment
   - Prevents overfitting to single workflows
   - Encourages faster exploration of workflow space

### 3. **Min Updates Before Early Stop: 10 → 5**
   - Reduced from 10 to 5 episodes minimum
   - Allows faster convergence for easy workflows
   - Still ensures meaningful training before compliance check

### 4. **Update Frequency: Unchanged**
   - Kept at 100 steps (1 full episode) per environment
   - Updates PPO after collecting one trajectory from each parallel environment
   - This means 200 trajectories per update with new configuration

## Impact on Training

### Data Efficiency:
- **Before**: 25 envs × 100 steps = 2,500 transitions per update
- **After**: 200 envs × 100 steps = 20,000 transitions per update
- **8x more data per update!**

### Training Speed:
- Faster convergence due to more diverse data per update
- Reduced maximum training time per workflow
- Earlier stopping for successful workflows

### Computational Requirements:
- Higher GPU/CPU usage during environment simulation
- More memory required for parallel environments
- But fewer total updates needed due to better data efficiency

## Usage Examples

### Default Settings (New):
```bash
python workflow_rl/parallel_train_workflow_rl.py
# Uses: 200 envs, 50 max episodes, 5 min episodes
```

### Custom Settings:
```bash
# For limited resources:
python workflow_rl/parallel_train_workflow_rl.py \
    --n-envs 50 \
    --max-episodes 100 \
    --min-episodes 10

# For aggressive exploration:
python workflow_rl/parallel_train_workflow_rl.py \
    --n-envs 500 \
    --max-episodes 20 \
    --min-episodes 3
```

## Expected Benefits

1. **Faster Training**: 8x more data per update should lead to faster convergence
2. **Better Exploration**: More diverse experiences from 200 parallel trajectories
3. **Reduced Overfitting**: Shorter per-workflow training prevents overfitting
4. **Quicker Iteration**: Faster workflow exploration in GP-UCB search

## Monitoring

Watch for:
- GPU/CPU utilization (should be higher)
- Memory usage (will increase with more environments)
- Convergence speed (should be faster)
- Final performance (should be similar or better)
