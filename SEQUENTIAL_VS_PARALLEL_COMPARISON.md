# Sequential vs Parallel Training Comparison

## Overview

Two different approaches for collecting training data in workflow-conditioned RL:

### 1. **Parallel Training** (`parallel_train_workflow_rl.py`)
- **100 parallel environments** running simultaneously
- Collect 1 episode from each environment (100 total) before PPO update
- Faster data collection through parallelization
- Higher memory usage

### 2. **Sequential Training** (`sequential_train_workflow_rl.py`)
- **1 single environment**
- Collect 100 episodes sequentially before PPO update
- Slower data collection but lower memory usage
- Simpler implementation

## Key Differences

| Aspect | Parallel (100 envs) | Sequential (1 env) |
|--------|-------------------|-------------------|
| **Environments** | 100 parallel | 1 sequential |
| **Episodes per update** | 100 (1 per env) | 100 (all from 1 env) |
| **Data collection** | Simultaneous | One at a time |
| **Memory usage** | ~100x higher | Minimal |
| **Speed** | Fast (~100-200 eps/sec) | Slower (~10-30 eps/sec) |
| **CPU usage** | High (100 processes) | Low (1 process) |
| **Implementation** | Complex (IPC) | Simple |
| **Trajectory diversity** | High (100 seeds) | Low (1 seed per update) |

## Mathematical Equivalence

Both approaches collect 100 episodes before each PPO update:
- **Parallel**: 100 envs × 1 episode = 100 episodes
- **Sequential**: 1 env × 100 episodes = 100 episodes

The PPO update sees the same amount of data (10,000 transitions with 100-step episodes).

## Advantages & Disadvantages

### Parallel Approach
**Pros:**
- ✅ Much faster training (5-10x speedup)
- ✅ Better exploration (different random seeds)
- ✅ More stable gradients (diverse trajectories)
- ✅ Efficient hardware utilization

**Cons:**
- ❌ High memory usage
- ❌ Complex implementation
- ❌ IPC overhead
- ❌ Can overwhelm system

### Sequential Approach
**Pros:**
- ✅ Low memory footprint
- ✅ Simple to implement and debug
- ✅ Works on resource-constrained systems
- ✅ No IPC overhead

**Cons:**
- ❌ Much slower training
- ❌ Less trajectory diversity
- ❌ Poor hardware utilization
- ❌ May need more updates to converge

## When to Use Which?

### Use **Parallel** when:
- You have sufficient RAM (>16GB)
- Speed is critical
- You have many CPU cores
- Training complex policies

### Use **Sequential** when:
- Limited memory (<8GB RAM)
- Debugging or prototyping
- Running on laptop/edge device
- Simple environments

## Performance Expectations

### Parallel (100 environments):
```
Episodes/second: 100-200
Time for 100k episodes: ~10-15 minutes
Memory usage: ~5-10 GB
CPU usage: 100% on all cores
```

### Sequential (1 environment):
```
Episodes/second: 10-30
Time for 100k episodes: ~60-90 minutes
Memory usage: ~500 MB
CPU usage: ~100% on 1 core
```

## Running Each Version

### Parallel:
```bash
python workflow_rl/parallel_train_workflow_rl.py \
    --n-envs 100 \
    --total-episodes 100000
```

### Sequential:
```bash
python workflow_rl/sequential_train_workflow_rl.py \
    --episodes-per-update 100 \
    --total-episodes 100000
```

Or use the convenience scripts:
```bash
./run_training_safe.sh      # Parallel
./run_sequential_training.sh # Sequential
```

## Theoretical Considerations

### Sample Efficiency
- **Parallel**: Better due to trajectory diversity
- **Sequential**: May need more updates due to correlated samples

### Convergence
- Both should converge to similar performance
- Parallel typically converges faster (in wall-clock time)
- Sequential may be more stable (less noisy gradients)

### Exploration
- **Parallel**: Natural exploration through different seeds
- **Sequential**: Relies more on policy entropy

## Recommendation

For most use cases, **parallel training is recommended** due to:
1. 5-10x faster training
2. Better exploration
3. More stable learning

Use sequential only when:
1. System resources are limited
2. Debugging is needed
3. Simplicity is preferred over speed
