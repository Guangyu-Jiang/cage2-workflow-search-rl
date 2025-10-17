# Training Acceleration Guide

## Current Bottlenecks
- Synchronous barrier: waiting for slowest environment
- Python GIL limitations even with shared memory
- Logging overhead with detailed CSV writes
- No GPU utilization
- Environment IPC overhead with shared memory

## 🎯 Quick Wins (Implement First)

### 1. **Switch to Vectorized Environments** (2.2x speedup)
```python
# Replace ParallelEnvSharedMemory with VectorizedCAGE2Envs
from workflow_rl.parallel_env_vectorized import VectorizedCAGE2Envs
envs = VectorizedCAGE2Envs(n_envs=100, scenario_path=..., red_agent_type=...)
```
Your benchmarks showed vectorized envs are 2.2x faster for pure stepping.

### 2. **Async Environment Stepping** (20-30% speedup)
Replace synchronous `step_all()` with async collection:
```python
# Instead of waiting for all envs to complete
results = envs.step_all(actions)

# Use async pattern - don't wait for slowest
ready_envs = envs.step_async(actions)
# Process ready ones immediately
```

### 3. **Reduce Update Frequency** (10-15% speedup)
```bash
# Update every 50 steps instead of 100
python workflow_rl/parallel_train_workflow_rl.py --update-steps 50
```
More frequent, smaller updates can converge faster.

## 🔧 Medium Effort Optimizations

### 4. **Keep K_epochs=4 for Stability**
```python
# Maintain K_epochs=4 for better convergence
K_epochs=4  # Better sample efficiency and stability
```
While K_epochs=2 would be faster, keeping it at 4 ensures more stable convergence.

### 5. **Batch Logging** (5-10% speedup)
Instead of writing to CSV every episode:
```python
# Buffer logs in memory
self.log_buffer = []
# Write in batches of 100
if len(self.log_buffer) >= 100:
    self.consolidated_csv_writer.writerows(self.log_buffer)
    self.log_buffer = []
```

### 6. **GPU Acceleration** (30-50% speedup if available)
```python
# Check GPU availability
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Move models to GPU in PPO agent
```

## 🚀 Advanced Optimizations

### 7. **JIT Compilation with TorchScript** (15-25% speedup)
```python
# Compile the policy network
self.policy = torch.jit.script(self.policy)
```

### 8. **Smaller Neural Networks** (10-20% speedup)
```python
# Reduce hidden layers from [64, 64] to [32, 32]
# Or use single layer [128]
```

### 9. **Experience Replay Buffer** (Better sample efficiency)
Instead of discarding old data, keep a replay buffer:
```python
# Reuse past experiences for multiple updates
self.replay_buffer = deque(maxlen=10000)
```

### 10. **Distributed Training** (Linear scaling)
Use multiple machines/processes for different workflows:
```python
# Run workflows 0-4 on process 1
# Run workflows 5-9 on process 2
# Aggregate results
```

## 📊 Recommended Configuration

For immediate 2-3x speedup, apply these changes:

```bash
# Optimal settings for speed
python workflow_rl/parallel_train_workflow_rl.py \
    --n-envs 200 \
    --update-steps 50 \
    --max-episodes 50 \
    --total-episodes 100000
```

And modify the code:
1. Use VectorizedCAGE2Envs (2.2x speedup)
2. Implement batch logging (10% I/O reduction)
3. More frequent updates (every 50 steps)
4. Keep K_epochs=4 for stability

## 🔬 Profiling to Find Bottlenecks

```python
import cProfile
import pstats

# Profile your training
profiler = cProfile.Profile()
profiler.enable()
# ... training code ...
profiler.disable()

# Show bottlenecks
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime').print_stats(20)
```

## 💡 Most Impactful Change

**Switch to VectorizedCAGE2Envs with K_epochs=4**
This will give you ~2-2.5x speedup:
- Vectorized: 2.2x faster environment stepping
- Batch logging: 10% I/O reduction
- More frequent updates: Better responsiveness
- Combined: ~2-2.5x overall speedup while maintaining stability
