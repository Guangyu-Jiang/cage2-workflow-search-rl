# Optimized Default Settings for 100,000 Episode Budget

## Updated Default Configuration

The training script has been optimized for large-scale workflow search with the following new defaults:

### Key Parameters (New Defaults)

| Parameter | Old Default | New Default | Reason |
|-----------|------------|-------------|---------|
| `--n-envs` | 200 | **100** | More stable, allows frequent updates |
| `--total-episodes` | 500 | **100,000** | Extensive exploration budget |
| `--max-episodes` | 50 | **100** | More training per workflow |

### Why These Changes?

1. **100 Parallel Environments**
   - Provides 100 trajectories per update (10,000 transitions)
   - More frequent model updates compared to 200 envs
   - Better gradient estimates with sufficient batch size
   - More stable process creation and management

2. **100,000 Total Episode Budget**
   - Allows exploration of ~1000 workflows (if each uses ~100 episodes)
   - Or ~10,000 workflows if they achieve compliance quickly (~10 episodes each)
   - Sufficient for thorough exploration of the workflow space

3. **100 Max Episodes per Workflow**
   - Gives difficult workflows more time to learn
   - With 100 parallel environments = up to 10,000 trajectories per workflow
   - Allows proper convergence even for challenging priority orderings

## Usage

### Default Run (100K episodes)
```bash
# Just run with defaults - optimized for 100K episode budget
python workflow_rl/parallel_train_workflow_rl.py
```

This will:
- Use 100 parallel environments
- Run for up to 100,000 total episodes
- Allow up to 100 episodes per workflow
- Stop each workflow early if 95% compliance is achieved

### Custom Configurations

```bash
# Quick test run
python workflow_rl/parallel_train_workflow_rl.py \
    --total-episodes 1000 \
    --max-episodes 20

# Even larger budget
python workflow_rl/parallel_train_workflow_rl.py \
    --total-episodes 500000 \
    --max-episodes 200

# Different red agent
python workflow_rl/parallel_train_workflow_rl.py \
    --red-agent bline
```

## Expected Training Behavior

With the new defaults:

1. **Training Speed**: 
   - ~100 episodes/update with 100 environments
   - ~1 update per second (depending on hardware)
   - ~100 updates per workflow maximum

2. **Workflow Exploration**:
   - Fast workflows (achieve compliance in 10 episodes): ~10,000 workflows possible
   - Average workflows (50 episodes): ~2,000 workflows possible  
   - Slow workflows (100 episodes): ~1,000 workflows possible

3. **Memory Requirements**:
   - ~15-20 GB RAM for 100 parallel environments
   - Stable operation without process termination

## Training Strategy

The optimized configuration enables:

1. **More Frequent Updates**: With 100 envs, PPO updates every 10,000 transitions
2. **Better Learning**: Up to 100 episodes allows proper convergence
3. **Extensive Search**: 100,000 episodes enables thorough workflow exploration
4. **Adaptive Allocation**: Episode budget automatically adapts to workflow difficulty

## Monitoring Progress

Watch for these indicators:
- Compliance rate approaching 95%
- Episode usage per workflow (efficient workflows use fewer)
- Total workflows explored
- Best workflow reward improving over time

## Recommended Hardware

- **CPU**: 100+ cores recommended (1 per environment)
- **RAM**: 32 GB minimum
- **GPU**: CUDA-capable GPU for PPO training
- **Storage**: 10+ GB for logs and checkpoints
