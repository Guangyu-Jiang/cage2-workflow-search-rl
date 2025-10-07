# Original train.py vs Workflow-Conditioned Training

## Original train.py Approach

### **Large Batch Collection Before Update**

The original train.py uses a **much larger batch size**:

```python
# Key parameters from train.py
update_timesteps = 20000  # Update after 20,000 steps!
max_timesteps = 100      # 100 steps per episode
K_epochs = 6              # More epochs than our 4

# This means:
# 20,000 steps ÷ 100 steps/episode = 200 episodes before update
```

### **Training Loop Structure**

```python
def train(env, ...):
    time_step = 0
    
    for i_episode in range(1, max_episodes + 1):  # 100,000 episodes
        state = env.reset()
        
        for t in range(max_timesteps):  # 100 steps
            time_step += 1
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            agent.store(reward, done)  # Just store, don't update
            
            # Update only after 20,000 steps collected
            if time_step % update_timestep == 0:  # Every 20,000 steps
                agent.train()  # PPO update on huge batch
                agent.clear_memory()
                time_step = 0
            
        agent.end_episode()  # Reset decoys, scan state
```

### **Batch Size Comparison**

| Aspect | Original train.py | Our Workflow Training |
|--------|------------------|----------------------|
| **Update Frequency** | Every 20,000 steps | Every episode (~50 steps) |
| **Batch Size** | 20,000 transitions | 30-80 transitions |
| **Episodes per Batch** | ~200 episodes | 1 episode |
| **K Epochs** | 6 | 4 |
| **Memory Usage** | Huge buffer (20k items) | Small buffer (1 episode) |
| **Parallel Envs** | No (still sequential) | No (sequential) |

## Key Differences

### 1. **Batch Size Impact**

**Original (20,000 steps):**
- **Pros:**
  - Very stable gradients (large batch)
  - Better sample efficiency
  - Closer to on-policy (all recent)
- **Cons:**
  - Slow initial learning (200 episodes before first update)
  - High memory usage
  - Delayed feedback

**Ours (1 episode):**
- **Pros:**
  - Immediate feedback after each episode
  - Low memory usage
  - Natural episode boundaries
  - Better for workflow compliance tracking
- **Cons:**
  - Higher gradient variance
  - Less sample efficient
  - More frequent updates (computational cost)

### 2. **Learning Dynamics**

**Original train.py:**
```
Episodes 1-200:    Collect data, no learning
                   → First update at step 20,000
Episodes 201-400:  Collect data, no learning  
                   → Second update at step 40,000
...
```

**Our Workflow Training:**
```
Episode 1:  Collect → Update immediately
Episode 2:  Collect → Update immediately
Episode 3:  Collect → Update immediately
...
```

### 3. **Memory Management**

**Original:**
```python
class Memory:
    def __init__(self):
        # Can hold 20,000 transitions
        self.states = []      # Size: [20,000, state_dim]
        self.actions = []     # Size: [20,000]
        self.rewards = []     # Size: [20,000]
        self.logprobs = []    # Size: [20,000]
        self.is_terminals = [] # Size: [20,000]
```

**Ours:**
```python
class Memory:
    def __init__(self):
        # Holds 1 episode
        self.states = []      # Size: [~50, state_dim]
        self.actions = []     # Size: [~50]
        self.rewards = []     # Size: [~50]
        self.logprobs = []    # Size: [~50]
        self.is_terminals = [] # Size: [~50]
```

## Why Original Uses Large Batch

### 1. **No Workflow Conditioning**
- Training a single, general policy
- Needs diverse experience for stability
- Not comparing multiple strategies

### 2. **Long Training Horizon**
- 100,000 episodes total
- Can afford to wait for updates
- Optimizing for final performance

### 3. **Standard PPO Practice**
- Large batches are typical for PPO
- Better approximation of policy gradient
- More stable learning

## Why We Use Small Batch

### 1. **Workflow Search Context**
- Training 20 different workflows
- Only 50 episodes per workflow
- Need quick adaptation

### 2. **Compliance Tracking**
- Episode-level compliance metrics
- Natural to update after episode
- Immediate feedback on workflow adherence

### 3. **Limited Budget**
- 50 episodes × 20 workflows = 1,000 total
- Can't wait 200 episodes for first update
- Need to learn quickly

## Performance Implications

### Original Approach Would Fail for Workflow Search:
```python
# If we used update_timesteps = 20,000:
# With 50 episodes × 100 steps = 5,000 steps per workflow
# We'd NEVER update! (5,000 < 20,000)
# Agent would never learn the workflow
```

### Our Approach is Necessary Because:
1. **Short training per workflow** (50 episodes)
2. **Need rapid adaptation** to workflow
3. **Workflow compliance** requires immediate feedback
4. **GP-UCB** needs consistent evaluation

## Hybrid Approach (Potential Improvement)

```python
# Collect multiple episodes before update
EPISODES_PER_UPDATE = 5  # Middle ground

for batch in range(train_episodes // EPISODES_PER_UPDATE):
    # Collect 5 episodes
    for episode in range(EPISODES_PER_UPDATE):
        collect_episode()
    
    # Update on 5 episodes (~250-400 transitions)
    agent.train()
    agent.clear_memory()
```

This would provide:
- Larger batch (250-400 vs 50-80)
- Still responsive (update every 5 episodes)
- Better gradient estimates
- Maintains episode boundaries

## Summary

The original train.py uses a **massive batch size of 20,000 steps** (200 episodes) before each update, optimized for long-term single-policy training. Our workflow-conditioned approach uses **episode-level batches** (30-80 steps) for rapid adaptation to different workflows within a limited 50-episode budget. This difference is necessary due to the fundamentally different training objectives: single optimal policy vs. workflow comparison.
