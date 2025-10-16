# Baseline PPO vs Workflow Search PPO Comparison

## 📊 Key Training Differences

### 1. Action Space

| Method | Action Space Size | Details |
|--------|------------------|---------|
| **Baseline PPO (train.py)** | **22 actions** | Hand-selected subset: restore, analyze, remove actions for specific hosts |
| **Baseline PPO Full (train_no_action_reduction.py)** | **145 actions** | Full action space (0-144) |
| **Workflow Search PPO** | **145 actions** | Full action space with workflow conditioning |

**Impact**: The original baseline has a **6.6x advantage** by using only 22 carefully selected actions vs the full 145 action space.

### 2. Training Data Collection

| Method | Data Collection | Updates |
|--------|----------------|---------|
| **Baseline PPO** | 20,000 steps (200 episodes) | Update after 200 episodes |
| **Workflow Search PPO** | 2,500 steps (100 episodes with 25 envs) | Update every 100 steps (4 per env) |

**Key Differences**:
- Baseline collects **8x more data** before each update (20,000 vs 2,500 steps)
- Baseline uses single environment vs 25 parallel environments
- Workflow search updates more frequently (every 100 steps vs 20,000 steps)

### 3. PPO Update Parameters

| Parameter | Baseline PPO | Workflow Search PPO |
|-----------|--------------|-------------------|
| K_epochs | 6 | 4 |
| eps_clip | 0.2 | 0.2 |
| gamma | 0.99 | 0.99 |
| lr | 0.002 | 0.002 |
| Update frequency | Every 200 episodes | Every 4 steps/env |

### 4. Environment Configuration

| Aspect | Baseline PPO | Workflow Search PPO |
|--------|--------------|-------------------|
| Red Agent | B_lineAgent (default) | RedMeanderAgent (default) |
| Max Steps | 100 | 100 |
| Parallel Envs | 1 | 25 |
| Episode Limit | 100,000 | 400 per env per workflow |

### 5. Additional Features

| Feature | Baseline PPO | Workflow Search PPO |
|---------|--------------|-------------------|
| Workflow Conditioning | ❌ No | ✅ Yes |
| Compliance Rewards | ❌ No | ✅ Yes (λ=30) |
| GP-UCB Search | ❌ No | ✅ Yes |
| Policy Inheritance | ❌ No | ✅ Yes |
| Action Masking | ❌ No | ❌ No (could add) |

---

## ⚖️ Fair Comparison Analysis

### **Current Unfair Advantages for Baseline:**

1. **Action Space Reduction (6.6x advantage)**
   - Baseline uses only 22 hand-picked actions
   - Workflow search uses full 145 actions
   - **Solution**: Use `train_no_action_reduction.py` for fair comparison

2. **More Data Per Update (8x advantage)**
   - Baseline: 20,000 steps before update
   - Workflow search: 2,500 steps before update
   - **Solution**: Either reduce baseline buffer or increase workflow search buffer

3. **More PPO Epochs (1.5x advantage)**
   - Baseline: K_epochs = 6
   - Workflow search: K_epochs = 4
   - **Solution**: Match K_epochs parameter

### **Advantages for Workflow Search:**

1. **Parallel Environments (25x advantage)**
   - More diverse experience collection
   - Faster wall-clock training time
   - Better exploration

2. **Workflow Guidance**
   - Additional conditioning information
   - Compliance-based reward shaping
   - Not directly comparable to baseline

3. **Policy Inheritance**
   - Transfer learning between workflows
   - Potentially faster convergence

---

## 🔧 Recommended Fair Comparison Setup

### Option 1: Minimal Changes (Most Fair)

**Baseline PPO (Modified)**:
```python
# Use train_no_action_reduction.py with:
action_space = list(range(145))  # Full action space
update_timesteps = 2500          # Match workflow search
K_epochs = 4                     # Match workflow search
red_agent = RedMeanderAgent      # Match workflow search
```

**Workflow Search PPO**:
```python
# Keep current settings:
n_envs = 25
max_episodes = 400
K_epochs = 4
update_every_steps = 100
```

### Option 2: Scale Baseline to Match Parallel Training

**Baseline PPO with Parallel Envs**:
- Implement parallel environment wrapper
- Use 25 environments
- Update every 100 steps (2500 total)
- Keep full action space

### Option 3: Isolate Workflow Benefit

Run three experiments:
1. **Baseline PPO**: Full action space, no workflow
2. **Random Workflow PPO**: Full action space, random workflow each episode
3. **GP-UCB Workflow PPO**: Full action space, GP-UCB selected workflows

---

## 📈 Metrics for Fair Comparison

### Primary Metrics:
1. **Final Performance**: Average reward over last 100 episodes
2. **Sample Efficiency**: Episodes to reach performance threshold
3. **Convergence Speed**: Time to stable performance
4. **Best Episode Score**: Maximum achieved reward

### Normalized Metrics:
- **Reward per Action**: Account for action space size difference
- **Reward per Update**: Account for update frequency difference
- **Reward per Sample**: Account for data efficiency

---

## 🚀 Running Fair Experiments

### 1. Baseline with Full Action Space:
```bash
python train_no_action_reduction.py --red-agent meander --episodes 10000
```

### 2. Workflow Search:
```bash
python workflow_rl/parallel_train_workflow_rl.py \
    --red-agent meander \
    --n-workflows 20 \
    --max-episodes 400
```

### 3. Analysis:
```python
# Compare learning curves
import pandas as pd
import matplotlib.pyplot as plt

# Load baseline results
baseline = pd.read_csv('Models/baseline_ppo_full_action_*/training_log.csv')

# Load workflow search results  
workflow = pd.read_csv('logs/exp_*/training_log.csv')

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(baseline['Episode'], baseline['Reward'].rolling(100).mean(), 
         label='Baseline PPO (Full Action)')
plt.plot(workflow['Episode'], workflow['Env_Reward'].rolling(100).mean(),
         label='Workflow Search PPO')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.legend()
plt.title('Fair Comparison: Baseline vs Workflow Search')
plt.show()
```

---

## 📝 Summary

The original baseline PPO (`train.py`) has several unfair advantages:
1. **6.6x smaller action space** (22 vs 145 actions)
2. **8x more data before updates** (20,000 vs 2,500 steps)  
3. **1.5x more PPO optimization** (K=6 vs K=4)

For fair comparison, use:
- `train_no_action_reduction.py` (full 145 actions)
- Match hyperparameters where possible
- Consider parallel environments for baseline
- Compare both sample efficiency and final performance

The workflow search approach should be evaluated on:
1. Whether it can match baseline with full action space
2. How much the workflow conditioning helps
3. Sample efficiency gains from policy inheritance
4. Benefits of GP-UCB exploration vs random
