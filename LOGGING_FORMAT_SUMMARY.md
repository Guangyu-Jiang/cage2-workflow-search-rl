# 📊 Logging Format - Three Reward Components

## CSV Columns

The training log now includes **three separate reward columns**:

### 1. **Env_Reward** (Original Environment Reward)
- The raw reward from CAGE2 environment
- Negative values indicate network compromise
- Example: `-862.32` (many hosts compromised)

### 2. **Alignment_Bonus** (Compliance Reward)
- Reward for following the workflow priority order
- Calculated as: `alignment_lambda × compliance_rate`
- Example: `+16.90` (56% compliance × 30 lambda)

### 3. **Total_Reward** (Customized Reward = Sum)
- The actual reward used for training
- `Total_Reward = Env_Reward + Alignment_Bonus`
- Example: `-862.32 + 16.90 = -845.42`

---

## CSV Structure

```csv
Workflow_ID,Workflow_Order,Update,Episodes,Env_Reward,Alignment_Bonus,Total_Reward,Compliance,Avg_Fixes,Collection_Time,Update_Time
1,user → defender → enterprise → op_server → op_host,1,25,-862.32,16.90,-845.42,0.5633,17.5,6.41,0.84
1,user → defender → enterprise → op_server → op_host,2,50,-717.07,18.04,-699.03,0.6013,17.4,1.89,0.02
```

### Column Descriptions:

| Column | Description | Example |
|--------|-------------|---------|
| **Workflow_ID** | Iteration number | 1 |
| **Workflow_Order** | Priority order being trained | user → defender → ... |
| **Update** | PPO update number | 1, 2, 3... |
| **Episodes** | Total episodes so far | 25, 50, 75... |
| **Env_Reward** | Original environment reward | -862.32 |
| **Alignment_Bonus** | Compliance reward | +16.90 |
| **Total_Reward** | Sum (used for training) | -845.42 |
| **Compliance** | Compliance rate (0-1) | 0.5633 (56.33%) |
| **Avg_Fixes** | Average fixes per episode | 17.5 |
| **Collection_Time** | Episode collection time (s) | 6.41 |
| **Update_Time** | PPO update time (s) | 0.84 |

---

## Example Data Interpretation

### Update 1:
```
Env_Reward: -862.32      ← Environment gave negative reward (bad state)
Alignment_Bonus: +16.90  ← Agent got bonus for 56% workflow compliance
Total_Reward: -845.42    ← Net reward used for training
```

**Interpretation**: The network had many compromised hosts (negative env reward), but the agent followed the workflow order 56% of the time, earning a +16.90 bonus. The total reward of -845.42 is what the PPO algorithm uses to update the policy.

### Update 2 (Improving):
```
Env_Reward: -717.07      ← Better (less negative)
Alignment_Bonus: +18.04  ← Higher bonus (60% compliance)
Total_Reward: -699.03    ← Better total reward
```

**Interpretation**: The agent improved! Network state is better (less negative reward) and compliance increased to 60%, resulting in a higher total reward.

---

## Reward Formula

### Alignment Bonus Calculation:
```python
alignment_bonus = alignment_lambda × compliance_rate
                = 30.0 × 0.5633
                = 16.90
```

### Total Reward:
```python
total_reward = env_reward + alignment_bonus
             = -862.32 + 16.90
             = -845.42
```

### Compliance Rate:
```python
compliance_rate = compliant_fix_actions / total_fix_actions
                = (number of fixes following workflow) / (total fix actions)
```

---

## Why Three Separate Columns?

### 1. **Analyze Environment Performance**
- Track how well agent defends the network
- Monitor if env reward improves over training
- Identify if policy is actually protecting hosts

### 2. **Analyze Compliance**
- See if agent learns to follow workflow priority
- Track alignment bonus changes
- Verify compliance-based reward shaping works

### 3. **Understand Total Optimization**
- See the actual reward PPO optimizes
- Balance between environment success and workflow compliance
- Tune `alignment_lambda` if needed

---

## Visualizing the Data

### Load and analyze:
```python
import pandas as pd

df = pd.read_csv('logs/exp_executor_async_*/training_log.csv')

# Plot environment reward trend
df.plot(x='Episodes', y='Env_Reward')

# Plot alignment bonus trend
df.plot(x='Episodes', y='Alignment_Bonus')

# Plot total reward (what PPO optimizes)
df.plot(x='Episodes', y='Total_Reward')

# Verify sum
assert (df['Env_Reward'] + df['Alignment_Bonus']).equals(df['Total_Reward'])
```

---

## Example Training Progression

| Update | Episodes | Env_Reward | Alignment_Bonus | Total_Reward | Compliance |
|--------|----------|------------|-----------------|--------------|------------|
| 1 | 25 | -862.32 | +16.90 | -845.42 | 56.3% |
| 2 | 50 | -717.07 | +18.04 | -699.03 | 60.1% |
| 3 | 75 | -650.12 | +21.30 | -628.82 | 71.0% |
| 4 | 100 | -580.45 | +25.50 | -554.95 | 85.0% |
| 5 | 125 | -520.30 | +28.50 | -491.80 | 95.0% ✅ |

**Progression shows:**
- Environment reward improving (less negative)
- Alignment bonus increasing (better compliance)
- Total reward improving overall
- Compliance reaching 95% threshold

---

## Comparison to Single Reward Logging

### Old (single reward):
```csv
Workflow_ID,Workflow_Order,Update,Episodes,Avg_Reward,Compliance
1,user → defender → ...,1,25,-845.42,0.5633
```

❌ Can't tell if improvement is from better defense or better compliance

### New (three rewards):
```csv
Workflow_ID,Workflow_Order,Update,Episodes,Env_Reward,Alignment_Bonus,Total_Reward,Compliance
1,user → defender → ...,1,25,-862.32,16.90,-845.42,0.5633
```

✅ Clear breakdown showing both components!

---

## Using the Logged Data

### Check if environment reward is improving:
```bash
# Should see Env_Reward becoming less negative
cut -d',' -f5 training_log.csv | tail -10
```

### Check if compliance is increasing:
```bash
# Should see Alignment_Bonus increasing
cut -d',' -f6 training_log.csv | tail -10
```

### Verify total reward = sum:
```bash
# All three columns together
cut -d',' -f5,6,7 training_log.csv
```

---

## ✅ Summary

The training log now provides complete visibility into:
1. **Original environment reward** - How well the network is defended
2. **Alignment bonus** - How well the workflow is followed
3. **Total reward** - Combined optimization objective

This allows you to:
- Debug training issues
- Tune `alignment_lambda` parameter
- Verify workflow learning
- Analyze training progression
- Compare different workflows

All logging is committed to git and production-ready!
