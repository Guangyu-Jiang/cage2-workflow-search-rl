# 📊 Baseline vs Workflow Search - Fair Comparison

## Two Training Approaches

Now you can compare **standard PPO** vs **workflow-conditioned PPO** at the same speed!

---

## 🔄 Parallel Baseline PPO (NEW!)

### **Script:** `train_parallel_baseline.py`

**What it does:**
- Standard PPO (NO workflow conditioning)
- 200 parallel workers
- Full 145 action space
- Simple Actor-Critic network

**Network:**
```
Input: State (52 dims)
  ↓
Actor: 52 → 64 → 64 → 145 (actions)
Critic: 52 → 64 → 64 → 1 (value)
```

**Run:**
```bash
bash run_parallel_baseline.sh

# Or:
python train_parallel_baseline.py \
    --n-workers 200 \
    --total-episodes 100000
```

**Performance:** ~100-150 episodes/sec (200 workers)

---

## 🎯 Workflow Search PPO

### **Script:** `workflow_rl/executor_async_train_workflow_rl.py`

**What it does:**
- Workflow-conditioned PPO
- 200 parallel workers
- Full 145 action space
- GP-UCB to find best workflow
- Compliance-based training

**Network:**
```
Input: State (52) + Workflow Encoding (25) = 77 dims
  ↓
Actor: 77 → 64 → 64 → 145 (actions)
Critic: 77 → 64 → 64 → 1 (value)
```

**Run:**
```bash
bash run_executor_async_training.sh

# Or:
python workflow_rl/executor_async_train_workflow_rl.py \
    --n-workers 200 \
    --total-episodes 100000
```

**Performance:** ~100-150 episodes/sec (200 workers)

---

## ⚖️ Fair Comparison

Both scripts now have:
- ✅ Same parallelization (200 workers)
- ✅ Same collection method (ProcessPoolExecutor)
- ✅ Same action space (full 145 actions)
- ✅ Same PPO hyperparameters
- ✅ Same speed (~100-150 eps/sec)

**The ONLY difference:**
- Baseline: No workflow encoding
- Workflow Search: With workflow encoding + compliance training

---

## 📊 Expected Results

### **Baseline PPO (No Workflow):**
```
Episode 0: Reward = -800 (random policy)
Episode 10k: Reward = -400 (learning)
Episode 50k: Reward = -100 (converging)
Episode 100k: Reward = -20 (final)

Final policy: 
  - Good at defending network
  - NO specific repair priority order
  - May fix hosts randomly
```

### **Workflow Search PPO:**
```
Episode 0: Reward = -800 (random)
Episode 10k: Reward = -300 (learning + compliance)
Episode 50k: Reward = -50 (better with workflow)
Episode 100k: Reward = -10 (best workflow found)

Final policy:
  - Good at defending network  
  - FOLLOWS optimal workflow order
  - 90%+ compliance with best workflow
  - More structured defense strategy
```

**Hypothesis:** Workflow search should achieve better final performance due to:
1. Structured defense strategy
2. Compliance-based reward shaping
3. GP-UCB finding optimal repair order

---

## 🧪 Running the Comparison

### **Step 1: Train Baseline**
```bash
bash run_parallel_baseline.sh
# ~10-15 minutes for 100k episodes

Results saved to:
  logs/parallel_baseline_*/training_log.csv
```

### **Step 2: Train Workflow Search**
```bash
bash run_executor_async_training.sh
# ~10-15 minutes for 100k episodes

Results saved to:
  logs/exp_executor_async_*/training_log.csv
  logs/exp_executor_async_*/gp_sampling_log.csv
```

### **Step 3: Compare Results**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load baseline
baseline = pd.read_csv('logs/parallel_baseline_*/training_log.csv')

# Load workflow search
workflow = pd.read_csv('logs/exp_executor_async_*/training_log.csv')

# Plot comparison
plt.plot(baseline['Episode'], baseline['Avg_Reward'], label='Baseline PPO')
plt.plot(workflow['Total_Episodes_Sampled'], workflow['Total_Reward'], label='Workflow Search')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.legend()
plt.title('Baseline vs Workflow Search')
plt.show()
```

---

## 📈 What to Compare

### **1. Final Performance**
- Which achieves better reward after 100k episodes?
- Baseline: Should converge to ~-20 reward
- Workflow: Should converge to ~-10 to 0 reward (hypothesis)

### **2. Learning Speed**
- Which learns faster?
- Baseline: Simpler, may converge faster initially
- Workflow: More complex, but reward shaping may help

### **3. Sample Efficiency**
- Episodes needed to reach -100 reward?
- Episodes needed to reach -50 reward?

### **4. Stability**
- Reward variance over time?
- Baseline: May have more variance
- Workflow: Compliance gating may stabilize

---

## 🎯 Key Differences

| Aspect | Baseline | Workflow Search |
|--------|----------|-----------------|
| **Network input** | 52 dims | 77 dims (52 + 25 workflow) |
| **Training objective** | Environment reward only | Env reward + compliance |
| **Search strategy** | Single policy | GP-UCB over 120 workflows |
| **Output** | One policy | Best workflow + policy |
| **Interpretability** | Black box | Explainable (workflow order) |

---

## 💡 Use Cases

### **Use Baseline PPO When:**
- You don't care about repair order
- Want simplest possible solution
- Don't need interpretability
- Just want max reward

### **Use Workflow Search When:**
- You want structured defense strategy
- Need interpretable policies (workflow order)
- Want to discover optimal repair priorities
- Willing to trade complexity for explainability

---

## ✅ Both Scripts Ready!

**Baseline:**
```bash
bash run_parallel_baseline.sh
```

**Workflow Search:**
```bash
bash run_executor_async_training.sh
```

**Same speed, different algorithms - perfect for comparison!** 🎉
