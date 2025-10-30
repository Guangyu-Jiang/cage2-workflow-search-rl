# 🚀 How to Launch All Baseline Trainings

## Complete Training Suite for L4DC Paper

All baselines configured for **100,000 episodes** with full logging.

---

## 📋 Quick Reference

| Baseline | Script | Time (est) | Log Location |
|----------|--------|------------|--------------|
| **Random** | `baselines/random_policy.py` | 30 min | `logs/random_policy_*/` |
| **Greedy** | `baselines/greedy_heuristic.py` | 30 min | `logs/greedy_heuristic_*/` |
| **Fixed Workflows** | `baselines/fixed_priority_workflows.py` | 3 hours | `logs/fixed_workflows_*/` |
| **PPO Baseline** | `train_parallel_baseline.py` | 15-20 min | `logs/parallel_baseline_*/` |
| **SAC Baseline** | `train_parallel_sac.py` | 20-25 min | `logs/parallel_sac_*/` |
| **Fixed Episodes** | `workflow_rl/executor_async_fixed_episodes.py` | 60-90 min | `logs/exp_fixed_episodes_*/` |
| **Your Method** | `workflow_rl/executor_async_train_workflow_rl.py` | 30-40 min | `logs/exp_executor_async_*/` |

---

## 🎯 Launch Commands (100k Episodes Each)

### **1. Random Policy Baseline**

```bash
python baselines/random_policy.py \
    --n-episodes 100000 \
    --red-agent B_lineAgent
```

**Logs:**
- `logs/random_policy_TIMESTAMP/training_log.csv`
- Columns: Episode, Reward, Steps

---

### **2. Greedy Heuristic (Best Fixed Workflow)**

```bash
# Test with best workflow (you can try different orders)
python baselines/greedy_heuristic.py \
    --n-episodes 100000 \
    --workflow defender,enterprise,op_server,op_host,user \
    --red-agent B_lineAgent
```

**Logs:**
- `logs/greedy_heuristic_TIMESTAMP/training_log.csv`
- Columns: Episode, Reward, Steps, Compliance

---

### **3. All Fixed Priority Workflows**

```bash
# Tests 6 workflows × 100k episodes each = 600k total
python baselines/fixed_priority_workflows.py \
    --n-episodes 100000 \
    --red-agent B_lineAgent
```

**Logs:**
- `logs/greedy_heuristic_TIMESTAMP/` (6 directories, one per workflow)
- `logs/fixed_workflows_TIMESTAMP/summary.csv` (comparison)

---

### **4. PPO Baseline (No Workflow)**

```bash
bash run_parallel_baseline.sh

# Or with custom settings:
python train_parallel_baseline.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --episodes-per-update 50 \
    --red-agent B_lineAgent
```

**Logs:**
- `logs/parallel_baseline_TIMESTAMP/training_log.csv`
- `logs/parallel_baseline_TIMESTAMP/experiment_config.json`
- Columns: Episode, Avg_Reward, Std_Reward, Min_Reward, Max_Reward, Collection_Time, Update_Time

---

### **5. SAC Baseline (No Workflow)**

```bash
bash run_parallel_sac.sh

# Or with custom settings:
python train_parallel_sac.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --episodes-per-update 50 \
    --batch-size 256 \
    --red-agent B_lineAgent
```

**Logs:**
- `logs/parallel_sac_TIMESTAMP/training_log.csv`
- `logs/parallel_sac_TIMESTAMP/experiment_config.json`
- Columns: Episode, Avg_Reward, Std_Reward, Min_Reward, Max_Reward, Collection_Time, Update_Time

---

### **6. Fixed-Episodes Training (Ablation)**

```bash
bash run_fixed_episodes_training.sh

# Or with custom settings:
python workflow_rl/executor_async_fixed_episodes.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --fixed-episodes-per-workflow 2500 \
    --episodes-per-update 50 \
    --alignment-lambda 30.0 \
    --red-agent B_lineAgent
```

**Logs:**
- `logs/exp_fixed_episodes_TIMESTAMP/training_log.csv`
- `logs/exp_fixed_episodes_TIMESTAMP/gp_sampling_log.csv`
- `logs/exp_fixed_episodes_TIMESTAMP/experiment_config.json`

---

### **7. Your Method (Adaptive Termination)**

```bash
bash run_executor_async_training.sh

# Or with custom settings:
python workflow_rl/executor_async_train_workflow_rl.py \
    --n-workers 200 \
    --total-episodes 100000 \
    --max-episodes-per-workflow 10000 \
    --episodes-per-update 200 \
    --alignment-lambda 30.0 \
    --compliance-threshold 0.90 \
    --red-agent B_lineAgent
```

**Logs:**
- `logs/exp_executor_async_TIMESTAMP/training_log.csv`
- `logs/exp_executor_async_TIMESTAMP/gp_sampling_log.csv`
- `logs/exp_executor_async_TIMESTAMP/experiment_config.json`

---

## 🔄 Run All at Once (Sequential)

Create a master script:

```bash
#!/bin/bash
# run_all_experiments.sh

echo "🚀 Running All Baselines for L4DC Paper"
echo "========================================"
echo ""

# 1. Random (fast)
echo "1/7 - Random Policy..."
python baselines/random_policy.py --n-episodes 100000 --red-agent B_lineAgent

# 2. Greedy (fast)
echo "2/7 - Greedy Heuristic..."
python baselines/greedy_heuristic.py --n-episodes 100000 \
    --workflow defender,enterprise,op_server,op_host,user \
    --red-agent B_lineAgent

# 3. PPO Baseline
echo "3/7 - PPO Baseline..."
python train_parallel_baseline.py --n-workers 50 --total-episodes 100000 \
    --episodes-per-update 50 --red-agent B_lineAgent

# 4. SAC Baseline
echo "4/7 - SAC Baseline..."
python train_parallel_sac.py --n-workers 50 --total-episodes 100000 \
    --episodes-per-update 50 --red-agent B_lineAgent

# 5. Fixed Episodes
echo "5/7 - Fixed Episodes..."
python workflow_rl/executor_async_fixed_episodes.py --n-workers 50 \
    --total-episodes 100000 --fixed-episodes-per-workflow 2500 \
    --episodes-per-update 50 --red-agent B_lineAgent

# 6. Your Method
echo "6/7 - Adaptive Termination (Your Method)..."
python workflow_rl/executor_async_train_workflow_rl.py --n-workers 200 \
    --total-episodes 100000 --episodes-per-update 200 --red-agent B_lineAgent

echo ""
echo "✅ All baselines complete!"
echo ""
echo "Results in logs/:"
ls -lth logs/ | head -10
```

---

## 📊 Log File Verification

### **All RL Baselines Save:**

1. **training_log.csv** - Episode-by-episode or update-by-update metrics
2. **experiment_config.json** - All hyperparameters
3. **Checkpoints** - Saved model weights

### **CSV Formats:**

**PPO/SAC Baseline:**
```csv
Episode,Avg_Reward,Std_Reward,Min_Reward,Max_Reward,Collection_Time,Update_Time
100,-750.23,245.12,-1100.50,-200.30,5.2,0.8
200,-680.45,220.34,-1050.20,-180.10,5.1,0.1
...
```

**Fixed-Episodes / Your Method:**
```csv
Workflow_ID,Workflow_Order,Update,Episodes,Total_Episodes_Sampled,Env_Reward,Alignment_Bonus,Total_Reward,Compliance,...
1,defender → enterprise → ...,1,50,50,-650.30,15.20,-635.10,0.3820,...
1,defender → enterprise → ...,2,100,100,-580.12,18.40,-561.72,0.4560,...
...
```

---

## ⚙️ Configuration Files

All RL methods save `experiment_config.json`:

```json
{
  "experiment_name": "parallel_sac_20251030_182350",
  "timestamp": "2025-10-30 18:23:50",
  "algorithm": "Parallel SAC",
  "environment": {
    "n_workers": 50,
    "red_agent_type": "B_lineAgent"
  },
  "training": {
    "total_episodes": 100000,
    "episodes_per_update": 50
  },
  "hyperparameters": {...}
}
```

---

## ✅ Verification Checklist

Before running, verify:

- [ ] All scripts have `--n-episodes` or `--total-episodes` = 100000
- [ ] All scripts save to `logs/` directory
- [ ] All scripts save `training_log.csv`
- [ ] RL methods save `experiment_config.json`
- [ ] Red agent set to B_lineAgent (consistent across all)

---

## 🎓 L4DC Paper Baselines

**You now have:**

✅ **3 Non-learning** (Random, Greedy, Fixed)  
✅ **2 RL algorithms** (PPO, SAC)  
✅ **2 Ablations** (Fixed-episodes, Your method)  

**This satisfies L4DC requirements!**

---

## 📁 All Launch Scripts:

```bash
run_parallel_baseline.sh       # PPO baseline
run_parallel_sac.sh            # SAC baseline  
run_fixed_episodes_training.sh # Fixed-episodes ablation
run_executor_async_training.sh # Your method (adaptive)
baselines/run_all_baselines.sh # All non-learning baselines
```

---

**Ready to run all experiments for your L4DC paper!** 🎓

All changes pushed to GitHub! ✅
