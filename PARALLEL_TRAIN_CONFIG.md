# ⚙️ parallel_train_workflow_rl.py Configuration Guide

## How to Specify Red Agent Type

### **Command Line:**

```bash
# Option 1: Moderate difficulty (B_lineAgent)
python workflow_rl/parallel_train_workflow_rl.py --red-agent bline

# Option 2: Aggressive (RedMeanderAgent) - Default
python workflow_rl/parallel_train_workflow_rl.py --red-agent meander

# Option 3: No attacks (SleepAgent)
python workflow_rl/parallel_train_workflow_rl.py --red-agent sleep
```

### **Red Agent Options:**

| Argument | Class | Difficulty | When to Use |
|----------|-------|------------|-------------|
| `--red-agent bline` | B_lineAgent | Moderate | Default training |
| `--red-agent meander` | RedMeanderAgent | High | Production/robust |
| `--red-agent sleep` | SleepAgent | None | Debugging |

---

## Configuration File Location

**Automatically saved to:**
```
logs/exp_TIMESTAMP_PID/experiment_config.json
```

### **Example:**
```
logs/exp_20251017_045342_2400107/
├── experiment_config.json  ← Hyperparameters saved here!
├── training_log.csv
└── gp_sampling_log.csv
```

---

## Configuration File Contents

The `experiment_config.json` includes all hyperparameters:

```json
{
  "experiment_name": "exp_20251017_045342_2400107",
  "pid": 2400107,
  "timestamp": "2025-10-17 04:53:42",
  
  "environment": {
    "n_envs": 25,
    "max_steps": 100,
    "red_agent_type": "B_lineAgent",
    "scenario": "/home/ubuntu/CAGE2/cage-challenge-2/CybORG/..."
  },
  
  "training": {
    "total_episode_budget": 100000,
    "max_train_episodes_per_env": 100,
    "compliance_threshold": 0.95,
    "update_every_steps": 100
  },
  
  "rewards": {
    "alignment_lambda": 30.0
  },
  
  "search": {
    "gp_beta": 2.0
  }
}
```

---

## All Command Line Arguments

### **Environment:**
```bash
--n-envs 100              # Parallel environments (default: 100)
--max-steps 100           # Steps per episode (default: 100)
--red-agent bline         # Red agent type (default: meander)
--total-episodes 100000   # Episode budget (default: 100000)
--max-episodes 100        # Max per workflow (default: 100)
```

### **Training:**
```bash
--alignment-lambda 30.0         # Compliance reward weight (default: 30.0)
--compliance-threshold 0.95     # Target compliance (default: 0.95)
--update-steps 100              # PPO update frequency (default: 100)
```

### **Search:**
```bash
--gp-beta 2.0            # GP-UCB exploration (default: 2.0)
--n-eval-episodes 20     # Eval episodes (default: 20)
```

### **Output:**
```bash
--checkpoint-dir logs    # Where to save (default: compliance_checkpoints)
--seed 42                # Random seed (default: 42)
```

---

## Example Usage

### **Default (Aggressive Red Agent):**
```bash
python workflow_rl/parallel_train_workflow_rl.py
```

### **Moderate Red Agent:**
```bash
python workflow_rl/parallel_train_workflow_rl.py \
    --red-agent bline \
    --n-envs 100 \
    --total-episodes 100000
```

### **High Compliance Focus:**
```bash
python workflow_rl/parallel_train_workflow_rl.py \
    --red-agent meander \
    --alignment-lambda 50.0 \
    --compliance-threshold 0.95
```

### **Quick Test:**
```bash
python workflow_rl/parallel_train_workflow_rl.py \
    --red-agent sleep \
    --n-envs 10 \
    --total-episodes 1000 \
    --max-episodes 50
```

---

## Configuration File Auto-Generated

✅ **Already implemented!**

The configuration file is **automatically saved** when training starts:

1. **Location:** `logs/exp_*/experiment_config.json`
2. **When:** Created at initialization
3. **Contents:** All hyperparameters
4. **Format:** JSON (easy to read/parse)

### **To view:**
```bash
# Find your experiment
ls -lt logs/

# View config
cat logs/exp_TIMESTAMP_PID/experiment_config.json

# Or use jq for pretty printing
jq . logs/exp_*/experiment_config.json | head -30
```

---

## Differences: parallel_train vs executor_async

### **parallel_train_workflow_rl.py** (Synchronous):
```bash
# Red agent via string
--red-agent meander     # 'meander', 'bline', 'sleep'

# Config saved to:
logs/exp_TIMESTAMP_PID/experiment_config.json
```

### **executor_async_train_workflow_rl.py** (Async):
```bash
# Red agent via string
--red-agent RedMeanderAgent  # Full class name

# Config saved to:
logs/exp_executor_async_TIMESTAMP_PID/experiment_config.json
```

---

## Summary

**Red Agent Selection:**
```bash
# For parallel_train_workflow_rl.py
--red-agent bline      # B_lineAgent
--red-agent meander    # RedMeanderAgent (default)
--red-agent sleep      # SleepAgent
```

**Configuration File:**
✅ Already saved automatically to `logs/exp_*/experiment_config.json`  
✅ Includes all hyperparameters  
✅ Created at training start  
✅ JSON format  

No changes needed - it's already working! 🎉
