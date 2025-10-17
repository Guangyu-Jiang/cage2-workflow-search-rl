# 🎮 Red Agent Selection Guide

## Available Red Agents

The training system supports **three different red agent types**, each with different attack behaviors and difficulty levels:

### 1. **B_lineAgent** (Default, Moderate Difficulty)
**Description**: Baseline red agent with structured attack pattern

**Characteristics:**
- Follows a predefined attack sequence
- Moderate aggressiveness
- Predictable attack patterns
- Good for initial training

**When to use:**
- Default training runs
- Benchmarking
- Testing new features

### 2. **RedMeanderAgent** (High Difficulty)
**Description**: Aggressive red agent with exploratory behavior

**Characteristics:**
- More aggressive attacks
- Explores network randomly ("meanders")
- Less predictable
- Harder to defend against

**When to use:**
- Advanced training
- Testing robustness
- Real-world scenarios (unpredictable attackers)

### 3. **SleepAgent** (Easy/No Attacks)
**Description**: Passive red agent that doesn't attack

**Characteristics:**
- No attacks
- Environment remains clean
- Useful for debugging
- Minimal difficulty

**When to use:**
- Debugging workflows
- Testing compliance tracking without interference
- Baseline performance measurement

---

## How to Select Red Agents

### **Command Line:**

```bash
# B_lineAgent (default - moderate)
python workflow_rl/executor_async_train_workflow_rl.py --red-agent B_lineAgent

# RedMeanderAgent (hard - aggressive)
python workflow_rl/executor_async_train_workflow_rl.py --red-agent RedMeanderAgent

# SleepAgent (easy - no attacks)
python workflow_rl/executor_async_train_workflow_rl.py --red-agent SleepAgent
```

### **Using Shell Scripts:**

```bash
# Edit run_executor_async_training.sh
# Change this line:
    --red-agent B_lineAgent \
# To:
    --red-agent RedMeanderAgent \
```

### **Full Examples:**

```bash
# Train against baseline attacker
python workflow_rl/executor_async_train_workflow_rl.py \
    --n-workers 100 \
    --total-episodes 100000 \
    --red-agent B_lineAgent

# Train against aggressive attacker
python workflow_rl/executor_async_train_workflow_rl.py \
    --n-workers 100 \
    --total-episodes 100000 \
    --red-agent RedMeanderAgent

# Debug mode with no attacks
python workflow_rl/executor_async_train_workflow_rl.py \
    --n-workers 10 \
    --total-episodes 1000 \
    --red-agent SleepAgent
```

---

## Red Agent Comparison

| Agent | Difficulty | Attack Pattern | Episodes to 95% Compliance | Best For |
|-------|------------|----------------|----------------------------|----------|
| **SleepAgent** | None | No attacks | Quick | Debugging |
| **B_lineAgent** | Moderate | Structured | ~100-300 | Training |
| **RedMeanderAgent** | High | Random/exploratory | ~300-500 | Production |

---

## Impact on Training

### Training Speed:
All agents have similar environment step times, so **performance is the same** regardless of red agent choice (~74 eps/sec with 100 workers).

### Compliance Achievement:
- **SleepAgent**: Reaches 95% very quickly (no attacks to worry about)
- **B_lineAgent**: Moderate time to 95% compliance
- **RedMeanderAgent**: Takes longer to reach 95% (more challenging)

### Final Policy Quality:
- **SleepAgent**: Policy may not generalize (too easy)
- **B_lineAgent**: Good generalization (recommended)
- **RedMeanderAgent**: Best generalization (most robust)

---

## Recommended Usage

### For Development/Testing:
```bash
--red-agent B_lineAgent  # Fast training, good results
```

### For Production/Deployment:
```bash
--red-agent RedMeanderAgent  # Robust policy, handles unpredictable attacks
```

### For Debugging:
```bash
--red-agent SleepAgent  # No interference, test pure workflow logic
```

---

## Training Different Agents Separately

You can train separate policies for each red agent type:

```bash
# Train against B_lineAgent
python workflow_rl/executor_async_train_workflow_rl.py \
    --red-agent B_lineAgent \
    --total-episodes 100000

# Train against RedMeanderAgent (separate run)
python workflow_rl/executor_async_train_workflow_rl.py \
    --red-agent RedMeanderAgent \
    --total-episodes 100000

# Compare results in logs/exp_*/ directories
```

---

## Agent Implementation Details

From the CAGE2 codebase:

### **B_lineAgent** (`CybORG.Agents.SimpleAgents.BaseAgent`)
- Baseline attack agent
- Follows predefined attack steps
- Consistent behavior across episodes

### **RedMeanderAgent** (`CybORG.Agents.SimpleAgents.Meander`)
- Meanders through network
- Exploratory attack strategy
- More realistic attacker model

### **SleepAgent** (`CybORG.Agents.SleepAgent`)
- Does nothing (Sleep action)
- Environment stays static
- Useful for testing

---

## Quick Reference

| Command | Agent Type | Description |
|---------|------------|-------------|
| `--red-agent B_lineAgent` | Moderate | **Default, recommended** |
| `--red-agent RedMeanderAgent` | Hard | Aggressive, best for robustness |
| `--red-agent SleepAgent` | Easy | No attacks, debugging only |

---

## Examples in All Training Scripts

All training scripts support the same `--red-agent` parameter:

```bash
# Synchronous parallel
python workflow_rl/parallel_train_workflow_rl.py --red-agent RedMeanderAgent

# Executor async (recommended!)
python workflow_rl/executor_async_train_workflow_rl.py --red-agent RedMeanderAgent

# Sequential
python workflow_rl/sequential_train_workflow_rl.py --red-agent B_lineAgent
```

---

## 🎯 Recommendation

**For production training:**
```bash
python workflow_rl/executor_async_train_workflow_rl.py \
    --red-agent RedMeanderAgent \
    --n-workers 100 \
    --total-episodes 100000
```

This trains against the most challenging red agent, resulting in the most robust defense policy!
