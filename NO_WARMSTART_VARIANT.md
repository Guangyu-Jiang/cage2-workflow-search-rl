# Fixed-Episodes Training Variants

You now have **TWO versions** of fixed-episodes training to test different aspects:

---

## 1. WITH Warm Starting (Original)

**File:** `workflow_rl/executor_async_fixed_episodes.py`

**Features:**
- ✅ Warm starting enabled
- ✅ Reuses policy from previously trained workflows
- ✅ Compliance rewards
- ❌ NO early stopping

**How it works:**
```
Workflow 1: Train from scratch (random init) → Policy_1
Workflow 2: Start from Policy_1 → Fine-tune → Policy_2  
Workflow 3: Start from closest (Policy_1 or Policy_2) → Policy_3
```

**Run:**
```bash
bash run_fixed_episodes_training.sh

# Or:
python workflow_rl/executor_async_fixed_episodes.py \
    --n-workers 25 \
    --fixed-episodes-per-workflow 2500
```

**Logs:** `logs/exp_fixed_episodes_*/`

---

## 2. WITHOUT Warm Starting (NEW!)

**File:** `workflow_rl/executor_async_fixed_episodes_no_warmstart.py`

**Features:**
- ❌ NO warm starting
- ❌ NO policy reuse between workflows
- ✅ Compliance rewards
- ❌ NO early stopping

**How it works:**
```
Workflow 1: Train from scratch (random init) → Policy_1
Workflow 2: Train from scratch (random init) → Policy_2
Workflow 3: Train from scratch (random init) → Policy_3
```

**Run:**
```bash
bash run_fixed_no_warmstart.sh

# Or:
python workflow_rl/executor_async_fixed_episodes_no_warmstart.py \
    --n-workers 25 \
    --fixed-episodes-per-workflow 2500
```

**Logs:** `logs/exp_fixed_no_warmstart_*/`

---

## What Each Tests

### **WITH Warm Start**
Tests: "Does warm starting help with fixed episodes?"

**Expected:**
- Faster convergence (leverages previous learning)
- Higher average final compliance
- More workflows explored (faster training per workflow)

### **WITHOUT Warm Start**  
Tests: "Pure fixed-episode performance from scratch"

**Expected:**
- Slower convergence (learns everything fresh)
- Lower average final compliance
- Fewer workflows explored (each takes full 2500 episodes)

---

## Use Cases

### **Use WITH Warm Start When:**
- You want to maximize sample efficiency
- Transfer learning is acceptable
- Comparing with adaptive method that also uses warm start

### **Use WITHOUT Warm Start When:**
- You want pure comparison (no confounds)
- Testing if warm start helps
- Comparing with methods that don't use warm start

---

## Differences Summary

| Feature | WITH Warm Start | WITHOUT Warm Start |
|---------|----------------|-------------------|
| **Policy Initialization** | From similar workflows | Random weights |
| **Training Speed** | Faster (warm start helps) | Slower (learns from scratch) |
| **Episode Efficiency** | Higher (reuses knowledge) | Lower (each independent) |
| **Workflows Explored** | More | Fewer |
| **Comparison Fairness** | Depends on baseline | Pure comparison |

---

## Recommendation

**For L4DC paper:**

If your adaptive method DOES use warm starting:
- Use `executor_async_fixed_episodes.py` (WITH warm start)
- Fair comparison (both use transfer learning)

If your adaptive method DOESN'T use warm starting:
- Use `executor_async_fixed_episodes_no_warmstart.py`
- Fair comparison (neither uses transfer learning)

**Or run BOTH** and report:
- Shows value of warm starting separately
- More complete ablation study

---

## All Variants Summary

You now have **THREE** fixed-episode configurations:

1. **executor_async_fixed_episodes.py**
   - With warm start
   - Default: 50 workers, 2500 episodes/workflow

2. **executor_async_fixed_episodes_no_warmstart.py** (NEW!)
   - NO warm start
   - Default: 25 workers, 2500 episodes/workflow

3. **executor_async_train_workflow_rl.py** (Main method)
   - With warm start
   - Adaptive early stopping
   - Default: 200 workers, variable episodes

Perfect for comprehensive L4DC experiments! 🎓

