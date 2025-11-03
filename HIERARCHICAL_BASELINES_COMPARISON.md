# Hierarchical RL Baselines - Domain-Specific vs General

You now have **TWO hierarchical RL baselines** for comparison:

---

## 1. Domain-Specific Hierarchical RL ❌ (NOT Recommended for L4DC)

**File:** `train_hierarchical_baseline.py`

**Architecture:**
- High-level: Selects **unit type** to fix (5 hard-coded choices)
  - defender, enterprise, op_server, op_host, user
- Low-level: Selects action from that unit type's action set

**Problem:** Encodes domain knowledge!
- Unit types are CAGE2-specific
- Not a fair baseline (uses task-specific structure)
- Reviewer complaint: "This baseline cheats by using domain knowledge"

**Use Case:** Don't use for L4DC paper (unfair advantage)

---

## 2. General Hierarchical RL ✅ (Recommended for L4DC)

**File:** `train_hierarchical_general.py`

**Architecture:**
- High-level: Selects from **K learned options** (abstract, not pre-defined)
  - Options are temporal abstractions
  - Learned from scratch, no domain knowledge
- Low-level: Executes primitive actions given selected option

**Advantages:**
- Domain-agnostic (could work on any RL task)
- Fair comparison (no task-specific encoding)
- Standard approach in hierarchical RL literature

**Based on:** Options Framework (Sutton, Precup, Singh, 1999)

---

## Architecture Comparison

### Domain-Specific:
```
High-level output: [defender, enterprise, op_server, op_host, user]
                    ↑ Hand-crafted based on CAGE2 knowledge

Low-level input: State + Selected unit type
                 ↑ Domain-specific conditioning

Action masking: Only allows actions for that unit type
                ↑ Requires knowing which actions belong to which unit
```

### General (Options):
```
High-level output: [Option_0, Option_1, ..., Option_7]
                    ↑ Learned abstractions, no domain knowledge

Low-level input: State + Option embedding
                 ↑ Generic conditioning

Action selection: Any action allowed
                  ↑ No domain-specific constraints
```

---

## How Options Work

### **Temporal Abstraction:**

```
Episode timeline:
Step 0-9:   High selects Option_2 → Low executes actions [a₁, a₂, ..., a₉]
Step 10-19: High selects Option_5 → Low executes actions [a₁₀, a₁₁, ..., a₁₉]
Step 20-29: High selects Option_2 → Low executes actions [a₂₀, a₂₁, ..., a₂₉]
```

**Key:** High-level makes decisions at slower timescale (every ~10 steps)

---

## Key Differences

| Aspect | Domain-Specific | General (Options) |
|--------|----------------|-------------------|
| **High-level Actions** | Unit types (5) | Learned options (8) |
| **Domain Knowledge** | ❌ Yes (unit types) | ✅ No (learned) |
| **Action Masking** | Required | Not required |
| **Fair Baseline** | ❌ No (cheats) | ✅ Yes (fair) |
| **L4DC Appropriate** | ❌ No | ✅ Yes |
| **Generalizability** | Only CAGE2 | Any RL task |

---

## For L4DC Paper

### **Use:**
```bash
bash run_hierarchical_general.sh
```

**Report as:**
"Hierarchical RL baseline using the Options Framework, with K learned temporal abstractions operating at a slower timescale than primitive actions."

### **Don't Use:**
`train_hierarchical_baseline.py` (domain-specific encoding)

---

## Why This Is Important

### **Reviewer Perspective:**

**Domain-Specific Baseline:**
- Reviewer: "This baseline has an unfair advantage - it already knows about unit types!"
- Reviewer: "Your method vs this baseline isn't a fair comparison"

**General Baseline:**
- Reviewer: "Fair comparison - both methods start with same knowledge"
- Reviewer: "Shows value of GP-UCB vs learned hierarchy"

---

## Parameters

### **General Hierarchical (Recommended):**
```bash
python train_hierarchical_general.py \
    --n-workers 50 \
    --total-episodes 100000 \
    --n-options 8 \              # Number of abstract actions
    --option-duration 10         # Steps per option
```

### **Hyperparameters:**
- `n_options`: How many abstract actions (default: 8)
  - More options = more flexibility
  - Fewer options = simpler hierarchy
  
- `option_duration`: How long each option lasts (default: 10 steps)
  - Longer = slower high-level decisions
  - Shorter = more frequent high-level decisions

---

## Expected Results

### **General Hierarchical:**
- Final reward: -50 to -100 (estimate)
- Compliance: 40-50% (learned, not forced)
- Should perform better than flat PPO
- Should perform worse than your method (which uses GP-UCB)

---

## Complete Baseline Suite for L4DC

1. ✅ Random Policy - Lower bound
2. ✅ Greedy Heuristic - Heuristic upper bound
3. ✅ PPO Baseline - Standard RL (no hierarchy)
4. ✅ SAC Baseline - State-of-the-art RL (no hierarchy)
5. ✅ **Hierarchical RL (General)** - Learned hierarchy ⭐ USE THIS
6. ❌ ~~Hierarchical RL (Domain-Specific)~~ - Don't use (unfair)
7. ✅ Your Method - GP-UCB hierarchy + compliance

**Perfect for L4DC!** 🎓

The general hierarchical baseline is the right choice because:
- Fair comparison (no domain knowledge)
- Standard approach in literature
- Shows your method beats learned hierarchy

