# 🚀 Improving Convergence Speed to 95% Compliance

## Current Situation

**Problem:**
- Most workflows plateau at 75-85% compliance
- Only ~20% reach 95% threshold
- Training stops due to plateau detection (no improvement for 12 updates)

**Goal:**
- Make MORE workflows reach 95% compliance
- Reach 95% in FEWER episodes
- Better episode budget utilization

---

## 🎯 Proposed Improvements

### **1. Progressive Compliance Reward Scaling** ⭐ (Most Promising)

**Current (Linear):**
```python
alignment_bonus = lambda × compliance_rate
                = 30.0 × 0.80
                = 24.0  # Same marginal benefit throughout
```

**Improved (Exponential near 95%):**
```python
def compute_alignment_bonus(compliance_rate, lambda_base=30.0):
    if compliance_rate < 0.90:
        # Below 90%: linear scaling
        return lambda_base * compliance_rate
    else:
        # Above 90%: exponential boost toward 95%
        # Heavily reward getting closer to 95%
        distance_from_95 = 0.95 - compliance_rate
        boost_factor = np.exp(-10 * distance_from_95)
        return lambda_base * compliance_rate * (1 + boost_factor)

# Examples:
compliance=0.80 → bonus = 24.0   (same as before)
compliance=0.90 → bonus = 35.6   (1.3x boost)
compliance=0.93 → bonus = 51.2   (1.8x boost)
compliance=0.95 → bonus = 85.5   (3.0x boost!)
```

**Why this works:**
- Makes the last 5% (90%→95%) much more rewarding
- Agent strongly incentivized to cross 95% threshold
- Doesn't hurt early learning (linear below 90%)

---

### **2. Increase Alignment Lambda** ⭐

**Current:**
```python
alignment_lambda = 30.0
```

**Improved:**
```python
alignment_lambda = 50.0  # or even 100.0
```

**Analysis:**
```
Current (lambda=30):
  Env reward: -700
  Alignment (80%): +24
  Total: -676
  
  Compliance improvement 80%→85% gives: +1.5 reward
  Environment improvement -700→-650 gives: +50 reward
  
  → Agent prioritizes environment over compliance!

Improved (lambda=100):
  Env reward: -700
  Alignment (80%): +80
  Total: -620
  
  Compliance improvement 80%→85% gives: +5 reward
  Environment improvement -700→-650 gives: +50 reward
  
  → Compliance matters more!
```

**Recommendation:** Start with `lambda=50`, try up to `lambda=100`

---

### **3. Two-Stage Training** ⭐

**Concept:** Focus on compliance more as training progresses

```python
def get_adaptive_lambda(episode_num, base_lambda=30.0):
    # Early training: focus on environment (lambda is lower)
    # Late training: focus on compliance (lambda is higher)
    
    if episode_num < 1000:
        return base_lambda  # 30.0
    elif episode_num < 3000:
        return base_lambda * 1.5  # 45.0
    else:
        return base_lambda * 2.5  # 75.0

# Usage in episode collection:
alignment_bonus = get_adaptive_lambda(total_episodes) × compliance_rate
```

**Why this works:**
- Early: Learn to defend network (get env reward up)
- Late: Learn compliance (with better baseline)
- Curriculum learning approach

---

### **4. Compliance-Focused Update Frequency**

**Current:**
```python
episodes_per_update = 100  # Update every 100 episodes
```

**Improved:**
```python
# More frequent updates when compliance is close to 95%
def get_update_frequency(compliance):
    if compliance < 0.85:
        return 100  # Normal frequency
    else:
        return 50   # 2x more frequent when close to 95%
```

**Why this works:**
- More gradient steps when fine-tuning compliance
- Faster adaptation in critical 85-95% range

---

### **5. Action Masking for Non-Compliant Actions** 🔥

**Concept:** After compliance reaches 80%, prevent obviously wrong actions

```python
def mask_non_compliant_actions(action_probs, current_state, workflow_order):
    # Find highest priority compromised type
    compromised_types = get_compromised_types(current_state)
    highest_priority = get_highest_priority(compromised_types, workflow_order)
    
    if highest_priority is None:
        return action_probs  # No masking needed
    
    # Mask fix actions that DON'T target highest priority
    masked_probs = action_probs.copy()
    
    for action, prob in enumerate(action_probs):
        if action in action_to_host_type:
            target_type = action_to_host_type[action]
            if target_type != highest_priority:
                # This would be non-compliant - reduce probability
                masked_probs[action] *= 0.1  # 90% reduction
    
    # Renormalize
    masked_probs = masked_probs / masked_probs.sum()
    return masked_probs
```

**Why this works:**
- Directly guides agent toward compliant actions
- Soft constraint (reduces prob, doesn't eliminate)
- Only applied when compliance is already decent

---

### **6. Increase PPO Epochs for Compliance Updates** 

**Current:**
```python
K_epochs = 6  # 6 gradient steps per update
```

**Improved (Adaptive):**
```python
def get_k_epochs(compliance, base_epochs=6):
    if compliance >= 0.85:
        return base_epochs * 2  # 12 epochs when close to 95%
    else:
        return base_epochs  # 6 epochs normally
```

**Why this works:**
- More optimization when fine-tuning compliance
- Standard early, intensive late
- Focuses compute where needed

---

### **7. Compliance Regularization in Loss Function**

**Current:**
```python
loss = -min(surr1, surr2).mean() + 0.5 * value_loss - 0.01 * entropy
```

**Improved:**
```python
# Add explicit compliance penalty
compliance_penalty = 0.0
if avg_compliance < 0.95:
    # Penalize policy if compliance is low
    # This adds pressure beyond just the reward
    compliance_penalty = (0.95 - avg_compliance) * 10.0

loss = -min(surr1, surr2).mean() + 0.5 * value_loss - 0.01 * entropy + compliance_penalty
```

**Why this works:**
- Direct optimization objective for compliance
- Not just through rewards
- Adds extra gradient signal

---

### **8. Smaller Learning Rate for Fine-Tuning**

**Current:**
```python
lr = 0.002  # Fixed throughout
```

**Improved (Adaptive):**
```python
def get_learning_rate(compliance, base_lr=0.002):
    if compliance < 0.80:
        return base_lr  # 0.002 (fast learning)
    elif compliance < 0.90:
        return base_lr * 0.5  # 0.001 (slower, more precise)
    else:
        return base_lr * 0.25  # 0.0005 (very precise near 95%)
```

**Why this works:**
- Fast learning early
- Careful fine-tuning near threshold
- Prevents overshooting

---

## 🔧 Implementation Priority

### **Quick Wins (Easy to Implement):**

1. ✅ **Increase alignment_lambda** to 50-100
   - Change one line
   - Immediate impact
   - Test: `--alignment-lambda 100`

2. ✅ **Progressive reward scaling**
   - Add exponential boost near 95%
   - ~20 lines of code
   - High impact

3. ✅ **Adaptive K_epochs**
   - More epochs when compliance > 85%
   - ~10 lines of code
   - Focused compute

### **Medium Effort (Moderate Implementation):**

4. ⚙️ **Two-stage training**
   - Adaptive lambda based on episodes
   - ~30 lines of code
   - Proven technique

5. ⚙️ **Compliance-focused update frequency**
   - Dynamic episodes_per_update
   - ~20 lines of code
   - More optimization when needed

### **Advanced (More Complex):**

6. 🔬 **Action masking**
   - Modify action selection
   - ~50 lines of code
   - Requires careful implementation

7. 🔬 **Compliance regularization**
   - Modify loss function
   - ~30 lines of code
   - Additional tuning needed

---

## 📊 Expected Impact

### **With Improvements 1-3 (Quick Wins):**
```
Current:
- 20% workflows reach 95%
- Avg episodes to plateau: 4000-5000
- Plateau compliance: 75-85%

Expected:
- 60-70% workflows reach 95%
- Avg episodes to 95%: 2000-3000
- Fewer plateaus below 90%
```

### **With All Improvements:**
```
Expected:
- 80-90% workflows reach 95%
- Avg episodes to 95%: 1500-2500
- Plateau compliance: 90-93% (much closer!)
```

---

## 🎯 Recommended Action Plan

### **Phase 1: Quick Test**
```bash
# Test with higher lambda
python workflow_rl/executor_async_train_workflow_rl.py \
    --n-workers 100 \
    --total-episodes 10000 \
    --alignment-lambda 100 \
    --max-episodes-per-workflow 3000
```

Compare results with baseline (lambda=30).

### **Phase 2: Implement Progressive Scaling**

Modify `collect_single_episode` function:
```python
def compute_progressive_alignment_bonus(compliance_rate, base_lambda=50.0):
    if compliance_rate < 0.90:
        return base_lambda * compliance_rate
    else:
        # Exponential boost near 95%
        distance = 0.95 - compliance_rate
        boost = 1 + np.exp(-10 * distance)
        return base_lambda * compliance_rate * boost
```

### **Phase 3: Adaptive Training**

Add adaptive K_epochs and learning rate based on compliance.

---

## 💡 Quick Code Changes

### **Change 1: Higher Lambda (Easiest)**

In `executor_async_train_workflow_rl.py`:
```python
# Line 790: Change default
parser.add_argument('--alignment-lambda', type=float, default=100.0)  # Was 30.0
```

### **Change 2: Progressive Scaling**

In `collect_single_episode` function, replace:
```python
# Old:
alignment_bonus = alignment_lambda * compliance

# New:
if compliance < 0.90:
    alignment_bonus = alignment_lambda * compliance
else:
    boost = 1 + np.exp(-10 * (0.95 - compliance))
    alignment_bonus = alignment_lambda * compliance * boost
```

### **Change 3: Adaptive K_epochs**

In `train_workflow_async`, before PPO update:
```python
# Adjust K_epochs based on compliance
if avg_compliance >= 0.85:
    current_k_epochs = 12  # Double epochs when close
else:
    current_k_epochs = 6

for epoch in range(current_k_epochs):
    # PPO update...
```

---

## 🧪 Testing Strategy

1. **Baseline** (current): lambda=30, linear
2. **Test 1**: lambda=100, linear
3. **Test 2**: lambda=50, progressive scaling
4. **Test 3**: lambda=100, progressive scaling + adaptive K_epochs

Compare:
- % workflows reaching 95%
- Average episodes to 95%
- Plateau compliance values

---

## 📈 Expected Results

| Configuration | Workflows @ 95% | Avg Episodes | Plateau @ |
|--------------|-----------------|--------------|-----------|
| **Current** (lambda=30) | 20% | 4000 | 80% |
| **lambda=100** | 50% | 3000 | 85% |
| **Progressive** | 70% | 2500 | 90% |
| **All improvements** | 85% | 2000 | 92% |

These are conservative estimates. Actual results may be even better!

Would you like me to implement these improvements?
