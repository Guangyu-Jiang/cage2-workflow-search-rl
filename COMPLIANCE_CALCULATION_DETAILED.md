# 📊 Compliance Rate Calculation - Detailed Explanation

## Overview

Compliance rate measures **how well the agent follows the workflow priority order** when performing fix actions.

**Formula:**
```
Compliance Rate = compliant_fix_actions / total_fix_actions
```

---

## 🔧 Step-by-Step Code Walkthrough

### **1. Action-to-Host-Type Mapping**

First, we define which actions affect which host types:

```python
action_to_host_type = {
    # Remove actions (15-27)
    15: 'defender', 16: 'enterprise', 17: 'enterprise', 18: 'enterprise',
    19: 'op_host', 20: 'op_host', 21: 'op_host', 22: 'op_server',
    23: 'user', 24: 'user', 25: 'user', 26: 'user', 27: 'user',
    
    # Restore actions (132-144)
    132: 'defender', 133: 'enterprise', 134: 'enterprise', 135: 'enterprise',
    136: 'op_host', 137: 'op_host', 138: 'op_host', 139: 'op_server',
    140: 'user', 141: 'user', 142: 'user', 143: 'user', 144: 'user'
}
```

**Example:**
- Action 132 (Restore) → targets `defender`
- Action 136 (Restore) → targets `op_host`
- Action 23 (Remove) → targets `user`

---

### **2. Initialize Tracking Variables**

At the start of each episode:

```python
total_fix_actions = 0        # Count ALL fix actions
compliant_fix_actions = 0    # Count only COMPLIANT fix actions
fixed_types = set()          # Track which types we've fixed
```

---

### **3. During Episode - Track Each Fix Action**

For every action taken in the episode:

```python
action = agent.select_action(state)

# Check if this is a fix action (Remove or Restore)
if action in action_to_host_type:
    total_fix_actions += 1
    target_type = action_to_host_type[action]
    
    # Determine if this fix is compliant...
```

---

### **4. Compliance Check Logic**

The key question: **"Is this fix action compliant with the workflow order?"**

```python
# Get the priority of the type being fixed
target_priority = workflow_order.index(target_type)

# Check if any HIGHER priority types are still unfixed
violation = False
for priority_idx in range(target_priority):
    priority_type = workflow_order[priority_idx]
    if priority_type not in fixed_types:
        # Found a higher priority type that hasn't been fixed yet!
        violation = True
        break

# If no violation, this fix is compliant
if not violation:
    compliant_fix_actions += 1

# Mark this type as fixed (for future checks)
fixed_types.add(target_type)
```

---

## 📝 Concrete Example

### **Scenario:**

**Workflow Order:** `defender → enterprise → op_server → op_host → user`

**Priorities:**
1. defender (highest)
2. enterprise
3. op_server
4. op_host
5. user (lowest)

---

### **Episode Execution:**

#### **Action 1: Agent selects action 136 (Restore op_host)**

```python
action = 136
target_type = action_to_host_type[136]  # 'op_host'
total_fix_actions = 1

# Check compliance
target_priority = workflow_order.index('op_host')  # Priority = 3 (4th position)

# Check if higher priority types (0, 1, 2) are fixed
for priority_idx in [0, 1, 2]:  # Check defender, enterprise, op_server
    priority_type = workflow_order[priority_idx]
    if priority_type not in fixed_types:
        # 'defender' not in {} → VIOLATION!
        violation = True
        break

# violation = True → DON'T count as compliant
compliant_fix_actions = 0

# Mark op_host as fixed
fixed_types = {'op_host'}
```

**Result:** ❌ **NOT compliant** (fixed op_host before defender!)

---

#### **Action 2: Agent selects action 132 (Restore defender)**

```python
action = 132
target_type = action_to_host_type[132]  # 'defender'
total_fix_actions = 2

# Check compliance
target_priority = workflow_order.index('defender')  # Priority = 0 (1st position)

# Check if higher priority types exist
for priority_idx in []:  # range(0) = empty!
    # No loop iterations - no higher priority types
    pass

violation = False  # No violations found

# violation = False → COUNT as compliant!
compliant_fix_actions = 1

# Mark defender as fixed
fixed_types = {'op_host', 'defender'}
```

**Result:** ✅ **Compliant** (defender is highest priority!)

---

#### **Action 3: Agent selects action 133 (Restore enterprise)**

```python
action = 133
target_type = action_to_host_type[133]  # 'enterprise'
total_fix_actions = 3

# Check compliance
target_priority = workflow_order.index('enterprise')  # Priority = 1 (2nd position)

# Check if higher priority types are fixed
for priority_idx in [0]:  # Check defender only
    priority_type = workflow_order[0]  # 'defender'
    if 'defender' not in fixed_types:
        violation = True
        break

# 'defender' IS in fixed_types → no violation
violation = False

compliant_fix_actions = 2

fixed_types = {'op_host', 'defender', 'enterprise'}
```

**Result:** ✅ **Compliant** (defender already fixed!)

---

#### **Action 4: Agent selects action 140 (Restore user)**

```python
action = 140
target_type = action_to_host_type[140]  # 'user'
total_fix_actions = 4

# Check compliance
target_priority = workflow_order.index('user')  # Priority = 4 (5th position)

# Check if higher priority types are fixed
for priority_idx in [0, 1, 2, 3]:  # Check defender, enterprise, op_server, op_host
    priority_type = workflow_order[2]  # 'op_server'
    if 'op_server' not in fixed_types:
        # op_server not fixed yet → VIOLATION!
        violation = True
        break

violation = True

compliant_fix_actions = 2  # No change

fixed_types = {'op_host', 'defender', 'enterprise', 'user'}
```

**Result:** ❌ **NOT compliant** (fixed user before op_server!)

---

#### **Action 5: Agent selects action 139 (Restore op_server)**

```python
action = 139
target_type = action_to_host_type[139]  # 'op_server'
total_fix_actions = 5

# Check compliance
target_priority = workflow_order.index('op_server')  # Priority = 2 (3rd position)

# Check if higher priority types are fixed
for priority_idx in [0, 1]:  # Check defender, enterprise
    # defender: in fixed_types ✓
    # enterprise: in fixed_types ✓
    pass

violation = False  # All higher priorities fixed!

compliant_fix_actions = 3

fixed_types = {'op_host', 'defender', 'enterprise', 'user', 'op_server'}
```

**Result:** ✅ **Compliant** (defender and enterprise already fixed!)

---

### **End of Episode:**

```python
# Calculate compliance rate
if total_fix_actions > 0:
    episode_compliance = compliant_fix_actions / total_fix_actions
else:
    episode_compliance = 0.5  # No fixes = neutral

# In our example:
episode_compliance = 3 / 5 = 0.6 = 60%
```

---

## 📊 Summary of Example

| Action | Type | Target | Priority | Higher Unfixed? | Compliant? |
|--------|------|--------|----------|-----------------|------------|
| 136 | Restore | op_host | 4th | defender, enterprise, op_server | ❌ No |
| 132 | Restore | defender | 1st | none | ✅ Yes |
| 133 | Restore | enterprise | 2nd | none (defender fixed) | ✅ Yes |
| 140 | Restore | user | 5th | op_server | ❌ No |
| 139 | Restore | op_server | 3rd | none (def, ent fixed) | ✅ Yes |

**Final Compliance:** 3 compliant / 5 total = **60%**

---

## 🎯 Compliance Rules

### **A fix action is COMPLIANT if:**
✅ All higher-priority unit types have already been fixed

### **A fix action is a VIOLATION if:**
❌ Any higher-priority unit type is still unfixed

---

## 🔢 The Complete Algorithm

```python
# Initialize
total_fix_actions = 0
compliant_fix_actions = 0
fixed_types = set()

# For each action in episode:
for action in episode_actions:
    if action in action_to_host_type:  # Is this a fix action?
        total_fix_actions += 1
        target_type = action_to_host_type[action]
        
        # Get priority of this type
        target_priority = workflow_order.index(target_type)
        
        # Check all higher priorities
        violation = False
        for priority_idx in range(target_priority):
            priority_type = workflow_order[priority_idx]
            if priority_type not in fixed_types:
                violation = True  # Higher priority not fixed yet!
                break
        
        # Update counters
        if not violation:
            compliant_fix_actions += 1
        
        # Mark type as fixed
        fixed_types.add(target_type)

# Compute final compliance
compliance_rate = compliant_fix_actions / max(total_fix_actions, 1)
```

---

## 🎓 Why This Makes Sense

### **Workflow Priority Order Defines Best Practice:**

If workflow is: `defender → enterprise → op_server → op_host → user`

This means:
1. **First**, fix defender hosts (most critical)
2. **Then**, fix enterprise hosts
3. **Then**, fix op_server hosts
4. **Then**, fix op_host hosts
5. **Finally**, fix user hosts (least critical)

### **Compliance Measures Adherence:**

**Good Agent (High Compliance):**
```
Fix order: defender → enterprise → op_server → op_host → user
All fixes follow priority → 100% compliant ✅
```

**Bad Agent (Low Compliance):**
```
Fix order: user → op_host → enterprise → defender → op_server
Violates priority constantly → 0-20% compliant ❌
```

**Learning Agent (Improving Compliance):**
```
Early training: Random fix order → 30-50% compliant
Mid training: Better order → 60-70% compliant
Late training: Following priority → 85-95% compliant ✅
```

---

## 💡 Edge Cases

### **Case 1: No Fix Actions**
```python
if total_fix_actions == 0:
    compliance = 0.5  # Neutral (not good or bad)
```

### **Case 2: All Fix Actions Compliant**
```python
# Perfect workflow following
compliance = compliant_fix_actions / total_fix_actions
           = 20 / 20
           = 1.0  # 100%
```

### **Case 3: Fixing Same Type Multiple Times**
```python
Action 1: Fix defender (compliant ✅)
Action 2: Fix defender again (compliant ✅ - defender already in fixed_types)
Action 3: Fix enterprise (compliant ✅ - defender fixed)

# All counted as compliant
compliance = 3 / 3 = 100%
```

Once a type is in `fixed_types`, all future fixes to that type are automatically compliant (higher priorities already satisfied).

---

## 📈 Training Progression Example

### **Early Training (Episode 1):**
```
Actions: [55, 136(op_host), 12, 140(user), 132(defender), 88, 133(ent), ...]
         
Fix actions: 136(op_host), 140(user), 132(defender), 133(ent)
Workflow: defender → enterprise → op_server → op_host → user

136(op_host): Priority 4, defender unfixed → ❌ violation
140(user): Priority 5, defender unfixed → ❌ violation  
132(defender): Priority 1, no higher → ✅ compliant
133(enterprise): Priority 2, defender fixed → ✅ compliant

Compliance: 2 / 4 = 50%
```

### **Late Training (Episode 100):**
```
Actions: [12, 132(defender), 88, 133(ent), 139(op_server), 136(op_host), ...]

Fix actions: 132(def), 133(ent), 139(op_server), 136(op_host)
Workflow: defender → enterprise → op_server → op_host → user

132(defender): Priority 1, no higher → ✅ compliant
133(enterprise): Priority 2, defender fixed → ✅ compliant
139(op_server): Priority 3, def+ent fixed → ✅ compliant
136(op_host): Priority 4, all higher fixed → ✅ compliant

Compliance: 4 / 4 = 100%
```

**Agent learned to follow the workflow!**

---

## 🔍 Code Location in executor_async_train_workflow_rl.py

### **Lines 43-53: Action Mapping**
```python
action_to_host_type = {
    15: 'defender', 16: 'enterprise', ..., 144: 'user'
}
```

### **Lines 80-82: Initialize Tracking**
```python
total_fix_actions = 0
compliant_fix_actions = 0
fixed_types = set()
```

### **Lines 110-130: Per-Action Compliance Check**
```python
if action in action_to_host_type:
    total_fix_actions += 1
    target_type = action_to_host_type[action]
    target_priority = workflow_order.index(target_type)
    
    violation = False
    for priority_idx in range(target_priority):
        if workflow_order[priority_idx] not in fixed_types:
            violation = True
            break
    
    if not violation:
        compliant_fix_actions += 1
    
    fixed_types.add(target_type)
```

### **Lines 148-152: Final Calculation**
```python
if total_fix_actions > 0:
    episode_compliance = compliant_fix_actions / total_fix_actions
else:
    episode_compliance = 0.5
```

---

## 🎯 Why This Metric?

### **Measures Workflow Learning:**
- Agent needs to learn the priority order
- High compliance = follows workflow correctly
- Low compliance = random/wrong order

### **Used for Two Purposes:**

1. **Alignment Reward** (during training)
```python
alignment_bonus = alignment_lambda × compliance_rate
                = 30.0 × 0.60
                = 18.0
```

2. **Early Stopping** (when to stop training)
```python
if avg_compliance >= 0.95:
    # 95% of fixes follow workflow → good enough!
    stop_training()
```

---

## 📊 Typical Compliance Values

Based on actual training runs:

| Training Stage | Compliance Range | Interpretation |
|----------------|------------------|----------------|
| **Early (Updates 1-5)** | 20-40% | Random/learning |
| **Mid (Updates 6-15)** | 50-70% | Improving |
| **Late (Updates 16-25)** | 75-90% | Good adherence |
| **Converged** | 90-100% | Excellent! |

**Threshold:** 95% compliance is considered "good enough" to stop training.

---

## 🔬 Comparison: Different Workflow Orders

### **Workflow A:** `defender → enterprise → op_server → op_host → user`

Agent fixes in order: defender, enterprise, op_server, op_host, user
```
All fixes follow priority → Compliance: 100%
```

### **Workflow B:** `user → op_host → op_server → enterprise → defender`

Same agent (same fix order): defender, enterprise, op_server, op_host, user
```
Fixes defender first (priority 5) when user unfixed (priority 1) → Violation!
Fixes enterprise (priority 4) when user unfixed → Violation!
...
Compliance: 20% (only last fix compliant)
```

**Same actions, different workflow → different compliance!**

This shows the agent must learn the SPECIFIC workflow order.

---

## 💻 Full Code Example

Here's the complete compliance calculation from the actual code:

```python
# From executor_async_train_workflow_rl.py, lines 80-152

# Initialize
total_fix_actions = 0
compliant_fix_actions = 0
fixed_types = set()

for step in range(100):
    action = select_action(state)
    next_state, reward, done, info = env.step(action)
    
    # Track compliance
    if action in action_to_host_type:  # Fix action?
        total_fix_actions += 1
        target_type = action_to_host_type[action]
        target_priority = workflow_order.index(target_type)
        
        # Check violation
        violation = False
        for priority_idx in range(target_priority):
            priority_type = workflow_order[priority_idx]
            if priority_type not in fixed_types:
                violation = True
                break
        
        # Update
        if not violation:
            compliant_fix_actions += 1
        fixed_types.add(target_type)
    
    if done:
        break

# Calculate compliance
if total_fix_actions > 0:
    compliance = compliant_fix_actions / total_fix_actions
else:
    compliance = 0.5
```

---

## ✅ Summary

**Compliance Rate = Percentage of fix actions that follow workflow priority**

**Calculation:**
1. Track all fix actions (Remove/Restore: actions 15-27, 132-144)
2. For each fix, check if higher-priority types are already fixed
3. Count as compliant if no higher-priority types remain unfixed
4. Compliance = compliant_fixes / total_fixes

**Used for:**
- Alignment reward during training
- Early stopping criterion (95% threshold)
- Measuring if agent learned the workflow

This ensures the agent learns to follow the specified repair priority order!
