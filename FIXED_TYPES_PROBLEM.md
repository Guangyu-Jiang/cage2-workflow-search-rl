# 🐛 Problem with `fixed_types` Variable

## Current Implementation (WRONG!)

```python
fixed_types = set()  # e.g., {'defender', 'enterprise'}

# When agent does action 132 (Restore defender)
if action == 132:
    fixed_types.add('defender')
    # Assumes defender is now fixed!
```

### **Problems:**

1. **Doesn't check actual environment state**
   - Assumes restore action succeeded
   - Doesn't verify host is actually clean
   
2. **Doesn't account for re-compromise**
   - Red agent might re-compromise the host immediately
   - `fixed_types` never removes types
   
3. **Not grounded in reality**
   - Just tracks which actions were taken
   - Not which hosts are actually fixed

---

## What We SHOULD Do

### **Check True State at Each Step:**

```python
# Get current compromised hosts from environment
true_state = cyborg.get_agent_state('True')

# Find which unit TYPES are currently compromised
compromised_types = set()
for hostname, host_info in true_state.items():
    if hostname != 'success':
        is_compromised = check_if_compromised(host_info)
        if is_compromised:
            unit_type = get_unit_type(hostname)  # 'defender', 'enterprise', etc.
            compromised_types.add(unit_type)

# Now we know: compromised_types = {'enterprise', 'op_server', 'user'}

# Find highest priority compromised type
highest_priority_compromised = None
for unit_type in workflow_order:
    if unit_type in compromised_types:
        highest_priority_compromised = unit_type
        break  # Found the first (highest priority) compromised type

# Check if agent's fix action targets this highest priority type
if action in action_to_host_type:
    target_type = action_to_host_type[action]
    
    if target_type == highest_priority_compromised:
        compliant_fix_actions += 1  # ✅ Fixing the right thing!
    else:
        # ❌ Fixing something else when higher priority exists
        pass
```

---

## Example

### **Current State (from true_state):**
```
Compromised hosts:
- enterprise_1 (compromised)
- op_server_0 (compromised)  
- user_0 (compromised)
- defender_0 (clean) ✓
- op_host_1 (clean) ✓

Compromised types: {enterprise, op_server, user}
```

### **Workflow:** `defender → enterprise → op_server → op_host → user`

### **Highest Priority Compromised:**
```
Check in priority order:
1. defender: not in compromised_types ✓ (clean)
2. enterprise: in compromised_types ❌ (compromised!)

→ highest_priority_compromised = 'enterprise'
```

### **Agent's Action: 139 (Restore op_server)**
```python
target_type = action_to_host_type[139]  # 'op_server'

if target_type == highest_priority_compromised:
    # 'op_server' == 'enterprise'?  NO!
    # ❌ NOT COMPLIANT
    pass
```

**Result:** Agent should fix enterprise (priority 2) but fixed op_server (priority 3) → **Violation!**

---

## Why Current Implementation Fails

### **Scenario:**
```
Workflow: defender → enterprise → op_server → op_host → user

Step 1: Action 132 (Restore defender)
  fixed_types.add('defender')
  fixed_types = {'defender'}

Step 2: Action 133 (Restore enterprise)  
  Check: defender in fixed_types? Yes ✓
  Marked as COMPLIANT ✅
  fixed_types = {'defender', 'enterprise'}

Step 3: Red agent compromises defender again!
  Environment state: defender is compromised again
  But fixed_types still has 'defender'!
  
Step 4: Action 139 (Restore op_server)
  Check: defender and enterprise in fixed_types? Yes ✓
  Marked as COMPLIANT ✅
  
  But WRONG! Defender is compromised again!
  Should fix defender, not op_server!
```

**The `fixed_types` variable doesn't reflect actual environment state!**

---

## Correct Implementation

We need to check the actual environment state at each step:

```python
def get_compromised_types(true_state, workflow_order):
    """Get currently compromised unit types from true state"""
    compromised_types = set()
    
    for hostname, host_info in true_state.items():
        if hostname == 'success':
            continue
            
        # Check if host is compromised
        is_compromised = (
            host_info.get('System info', {}).get('Compromised', False) or
            (host_info.get('Interface', [{}])[0].get('Compromised', False)
             if host_info.get('Interface') else False)
        )
        
        if is_compromised:
            # Determine unit type from hostname
            for unit_type in ['defender', 'enterprise', 'op_server', 'op_host', 'user']:
                if unit_type in hostname.lower():
                    compromised_types.add(unit_type)
                    break
    
    return compromised_types


def get_highest_priority_compromised(compromised_types, workflow_order):
    """Find the highest priority type that is compromised"""
    for unit_type in workflow_order:
        if unit_type in compromised_types:
            return unit_type
    return None


def is_fix_compliant(action, true_state, workflow_order):
    """Check if fix action targets highest priority compromised type"""
    if action not in action_to_host_type:
        return None  # Not a fix action
    
    target_type = action_to_host_type[action]
    compromised_types = get_compromised_types(true_state, workflow_order)
    highest_priority = get_highest_priority_compromised(compromised_types, workflow_order)
    
    if highest_priority is None:
        return True  # No compromised hosts, any fix is fine
    
    return target_type == highest_priority  # Compliant if targeting highest priority
```

This is the CORRECT approach!
