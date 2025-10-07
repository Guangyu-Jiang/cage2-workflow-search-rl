# Compliance Calculation in Workflow-Conditioned RL

## Overview
Compliance measures how well the PPO agent follows the prescribed workflow priority order when fixing compromised hosts. It's calculated as a percentage of actions that align with the workflow.

## Compliance Calculation Formula

```
Compliance Rate = (Compliant Actions / Total Fix Actions) × 100%
```

## Step-by-Step Process

### 1. Workflow Priority Order
Each workflow defines a priority order for unit types:
```python
workflow_order = ['defender', 'op_server', 'enterprise', 'op_host', 'user']
```
This means:
- Priority 1 (highest): Defender
- Priority 2: Op_Server
- Priority 3: Enterprise
- Priority 4: Op_Host
- Priority 5 (lowest): User

### 2. Action Classification
For each action the agent takes, we determine:
- **Is it a fix action?** (Analyse, Remove, or Restore)
- **Which unit type does it target?**

### 3. Compliance Check Algorithm

```python
def check_compliance(action, true_state, workflow_order):
    # Step 1: Identify what unit type this action targets
    target_type = get_action_unit_type(action)
    
    # If not a fix action, no compliance check needed
    if target_type is None:
        return None  # Not counted
    
    # Step 2: Get all currently compromised units
    compromised_units = get_compromised_units(true_state)
    
    # Step 3: Check if any higher priority units are compromised
    target_priority = workflow_order.index(target_type)
    
    for unit_type, units in compromised_units.items():
        if units:  # This type has compromised units
            unit_priority = workflow_order.index(unit_type)
            if unit_priority < target_priority:
                # Found a higher priority compromised unit
                return False  # VIOLATION
    
    return True  # COMPLIANT
```

## Example Scenarios

### Scenario 1: Perfect Compliance ✓
**Workflow**: `['defender', 'enterprise', 'op_server', 'op_host', 'user']`

**Situation**: Enterprise1 and User2 are compromised
- Agent fixes Enterprise1 first → **Compliant** (Enterprise has higher priority than User)
- Agent fixes User2 next → **Compliant** (No higher priority units compromised)
- **Compliance Rate**: 2/2 = 100%

### Scenario 2: Violation ✗
**Workflow**: `['defender', 'enterprise', 'op_server', 'op_host', 'user']`

**Situation**: Defender and User1 are compromised
- Agent fixes User1 first → **Violation** (Defender has higher priority)
- Agent fixes Defender next → **Compliant** (Highest priority)
- **Compliance Rate**: 1/2 = 50%

### Scenario 3: Mixed Actions
**Workflow**: `['op_server', 'defender', 'enterprise', 'user', 'op_host']`

**Situation**: Multiple hosts compromised over time
```
Step 1: Op_Server0 compromised
  → Agent: Restore Op_Server0 ✓ (Compliant - highest priority)
  
Step 2: User1 and Enterprise0 compromised
  → Agent: Restore User1 ✗ (Violation - Enterprise has higher priority)
  
Step 3: Enterprise0 still compromised
  → Agent: Restore Enterprise0 ✓ (Compliant - no higher priority)
  
Step 4: Defender compromised
  → Agent: Analyse Defender ✓ (Compliant - highest among compromised)
```
**Compliance Rate**: 3/4 = 75%

## Alignment Rewards

Based on compliance, the agent receives alignment rewards:

```python
if action_is_compliant:
    alignment_reward = +α  # Default: α = 0.1
else:
    alignment_reward = -β  # Default: β = 0.2
```

This shapes the total reward:
```python
total_reward = environment_reward + alignment_reward
```

## Key Points

1. **Only Fix Actions Count**: Monitor, Decoy, and Sleep actions don't affect compliance
2. **Dynamic Evaluation**: Compliance is checked against the current true state
3. **Priority Matters**: A fix is compliant only if no higher-priority units are compromised
4. **Partial Credit**: Each action is evaluated independently

## Tracking Over Episodes

During training, we track:
- **Per-Episode Compliance**: Individual episode rates
- **Running Average**: Mean compliance over last N episodes
- **Overall Compliance**: Total compliant actions / total fix actions across all episodes

## Example Output
```
Episode 0: Avg Reward=-48.80, Compliance=16.67%
Episode 10: Avg Reward=-145.08, Compliance=34.77%
Episode 20: Avg Reward=-120.50, Compliance=67.23%
```

This shows the agent is learning to better follow the workflow over time.

## Why Compliance Matters

1. **Workflow Verification**: Ensures the agent actually follows the prescribed strategy
2. **Learning Signal**: Low compliance suggests the workflow may be suboptimal
3. **GP-UCB Update**: Compliance rate affects how we update the Gaussian Process:
   ```python
   adjusted_reward = reward * (0.7 + 0.3 * compliance_rate)
   ```
   High compliance = more reliable sample of workflow performance

## Edge Cases

### No Fix Actions
If an episode has no fix actions (agent only monitors/sleeps):
- Compliance = 100% (default, as there were no violations)

### All Units Compromised
If all unit types are compromised:
- Agent should fix in exact workflow order for 100% compliance
- Any deviation reduces compliance rate

### Simultaneous Compromise
If multiple units of the same type are compromised:
- Fixing any of them is equally compliant
- The type priority matters, not the specific unit
