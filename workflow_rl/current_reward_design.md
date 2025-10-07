# Current Customized Reward Design

## Overview
The reward system combines the original CAGE2 environment reward with a custom alignment reward that encourages adherence to a prescribed workflow (priority order) for fixing compromised hosts.

## Total Reward Formula
```
Total Reward = Environment Reward + Alignment Reward
```

## Components

### 1. Environment Reward
- The original reward from CAGE2 environment
- Based on the availability and confidentiality values of hosts
- Negative when hosts are compromised
- Typically ranges from -100 to -1000 per step

### 2. Alignment Reward
The alignment reward is computed based on whether fix actions (Remove and Restore) follow the prescribed workflow order.

#### Default Parameters
- **α (alpha)** = 0.1 (bonus for compliant fixes)
- **β (beta)** = 0.2 (penalty for violations)

#### Calculation Logic

```python
For each Remove or Restore action:
    1. Identify the target host type (defender, enterprise, op_server, op_host, user)
    2. Check if this violates the workflow order:
       - Violation occurs if any higher-priority type hasn't been fixed yet
       - We track fixed types using env_fixed_types set
    3. Apply reward:
       - If violation: alignment_reward -= β (subtract 0.2)
       - If compliant: alignment_reward += α (add 0.1)
    4. Mark this type as fixed in env_fixed_types
```

## Action Mapping

### Remove Actions (indices 15-27)
- 15: Defender
- 16-18: Enterprise (0,1,2)
- 19-21: Op_Host (0,1,2)
- 22: Op_Server
- 23-27: User (0,1,2,3,4)

### Restore Actions (indices 132-144)
- 132: Defender
- 133-135: Enterprise (0,1,2)
- 136-138: Op_Host (0,1,2)
- 139: Op_Server
- 140-144: User (0,1,2,3,4)

## Example Scenarios

### Scenario 1: Compliant Fix Sequence
Workflow order: `['defender', 'enterprise', 'op_server', 'op_host', 'user']`

1. Agent removes malware from Defender → +0.1 (compliant, first priority)
2. Agent restores Enterprise0 → +0.1 (compliant, defender already fixed)
3. Agent removes malware from Op_Server → +0.1 (compliant, higher priorities fixed)

**Total alignment reward**: +0.3

### Scenario 2: Violation
Workflow order: `['defender', 'enterprise', 'op_server', 'op_host', 'user']`

1. Agent restores User0 → -0.2 (violation, defender not fixed yet)
2. Agent removes malware from Defender → +0.1 (compliant, highest priority)
3. Agent restores Enterprise1 → +0.1 (compliant, defender fixed)

**Total alignment reward**: 0.0 (-0.2 + 0.1 + 0.1)

## Key Design Decisions

### 1. Order-Based Tracking
- We track which **types** have been fixed, not individual hosts
- Once any host of a type is fixed, that type is considered "handled"
- This prevents repeated penalties for fixing multiple hosts of the same type

### 2. Asymmetric Rewards
- Penalty (β = 0.2) is larger than bonus (α = 0.1)
- This makes violations more costly, encouraging strict adherence to the workflow

### 3. Fix-Action Focus
- Only Remove (15-27) and Restore (132-144) actions affect alignment reward
- Other actions (Monitor, Analyze, etc.) have no alignment impact
- This focuses the learning on the critical defensive actions

### 4. Episode Reset
- The `env_fixed_types` set is cleared at episode boundaries
- Each episode starts fresh with no types marked as fixed
- This ensures consistent learning across episodes

## Impact on Training

### Reward Magnitude Comparison
- **Environment reward**: Typically -200 to -600 per episode
- **Alignment component**: Can add -50 to +20 per episode
- **Total reward (PPO optimizes)**: Can be -15,000 to -35,000 per episode

The large negative total rewards show that alignment penalties significantly affect the optimization, pushing the agent to learn workflow-compliant behavior even if it means slightly worse immediate environment rewards.

### Learning Dynamics
1. **Early Training**: Agent explores randomly, many violations, large negative total rewards
2. **Mid Training**: Agent learns to avoid penalties, compliance improves
3. **Late Training**: Agent balances environment performance with workflow compliance

## Current Performance Metrics
From recent tests:
- **Environment Reward**: ~-300 to -500 (good performance)
- **Total Reward**: ~-15,000 to -25,000 (includes alignment penalties)
- **Compliance Rate**: 40-75% (varies based on workflow difficulty)
- **Average Fixes/Environment**: 200-350 actions per training

## Tuning Considerations

### To Increase Compliance:
- Increase β (penalty) to make violations more costly
- Increase α (bonus) to make compliance more rewarding
- Increase the α/β ratio if compliance is too low

### To Improve Environment Performance:
- Decrease both α and β to reduce alignment influence
- Allow more exploration by reducing penalty magnitude

### Current Balance:
The 0.1/0.2 (α/β) ratio provides moderate pressure toward compliance while still allowing the agent to optimize for environment performance when beneficial.
