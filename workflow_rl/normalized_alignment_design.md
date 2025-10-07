# Normalized Alignment Reward Design

## Problem Solved
The original per-action alignment reward design encouraged agents to perform excessive fix actions to maximize rewards, regardless of actual effectiveness. This led to suboptimal behavior where agents would spam fix actions.

## Solution: Episode-Level Normalized Rewards

### Key Features
1. **Compliance Rate Based**: Reward depends on the percentage of compliant fixes, not the total count
2. **Episode-End Only**: Single reward given at episode termination
3. **No Gaming Incentive**: Agent cannot increase reward by doing more fixes

### Reward Formula
At episode end:
```python
if total_fix_actions > 0:
    compliance_rate = compliant_actions / total_fix_actions
    alignment_reward = α * compliance_rate - β * (1 - compliance_rate)
else:
    alignment_reward = -β * 0.5  # Small penalty for no fixes
```

### Example with α=50, β=50:
- 100% compliance: +50 bonus
- 75% compliance: +25 bonus (0.75*50 - 0.25*50)
- 50% compliance: 0 (neutral)
- 25% compliance: -25 penalty
- 0% compliance: -50 penalty
- No fixes: -25 penalty

### Implementation Details

#### In `parallel_order_conditioned_ppo.py`:
- `compute_alignment_rewards()` tracks compliance during episode
- Returns non-zero reward only when `dones[env_idx] == True`
- Reward based on overall compliance rate for the episode

#### In `parallel_train_workflow_rl.py`:
- Alignment rewards computed after each step
- Added to environment rewards in the buffer
- Affects only the final step of each episode

### Benefits
1. **No Excessive Fixing**: Agent learns to fix efficiently, not excessively
2. **Quality Over Quantity**: Focuses on following the workflow correctly
3. **Balanced Incentives**: Encourages fixing (penalty for no fixes) but prioritizes compliance
4. **Stable Learning**: Single large reward at episode end provides clear signal

### Tuning Parameters
- **α (alignment_alpha)**: Bonus weight for compliance (default: 10.0)
- **β (alignment_beta)**: Penalty weight for violations (default: 10.0)
- Increase both for stronger workflow adherence
- Decrease both to prioritize environment performance

### Current Status
The normalized alignment reward system is fully implemented and working. The agent receives:
1. Environment rewards at each step
2. Alignment bonus/penalty at episode end based on compliance rate

This design successfully prevents gaming while encouraging workflow adherence.
