# Logging Improvements for Parallel Training

## Changes Made

### 1. Dual Reward Tracking
- **Environment Reward**: The pure reward from the CAGE2 environment (what we ultimately care about)
- **Total Reward**: Environment reward + alignment reward (what PPO actually optimizes)

Both rewards are now tracked and displayed:
- During training progress updates
- In final performance summary
- Helps understand the impact of the alignment reward

### 2. Episode Counting Fixed
**Before**: 
- Showed "Avg Episodes" which would display confusing values like 1, 11, 21...
- This was because it averaged across environments at specific update points

**After**:
- Shows "Total Episodes" - the cumulative count across all parallel environments
- Progress shows: `Episodes: 8, 20, 28, 40...` (for 4 environments)
- More intuitive and easier to understand training progress

### 3. Fixes Per Environment
**Before**:
- Showed "Total Fixes" which could be very large with many environments

**After**:
- Shows "Avg Fixes/Env" - average number of fix actions per environment
- More meaningful metric that's independent of the number of parallel environments
- Easier to compare across different training configurations

## Implementation Details

### Modified Variables
```python
# Added tracking for both reward types
episode_rewards = [[] for _ in range(self.n_envs)]  # Pure env rewards
episode_total_rewards = [[] for _ in range(self.n_envs)]  # Env + alignment rewards

# Track both during episodes
current_episode_rewards += env_rewards  # Pure environment rewards
current_episode_total_rewards += total_rewards  # Total rewards (what PPO optimizes)
```

### Updated Logging
```python
# Progress updates now show:
print(f"  Episodes: {int(total_episodes)}, "
      f"Env Reward: {avg_env_reward:.2f}, "
      f"Total Reward: {avg_total_reward:.2f}, "
      f"Compliance: {avg_compliance:.2%}, "
      f"Avg Fixes/Env: {avg_fixes_per_env:.1f}")

# Final summary shows:
print(f"  Final performance (last 10 eps/env):")
print(f"    Env Reward: {final_avg_env_reward:.2f}, Total Reward: {final_avg_total_reward:.2f}")
print(f"    Compliance: {final_avg_compliance:.2%}, Avg Fixes/Env: {avg_fixes_per_env:.1f}")
print(f"    Total Episodes: {int(total_episodes)} (across {self.n_envs} envs)")
```

## Example Output

```
Episodes: 20, Env Reward: -498.86, Total Reward: -24113.69, Compliance: 74.75%, Avg Fixes/Env: 90.5
```

This shows:
- 20 total episodes completed (across all environments)
- Average environment reward: -498.86
- Average total reward (with alignment): -24113.69 (much more negative due to alignment penalties)
- Compliance rate: 74.75%
- Average fixes per environment: 90.5

## Benefits

1. **Clarity**: Users can see exactly what rewards are being optimized vs. what the actual game performance is
2. **Consistency**: Episode counting now makes intuitive sense
3. **Comparability**: Metrics are normalized per-environment, making it easier to compare different configurations
4. **Debugging**: The large difference between env and total rewards helps identify when alignment rewards are dominating
