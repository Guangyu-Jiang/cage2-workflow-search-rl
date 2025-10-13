# Compliance-Gated Workflow Search

## Overview

This document describes the **compliance-gated workflow search** strategy, which ensures that each workflow evaluation is based on meaningful, high-compliance policy behavior.

## Core Problem

In workflow search, we want to:
1. **Find the best workflow order** for fixing compromised hosts
2. **Ensure each workflow is properly tested** with compliant agent behavior

The challenge: If the agent doesn't follow the prescribed workflow (low compliance), then the measured reward doesn't reflect the true performance of that workflow.

## Solution: Compliance-Gated Training and Evaluation

### Strategy

For each workflow candidate:

1. **Training Phase**: Train PPO with alignment rewards until compliance ≥ 95%
   - Use customized reward: `Total Reward = Environment Reward + Alignment Reward`
   - Alignment reward encourages following the workflow order
   - Continue training until compliance threshold is met (or max episodes)

2. **Evaluation Phase**: Evaluate on pure environment reward
   - Run policy without alignment rewards
   - Measure only the original CAGE2 environment reward
   - This reflects the true performance of the workflow

3. **GP-UCB Update**: Use evaluation reward for workflow search
   - Only the pure environment reward is used for GP-UCB
   - This ensures fair comparison across workflows

### Key Insight

**Training** uses alignment rewards to learn compliance.  
**Evaluation** uses pure environment rewards to measure performance.  
**GP-UCB** searches based on evaluation performance.

This separates the learning objective (compliance) from the search objective (environment performance).

## Implementation Details

### Training Loop

```python
while min_episodes < max_episodes:
    # 1. Get actions from policy
    actions = agent.get_actions(observations)
    
    # 2. Step environment
    observations, env_rewards, dones = envs.step(actions)
    
    # 3. Compute alignment rewards (compliance-delta)
    alignment_rewards = agent.compute_alignment_rewards(actions, states, dones)
    
    # 4. Combine for PPO training
    total_rewards = env_rewards + alignment_rewards
    agent.store_transition(observations, actions, total_rewards, dones)
    
    # 5. Check compliance after sufficient episodes
    if episodes >= min_episodes:
        avg_compliance = compute_recent_compliance()
        if avg_compliance >= compliance_threshold:
            break  # Training successful!
```

### Evaluation Loop

```python
# Evaluate on pure environment reward
eval_rewards = []
for episode in range(n_eval_episodes):
    episode_reward = 0
    observation = env.reset()
    
    while not done:
        action = agent.get_action(observation, deterministic=True)
        observation, reward, done = env.step(action)
        episode_reward += reward  # Only environment reward!
    
    eval_rewards.append(episode_reward)

eval_reward = mean(eval_rewards)
```

### GP-UCB Update

```python
if compliance_achieved:
    # Use evaluation reward for GP-UCB
    gp_ucb.add_observation(workflow, eval_reward)
else:
    # Do NOT record failed samples - skip this workflow
    # (No observation added, workflow will not influence search)
    print("Compliance not achieved - sample NOT recorded in GP-UCB")
```

## Parameters

### Training Parameters

- **`alignment_lambda`**: 30.0 (increased from 10.0 for stricter compliance)
  - Controls the strength of alignment rewards
  - Formula: `S = λ * compliance_rate`
  - Higher λ → stronger compliance pressure

- **`compliance_threshold`**: 0.95
  - Required compliance rate before evaluation
  - 95% means agent follows workflow order in 95% of fix actions

- **`min_episodes`**: 25
  - Minimum episodes before checking compliance
  - Gives agent time to learn before evaluation

- **`max_train_episodes_per_env`**: 100
  - Maximum episodes per environment per workflow before giving up
  - Early stopping at 95% compliance (typically 25-75 episodes)
  - If compliance not achieved, workflow is marked as failed and NOT recorded
  - **Policy inheritance**: Each workflow inherits weights from previous workflow

### Evaluation Parameters

- **`n_eval_episodes`**: 20
  - Number of episodes per environment for evaluation
  - More episodes = more reliable performance estimate

- **`n_envs`**: 25
  - Parallel environments for training and evaluation
  - Total evaluation rollouts = 20 × 25 = 500 episodes

## Policy Inheritance Across Workflows

### Key Feature: Continual Learning

Instead of training each workflow from scratch, the system uses **policy inheritance**:

1. **First workflow**: Creates a new policy from random initialization
2. **Subsequent workflows**: Inherits policy weights from previous workflow
   - Policy network weights are copied
   - Only the workflow order encoding changes
   - Continues learning from where previous workflow left off

### Benefits

✅ **Faster convergence**: Each workflow starts with knowledge from previous workflows  
✅ **Better generalization**: Agent learns general defense strategies, not just workflow-specific tactics  
✅ **Reduced training time**: Later workflows train faster due to knowledge transfer  
✅ **Cumulative learning**: Knowledge accumulates across the entire search

### Implementation

```python
# First workflow
if self.shared_agent is None:
    agent = create_new_agent()  # Random initialization
else:
    # Later workflows
    agent = create_new_agent()
    agent.policy.load_state_dict(self.shared_agent.policy.state_dict())  # Inherit weights
    # Only workflow encoding changes, core policy is preserved

# After training
self.shared_agent = agent  # Save for next workflow
```

### Example

| Workflow | Starting Point | Training Episodes | Final Compliance |
|----------|---------------|-------------------|------------------|
| 1: `user → ...` | Random init | 75 episodes | 90% (failed) |
| 2: `defender → ...` | Inherits from #1 | 35 episodes | 96% (success!) |
| 3: `enterprise → ...` | Inherits from #2 | 28 episodes | 97% (success!) |

Notice how later workflows train faster due to inherited knowledge!

---

## Alignment Reward Design

### Compliance-Delta Shaping

The alignment reward uses a **compliance-delta** approach:

```python
# Current compliance score
current_score = λ * (compliant_fixes / total_fixes)

# Alignment reward = change in score
alignment_reward = current_score - previous_score
```

**Benefits**:
1. **Per-step feedback**: Reward is given immediately when compliance changes
2. **Normalized**: Score is normalized by total fixes (0 to λ)
3. **Smooth learning**: Avoids sparse, episode-end rewards

**Example** (λ = 30):
- Agent makes compliant fix: compliance 4/5 → 5/6
  - Score change: 30×0.80 → 30×0.833 = +1.0 reward
- Agent violates order: compliance 5/6 → 5/7
  - Score change: 30×0.833 → 30×0.714 = -3.6 penalty

## Success Criteria

A workflow training is considered **successful** if:

1. ✓ Compliance rate ≥ 95% (averaged over last 5 episodes)
2. ✓ At least 10 total fix actions detected
3. ✓ Within max episode limit

If successful:
- Agent is saved as `workflow_{id}_compliant_agent.pth`
- Evaluation phase runs
- Evaluation reward is used for GP-UCB

If failed:
- No evaluation phase
- No observation added to GP-UCB
- Workflow is skipped entirely (will not influence search)

## Training Output

### Phase 1: Training

```
============================================================
Training with workflow: defender → enterprise → op_server → op_host → user
Goal: Train until compliance >= 95.0%
Using 25 parallel environments
============================================================

  Update 0: Episodes: 25 total
    Env Reward/Episode: -529.70
    Total Reward/Episode: -512.35
    Alignment Bonus (episode-end): +17.35
    Compliance: 82.50%
    Avg Fixes/Episode: 18.9

  Update 1: Episodes: 50 total
    Env Reward/Episode: -498.22
    Total Reward/Episode: -470.15
    Alignment Bonus (episode-end): +28.07
    Compliance: 91.20%
    Avg Fixes/Episode: 17.2

  Update 2: Episodes: 75 total
    Env Reward/Episode: -485.10
    Total Reward/Episode: -456.33
    Alignment Bonus (episode-end): +28.77
    Compliance: 95.90%
    Avg Fixes/Episode: 16.8

  ✓ Compliance threshold achieved!
    Episodes trained: 78 per env
    Compliance: 95.90%
    Total fixes detected: 421
```

### Phase 2: Evaluation

```
============================================================
EVALUATION PHASE
============================================================
Evaluating agent on PURE ENVIRONMENT REWARD (no alignment)
Running 20 episodes per environment...

Evaluation Results:
  Environment Reward: -472.35
  Compliance (eval): 96.20%
  → This reward will be used for GP-UCB
============================================================

✓ GP-UCB updated with evaluation reward: -472.35
```

### Iteration Summary

```
============================================================
ITERATION 1 SUMMARY
============================================================
  Workflow: defender → enterprise → op_server → op_host → user
  Training Episodes: 78 per env
  Final Compliance: 95.90%
  Success: ✓ Yes
  Evaluation Reward (for GP-UCB): -472.35
============================================================
```

## Advantages Over Previous Approach

### Old Approach
- Train for fixed number of episodes (e.g., 2500 per env)
- Use training reward (env + alignment) for GP-UCB
- No guarantee of high compliance
- Difficult to compare workflows fairly

**Problems**:
- Low compliance → not testing the intended workflow
- Alignment rewards bias GP-UCB search
- Wasted training on unpromising workflows

### New Approach (Compliance-Gated)
- Train until compliance ≥ 95%
- Use evaluation reward (pure env) for GP-UCB
- Guaranteed high compliance for all comparisons
- Early stopping saves computation

**Benefits**:
- High compliance → meaningful workflow tests
- Fair comparison (same evaluation metric)
- Efficient (stop early when compliant)
- Clear separation of learning vs. search objectives

## Tuning Guidelines

### If Compliance is Too Hard to Achieve

**Symptoms**:
- Many workflows fail to reach 95% compliance
- Training often hits max episode limit

**Solutions**:
1. **Increase `alignment_lambda`** (e.g., 30 → 50)
   - Stronger compliance pressure
2. **Decrease `compliance_threshold`** (e.g., 95% → 90%)
   - Lower bar for success
3. **Increase `max_train_episodes_per_env`** (e.g., 200 → 500)
   - More time to learn

### If Training is Too Slow

**Symptoms**:
- Takes many episodes to reach compliance
- Excessive training time per workflow

**Solutions**:
1. **Increase `alignment_lambda`** (faster learning)
2. **Decrease `min_episodes`** (check compliance earlier)
3. **Reduce `n_eval_episodes`** (faster evaluation)
   - Trade-off: less reliable performance estimate

### If Evaluation Reward is Unstable

**Symptoms**:
- High variance in evaluation rewards
- GP-UCB can't identify best workflow

**Solutions**:
1. **Increase `n_eval_episodes`** (e.g., 20 → 50)
   - More samples → lower variance
2. **Use more environments** (e.g., 25 → 50)
   - Total evaluation: 50 × 50 = 2500 episodes

## Expected Behavior

### Workflow Difficulty Variation

Some workflows are naturally harder to learn:

**Easy workflows** (natural priority):
- `defender → enterprise → op_server → op_host → user`
- Compliance achieved in ~50-100 episodes
- High evaluation performance

**Hard workflows** (unnatural priority):
- `user → op_host → op_server → enterprise → defender`
- May require 150-200 episodes for compliance
- Lower evaluation performance (harder to defend in this order)

**Impossible workflows**:
- Some workflows may never achieve 95% compliance within 100 episodes
- Not recorded in GP-UCB
- GP-UCB ignores them (as if they were never explored)
- However, the learned policy is still inherited by next workflow

### GP-UCB Learning

Over iterations, GP-UCB should:
1. **Explore diverse workflows** early (high uncertainty)
2. **Identify promising regions** (high mean reward)
3. **Exploit best workflows** later (high UCB score)
4. **Skip failed workflows** (not recorded, so GP-UCB may re-explore similar ones)

**Typical progression**:
- Iteration 1-5: Random/diverse exploration
- Iteration 6-10: Start exploiting good workflows
- Iteration 11-15: Refinement around best region
- Iteration 16-20: Final exploitation of best workflow

## Code Location

**Main training file**:
- `/home/ubuntu/CAGE2/-cyborg-cage-2/workflow_rl/parallel_train_workflow_rl.py`

**Key methods**:
- `train_workflow_parallel()`: Training phase (compliance-gated)
- `evaluate_pure_performance_parallel()`: Evaluation phase
- `run_workflow_search()`: Main GP-UCB loop

**PPO agent**:
- `/home/ubuntu/CAGE2/-cyborg-cage-2/workflow_rl/parallel_order_conditioned_ppo.py`

**Key methods**:
- `compute_alignment_rewards()`: Compliance-delta reward
- `get_compliance_rates()`: Current compliance per environment
- `reset_episode_compliance()`: Reset tracking at episode end

## Running the Training

```bash
cd /home/ubuntu/CAGE2/-cyborg-cage-2
conda activate CAGE2
export PYTHONPATH=/home/ubuntu/CAGE2/-cyborg-cage-2:$PYTHONPATH
python workflow_rl/parallel_train_workflow_rl.py
```

**Expected runtime**:
- ~5-15 minutes per workflow (depending on difficulty)
- ~2-5 hours total for 20 workflows
- Checkpoints saved in `compliance_checkpoints/`
- Training history saved in `parallel_training_history.json`

## Results Interpretation

### Best Workflow

The workflow with the **highest evaluation reward** is the best.

**Example**:
```
Best Workflow: defender → enterprise → op_server → op_host → user
Best Reward: -465.20
```

**Interpretation**:
- This workflow order results in the least negative total episode reward
- When agent follows this order, hosts stay protected longer
- Lower magnitude = better defense performance

### Success Rate

Track how many workflows achieved compliance:

```
Successful workflows: 18/20 (90%)
```

**If success rate is low** (<50%):
- Workflows may be too difficult
- Increase `alignment_lambda` or adjust compliance threshold

### Compliance vs. Performance

Compare training compliance with evaluation performance:

```
Workflow: user → defender → enterprise → op_server → op_host
  Compliance: 96.5%  ← High compliance achieved
  Eval Reward: -712.3  ← But poor performance
```

**Interpretation**:
- Agent successfully learned to follow this workflow
- However, this workflow order is ineffective for defense
- GP-UCB will learn to avoid similar workflows

## Future Enhancements

### Adaptive Lambda

Automatically adjust `alignment_lambda` based on compliance progress:

```python
if compliance < 80% and episodes > 50:
    alignment_lambda *= 1.5  # Increase pressure
```

### Two-Phase Training

1. **Phase 1**: High lambda for rapid compliance learning
2. **Phase 2**: Lower lambda for performance optimization

### Action Masking

Instead of reward shaping, directly mask non-compliant actions:

```python
if not is_compliant(action, workflow):
    action_probs[action] = 0  # Disable this action
```

This guarantees 100% compliance but is more restrictive.

## Summary

The **compliance-gated workflow search** ensures meaningful evaluation by:

1. ✓ Training with alignment rewards until high compliance
2. ✓ Evaluating on pure environment reward
3. ✓ Using evaluation reward for GP-UCB search

This approach provides:
- **Fair comparison** across workflows
- **Reliable performance** estimates
- **Efficient training** via early stopping
- **Clear separation** of learning and search objectives

**Key takeaway**: We train for compliance, evaluate for performance, and search based on evaluation.

