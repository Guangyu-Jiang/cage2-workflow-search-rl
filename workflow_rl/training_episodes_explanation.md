# Training Episodes and Stopping Criteria

## Episode Configuration

### Default Settings (Full Training)
```python
train_episodes = 50      # Episodes per workflow
eval_episodes = 5        # Episodes for evaluation
max_steps = 100          # Steps per episode
n_iterations = 20        # Number of workflows to explore
```

### Demo Settings (Quick Test)
```python
train_episodes = 10      # Fewer episodes for quick demo
eval_episodes = 2        # Minimal evaluation
max_steps = 30           # Shorter episodes
n_iterations = 5         # Test fewer workflows
```

## Training Process for Each Workflow

### 1. **Inner Loop: PPO Training (50 episodes per workflow)**

For each selected workflow, the PPO agent trains for exactly **50 episodes**:

```python
def train_workflow(self, workflow_order, ...):
    agent = WorkflowConditionedPPO(workflow=workflow_order)
    
    for episode in range(self.train_episodes):  # 50 episodes
        obs = env.reset()
        
        for step in range(self.max_steps):  # 100 steps max
            action = agent.get_action(obs)
            next_obs, env_reward, done, _ = env.step(action)
            
            # Compute alignment reward
            align_reward = agent.compute_alignment_reward(...)
            total_reward = env_reward + align_reward
            
            agent.update(total_reward, done)
            
            if done:
                break
        
        # Log progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode}: Avg Reward={avg_reward:.2f}")
```

### 2. **Stopping Criteria for Current Workflow**

The training for a specific workflow stops after **exactly 50 episodes**. There's **no early stopping** based on performance or compliance. This is intentional because:

1. **Fixed Budget**: Ensures fair comparison between workflows
2. **Exploration**: Allows each workflow sufficient time to learn
3. **GP-UCB Needs Consistent Data**: Variable training lengths would bias the search

### 3. **When Training Moves to Next Workflow**

After completing 50 training episodes:
1. **Save checkpoint**: Agent saved as `workflow_{id}_agent.pth`
2. **Evaluate**: Run 5 evaluation episodes
3. **Update GP-UCB**: Add (workflow, reward) to observations
4. **Select next**: GP-UCB picks next workflow to explore

## Total Training Budget

### Full Training Run:
- **Workflows explored**: 20
- **Episodes per workflow**: 50 training + 5 evaluation = 55
- **Total episodes**: 20 × 55 = 1,100 episodes
- **Total environment steps**: ~1,100 × 100 = 110,000 steps

### Demo Run:
- **Workflows explored**: 5
- **Episodes per workflow**: 10 training + 2 evaluation = 12
- **Total episodes**: 5 × 12 = 60 episodes
- **Total environment steps**: ~60 × 30 = 1,800 steps

## Why Fixed Episodes (No Early Stopping)?

### 1. **Fair Comparison**
- All workflows get equal training opportunity
- Performance differences reflect workflow quality, not training time

### 2. **Compliance Learning Takes Time**
```
Episodes 1-10:  Agent learns basic defense actions
Episodes 11-30: Agent starts following workflow order
Episodes 31-50: Agent refines compliance while maintaining defense
```

### 3. **Avoiding Local Minima**
- Early high compliance might mean doing nothing
- Need time to balance defense effectiveness with workflow compliance

### 4. **GP-UCB Consistency**
- Variable training would introduce another source of uncertainty
- Fixed budget ensures rewards are comparable

## Monitoring During Training

### Progress Indicators (Every 10 Episodes):
```
Episode 10: Avg Reward=-45.32, Compliance=42%
Episode 20: Avg Reward=-38.21, Compliance=65%
Episode 30: Avg Reward=-35.87, Compliance=78%
Episode 40: Avg Reward=-34.12, Compliance=85%
Episode 50: Avg Reward=-33.45, Compliance=88%
```

### Key Metrics:
- **Average Reward**: Rolling mean of last 10 episodes
- **Compliance Rate**: Percentage of fixes following workflow order
- **Alignment Rewards**: Tracking bonus/penalty distribution

## Alternative Stopping Strategies (Not Used)

### Could Consider (But Don't):

1. **Performance Plateau**
   ```python
   if abs(current_avg - previous_avg) < threshold:
       stop_training()
   ```
   Problem: Might stop before learning compliance

2. **High Compliance Target**
   ```python
   if compliance_rate > 0.95:
       stop_training()
   ```
   Problem: High compliance ≠ good defense

3. **Adaptive Episodes**
   ```python
   episodes = base_episodes * (1 + uncertainty_factor)
   ```
   Problem: Complicates workflow comparison

## Revisiting Workflows

### How GP-UCB Handles Revisits:
```python
# Reduce exploration bonus for frequently visited workflows
revisit_count = observation_counts[workflow]
adjusted_β = β / (1 + 0.5 × revisit_count)
```

- First visit: Full exploration bonus (β = 2.0)
- Second visit: Reduced bonus (β ≈ 1.33)
- Third visit: Further reduced (β = 1.0)

This naturally causes the search to explore new workflows unless a revisited one shows exceptional promise.

## Summary

- **50 episodes per workflow** - Fixed, no early stopping
- **Move to next workflow** - After exactly 50 training + 5 eval episodes
- **Total exploration** - 20 workflows in full run
- **Why fixed?** - Fair comparison, sufficient learning, GP-UCB consistency
- **Revisits allowed** - But with reduced exploration bonus

This design ensures each workflow gets a fair chance to demonstrate its potential while maintaining efficient exploration of the workflow space.
