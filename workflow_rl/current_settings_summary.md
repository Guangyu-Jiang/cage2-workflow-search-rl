# Current Settings in train_workflow_rl.py

## Main Training Configuration

### WorkflowRLTrainer Default Settings
```python
# From __init__ method (lines 35-43):
max_steps = 100              # Max steps per episode
eval_episodes = 5            # Episodes for evaluation after training
train_episodes = 50          # Training episodes per workflow
alignment_alpha = 0.1        # Bonus for workflow-compliant fixes
alignment_beta = 0.2         # Penalty for workflow violations
checkpoint_dir = "workflow_rl_checkpoints"
```

### GP-UCB Search Settings
```python
# Line 59:
beta = 2.0                   # Exploration parameter for GP-UCB
```

### Workflow Search Settings
```python
# Line 233:
n_iterations = 20            # Number of workflows to explore
```

## PPO Agent Settings

### From OrderConditionedPPO (order_conditioned_ppo.py lines 128-133):
```python
# PPO Hyperparameters (defaults)
lr = 0.002                   # Learning rate
betas = [0.9, 0.990]        # Adam optimizer betas
gamma = 0.99                 # Discount factor
K_epochs = 4                 # PPO update epochs per batch
eps_clip = 0.2              # PPO clipping parameter

# Workflow-specific (defaults match trainer)
alignment_alpha = 0.1        # Compliance bonus weight
alignment_beta = 0.2         # Violation penalty weight

# Fixed settings
input_dims = 52              # CAGE2 observation space
action_space = list(range(145))  # Full action space (NOT reduced)
order_dims = 25              # One-hot encoding dimensions
```

## Network Architecture

### Actor-Critic Networks (from order_conditioned_ppo.py):
```python
# Input augmentation
augmented_input = 77         # 52 (state) + 25 (order encoding)

# Actor Network
Actor: 77D → 64 → 64 → 145 (Softmax)

# Critic Network  
Critic: 77D → 64 → 64 → 1
```

## Environment Settings

```python
# From train_workflow_rl.py:
scenario_path = '.../Scenarios/Scenario2.yaml'
red_agent_type = B_lineAgent    # Default red agent
agent_name = "Blue"              # We control Blue agent
```

## Main Script Execution Settings

### When running main() (lines 343-355):
```python
# DEMO/TEST SETTINGS (overrides defaults)
max_steps = 50               # Shorter episodes (was 100)
eval_episodes = 3            # Fewer eval episodes (was 5)
train_episodes = 20          # Fewer training episodes (was 50)
alignment_alpha = 0.1        # Same as default
alignment_beta = 0.2         # Same as default
n_iterations = 10            # Fewer workflows to explore (was 20)
```

## Training Process Summary

### Per Workflow:
- **Training**: 50 episodes (or 20 in demo mode)
- **Evaluation**: 5 episodes (or 3 in demo mode)
- **Total steps**: ~50 × 100 = 5,000 steps per workflow

### Full Search:
- **Workflows explored**: 20 (or 10 in demo)
- **Total episodes**: 20 × (50+5) = 1,100 episodes
- **Total environment steps**: ~110,000 steps

### Update Schedule:
- **PPO Update**: After each episode (batch size = 1 episode)
- **K epochs**: 4 passes over episode data
- **GP-UCB Update**: After each workflow evaluation

## Reward Structure

```python
# Total reward per step:
total_reward = env_reward + alignment_reward

# Where alignment_reward:
if fix_aligns_with_workflow:
    alignment_reward = +0.1  # α bonus
elif fix_violates_workflow:
    alignment_reward = -0.2  # β penalty
else:
    alignment_reward = 0.0   # Non-fix action
```

## Key Differences from Original train.py

| Setting | Original train.py | Our Workflow Training |
|---------|------------------|----------------------|
| Episodes | 100,000 | 50 per workflow |
| Update frequency | Every 20,000 steps | Every episode |
| Batch size | 20,000 transitions | ~50 transitions |
| K epochs | 6 | 4 |
| Action space | 28 actions (reduced) | 145 actions (full) |
| Input dims | 52 + 10 (scan state) | 52 + 25 (order encoding) |
| Alignment rewards | None | α=0.1, β=0.2 |

## Configuration for Different Use Cases

### Quick Test (demo_train.py):
```python
train_episodes = 10
eval_episodes = 2
max_steps = 30
n_iterations = 5
```

### Full Training (default):
```python
train_episodes = 50
eval_episodes = 5
max_steps = 100
n_iterations = 20
```

### Extended Training (potential):
```python
train_episodes = 100
eval_episodes = 10
max_steps = 100
n_iterations = 50
```

## Files and Outputs

### Checkpoints:
- Location: `workflow_rl_checkpoints_YYYYMMDD_HHMMSS/`
- Format: `workflow_{id}_agent.pth`
- Contents: Policy weights, optimizer state, workflow order

### Results:
- `training_history.json`: All workflow evaluations
- `best_workflow.json`: Best performing workflow
- `search_statistics.json`: GP-UCB search stats

## Random Seeds
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

These settings balance exploration efficiency with computational constraints, allowing meaningful workflow comparison within reasonable training time.
