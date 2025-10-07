# PPO Batch Size and Parallelization Details

## Current Implementation: Sequential, Episode-Based Updates

### **NO Parallel Environments Used**

The current implementation uses **1 environment** running **sequentially**:

```python
def train_workflow(self, workflow_order, ...):
    for episode in range(self.train_episodes):  # 50 episodes
        env, cyborg = self.create_env()  # Single environment
        obs = env.reset()
        
        for step in range(self.max_steps):  # Up to 100 steps
            action = agent.get_action(obs)
            next_obs, reward, done, _ = env.step(action)
            # Store in memory buffer
            
        # Update after each episode
        agent.update()  # PPO update on collected data
```

### **Batch Size = 1 Episode**

The effective batch size is the data from **one complete episode**:
- **Steps per episode**: Variable, up to 100 (episode may end early)
- **Typical episode length**: 30-80 steps
- **Batch size**: ~30-80 transitions per update

### **Memory Buffer Pattern**

```python
class Memory:
    def __init__(self):
        self.actions = []      # Collected over 1 episode
        self.states = []       # Size: [episode_length]
        self.logprobs = []     
        self.rewards = []      
        self.is_terminals = []
    
    def clear_memory(self):
        # Cleared after each PPO update (every episode)
```

### **PPO Update Process**

After each episode:
1. **Collect trajectory**: 30-80 state-action-reward tuples
2. **Compute returns**: Monte Carlo discounted rewards
3. **K epochs of updates**: Default K=4
4. **Clear memory**: Start fresh for next episode

```python
def update(self):
    # Process entire episode's data
    old_states = torch.stack(self.memory.states)  # [episode_length, 77]
    old_actions = torch.tensor(self.memory.actions)  # [episode_length]
    
    # Update K times on same data
    for _ in range(self.K_epochs):  # K = 4
        # Compute PPO loss on full episode
        # Update network weights
```

## Comparison with Standard PPO

### Standard PPO (Typical Implementation):
```python
# Parallel environments
envs = VectorEnv(num_envs=8)  # 8 parallel environments
batch_size = 2048  # Large batch

for iteration in range(num_iterations):
    # Collect batch_size transitions
    batch = collect_rollout(envs, batch_size)  
    
    # Multiple epochs on minibatches
    for epoch in range(K_epochs):
        for minibatch in batch.split(minibatch_size=64):
            update_network(minibatch)
```

### Our Implementation:
```python
# Single environment
env = CybORG(...)  # 1 environment
batch_size = ~50  # One episode

for episode in range(train_episodes):
    # Collect one episode
    trajectory = collect_episode(env)  
    
    # K epochs on full episode
    for epoch in range(K_epochs):
        update_network(trajectory)  # No minibatches
```

## Why No Parallelization?

### 1. **CAGE2 Environment Constraints**
- CybORG is not trivially parallelizable
- Each instance maintains complex internal state
- Red agent behavior depends on history

### 2. **Workflow Conditioning**
- Each PPO agent is conditioned on specific workflow
- Parallel envs would need same workflow
- Limited benefit for workflow search

### 3. **Computational Trade-offs**
- Single env: Simple, debuggable, reproducible
- Parallel envs: Complex synchronization, memory overhead

## Actual Data Flow

### Per Episode:
```
Step 1:  State_1 → Action_1 → Reward_1
Step 2:  State_2 → Action_2 → Reward_2
...
Step 50: State_50 → Action_50 → Reward_50
[Episode ends or reaches max_steps]

Batch = [State_1...State_50, Action_1...Action_50, Reward_1...Reward_50]
Update PPO with Batch (K=4 epochs)
Clear memory, start next episode
```

### Per Workflow Training (50 episodes):
```
Episodes 1-10:   50-80 steps each → 500-800 total transitions
Episodes 11-20:  50-80 steps each → 500-800 total transitions  
Episodes 21-30:  50-80 steps each → 500-800 total transitions
Episodes 31-40:  50-80 steps each → 500-800 total transitions
Episodes 41-50:  50-80 steps each → 500-800 total transitions

Total: ~2500-4000 transitions per workflow
```

## Implications for Learning

### Advantages of Episode-Based Updates:
1. **Natural boundaries**: Update at episode completion
2. **Complete trajectories**: Full return calculation
3. **Workflow compliance**: Can track full episode compliance
4. **Simpler implementation**: No parallel sync issues

### Disadvantages:
1. **Smaller batches**: Higher variance in gradients
2. **Slower wall-clock time**: No parallel speedup
3. **Correlated samples**: All from same episode
4. **Less diverse experience**: Single environment trajectory

## Potential Improvements (Not Implemented)

### Option 1: Collect Multiple Episodes Before Update
```python
UPDATE_FREQUENCY = 5  # Episodes

for episode in range(train_episodes):
    collect_episode()  # Add to memory
    
    if episode % UPDATE_FREQUENCY == 0:
        agent.update()  # Batch = 5 episodes
        agent.memory.clear()
```

### Option 2: Parallel Workflow Training
```python
# Train multiple workflows simultaneously
workflows = [workflow_1, workflow_2, workflow_3, workflow_4]
envs = [create_env() for _ in range(4)]
agents = [PPO(workflow=w) for w in workflows]

for step in range(max_steps):
    actions = [agent.get_action(obs) for agent, obs in zip(agents, observations)]
    observations, rewards, dones = parallel_step(envs, actions)
```

### Option 3: Vectorized Single Workflow
```python
# Multiple copies of same workflow
num_envs = 4
envs = VectorEnv([create_env() for _ in range(num_envs)])
agent = PPO(workflow=selected_workflow)

# Collect from all envs
for step in range(steps_per_update // num_envs):
    actions = agent.get_actions(observations)  # [num_envs] actions
    observations, rewards, dones = envs.step(actions)
```

## Summary

- **Batch Size**: 1 episode (~30-80 transitions)
- **Parallel Environments**: None (1 sequential environment)
- **Update Frequency**: After every episode
- **K Epochs**: 4 updates on same episode data
- **Total Data per Workflow**: ~2500-4000 transitions over 50 episodes

This sequential, episode-based approach trades computational efficiency for implementation simplicity and is sufficient for the workflow search task where the focus is on comparing different workflow strategies rather than achieving maximum sample efficiency.
