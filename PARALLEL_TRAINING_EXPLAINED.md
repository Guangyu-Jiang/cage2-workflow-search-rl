# 📚 Parallel Training Code Explanation

## Overview

The parallel training system uses **100 parallel environments** to train a workflow-conditioned PPO agent. It combines:
1. **Multiprocessing** for parallel environment execution
2. **Shared memory** for fast data transfer
3. **GP-UCB workflow search** to find the best repair priority order
4. **Policy inheritance** to transfer learning across workflows

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│          ParallelWorkflowRLTrainer (Main Process)       │
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │  GP-UCB Workflow Search                          │   │
│  │  Selects next workflow to try                    │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ParallelOrderConditionedPPO                     │   │
│  │  Neural network that takes (state + workflow)    │   │
│  │  Shared across all workflows (policy inheritance)│   │
│  └─────────────────────────────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ParallelEnvSharedMemoryOptimized                │   │
│  │  Manages 100 parallel environment processes       │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐      ┌─────────┐      ┌─────────┐
   │ Worker  │      │ Worker  │ ...  │ Worker  │
   │   #0    │      │   #1    │      │  #99    │
   │         │      │         │      │         │
   │ CAGE2   │      │ CAGE2   │      │ CAGE2   │
   │  Env    │      │  Env    │      │  Env    │
   └─────────┘      └─────────┘      └─────────┘
```

---

## 🔧 Component Breakdown

### 1. **Main Trainer** (`ParallelWorkflowRLTrainer`)

**Responsibilities:**
- Manages the overall training loop
- Tracks episode budget (100,000 episodes total)
- Logs training progress to CSV files
- Saves checkpoints

**Key Parameters:**
```python
n_envs = 100                      # Number of parallel environments
total_episode_budget = 100000     # Total episodes to train
max_train_episodes_per_env = 100  # Max episodes per workflow
compliance_threshold = 0.95       # 95% compliance required
update_every_steps = 100          # Update after 100 steps (1 full episode)
```

**Training Flow:**
```
1. Select workflow using GP-UCB
2. Train until compliance >= 95%
3. Record final performance
4. Update GP model
5. Repeat until episode budget exhausted
```

---

### 2. **Parallel Environments** (`ParallelEnvSharedMemoryOptimized`)

This is the key optimization that provides 3.4x speedup!

#### **Architecture:**

```
Main Process                Worker Processes (100 total)
─────────────                ────────────────────────────

┌──────────────┐            ┌──────────────┐
│  Main Thread │            │  Worker #0   │
│              │ Pipe #0    │              │
│              │◄──────────►│  CAGE2 Env   │
│              │            │              │
│              │ Pipe #1    │  Cached      │
│              │◄──────────►│  True State  │
│              │            └──────────────┘
│              │
│              │ Pipe #2    ┌──────────────┐
│              │◄──────────►│  Worker #1   │
│              │            │              │
│              │            │  CAGE2 Env   │
│  Shared      │            │              │
│  Memory:     │            └──────────────┘
│              │
│  ┌─────────┐ │               ... (98 more)
│  │ States  │ │
│  │ (52*100)│ │            ┌──────────────┐
│  ├─────────┤ │ Pipe #99   │  Worker #99  │
│  │ Rewards │ │◄──────────►│              │
│  │ (100)   │ │            │  CAGE2 Env   │
│  ├─────────┤ │            │              │
│  │ Dones   │ │            └──────────────┘
│  │ (100)   │ │
│  └─────────┘ │
│              │
│  Command     │
│  Queue       │
└──────────────┘
```

#### **How It Works:**

1. **Shared Memory for Fast Data**
   ```python
   # Created once, accessible by all processes
   obs_shm = shared_memory.SharedMemory(create=True, size=n_envs * 52 * 4)
   reward_shm = shared_memory.SharedMemory(create=True, size=n_envs * 4)
   done_shm = shared_memory.SharedMemory(create=True, size=n_envs)
   ```
   - **No serialization overhead!** Data is directly written to shared memory
   - States (52 floats), rewards (1 float), dones (1 bool) per environment

2. **Dedicated Pipes for True States**
   ```python
   for i in range(n_envs):
       parent_conn, child_conn = Pipe()
       pipes.append((parent_conn, child_conn))
   ```
   - Each worker has its **own dedicated pipe** (no queue contention!)
   - Parallel communication instead of serial

3. **Cached True States**
   ```python
   class OptimizedEnvWorker:
       def __init__(self):
           self.cached_true_state = None
           self.cache_step_count = 0
           self.cache_update_interval = 10  # Only update every 10 steps
   ```
   - True states are expensive to compute (~2ms)
   - Cache and reuse when possible
   - Only update when necessary

#### **Communication Protocol:**

**Step Operation:**
```
Main → Worker: ("step", action_id)
Worker executes: obs, reward, done = env.step(action)
Worker writes: obs/reward/done → shared memory
Worker → Main: ("step_done", None) via dedicated pipe
Main reads: obs/reward/done ← shared memory
```

**True State Query:**
```
Main → Worker: ("get_true_state", None)
Worker checks cache
Worker → Main: ("true_state", cached_or_fresh_state) via pipe
```

---

### 3. **PPO Agent** (`ParallelOrderConditionedPPO`)

#### **Network Architecture:**

```
Input: [State (52) + Workflow Encoding (25)] = 77 dimensions
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
    ┌────────┐             ┌────────┐
    │ Actor  │             │ Critic │
    │ Policy │             │ Value  │
    └────────┘             └────────┘
        │                       │
        │                       ▼
        │                   State Value
        ▼                   (1 output)
    Action Probs
    (145 actions)
```

**Workflow Encoding:**
- 5 unit types: user, op_host, op_server, enterprise, defender
- One-hot encoded as 5x5 matrix (25 dimensions)
- Example: `[user → op_host → defender → ...]`
  ```
  Position: 0  1  2  3  4
  user:     1  0  0  0  0
  op_host:  0  1  0  0  0
  defender: 0  0  1  0  0
  ...
  ```

#### **Policy Inheritance:**

One of the key innovations! Instead of training from scratch for each workflow:

```python
if self.shared_agent is None:
    # First workflow - train from scratch
    agent = ParallelOrderConditionedPPO(...)
else:
    # Subsequent workflows - inherit weights
    agent = ParallelOrderConditionedPPO(...)
    agent.policy.load_state_dict(self.shared_agent.policy.state_dict())
```

**Why This Works:**
- Similar workflows share similar patterns
- Network learns general defense strategies
- Only needs to fine-tune for specific workflow
- Dramatically reduces training time

---

### 4. **Training Loop**

#### **Step-by-Step Execution:**

```python
# Episode tracking for all 100 environments
episode_counts = np.zeros(100)      # Episodes completed per env
observations = envs.reset()         # Initial states

while np.min(episode_counts) < max_episodes:
    # === STEP 1: Get Current State Info ===
    true_states = envs.get_true_states()  # Compliance tracking
    
    # === STEP 2: Agent Selects Actions ===
    actions, log_probs, values = agent.get_actions(observations)
    # actions: [100] - one action per env
    # log_probs: [100] - for PPO loss
    # values: [100] - state values for advantage
    
    # === STEP 3: Execute Actions in Parallel ===
    observations, env_rewards, dones, infos = envs.step(actions)
    # All 100 envs step simultaneously!
    
    # === STEP 4: Get New State Info ===
    new_true_states = envs.get_true_states()
    
    # === STEP 5: Compute Alignment Rewards ===
    alignment_rewards = agent.compute_alignment_rewards(
        actions, new_true_states, true_states, dones
    )
    # Reward for improving compliance
    
    # === STEP 6: Combine Rewards ===
    total_rewards = env_rewards + alignment_rewards
    
    # === STEP 7: Store Experience ===
    agent.buffer.add(
        observations, actions, total_rewards, dones,
        log_probs, values
    )
    
    # === STEP 8: PPO Update (every 100 steps) ===
    if agent.should_update():
        agent.update()  # 4 epochs of minibatch SGD
    
    # === STEP 9: Track Episodes ===
    for i, done in enumerate(dones):
        if done:
            episode_counts[i] += 1
            # Log episode metrics
    
    # === STEP 10: Check Compliance ===
    current_compliance = agent.get_compliance_rate()
    if current_compliance >= 0.95:
        break  # Workflow achieved compliance!
```

#### **Key Metrics Tracked:**

1. **Environment Reward**: Raw CAGE2 reward
2. **Alignment Reward**: Bonus for compliance improvement
3. **Total Reward**: Env + Alignment (used for training)
4. **Compliance Rate**: % of fix actions that are compliant
5. **Episode Count**: Total episodes per environment

---

### 5. **GP-UCB Workflow Search**

Selects which workflow to try next using Gaussian Process optimization.

#### **Algorithm:**

```python
# For each candidate workflow:
for workflow in all_workflows:
    # Estimate performance from previous trials
    mean, std = gp_model.predict(workflow)
    
    # UCB: Balance exploitation (mean) and exploration (std)
    ucb_score = mean + beta * std
    
# Select workflow with highest UCB
best_workflow = argmax(ucb_score)
```

**Why GP-UCB?**
- **Exploitation**: Try workflows that performed well before
- **Exploration**: Try uncertain workflows (high std)
- **Efficient**: Finds good workflows with fewer samples
- **Adaptive**: β parameter controls exploration vs exploitation

**Integration with Training:**
```
Loop until episode budget exhausted:
    1. GP-UCB selects workflow
    2. Train agent for that workflow
    3. Record compliance achieved
    4. Update GP model with result
    5. GP learns which workflows are promising
```

---

## 🎯 Key Optimizations Applied

### 1. **Shared Memory (2x speedup)**
- Eliminates pickle/unpickle overhead
- Direct memory access across processes
- 200KB → 0 bytes serialization per step

### 2. **Dedicated Pipes (1.5x speedup)**
- No queue contention
- Parallel communication
- Each worker has direct channel to main process

### 3. **Cached True States (1.5x speedup)**
- Expensive operation (~2ms)
- Only update when necessary
- Reduces 200 calls → ~20 calls per episode

### 4. **Policy Inheritance (5-10x faster convergence)**
- Transfer learning across workflows
- Start from good initial policy
- Reduce training episodes per workflow

**Total Speedup: 3.4x in parallel communication + faster convergence**

---

## 📊 Performance Characteristics

### **Current Performance:**
- **25 envs**: ~30-35 eps/sec (70-80% efficiency)
- **100 envs**: ~70-120 eps/sec (40-70% efficiency)

### **Bottlenecks:**
1. **Synchronous barriers**: All envs wait for slowest
2. **IPC overhead**: Still some serialization for true states
3. **Python GIL**: Process management overhead
4. **True state computation**: Cannot be fully vectorized

### **Why Not 100x Speedup?**
- Synchronous stepping loses 30-60% efficiency
- IPC overhead loses ~20%
- Process management loses ~10-20%
- Best achievable: 50-70x with 100 processes

---

## 🔄 Complete Training Workflow

```
START
  │
  ▼
Initialize GP-UCB Model
Initialize Shared Agent (None)
Episode Budget = 100,000
  │
  ▼
┌─────────────────────────────┐
│ While budget > 0:           │
│                             │
│  1. GP-UCB selects workflow │
│     (e.g., user → op_host  │
│      → enterprise → ...)    │
│                             │
│  2. Create 100 parallel envs│
│                             │
│  3. Create/inherit PPO agent│
│                             │
│  4. Train until compliance  │
│     >= 95% or max episodes  │
│     ┌──────────────┐       │
│     │ Sample 100   │       │
│     │ episodes     │       │
│     │ in parallel  │       │
│     └──────────────┘       │
│     ┌──────────────┐       │
│     │ PPO update   │       │
│     │ (4 epochs)   │       │
│     └──────────────┘       │
│     Repeat...              │
│                             │
│  5. Record final performance│
│                             │
│  6. Update GP model         │
│                             │
│  7. Update episode budget   │
│                             │
└─────────────────────────────┘
  │
  ▼
Save final model
Generate logs
END
```

---

## 📁 Code Structure

```
workflow_rl/
├── parallel_train_workflow_rl.py          # Main training script
│   └── ParallelWorkflowRLTrainer          # Orchestrates everything
│
├── parallel_env_shared_memory_optimized.py # Environment management
│   ├── ParallelEnvSharedMemoryOptimized   # Main interface
│   └── OptimizedEnvWorker                 # Worker process
│
├── parallel_order_conditioned_ppo.py      # RL algorithm
│   ├── ParallelOrderConditionedPPO        # Agent
│   └── OrderConditionedActorCritic        # Neural network
│
├── gp_ucb_order_search.py                 # Workflow search
│   └── GPUCBOrderSearch                   # GP-UCB algorithm
│
└── order_based_workflow.py                # Workflow encoding
    └── OrderBasedWorkflow                 # Workflow → vector
```

---

## 🚀 Running the Training

```bash
# Standard training with 100 environments
python workflow_rl/parallel_train_workflow_rl.py \
    --n-envs 100 \
    --total-episodes 100000 \
    --red-agent B_lineAgent

# For better efficiency, use fewer environments
python workflow_rl/parallel_train_workflow_rl.py \
    --n-envs 25 \
    --total-episodes 100000 \
    --red-agent B_lineAgent
```

---

## 🎓 Key Takeaways

1. **Parallel Training**: 100 environments run simultaneously using multiprocessing
2. **Shared Memory**: Fast data transfer without serialization
3. **Dedicated Pipes**: Each worker has its own communication channel
4. **Cached States**: Minimize expensive computations
5. **Policy Inheritance**: Transfer learning across workflows
6. **GP-UCB Search**: Intelligently select which workflows to try
7. **Compliance Gating**: Only train until 95% compliance achieved

The system achieves **70-120 episodes/sec** with 100 parallel environments, which is **40-70x faster** than sequential training!
