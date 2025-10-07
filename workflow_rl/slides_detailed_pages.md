# Detailed Slides for Workflow-Conditioned RL

## Page 1: High-Level Workflow Search (GP-UCB)

### Title: Gaussian Process Upper Confidence Bound for Workflow Discovery

**Overview**: The high-level search explores the space of defense priority orderings to find optimal strategies.

### Search Space
- **120 possible workflows**: All permutations of 5 unit types
  - Example: `[defender, enterprise, op_server, op_host, user]`
- **Direct order representation**: No embedding needed
  - Work with permutations directly
  - Kendall tau distance for similarity in GP kernel

### GP-UCB Algorithm

#### Acquisition Function
```python
UCB(workflow) = μ(workflow) + β × σ(workflow)
```

#### Components Breakdown:

**1. Mean Prediction μ(workflow)**
- Gaussian Process posterior mean at workflow location
- Represents expected performance based on observed data
- Computed using kernel-weighted combination of nearby observations:
  ```python
  μ(x) = k(x, X) @ K_inv @ y
  where:
    k(x, X) = kernel between new point and observed points
    K_inv = inverse of kernel matrix of observations
    y = observed rewards
  ```
- Higher μ → Workflow expected to perform well (exploitation)

**2. Uncertainty Estimate σ(workflow)**
- Gaussian Process posterior standard deviation
- Measures our uncertainty about workflow performance
- Computed as:
  ```python
  σ²(x) = k(x, x) - k(x, X) @ K_inv @ k(X, x)
  where:
    k(x, x) = kernel self-similarity (usually 1)
    k(x, X) = kernel between new and observed points
  ```
- Higher σ → Less explored region (exploration opportunity)
- Decreases as we sample near this workflow

**3. Exploration Parameter β**
- Controls exploration vs exploitation trade-off
- Default β = 2.0 (moderately exploratory)
- Theoretical choice: β = √(2 log(t²π²/6δ)) for regret bounds
- Practical tuning:
  - β < 1: Exploit known good workflows
  - β > 2: Aggressive exploration of uncertain regions
  
**4. Adaptive β for Revisits**
```python
# Reduce exploration bonus for frequently visited workflows
revisit_count = observation_counts[workflow]
adjusted_β = β / (1 + 0.5 × revisit_count)

# Final acquisition:
UCB(workflow) = μ(workflow) + adjusted_β × σ(workflow)
```

#### Example Calculation:

**Iteration 5** - Evaluating workflow `[defender, enterprise, op_server, op_host, user]`

```python
# From GP prediction:
μ = -45.2  # Expected reward (negative is normal in CAGE2)
σ = 12.3   # Uncertainty

# Standard UCB:
UCB = -45.2 + 2.0 × 12.3 = -45.2 + 24.6 = -20.6

# If this workflow was visited once before:
adjusted_β = 2.0 / (1 + 0.5 × 1) = 1.33
UCB_adjusted = -45.2 + 1.33 × 12.3 = -45.2 + 16.4 = -28.8
```

#### Why GP-UCB Works:

1. **Optimism in Face of Uncertainty**: High σ regions get exploration bonus
2. **Convergence**: As σ → 0 with more samples, focuses on high μ regions  
3. **No Gradient Required**: Works with black-box reward functions
4. **Theoretical Guarantees**: Sublinear regret bounds under assumptions

### Search Process

1. **Initialize with Canonical Workflows**
   - Critical-first: `[defender, op_server, enterprise, op_host, user]`
   - Enterprise-focus: `[enterprise, defender, op_server, op_host, user]`
   - User-priority: `[user, defender, enterprise, op_server, op_host]`
   - 3 more strategic baselines

2. **Iterative Workflow Selection**
   ```
   For each iteration:
     1. Fit Gaussian Process on observed (workflow, reward) pairs
     2. Generate 500 candidate workflows:
        - 33% random exploration
        - 33% variations of top performers
        - 33% strategy-based samples
     3. Select workflow with highest UCB score
     4. Train PPO agent with selected workflow
     5. Update GP with evaluation results
   ```

3. **Adaptive Exploration**
   - Revisit penalty: `adjusted_β = β / (1 + 0.5 × revisit_count)`
   - Encourages diversity in exploration
   - Balances refinement of promising workflows

### Key Features

**Distance Metric**: Kendall's tau distance between orderings
- Measures disagreements in relative rankings
- Normalized to [0, 1] for kernel computation

**Kernel Function**: Matérn (ν=2.5)
- Captures smooth performance landscape
- Length scale = 0.5 for local correlation

**Performance Adjustment**:
```python
adjusted_reward = raw_reward × (0.7 + 0.3 × compliance_rate)
```
- Accounts for how well the policy follows the workflow
- Low compliance → less informative sample

### Convergence Behavior
- Early iterations: High exploration (high σ)
- Later iterations: Exploitation of promising regions
- Typically converges to top workflows within 15-20 iterations

---

## Page 2: Low-Level Policy Training (Reward Design)

### Title: Workflow-Conditioned PPO with Shaped Rewards

**Core Principle**: Train PPO agents to execute specific defense priorities through carefully designed rewards.

### Reward Structure

```python
Total Reward = Environment Reward + Alignment Reward
```

### 1. Environment Reward (Base Game)
- **Negative per-step penalty**: -0.1 for red agent presence
- **Host compromise penalty**: Varies by importance
  - Defender: High penalty
  - Op_Server: High penalty  
  - Enterprise: Medium penalty
  - Op_Host: Medium penalty
  - User: Low penalty
- **Impact penalty**: -10 when red achieves objective
- **Restore cost**: -1 for each restoration action

### 2. Alignment Reward (Workflow Compliance)

```python
def compute_alignment_reward(action, true_state, workflow):
    target_type = get_action_unit_type(action)
    
    if target_type is None:  # Not a fix action
        return 0.0
    
    compromised = get_compromised_units(true_state)
    target_priority = workflow.index(target_type)
    
    # Check for higher priority compromised units
    for unit_type, units in compromised.items():
        if units and workflow.index(unit_type) < target_priority:
            return -β  # Violation penalty (default: -0.2)
    
    return +α  # Compliance bonus (default: +0.1)
```

### Reward Shaping Properties

**Invariant Shaping**: Preserves optimal policy structure
- Only changes learning dynamics, not final solution
- Accelerates convergence to workflow-aligned behavior

**Dynamic Feedback**: Real-time compliance checking
- Immediate penalty for workflow violations
- Immediate reward for correct prioritization

### Workflow Conditioning

**Network Input**:
```python
# State: 52D observation vector from CAGE2
# Order: 25D one-hot encoding (5 positions × 5 unit types)
#   - Position 0-4: which unit type is first (one-hot)
#   - Position 5-9: which unit type is second (one-hot)
#   - ... etc for all 5 positions
augmented_input = concat(state, order_encoding)  # 77D total
```

**Neural Network Architecture**:
```python
Actor Network: 77D → 64 → 64 → 145 (full action space)
Critic Network: 77D → 64 → 64 → 1 (value estimate)
```

### Training Process with Early Stopping

```python
for episode in range(train_episodes):
    state = env.reset()
    for step in range(max_steps):
        # 1. Get action from workflow-conditioned policy
        action = agent.get_action(state, workflow_order)
        
        # 2. Execute in environment
        next_state, env_reward, done = env.step(action)
        
        # 3. Compute alignment reward
        align_reward = compute_alignment(action, true_state, workflow)
        
        # 4. Combined reward signal
        total_reward = env_reward + align_reward
        
        # 5. Update PPO with shaped reward
        agent.update(state, action, total_reward)
    
    # Early Stopping (NEW)
    if episode >= 10 and mean(last_5_episodes_compliance) >= 0.95:
        print(f"Early stop at episode {episode}: 95% compliance achieved")
        # Evaluate on pure environment reward
        final_reward = evaluate_pure_performance(agent, episodes=10)
        break  # Stop training, use pure reward for GP update
```

### Network Architecture

**Input**: State (52D) + Order Encoding (25D) = 77D
**Actor Network**: 77 → 64 → 64 → 145 (softmax)
**Critic Network**: 77 → 64 → 64 → 1 (value)

### Hyperparameters
- Learning rate: 0.002
- PPO clip: 0.2
- Discount γ: 0.99
- Alignment α: 0.1 (compliance bonus)
- Alignment β: 0.2 (violation penalty)

### Expected Learning Behavior
- **Early training**: Low compliance, exploring actions
- **Mid training**: Learning workflow priorities
- **Late training**: High compliance, optimized execution

---

## Page 3: Workflow Compliance Analysis

### Title: Measuring and Ensuring Workflow Adherence

**Definition**: Compliance measures how well an agent's trajectory follows the prescribed defense priority order.

### Compliance Calculation

```python
Compliance Rate = (Compliant Fix Actions / Total Fix Actions) × 100%
```

### Step-by-Step Compliance Checking

**For each timestep in a trajectory:**

1. **Action Classification**
   ```python
   def classify_action(action_id):
       if action_id in [2, 15, 132]:  # Defender actions
           return 'defender'
       elif action_id in [3-5, 16-18, 133-135]:  # Enterprise
           return 'enterprise'
       elif action_id in [9, 22, 139]:  # Op_Server
           return 'op_server'
       elif action_id in [6-8, 19-21, 136-138]:  # Op_Host
           return 'op_host'
       elif action_id in [10-14, 23-27, 140-144]:  # User
           return 'user'
       else:
           return None  # Not a fix action
   ```

2. **True State Analysis**
   ```python
   def get_compromised_units(true_state):
       compromised = {'defender': [], 'enterprise': [], 
                     'op_server': [], 'op_host': [], 'user': []}
       
       for hostname, info in true_state.items():
           if has_red_session(info):
               unit_type = get_unit_type(hostname)
               compromised[unit_type].append(hostname)
       
       return compromised
   ```

3. **Compliance Verdict**
   ```python
   def check_step_compliance(action, true_state, workflow_order):
       target_type = classify_action(action)
       if not target_type:
           return None  # Skip non-fix actions
       
       compromised = get_compromised_units(true_state)
       target_priority = workflow_order.index(target_type)
       
       # Check all compromised units
       for unit_type, units in compromised.items():
           if units:  # This type has compromised units
               if workflow_order.index(unit_type) < target_priority:
                   return 'VIOLATION'  # Higher priority unit needs attention
       
       return 'COMPLIANT'  # Following workflow
   ```

### Trajectory-Level Metrics

**Example Trajectory Analysis:**
```
Workflow: [defender, enterprise, op_server, op_host, user]

Step | Action        | Compromised        | Verdict    | Running
-----|--------------|-------------------|------------|----------
  5  | Restore User | User1, Enterprise0 | VIOLATION  | 0/1 (0%)
  8  | Analyze Ent. | Enterprise0        | COMPLIANT  | 1/2 (50%)
 12  | Remove Ent.  | Enterprise0        | COMPLIANT  | 2/3 (67%)
 15  | Restore Def. | Defender, User2    | COMPLIANT  | 3/4 (75%)
 18  | Restore User | User2              | COMPLIANT  | 4/5 (80%)

Final Compliance: 4/5 = 80%
```

### Compliance Patterns

**High Compliance (>80%)**
- Agent successfully learned workflow
- Reliable evaluation of workflow quality
- Strong alignment signal during training

**Medium Compliance (40-80%)**
- Partial workflow understanding
- May indicate workflow-environment mismatch
- Moderate GP-UCB confidence

**Low Compliance (<40%)**
- Agent struggling with workflow
- Possible causes:
  - Workflow is suboptimal for environment
  - Insufficient training
  - Conflicting priorities

### Visualization

```python
# Compliance over training episodes
Episodes:     [0    10   20   30   40   50]
Compliance:   [15%  28%  45%  67%  78%  85%]
              ↑                            ↑
         Random actions            Learned workflow
```

### Usage in GP-UCB Update

```python
def update_gp_with_compliance(workflow, raw_reward, compliance_rate):
    # Adjust reward based on compliance
    confidence_weight = 0.7 + 0.3 * compliance_rate
    adjusted_reward = raw_reward * confidence_weight
    
    # High compliance → High confidence in reward estimate
    # Low compliance → Discount the reward signal
    
    gp.update(workflow, adjusted_reward)
```

### Key Insights

1. **Compliance ≠ Performance**: High compliance doesn't guarantee high reward
2. **Learning Signal**: Compliance improves over training episodes
3. **Workflow Validation**: Low compliance may indicate poor workflow choice
4. **Dynamic Adaptation**: Compliance checked against current game state

### Compliance Report Example
```
Workflow: defender → op_server → enterprise → op_host → user
Episodes: 50
Average Compliance: 72.3%
Compliance Trend: ↑ (improving)
Fix Actions: 1,247
Compliant Actions: 902
Violations: 345
Most Common Violation: Fixing user before enterprise (112 times)
```
