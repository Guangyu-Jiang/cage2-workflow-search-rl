# Workflow-Conditioned Reinforcement Learning for CAGE2

## Slide 1: Problem Overview
**Title: Adaptive Defense Strategy Learning in CAGE2**

- **Challenge**: Learn optimal defense policies against diverse cyber attacks
- **Environment**: CAGE2 - Multi-host network defense simulation
- **Key Issue**: Large action space with complex dependencies
- **Goal**: Discover effective defense workflows through hierarchical RL

## Slide 2: Key Insight
**Title: Defense as Prioritized Workflows**

- **Observation**: Effective defense follows strategic priorities
- **Workflow**: An ordering of unit types to defend
  - Example: [Defender → Op_Server → Enterprise → Op_Host → User]
- **Hypothesis**: Different attack patterns require different defense priorities
- **Approach**: Learn to select and execute optimal workflows

## Slide 3: Environment Analysis
**Title: CAGE2 Unit Types**

| Unit Type | Count | Value | Example Actions |
|-----------|-------|-------|-----------------|
| Defender | 1 | Critical | Analyze(2), Remove(15), Restore(132) |
| Enterprise | 3 | High | Analyze(3-5), Remove(16-18), Restore(133-135) |
| Op_Server | 1 | High | Analyze(9), Remove(22), Restore(139) |
| Op_Host | 3 | Medium | Analyze(6-8), Remove(19-21), Restore(136-138) |
| User | 5 | Low | Analyze(10-14), Remove(23-27), Restore(140-144) |

**Total**: 13 hosts, 5 unit types → 5! = 120 possible type orderings

## Slide 4: Hierarchical Architecture
**Title: Two-Level Learning Framework**

```
High Level: Workflow Search (GP-UCB)
    ↓ Selects workflow (priority order)
    ↓ 
Low Level: PPO Execution
    ↓ Executes actions following workflow
    ↓
Environment: CAGE2
```

- **Outer Loop**: Search over workflow space using Gaussian Process Upper Confidence Bound
- **Inner Loop**: PPO learns to execute given workflow with alignment rewards

## Slide 5: Workflow Representation
**Title: Encoding Defense Priorities**

**Workflow as Permutation**:
- Direct representation: Order of 5 unit types
- Example: `[defender, op_server, enterprise, user, op_host]`
- Total space: 5! = 120 possible orderings

**No Embedding Needed**:
- Work directly with permutations
- One-hot encoding for neural network (25D)
- Kendall tau distance for GP kernel
- Pure discrete optimization over orders

## Slide 6: Alignment Reward Design
**Title: Encouraging Workflow Compliance**

**Total Reward = Environment Reward + Alignment Reward**

```python
alignment_reward = α * correct_fixes - β * violations

where:
- correct_fixes: Fixes following priority order
- violations: Fixes violating priority order
- α, β: Positive constants (e.g., α=0.1, β=0.2)
```

**Example**:
- Workflow: Enterprise > User
- If User compromised → Fix User ✓ (+α)
- If both compromised → Fix Enterprise first ✓ (+α)
- If both compromised → Fix User first ✗ (-β)

## Slide 7: Compliance Checking Algorithm
**Title: Verifying Workflow Alignment**

```python
def check_alignment(action, true_state, workflow):
    # 1. Identify target of action
    target_type = get_unit_type(action)
    
    # 2. Get all compromised units
    compromised = get_compromised_units(true_state)
    
    # 3. Check if higher priority units exist
    for unit in compromised:
        if workflow.priority(unit) < workflow.priority(target_type):
            return False  # Violation
    
    return True  # Aligned
```

## Slide 8: Training Process
**Title: Workflow Search Loop**

```
for iteration in range(N):
    # 1. Select workflow using GP-UCB
    workflow = gp_ucb.select_next_workflow()
    
    # 2. Train PPO with workflow
    for episode in range(M):
        state = env.reset()
        for step in range(T):
            # Get action from PPO
            action = ppo.get_action(state, workflow)
            
            # Execute and get rewards
            next_state, env_reward, done = env.step(action)
            
            # Compute alignment reward
            align_reward = compute_alignment(action, true_state, workflow)
            
            # Update PPO with shaped reward
            total_reward = env_reward + align_reward
            ppo.update(state, action, total_reward)
    
    # 3. Evaluate workflow performance
    score = evaluate_workflow(workflow, ppo)
    
    # 4. Update GP-UCB
    gp_ucb.update(workflow, score)
```

## Slide 9: Expected Benefits
**Title: Why Workflow-Conditioned RL?**

1. **Reduced Search Space**
   - From: ~145 actions per step × 100 steps
   - To: 120 workflows × learned execution

2. **Interpretability**
   - Clear strategic priorities
   - Explainable defense decisions

3. **Adaptability**
   - Different workflows for different threats
   - Meta-learning which strategies work

4. **Sample Efficiency**
   - High-level search in small space
   - Low-level learning with guidance

## Slide 10: Implementation Details
**Title: Technical Specifications**

**PPO Configuration**:
- Network: [52+8 → 128 → 64 → 145] (state + workflow → actions)
- Learning rate: 0.002
- Batch size: 64
- γ: 0.99

**GP-UCB Parameters**:
- Kernel: Matérn (ν=2.5)
- Acquisition: UCB with β=2.0
- Workflow embedding: 8D continuous

**Alignment Reward**:
- α (compliance bonus): 0.1
- β (violation penalty): 0.2
- Checked against true state

## Slide 11: Experimental Setup
**Title: Evaluation Protocol**

**Baselines**:
1. Original PPO (no workflow)
2. Fixed workflow (expert-designed)
3. Random workflow selection

**Metrics**:
- Average episode reward
- Workflow compliance rate
- Convergence speed
- Robustness across red agents

**Test Scenarios**:
- B_lineAgent (aggressive)
- RedMeanderAgent (stealthy)
- Mixed strategies

## Slide 12: Key Contributions
**Title: Summary**

1. **Novel Framework**: Hierarchical RL for cyber defense
2. **Workflow Representation**: Interpretable strategy encoding
3. **Alignment Mechanism**: Ensures strategy execution
4. **Empirical Validation**: On realistic CAGE2 environment

**Future Work**:
- Dynamic workflow adaptation
- Multi-agent coordination
- Transfer to real networks
