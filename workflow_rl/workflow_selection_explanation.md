# Workflow Selection in GP-UCB

## How Initial Workflows are Chosen

### 1. **First Workflow (Iteration 1)**
- **Method**: Pure random selection from candidates
- **Reason**: No prior data exists for GP modeling
- **Example**: `user → op_host → op_server → enterprise → defender`
- **UCB Score**: 0.0 (no score computed)

### 2. **Second Workflow (Iteration 2)**
- **Method**: Diversity maximization
- **Reason**: Need at least 2 observations to fit Gaussian Process
- **Selection**: Choose workflow that maximizes Kendall tau distance from first
- **Example**: If first was `user → op_host → ...`, second might be `enterprise → defender → ...` (distance=0.70)
- **Purpose**: Explore different regions of the workflow space

### 3. **Subsequent Workflows (Iteration 3+)**
- **Method**: Full GP-UCB algorithm
- **Formula**: `UCB = μ(x) + β × σ(x)`
  - μ(x) = predicted mean reward (exploitation)
  - σ(x) = uncertainty/standard deviation (exploration)
  - β = exploration parameter (default: 2.0)

## GP-UCB Selection Details

### Information Displayed for Each Selection:

1. **UCB Score**: The upper confidence bound value used for selection
2. **Mean Reward**: Predicted average reward based on similar workflows
3. **Uncertainty (std)**: How uncertain the prediction is
4. **Exploration Bonus**: `β × σ(x)` - the exploration component
5. **Exploitation Value**: `μ(x)` - the expected reward component
6. **Selection Type**: 
   - "exploration" if exploration bonus > exploitation value
   - "exploitation" if exploitation value > exploration bonus
7. **Previous Visits**: How many times this exact workflow was evaluated
8. **Closest Known**: The most similar previously evaluated workflow
   - Shows Kendall tau distance and its reward
   - Helps understand the prediction basis

### Top 3 Candidates
Shows the top 3 workflows by UCB score:
- Helps understand why one was chosen over others
- Shows the competition between exploration and exploitation
- Each shows: UCB score, mean prediction, uncertainty

## Candidate Generation

### For First 5 Iterations:
Predefined diverse candidates to ensure good coverage:
```python
[
    ['defender', 'enterprise', 'op_server', 'op_host', 'user'],
    ['op_server', 'defender', 'enterprise', 'op_host', 'user'],
    ['enterprise', 'op_server', 'defender', 'user', 'op_host'],
    ['user', 'op_host', 'op_server', 'enterprise', 'defender'],
    ['op_host', 'user', 'defender', 'enterprise', 'op_server'],
]
```

### After 5 Iterations:
- 10 random permutations generated as candidates
- GP-UCB selects from these candidates

## Distance Metric: Kendall Tau

Measures how different two orderings are:
- Range: [0, 1] where 0 = identical, 1 = completely reversed
- Counts the number of pairwise disagreements between orderings
- Example:
  - `[A, B, C]` vs `[A, B, C]`: distance = 0.0
  - `[A, B, C]` vs `[C, B, A]`: distance = 1.0
  - `[A, B, C]` vs `[A, C, B]`: distance = 0.33

## Exploration vs Exploitation Balance

### High β (e.g., 2.0 - default):
- More exploration of uncertain workflows
- Willing to try workflows far from known good ones
- Better for finding global optimum

### Low β (e.g., 0.5):
- More exploitation of known good workflows
- Focuses on refining near best-known workflows
- Better for local optimization

### Adaptive β:
The implementation reduces β for frequently visited workflows:
```python
adjusted_beta = beta / (1 + 0.5 * visit_count)
```
This encourages exploring new workflows while still allowing revisits.

## Example Output Interpretation

```
GP-UCB Selection Details:
  Selected: enterprise → op_host → defender → op_server → user
  UCB Score: -670.541
  Mean Reward: -676.54        # Expected reward (negative is normal)
  Uncertainty (std): 3.000     # Low uncertainty (similar to known workflows)
  Exploration Bonus: 6.000     # β × std = 2.0 × 3.0
  Exploitation Value: -676.54  # Predicted reward
  Selection Type: exploration  # Bonus (6.0) helps overcome negative mean
  Previous Visits: 0           # Never tried this exact workflow
  Closest Known: enterprise → defender → op_host → user → op_server
                 (dist=0.20, reward=-526.25)
```

This tells us:
- The workflow is similar to a known good one (distance=0.20)
- It has moderate uncertainty (std=3.0)
- It's selected for exploration despite lower expected reward
- The closest known workflow performed well (-526.25)
