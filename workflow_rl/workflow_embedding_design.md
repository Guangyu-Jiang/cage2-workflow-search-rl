# Workflow Embedding Design for CAGE2

## Design Methodology

The workflow embedding was designed through analysis of optimal policies and identifying key strategic decisions that differentiate successful defense strategies.

## The 8-Dimensional Embedding

### Dimension 0: Fortification Timing [0.0 to 1.0]
- **0.0**: Late fortification (deploy decoys after threats detected)
- **1.0**: Early fortification (deploy decoys in first steps)
- **Rationale**: From meander model analysis - early decoy deployment (steps 0-6) was highly effective
- **Evidence**: Meander achieved -4.80 reward with early fortification vs bline's -10.70 with reactive approach

### Dimension 1: Fortification Intensity [0.0 to 1.0]
- **0.0**: No decoys
- **0.5**: Moderate decoy usage
- **1.0**: Maximum decoy deployment
- **Rationale**: Meander used 40% decoy actions vs bline's 6.7%
- **Optimal**: Around 0.3-0.4 based on analysis

### Dimension 2: Analysis Frequency [0.0 to 1.0]
- **0.0**: Minimal information gathering
- **1.0**: Continuous analysis
- **Rationale**: Meander used 46.7% analyse actions for better situational awareness
- **Trade-off**: Information vs action economy

### Dimension 3: Remove vs Restore Preference [-1.0 to 1.0]
- **-1.0**: Always prefer Remove (only works on User-level)
- **0.0**: Balanced approach
- **1.0**: Always prefer Restore (works on Privileged but costs -1)
- **Rationale**: Key strategic decision - bline wasted actions on Remove (66.7%) when hosts had Privileged access
- **Evidence**: Meander's immediate Restore strategy was more effective

### Dimension 4: Response Speed [0.0 to 1.0]
- **0.0**: Delayed response (wait and observe)
- **1.0**: Immediate response to threats
- **Rationale**: Meander's immediate restoration prevented escalation
- **Trade-off**: Quick action vs information gathering

### Dimension 5-7: Subnet Focus (sum to ~1.0)
- **Dim 5**: User subnet focus [0.0 to 1.0]
- **Dim 6**: Enterprise subnet focus [0.0 to 1.0]
- **Dim 7**: Operational subnet focus [0.0 to 1.0]
- **Rationale**: Different red agents target different subnets
- **B_line**: Direct path to operational (higher enterprise/operational focus needed)
- **Meander**: Thorough exploration (balanced focus works better)

## Extraction from Optimal Policies

### Bline Model Embedding:
```python
bline_embedding = [
    0.2,   # Late fortification (reactive)
    0.1,   # Low decoy intensity (6.7% of actions)
    0.2,   # Low analysis frequency (13.3% of actions)
    -0.5,  # Prefer Remove (66.7% Remove vs 13.3% Restore)
    0.8,   # Quick response (but to wrong targets)
    0.6,   # High user focus (Remove User3 repeatedly)
    0.3,   # Moderate enterprise focus
    0.1    # Low operational focus
]
```

### Meander Model Embedding:
```python
meander_embedding = [
    0.9,   # Early fortification (steps 0-6)
    0.4,   # Moderate decoy intensity (40% of actions)
    0.5,   # High analysis frequency (46.7% of actions)
    0.5,   # Prefer Restore (13.3% Restore, 0% Remove)
    0.9,   # Immediate response
    0.3,   # Moderate user focus
    0.4,   # Moderate enterprise focus
    0.3    # Moderate operational focus
]
```

## Why These Dimensions?

### 1. **Completeness**: Covers all major strategic decisions
   - When to act (timing)
   - How to act (intensity, preference)
   - Where to act (subnet focus)
   - How fast to act (response speed)

### 2. **Independence**: Each dimension captures orthogonal aspects
   - Fortification strategy (dims 0-1)
   - Information strategy (dim 2)
   - Remediation strategy (dim 3)
   - Temporal strategy (dim 4)
   - Spatial strategy (dims 5-7)

### 3. **Smoothness**: Continuous values enable gradient-based optimization
   - Small changes in embedding → small changes in behavior
   - Enables interpolation between strategies

### 4. **Interpretability**: Each dimension has clear meaning
   - Can explain why a workflow performs well
   - Can manually design workflows based on domain knowledge

## Validation Through Analysis

The embedding captures the key differences between successful strategies:

1. **Meander vs Bline Performance Gap**: 
   - Meander: -4.80 (early fortify, restore focus)
   - Bline: -10.70 (reactive, remove focus)
   - Difference explained by dims 0, 1, 3

2. **Action Distribution Patterns**:
   ```
   Meander: 40% Decoy, 47% Analyse, 13% Restore, 0% Remove
   Bline:   7% Decoy,  13% Analyse, 13% Restore, 67% Remove
   ```
   - Captured by dims 1, 2, 3

3. **Temporal Patterns**:
   - Meander: Heavy early fortification (7 decoys in first 6 steps)
   - Bline: Scattered fortification
   - Captured by dim 0

## Alternative Embeddings Considered

### Option 1: Action-based (Rejected)
```python
# Direct action probabilities
[p_sleep, p_analyse, p_remove, p_restore, p_decoy]
```
- **Problem**: Too low-level, doesn't capture strategy

### Option 2: Milestone-based (Rejected)
```python
# Ordering of milestones
[fortify_order, analyse_order, clean_order, ...]
```
- **Problem**: Discrete, not smooth for optimization

### Option 3: Rule-based (Rejected)
```python
# Threshold parameters
[scan_threshold, compromise_threshold, restore_threshold, ...]
```
- **Problem**: Too specific to CAGE2, not general

## Final Design Principles

1. **Strategy over Actions**: Encode high-level strategy, not specific actions
2. **Continuous over Discrete**: Enable smooth optimization
3. **General over Specific**: Could work for other cyber defense scenarios
4. **Empirically Grounded**: Based on actual performance analysis
5. **Compact**: 8 dimensions is tractable for search algorithms

This embedding design enables efficient workflow search while maintaining interpretability and generalizability.
