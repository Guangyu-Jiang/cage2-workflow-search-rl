# 25D One-Hot Encoding for Workflow Order

## Structure: 5 Positions × 5 Unit Types = 25 Dimensions

The encoding represents which unit type occupies each position in the priority order.

### Unit Types (5):
1. `defender`
2. `enterprise`
3. `op_server`
4. `op_host`
5. `user`

### Positions (5):
- Position 0: Highest priority (fix first)
- Position 1: Second priority
- Position 2: Third priority
- Position 3: Fourth priority
- Position 4: Lowest priority (fix last)

## Encoding Structure

The 25D vector is organized as 5 consecutive blocks of 5 dimensions each:

```
[Position_0_encoding, Position_1_encoding, Position_2_encoding, Position_3_encoding, Position_4_encoding]

Where each position encoding is:
[is_defender, is_enterprise, is_op_server, is_op_host, is_user]
```

## Example 1: Critical-First Order

**Order**: `[defender, op_server, enterprise, op_host, user]`

```python
# Position 0: defender     → [1,0,0,0,0]
# Position 1: op_server    → [0,0,1,0,0]
# Position 2: enterprise   → [0,1,0,0,0]
# Position 3: op_host      → [0,0,0,1,0]
# Position 4: user         → [0,0,0,0,1]

# Concatenated 25D vector:
[1,0,0,0,0, 0,0,1,0,0, 0,1,0,0,0, 0,0,0,1,0, 0,0,0,0,1]
```

### Visual Breakdown:
```
Dims 0-4:   [1,0,0,0,0] ← Position 0 has defender
Dims 5-9:   [0,0,1,0,0] ← Position 1 has op_server
Dims 10-14: [0,1,0,0,0] ← Position 2 has enterprise
Dims 15-19: [0,0,0,1,0] ← Position 3 has op_host
Dims 20-24: [0,0,0,0,1] ← Position 4 has user
```

## Example 2: User-Priority Order

**Order**: `[user, defender, enterprise, op_server, op_host]`

```python
# Position 0: user         → [0,0,0,0,1]
# Position 1: defender     → [1,0,0,0,0]
# Position 2: enterprise   → [0,1,0,0,0]
# Position 3: op_server    → [0,0,1,0,0]
# Position 4: op_host      → [0,0,0,1,0]

# Concatenated 25D vector:
[0,0,0,0,1, 1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,0,1,0]
```

## Example 3: Balanced Order

**Order**: `[defender, enterprise, op_server, user, op_host]`

```python
# Full 25D encoding:
[1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,0,0,1, 0,0,0,1,0]
```

## Properties of the Encoding

### 1. **Sparsity**
- Only 5 out of 25 dimensions are 1
- Exactly one 1 per position block
- 20 dimensions are always 0

### 2. **Mutual Exclusivity**
- Each position has exactly one unit type
- Each unit type appears in exactly one position

### 3. **Permutation Invariant**
- Every valid order produces a unique 25D vector
- 5! = 120 possible unique encodings

### 4. **Distance Properties**
```python
# Orders with similar priorities have similar encodings
order1 = [defender, enterprise, op_server, op_host, user]
order2 = [defender, enterprise, op_server, user, op_host]
# These differ only in last 10 dimensions (positions 3 and 4)

# Completely opposite orders have maximum distance
order1 = [defender, enterprise, op_server, op_host, user]
order2 = [user, op_host, op_server, enterprise, defender]
# All 25 dimensions differ
```

## Implementation Code

```python
def order_to_onehot(order: List[str]) -> np.ndarray:
    """
    Convert order to 25D one-hot encoding
    """
    unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
    onehot = np.zeros(25)
    
    for position, unit_type in enumerate(order):
        type_index = unit_types.index(unit_type)
        # Set the appropriate dimension to 1
        onehot[position * 5 + type_index] = 1.0
    
    return onehot

# Example usage:
order = ['defender', 'op_server', 'enterprise', 'op_host', 'user']
encoding = order_to_onehot(order)
print(encoding)
# Output: [1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1.]
```

## Reverse Mapping

```python
def onehot_to_order(onehot: np.ndarray) -> List[str]:
    """
    Convert 25D one-hot back to order
    """
    unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
    order = []
    onehot = onehot.reshape(5, 5)  # Reshape to [positions, types]
    
    for position in range(5):
        type_index = np.argmax(onehot[position])
        order.append(unit_types[type_index])
    
    return order

# Example:
encoding = [1,0,0,0,0, 0,0,1,0,0, 0,1,0,0,0, 0,0,0,1,0, 0,0,0,0,1]
order = onehot_to_order(np.array(encoding))
print(order)
# Output: ['defender', 'op_server', 'enterprise', 'op_host', 'user']
```

## Why One-Hot Instead of Other Encodings?

### Alternatives Considered:

1. **Integer Encoding** (5D):
   ```python
   # [defender, op_server, enterprise, op_host, user] → [0, 2, 1, 3, 4]
   ```
   Problem: Implies false ordinal relationships between unit types

2. **Ranking Vector** (5D):
   ```python
   # Position of each type: [defender_pos, enterprise_pos, op_server_pos, op_host_pos, user_pos]
   # [defender, op_server, enterprise, op_host, user] → [0, 2, 1, 3, 4]
   ```
   Problem: Neural network might struggle with permutation semantics

3. **Binary Encoding** (15D using pairwise comparisons):
   ```python
   # For each pair (i,j): 1 if type_i comes before type_j, else 0
   ```
   Problem: Redundant, harder to interpret

### Why One-Hot Works Best:

1. **Neural Network Friendly**: Clear, sparse signal
2. **No False Relationships**: Each dimension independent
3. **Easy Concatenation**: Works well with state vector
4. **Interpretable**: Can directly read which type is where
5. **Standard Practice**: Common for categorical data in deep learning

## Network Input

The full network input concatenates state and order encoding:

```python
# State: 52D observation from CAGE2
# Order: 25D one-hot encoding
augmented_input = concat(state, order_encoding)  # 77D total

# Fed to actor network:
Actor: 77D → 64 → 64 → 145 (action probabilities)

# Fed to critic network:
Critic: 77D → 64 → 64 → 1 (value estimate)
```

This 25D encoding provides a clear, unambiguous representation of the workflow priority order that the neural network can easily process.
