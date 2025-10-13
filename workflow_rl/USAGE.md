# Command-Line Usage Guide

## Basic Usage

### Default Training (RedMeanderAgent)
```bash
cd /home/ubuntu/CAGE2/-cyborg-cage-2
conda activate CAGE2
export PYTHONPATH=/home/ubuntu/CAGE2/-cyborg-cage-2:$PYTHONPATH
python workflow_rl/parallel_train_workflow_rl.py
```

### Show Help
```bash
python workflow_rl/parallel_train_workflow_rl.py --help
```

---

## Command-Line Arguments

### Environment Configuration

#### `--red-agent` (Red Agent Type)
```bash
# Use RedMeanderAgent (aggressive, recommended)
python workflow_rl/parallel_train_workflow_rl.py --red-agent meander

# Use B_lineAgent (moderate attacks)
python workflow_rl/parallel_train_workflow_rl.py --red-agent bline

# Use SleepAgent (no attacks, testing only)
python workflow_rl/parallel_train_workflow_rl.py --red-agent sleep
```

#### `--n-envs` (Parallel Environments)
```bash
# Use 50 parallel environments (more data per update)
python workflow_rl/parallel_train_workflow_rl.py --n-envs 50

# Use 10 parallel environments (faster for debugging)
python workflow_rl/parallel_train_workflow_rl.py --n-envs 10
```

#### `--n-workflows` (Number of Workflows to Explore)
```bash
# Quick test with 5 workflows
python workflow_rl/parallel_train_workflow_rl.py --n-workflows 5

# Thorough search with 50 workflows
python workflow_rl/parallel_train_workflow_rl.py --n-workflows 50
```

#### `--max-episodes` (Max Episodes per Workflow)
```bash
# Allow more training time (200 episodes)
python workflow_rl/parallel_train_workflow_rl.py --max-episodes 200

# Quick testing (50 episodes)
python workflow_rl/parallel_train_workflow_rl.py --max-episodes 50
```

### Learning Configuration

#### `--alignment-lambda` (Compliance Reward Strength)
```bash
# Stricter compliance pressure
python workflow_rl/parallel_train_workflow_rl.py --alignment-lambda 50.0

# Gentler compliance pressure
python workflow_rl/parallel_train_workflow_rl.py --alignment-lambda 10.0
```

#### `--compliance-threshold` (Required Compliance)
```bash
# Lower threshold (easier to achieve, 90%)
python workflow_rl/parallel_train_workflow_rl.py --compliance-threshold 0.90

# Higher threshold (stricter, 98%)
python workflow_rl/parallel_train_workflow_rl.py --compliance-threshold 0.98
```

#### `--min-episodes` (Min Episodes Before Checking)
```bash
# Check compliance earlier (15 episodes)
python workflow_rl/parallel_train_workflow_rl.py --min-episodes 15

# Wait longer before checking (50 episodes)
python workflow_rl/parallel_train_workflow_rl.py --min-episodes 50
```

### Search Configuration

#### `--gp-beta` (GP-UCB Exploration)
```bash
# More exploration
python workflow_rl/parallel_train_workflow_rl.py --gp-beta 3.0

# More exploitation
python workflow_rl/parallel_train_workflow_rl.py --gp-beta 1.0
```

#### `--n-eval-episodes` (Evaluation Episodes)
```bash
# More thorough evaluation (50 episodes)
python workflow_rl/parallel_train_workflow_rl.py --n-eval-episodes 50

# Quick evaluation (10 episodes)
python workflow_rl/parallel_train_workflow_rl.py --n-eval-episodes 10
```

### Output Configuration

#### `--checkpoint-dir` (Save Directory)
```bash
# Custom save directory
python workflow_rl/parallel_train_workflow_rl.py --checkpoint-dir my_experiment_1

# Separate directories for different experiments
python workflow_rl/parallel_train_workflow_rl.py --checkpoint-dir bline_agent_test
```

#### `--seed` (Random Seed)
```bash
# Different random seed for new experiment
python workflow_rl/parallel_train_workflow_rl.py --seed 123

# Reproduce exact results
python workflow_rl/parallel_train_workflow_rl.py --seed 42
```

---

## Example Configurations

### Quick Test Run
```bash
python workflow_rl/parallel_train_workflow_rl.py \
    --n-envs 10 \
    --n-workflows 3 \
    --max-episodes 30 \
    --compliance-threshold 0.90 \
    --checkpoint-dir quick_test
```

### Thorough Search (Strict Compliance)
```bash
python workflow_rl/parallel_train_workflow_rl.py \
    --n-envs 25 \
    --n-workflows 50 \
    --max-episodes 200 \
    --alignment-lambda 50.0 \
    --compliance-threshold 0.98 \
    --checkpoint-dir strict_search
```

### Compare Red Agents
```bash
# Experiment 1: RedMeanderAgent
python workflow_rl/parallel_train_workflow_rl.py \
    --red-agent meander \
    --checkpoint-dir exp_meander \
    --seed 42

# Experiment 2: B_lineAgent
python workflow_rl/parallel_train_workflow_rl.py \
    --red-agent bline \
    --checkpoint-dir exp_bline \
    --seed 42
```

### Fast Prototyping
```bash
python workflow_rl/parallel_train_workflow_rl.py \
    --n-envs 5 \
    --n-workflows 2 \
    --max-episodes 20 \
    --min-episodes 10 \
    --compliance-threshold 0.85 \
    --n-eval-episodes 5 \
    --checkpoint-dir prototype
```

---

## Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--red-agent` | `meander` | RedMeanderAgent (aggressive) |
| `--n-envs` | `25` | Parallel environments |
| `--n-workflows` | `20` | Workflows to explore |
| `--max-episodes` | `100` | Max episodes per workflow |
| `--max-steps` | `100` | Max steps per episode |
| `--alignment-lambda` | `30.0` | Compliance reward strength |
| `--compliance-threshold` | `0.95` | Required compliance (95%) |
| `--min-episodes` | `25` | Min episodes before checking |
| `--update-steps` | `100` | PPO update frequency |
| `--gp-beta` | `2.0` | GP-UCB exploration |
| `--n-eval-episodes` | `20` | Evaluation episodes |
| `--checkpoint-dir` | `compliance_checkpoints` | Save directory |
| `--seed` | `42` | Random seed |

---

## Tips

### 1. Start with Default Settings
For first-time users, run with defaults:
```bash
python workflow_rl/parallel_train_workflow_rl.py
```

### 2. Use Descriptive Checkpoint Directories
```bash
# Good: Describes the experiment
python workflow_rl/parallel_train_workflow_rl.py --checkpoint-dir lambda30_threshold95_meander

# Bad: Generic name
python workflow_rl/parallel_train_workflow_rl.py --checkpoint-dir test1
```

### 3. Adjust Compliance Threshold Based on Red Agent
```bash
# RedMeanderAgent is aggressive -> may need lower threshold
python workflow_rl/parallel_train_workflow_rl.py --red-agent meander --compliance-threshold 0.90

# B_lineAgent is moderate -> can use higher threshold
python workflow_rl/parallel_train_workflow_rl.py --red-agent bline --compliance-threshold 0.95
```

### 4. Use Different Seeds for Multiple Runs
```bash
# Run 1
python workflow_rl/parallel_train_workflow_rl.py --seed 1 --checkpoint-dir run1

# Run 2
python workflow_rl/parallel_train_workflow_rl.py --seed 2 --checkpoint-dir run2

# Run 3
python workflow_rl/parallel_train_workflow_rl.py --seed 3 --checkpoint-dir run3
```

---

## Troubleshooting

### Training Too Slow
- Reduce `--n-envs` (e.g., 10 instead of 25)
- Reduce `--max-episodes` (e.g., 50 instead of 100)
- Lower `--compliance-threshold` (e.g., 0.90 instead of 0.95)

### Can't Achieve Compliance
- Increase `--alignment-lambda` (e.g., 50.0 instead of 30.0)
- Lower `--compliance-threshold` (e.g., 0.85 instead of 0.95)
- Increase `--max-episodes` (e.g., 200 instead of 100)

### Want More Diverse Workflows
- Increase `--n-workflows` (e.g., 50 instead of 20)
- Increase `--gp-beta` for more exploration (e.g., 3.0 instead of 2.0)

---

## Full Example with All Options

```bash
python workflow_rl/parallel_train_workflow_rl.py \
    --red-agent meander \
    --n-envs 25 \
    --n-workflows 20 \
    --max-episodes 100 \
    --max-steps 100 \
    --alignment-lambda 30.0 \
    --compliance-threshold 0.95 \
    --min-episodes 25 \
    --update-steps 100 \
    --gp-beta 2.0 \
    --n-eval-episodes 20 \
    --checkpoint-dir my_experiment \
    --seed 42
```

