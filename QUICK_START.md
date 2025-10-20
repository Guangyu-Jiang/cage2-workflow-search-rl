# 🚀 Quick Start Guide - Workflow Search RL

## Production-Ready Training (Recommended!)

### **Best Performance: ProcessPoolExecutor Async**

```bash
# Run with 100 workers (74 episodes/sec)
bash run_executor_async_training.sh

# Or custom settings:
python workflow_rl/executor_async_train_workflow_rl.py \
    --n-workers 100 \
    --total-episodes 100000 \
    --red-agent B_lineAgent
```

**Performance**: 74.3 eps/sec, **100k episodes in ~22 minutes**

---

## Key Features

✅ **True Async Training** - Workers collect episodes independently  
✅ **All 120 Workflows Evaluated** - Complete UCB optimization  
✅ **Workflow-Specific Policies** - Each workflow trains independently  
✅ **Correct Compliance Tracking** - Checks actual environment state  
✅ **Three-Component Rewards** - Env, Alignment, Total logged separately  
✅ **67x Speedup** - vs synchronous parallel  

---

## File Structure (After Cleanup)

### **Main Training Scripts:**
```
workflow_rl/
├── executor_async_train_workflow_rl.py  ⭐ USE THIS (fastest!)
├── parallel_train_workflow_rl.py         (synchronous, slower)
└── sequential_train_workflow_rl.py       (single env, debugging)
```

### **Launch Scripts:**
```
run_executor_async_training.sh  ⭐ Main script
run_training_safe.sh             (synchronous version)
```

### **Core Documentation:**
```
README.md                                  - Project overview
QUICK_START.md                             - This file
COMPLIANCE_CORRECT_IMPLEMENTATION.md       - How compliance works
POLICY_INHERITANCE_EXPLAINED.md            - Policy storage per workflow
ALL_PERMUTATIONS_UCB.md                    - Why we evaluate all 120
RED_AGENT_GUIDE.md                         - Red agent selection
LOGGING_FORMAT_SUMMARY.md                  - CSV format explanation
```

---

## Training Options

### **Red Agents:**
```bash
--red-agent B_lineAgent        # Moderate (default)
--red-agent RedMeanderAgent    # Aggressive (recommended)
--red-agent SleepAgent         # No attacks (debugging)
```

### **Workers:**
```bash
--n-workers 25    # Good efficiency (40 eps/sec)
--n-workers 50    # Better (50 eps/sec)
--n-workers 100   # Best (74 eps/sec) ⭐
```

### **Episode Budget:**
```bash
--total-episodes 10000    # Quick test
--total-episodes 100000   # Full training ⭐
```

---

## Output

### **Console:**
```
Iteration 1
  Evaluating UCB for ALL 120 possible workflows...
  
  Selected: defender → op_server → enterprise → op_host → user
  
  Update 1: Episodes: 100 total
    Env Reward/Episode: -708.61
    Total Reward/Episode: -701.23
    Alignment Bonus (episode-end): +7.37
    Compliance: 24.57%
    ⏱️ Timing: Sampling=1.3s, Update=0.04s
```

### **Logs:**
```
logs/exp_executor_async_*/
├── training_log.csv        # Episode data with 3 reward components
├── gp_sampling_log.csv     # GP-UCB decisions
└── workflow_*_agent.pt     # Saved policies per workflow
```

---

## Performance Summary

| Implementation | Episodes/sec | Time for 100k |
|---------------|--------------|---------------|
| Sequential | 2.6 | 10.7 hours |
| Synchronous Parallel | 1.1 | 25.3 hours |
| **Executor Async** | **74.3** | **22 minutes** ⭐ |

---

## Quick Reference

```bash
# Production training (recommended)
bash run_executor_async_training.sh

# View logs
ls -lh logs/exp_executor_async_*/

# Plot results
python plot_baseline_training.py
```

---

## Git Repository

All code pushed to: `https://github.com/Guangyu-Jiang/cage2-workflow-search-rl.git`

Latest commit includes:
- ProcessPoolExecutor async implementation (74.3 eps/sec)
- Corrected compliance calculation
- All 120 workflows evaluated
- Workflow-specific policy storage
- Comprehensive documentation

---

## Next Steps

1. **Run training**: `bash run_executor_async_training.sh`
2. **Monitor logs**: Check `logs/exp_executor_async_*/training_log.csv`
3. **Analyze results**: Use the CSV to find best workflows
4. **Deploy policy**: Load best workflow's checkpoint

Happy training! 🚀
