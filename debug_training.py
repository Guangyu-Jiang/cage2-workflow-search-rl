#!/usr/bin/env python3
"""
Debug script to identify where the training is getting stuck
"""

import os
import sys
import time
import traceback

# Add parent directory to path for imports
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

def test_small_env_count():
    """Test with a small number of environments to see if it works"""
    print("="*60)
    print("TESTING WITH SMALL ENVIRONMENT COUNT")
    print("="*60)
    
    from workflow_rl.parallel_train_workflow_rl import ParallelWorkflowRLTrainer
    from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
    
    print("\n1. Creating trainer with 10 environments...")
    
    try:
        trainer = ParallelWorkflowRLTrainer(
            n_envs=10,  # Use only 10 environments instead of 200
            total_episode_budget=50,  # Small budget for testing
            max_train_episodes_per_env=10,
            red_agent_type=RedMeanderAgent
        )
        print("   ✓ Trainer created successfully")
        
        print("\n2. Starting workflow search...")
        trainer.run_workflow_search()
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False
    
    return True

def test_env_creation():
    """Test just the environment creation to see if that's the issue"""
    print("\n" + "="*60)
    print("TESTING ENVIRONMENT CREATION")
    print("="*60)
    
    from workflow_rl.parallel_env_shared_memory import ParallelEnvSharedMemory
    from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
    
    for n_envs in [1, 5, 10, 20, 50]:
        print(f"\nTesting with {n_envs} environments...")
        start_time = time.time()
        
        try:
            envs = ParallelEnvSharedMemory(
                n_envs=n_envs,
                scenario_path='/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                red_agent_type=RedMeanderAgent
            )
            creation_time = time.time() - start_time
            print(f"   ✓ Created in {creation_time:.2f} seconds")
            
            # Test reset
            print(f"   Testing reset...")
            reset_start = time.time()
            obs = envs.reset()
            reset_time = time.time() - reset_start
            print(f"   ✓ Reset in {reset_time:.2f} seconds")
            
            # Test step
            print(f"   Testing step...")
            import numpy as np
            actions = [np.random.randint(0, 100) for _ in range(n_envs)]
            step_start = time.time()
            obs, rewards, dones, infos = envs.step(actions)
            step_time = time.time() - step_start
            print(f"   ✓ Step in {step_time:.2f} seconds")
            
            # Close environments
            envs.close()
            print(f"   ✓ Closed successfully")
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            return False
    
    return True

def check_system_resources():
    """Check system resources to see if that's the issue"""
    print("\n" + "="*60)
    print("SYSTEM RESOURCES CHECK")
    print("="*60)
    
    import psutil
    
    # CPU info
    print(f"\nCPU:")
    print(f"  Cores (physical): {psutil.cpu_count(logical=False)}")
    print(f"  Cores (logical): {psutil.cpu_count(logical=True)}")
    print(f"  Usage: {psutil.cpu_percent(interval=1)}%")
    
    # Memory info
    mem = psutil.virtual_memory()
    print(f"\nMemory:")
    print(f"  Total: {mem.total / (1024**3):.1f} GB")
    print(f"  Available: {mem.available / (1024**3):.1f} GB")
    print(f"  Used: {mem.percent}%")
    
    # Check for zombie processes
    zombies = []
    for proc in psutil.process_iter(['pid', 'name', 'status']):
        try:
            if proc.info['status'] == psutil.STATUS_ZOMBIE:
                zombies.append(proc.info)
        except:
            pass
    
    if zombies:
        print(f"\n⚠ Found {len(zombies)} zombie processes")
    else:
        print(f"\n✓ No zombie processes found")
    
    # Check shared memory
    try:
        result = os.popen("ipcs -m 2>/dev/null | wc -l").read()
        shm_count = int(result.strip()) - 3  # Subtract header lines
        print(f"\nShared Memory Segments: {shm_count}")
    except:
        pass

def main():
    print("="*60)
    print("DEBUGGING PARALLEL TRAINING HANG")
    print("="*60)
    
    # Check system resources first
    check_system_resources()
    
    # Test environment creation with increasing counts
    if not test_env_creation():
        print("\n✗ Environment creation failed")
        return
    
    print("\n" + "="*60)
    print("Environment creation tests passed!")
    print("="*60)
    
    # Test with small environment count
    if test_small_env_count():
        print("\n" + "="*60)
        print("✓ Small environment count works!")
        print("The issue is likely with creating 200 processes.")
        print("\nRecommendation: Use fewer parallel environments")
        print("Try: --n-envs 50 or --n-envs 100")
        print("="*60)
    else:
        print("\n✗ Even small environment count failed")
        print("There may be a deeper issue with the code")


if __name__ == "__main__":
    main()
