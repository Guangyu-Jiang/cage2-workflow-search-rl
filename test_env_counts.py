#!/usr/bin/env python3
"""
Test different environment counts to find the optimal number
"""

import time
import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

from workflow_rl.parallel_env_shared_memory import ParallelEnvSharedMemory
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent

def test_env_count(n_envs):
    """Test a specific environment count"""
    print(f"\n{'='*60}")
    print(f"Testing with {n_envs} environments")
    print(f"{'='*60}")
    
    try:
        # Create environments
        start_time = time.time()
        envs = ParallelEnvSharedMemory(
            n_envs=n_envs,
            scenario_path='/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
            red_agent_type=RedMeanderAgent
        )
        creation_time = time.time() - start_time
        print(f"✓ Created in {creation_time:.2f} seconds")
        
        # Test reset
        reset_start = time.time()
        obs = envs.reset()
        reset_time = time.time() - reset_start
        print(f"✓ Reset in {reset_time:.2f} seconds")
        
        # Test 10 steps
        import numpy as np
        total_step_time = 0
        for step in range(10):
            actions = [np.random.randint(0, 100) for _ in range(n_envs)]
            step_start = time.time()
            obs, rewards, dones, infos = envs.step(actions)
            total_step_time += time.time() - step_start
        
        avg_step_time = total_step_time / 10
        print(f"✓ Average step time: {avg_step_time:.3f} seconds")
        print(f"✓ Throughput: {n_envs / avg_step_time:.1f} env-steps/second")
        
        # Close environments
        envs.close()
        print(f"✓ Closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("TESTING DIFFERENT ENVIRONMENT COUNTS")
    print("="*60)
    
    # Test different counts
    counts_to_test = [50, 75, 100, 150, 200]
    
    successful = []
    for n_envs in counts_to_test:
        if test_env_count(n_envs):
            successful.append(n_envs)
        else:
            print(f"\n⚠ Failed at {n_envs} environments")
            break
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Successful environment counts: {successful}")
    
    if successful:
        optimal = successful[-1]  # Highest successful count
        if optimal < 200:
            print(f"\n⚠ Maximum stable environments: {optimal}")
            print(f"Recommendation: Use --n-envs {optimal}")
        else:
            print(f"\n✓ All counts worked! Can use up to 200 environments")
            print(f"Recommendation: Use --n-envs 200 (or 100 for safety)")
    else:
        print("\n✗ All counts failed - there's a deeper issue")

if __name__ == "__main__":
    main()
