#!/usr/bin/env python3
"""
Test the optimized parallel environment implementation
"""

import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import time
import numpy as np
from CybORG.Agents import B_lineAgent

# Test both implementations
from workflow_rl.parallel_env_shared_memory import ParallelEnvSharedMemory
from workflow_rl.parallel_env_shared_memory_optimized import ParallelEnvSharedMemoryOptimized


def test_implementation(env_class, name, n_envs=25, n_episodes=10):
    """Test an environment implementation"""
    print(f"\nTesting {name} with {n_envs} environments...")
    print("-" * 60)
    
    # Create environments
    start = time.time()
    envs = env_class(n_envs=n_envs, red_agent_type=B_lineAgent)
    creation_time = time.time() - start
    print(f"  Creation time: {creation_time:.2f}s")
    
    # Time episodes with true state calls
    start = time.time()
    total_steps = 0
    
    for ep in range(n_episodes):
        obs = envs.reset()
        
        for step in range(100):
            # Get true states BEFORE action (as in training)
            true_states_before = envs.get_true_states()
            
            # Take actions
            actions = np.random.randint(0, 145, size=n_envs)
            obs, rewards, dones, infos = envs.step(actions)
            
            # Get true states AFTER action (as in training)
            true_states_after = envs.get_true_states()
            
            total_steps += n_envs
            
            if np.any(dones):
                break
    
    elapsed = time.time() - start
    
    # Calculate metrics
    episodes_per_sec = (n_episodes * n_envs) / elapsed
    steps_per_sec = total_steps / elapsed
    
    print(f"  Time for {n_episodes} episodes: {elapsed:.2f}s")
    print(f"  Episodes/sec: {episodes_per_sec:.1f}")
    print(f"  Steps/sec: {steps_per_sec:.1f}")
    print(f"  Time per episode: {elapsed/(n_episodes*n_envs)*1000:.1f}ms")
    
    # Clean up
    envs.close()
    
    return episodes_per_sec


def main():
    print("="*60)
    print("OPTIMIZED PARALLEL ENVIRONMENT TEST")
    print("="*60)
    print("\nComparing implementations with true state calls...")
    
    # Test with 25 environments (more reasonable scaling)
    n_envs = 25
    n_episodes = 5
    
    # Test original
    original_speed = test_implementation(
        ParallelEnvSharedMemory, 
        "Original (shared queue)",
        n_envs, n_episodes
    )
    
    time.sleep(2)  # Let system recover
    
    # Test optimized
    optimized_speed = test_implementation(
        ParallelEnvSharedMemoryOptimized,
        "Optimized (dedicated pipes + caching)",
        n_envs, n_episodes
    )
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Original:  {original_speed:.1f} episodes/sec")
    print(f"Optimized: {optimized_speed:.1f} episodes/sec")
    
    if optimized_speed > original_speed:
        speedup = optimized_speed / original_speed
        print(f"\n🚀 Speedup: {speedup:.2f}x faster!")
    else:
        print("\n⚠️ No improvement - check implementation")
    
    print("\nExpected improvements:")
    print("  - Dedicated pipes eliminate queue contention")
    print("  - Cached true states reduce redundant computation")
    print("  - Sparse updates only get states when needed")


if __name__ == "__main__":
    main()
