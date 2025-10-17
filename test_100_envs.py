#!/usr/bin/env python3
"""
Test specifically with 100 environments to see the scaling issue
"""

import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import time
import numpy as np
import psutil
import os

from workflow_rl.parallel_env_shared_memory import ParallelEnvSharedMemory
from CybORG.Agents import B_lineAgent


def monitor_resources():
    """Monitor CPU and memory usage"""
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=0.1)
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Count child processes
    try:
        children = process.children(recursive=True)
        n_children = len(children)
        child_cpu = sum(p.cpu_percent(interval=0) for p in children)
    except:
        n_children = 0
        child_cpu = 0
    
    return cpu_percent, memory_mb, n_children, child_cpu


def test_scaling():
    """Test scaling with different numbers of environments"""
    print("Testing environment scaling...")
    print("-" * 60)
    
    test_sizes = [1, 5, 10, 25, 50, 100]
    results = []
    
    for n_envs in test_sizes:
        print(f"\nTesting with {n_envs} environments:")
        
        # Monitor before
        cpu_before, mem_before, _, _ = monitor_resources()
        
        # Create environments
        start = time.time()
        try:
            envs = ParallelEnvSharedMemory(n_envs=n_envs, red_agent_type=B_lineAgent)
            creation_time = time.time() - start
            print(f"  Created in {creation_time:.2f}s")
        except Exception as e:
            print(f"  ERROR creating environments: {e}")
            continue
        
        # Monitor after creation
        cpu_after, mem_after, n_procs, child_cpu = monitor_resources()
        print(f"  Processes: {n_procs}, Memory: {mem_after:.0f}MB (+{mem_after-mem_before:.0f}MB)")
        print(f"  CPU: Main={cpu_after:.0f}%, Children={child_cpu:.0f}%")
        
        # Time reset
        start = time.time()
        obs = envs.reset()
        reset_time = time.time() - start
        print(f"  Reset time: {reset_time:.2f}s")
        
        # Time a single step
        start = time.time()
        actions = np.random.randint(0, 145, size=n_envs)
        obs, rewards, dones, infos = envs.step(actions)
        step_time = time.time() - start
        print(f"  Single step time: {step_time:.2f}s")
        
        # Time 10 steps
        start = time.time()
        for _ in range(10):
            actions = np.random.randint(0, 145, size=n_envs)
            obs, rewards, dones, infos = envs.step(actions)
        ten_steps_time = time.time() - start
        print(f"  10 steps time: {ten_steps_time:.2f}s ({ten_steps_time/10:.2f}s per step)")
        
        # Calculate throughput
        steps_per_sec = (10 * n_envs) / ten_steps_time
        print(f"  Throughput: {steps_per_sec:.1f} steps/sec total")
        print(f"             {steps_per_sec/n_envs:.1f} steps/sec per env")
        
        results.append({
            'n_envs': n_envs,
            'creation_time': creation_time,
            'reset_time': reset_time,
            'step_time': step_time,
            'steps_per_sec': steps_per_sec
        })
        
        # Clean up
        envs.close()
        time.sleep(1)  # Let processes clean up
    
    # Summary
    print("\n" + "="*60)
    print("SCALING SUMMARY")
    print("="*60)
    print(f"{'Envs':<10} {'Create':<10} {'Reset':<10} {'Step':<10} {'Steps/sec':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['n_envs']:<10} {r['creation_time']:<10.2f} {r['reset_time']:<10.2f} "
              f"{r['step_time']:<10.2f} {r['steps_per_sec']:<15.1f}")
    
    # Check for scaling issues
    if len(results) >= 2:
        small = results[0]['steps_per_sec']
        large = results[-1]['steps_per_sec']
        expected = small * results[-1]['n_envs']
        actual = large
        efficiency = (actual / expected) * 100
        
        print(f"\nScaling Efficiency:")
        print(f"  1 env: {small:.1f} steps/sec")
        print(f"  {results[-1]['n_envs']} envs: {large:.1f} steps/sec")
        print(f"  Expected: {expected:.1f} steps/sec")
        print(f"  Efficiency: {efficiency:.1f}%")
        
        if efficiency < 50:
            print("\n⚠️ SEVERE SCALING PROBLEM DETECTED!")
            print(f"   Only {efficiency:.1f}% of expected performance")


if __name__ == "__main__":
    print("="*60)
    print("100 ENVIRONMENT SCALING TEST")
    print("="*60)
    test_scaling()
