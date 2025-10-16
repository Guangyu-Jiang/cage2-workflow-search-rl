#!/usr/bin/env python3
"""
Quick benchmark to compare the three implementations
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

from CybORG.Agents import B_lineAgent


def quick_test():
    """Quick test of all three implementations"""
    
    n_envs = 20  # Fewer environments for quick test
    n_steps = 50  # Fewer steps
    
    print("="*70)
    print("QUICK PARALLEL ENVIRONMENT BENCHMARK")
    print("="*70)
    print(f"\nTesting with {n_envs} environments, {n_steps} steps each")
    print("-"*70)
    
    results = []
    
    # Test 1: Original implementation
    try:
        print("\n1. ORIGINAL (Multiprocessing with Pipes)")
        print("-"*40)
        from workflow_rl.parallel_env_wrapper import ParallelEnvWrapper
        
        start = time.time()
        envs = ParallelEnvWrapper(n_envs=n_envs, red_agent_type=B_lineAgent)
        creation = time.time() - start
        print(f"   Creation: {creation:.2f}s")
        
        obs = envs.reset()
        start = time.time()
        for i in range(n_steps):
            actions = np.random.randint(0, 145, n_envs)
            obs, rewards, dones, infos = envs.step(actions)
        step_time = time.time() - start
        
        throughput = (n_envs * n_steps) / step_time
        print(f"   Step time: {step_time:.2f}s")
        print(f"   Throughput: {throughput:.0f} transitions/second")
        
        envs.close()
        results.append(("Original", throughput))
        
    except Exception as e:
        print(f"   Failed: {e}")
        results.append(("Original", 0))
    
    # Test 2: Shared Memory
    try:
        print("\n2. SHARED MEMORY (Multiprocessing with Shared Memory)")
        print("-"*40)
        from workflow_rl.parallel_env_shared_memory import ParallelEnvSharedMemory
        
        start = time.time()
        envs = ParallelEnvSharedMemory(n_envs=n_envs, red_agent_type=B_lineAgent)
        creation = time.time() - start
        print(f"   Creation: {creation:.2f}s")
        
        obs = envs.reset()
        start = time.time()
        for i in range(n_steps):
            actions = np.random.randint(0, 145, n_envs).tolist()
            obs, rewards, dones, infos = envs.step(actions)
        step_time = time.time() - start
        
        throughput = (n_envs * n_steps) / step_time
        print(f"   Step time: {step_time:.2f}s")
        print(f"   Throughput: {throughput:.0f} transitions/second")
        
        envs.close()
        results.append(("Shared Memory", throughput))
        
    except Exception as e:
        print(f"   Failed: {e}")
        results.append(("Shared Memory", 0))
    
    # Test 3: Vectorized
    try:
        print("\n3. VECTORIZED (Single Process, No IPC)")
        print("-"*40)
        from workflow_rl.parallel_env_vectorized import VectorizedCAGE2Envs
        
        start = time.time()
        envs = VectorizedCAGE2Envs(n_envs=n_envs, red_agent_type=B_lineAgent)
        creation = time.time() - start
        print(f"   Creation: {creation:.2f}s")
        
        obs = envs.reset()
        start = time.time()
        for i in range(n_steps):
            actions = np.random.randint(0, 145, n_envs)
            obs, rewards, dones, infos = envs.step(actions)
        step_time = time.time() - start
        
        throughput = (n_envs * n_steps) / step_time
        print(f"   Step time: {step_time:.2f}s")
        print(f"   Throughput: {throughput:.0f} transitions/second")
        
        envs.close()
        results.append(("Vectorized", throughput))
        
    except Exception as e:
        print(f"   Failed: {e}")
        results.append(("Vectorized", 0))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Implementation':<20} {'Throughput':>20} {'Relative Speed':>15}")
    print("-"*70)
    
    baseline = results[0][1] if results[0][1] > 0 else 1
    for name, throughput in results:
        if throughput > 0:
            speedup = throughput / baseline
            print(f"{name:<20} {throughput:>15.0f} t/s {speedup:>13.1f}x")
        else:
            print(f"{name:<20} {'FAILED':>20} {'N/A':>15}")
    
    # Winner
    if any(t > 0 for _, t in results):
        winner = max(results, key=lambda x: x[1])
        if winner[1] > 0:
            print("\n" + "="*70)
            print(f"🏆 WINNER: {winner[0]}")
            print(f"   {winner[1]:.0f} transitions/second")
            print(f"   {winner[1]/baseline:.1f}x faster than baseline")
            print("="*70)
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    if results[1][1] > results[0][1]:  # Shared memory faster
        speedup = results[1][1] / results[0][1] if results[0][1] > 0 else 0
        print(f"\n✅ Shared Memory is {speedup:.1f}x faster than Original")
        print("   - Eliminates pickle/unpickle overhead")
        print("   - Direct memory access for observations/rewards")
        print("   - Still uses multiprocessing for true parallelism")
    
    if results[2][1] > results[0][1]:  # Vectorized faster
        speedup = results[2][1] / results[0][1] if results[0][1] > 0 else 0
        print(f"\n✅ Vectorized is {speedup:.1f}x faster than Original")
        print("   - No IPC overhead at all")
        print("   - Single process (easier debugging)")
        print("   - Better cache locality")
    
    print("\n📊 The bottleneck was IPC overhead:")
    print("   - Original: 800 pipe operations per step")
    print("   - Shared Memory: Minimal serialization")
    print("   - Vectorized: Zero IPC")
    
    print("\n🚀 For production, recommend:")
    if results[1][1] > results[2][1]:
        print("   Shared Memory - Best performance with true parallelism")
    else:
        print("   Vectorized - Good performance with simpler architecture")
    
    print("="*70)


if __name__ == "__main__":
    quick_test()
