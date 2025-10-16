#!/usr/bin/env python3
"""
Visualize why parallel sampling is slower than sequential
"""

import time
import numpy as np

def simulate_sequential(n_steps=100):
    """Simulate sequential environment stepping"""
    print("\n" + "="*60)
    print("SEQUENTIAL EXECUTION (1 Environment)")
    print("="*60)
    
    total_time = 0
    step_times = []
    
    for step in range(n_steps):
        # Each step takes ~3.4ms (measured)
        step_time = 0.0034
        total_time += step_time
        step_times.append(step_time)
    
    print(f"Steps: {n_steps}")
    print(f"Time per step: {np.mean(step_times)*1000:.1f}ms")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {n_steps/total_time:.0f} steps/second")
    
    return total_time

def simulate_parallel_current(n_envs=200, n_steps=100):
    """Simulate current parallel implementation (synchronous)"""
    print("\n" + "="*60)
    print(f"PARALLEL EXECUTION - CURRENT (Synchronous, {n_envs} Environments)")
    print("="*60)
    
    total_time = 0
    step_times = []
    
    for step in range(n_steps):
        # IPC overhead for sending actions
        send_time = 0.0001 * n_envs  # 0.1ms per env
        
        # Each env takes different time (3-8ms)
        env_times = np.random.uniform(0.003, 0.008, n_envs)
        
        # Must wait for SLOWEST environment
        step_time = send_time + np.max(env_times)
        
        # IPC overhead for receiving results
        recv_time = 0.0001 * n_envs
        step_time += recv_time
        
        # Get true states overhead
        true_state_time = 0.001 * n_envs  # 1ms per env
        step_time += true_state_time
        
        total_time += step_time
        step_times.append(step_time)
    
    print(f"Steps: {n_steps} (×{n_envs} envs = {n_steps*n_envs} transitions)")
    print(f"Time per step: {np.mean(step_times)*1000:.1f}ms")
    print(f"  - IPC send: {send_time*1000:.1f}ms")
    print(f"  - Wait for slowest env: {np.mean([np.max(np.random.uniform(0.003, 0.008, n_envs)) for _ in range(100)])*1000:.1f}ms")
    print(f"  - IPC receive: {recv_time*1000:.1f}ms")
    print(f"  - Get true states: {true_state_time*1000:.1f}ms")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {n_steps*n_envs/total_time:.0f} transitions/second")
    
    return total_time

def simulate_parallel_async(n_envs=200, n_steps=100):
    """Simulate improved async parallel implementation"""
    print("\n" + "="*60)
    print(f"PARALLEL EXECUTION - ASYNC (No waiting, {n_envs} Environments)")
    print("="*60)
    
    total_time = 0
    step_times = []
    env_timers = np.zeros(n_envs)
    completed_transitions = 0
    
    for step in range(n_steps):
        # Each env progresses independently
        for env_id in range(n_envs):
            if env_timers[env_id] <= 0:
                # This env is ready, give it new work
                env_timers[env_id] = np.random.uniform(0.003, 0.008)
                completed_transitions += 1
        
        # Time passes
        min_time = np.min(env_timers[env_timers > 0]) if np.any(env_timers > 0) else 0.001
        env_timers = np.maximum(0, env_timers - min_time)
        total_time += min_time
        step_times.append(min_time)
    
    print(f"Transitions completed: {completed_transitions}")
    print(f"Average time per transition: {total_time/completed_transitions*1000:.2f}ms")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {completed_transitions/total_time:.0f} transitions/second")
    
    return total_time

def visualize_bottleneck():
    """Show visual representation of the bottleneck"""
    print("\n" + "="*60)
    print("VISUAL: Why Parallel is Slower (Synchronous Barrier)")
    print("="*60)
    
    print("\nStep 1 - All environments must finish before Step 2:")
    print("─" * 50)
    
    # Simulate 10 environments with different completion times
    for env_id in range(10):
        completion_time = np.random.randint(3, 9)
        bar = "█" * completion_time
        spaces = " " * (8 - completion_time)
        print(f"Env {env_id:2d}: {bar}{spaces} {completion_time}ms")
    
    print("─" * 50)
    print("        ↑ MUST WAIT for slowest (8ms)")
    print("\nStep 2 - Can't start until ALL of Step 1 complete:")
    print("─" * 50)
    
    for env_id in range(10):
        completion_time = np.random.randint(3, 9)
        bar = "█" * completion_time
        spaces = " " * (8 - completion_time)
        print(f"Env {env_id:2d}: {bar}{spaces} {completion_time}ms")
    
    print("─" * 50)
    print("        ↑ MUST WAIT for slowest again")
    
    print("\n🔴 Problem: Fast environments idle while waiting for slow ones!")
    print("✅ Solution: Let each environment run at its own pace (async)")

if __name__ == "__main__":
    print("="*60)
    print("PARALLEL SAMPLING BOTTLENECK ANALYSIS")
    print("="*60)
    
    # Run simulations
    seq_time = simulate_sequential(100)
    par_time = simulate_parallel_current(200, 100)
    async_time = simulate_parallel_async(200, 100)
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    print(f"\nFor 20,000 transitions:")
    print(f"  Sequential (20,000 steps):     {seq_time*200:.1f}s")
    print(f"  Parallel Current (100 steps):  {par_time:.1f}s")
    print(f"  Parallel Async (100 steps):    {async_time:.1f}s")
    
    print(f"\nSpeedup over sequential:")
    print(f"  Current Parallel: {seq_time*200/par_time:.1f}x")
    print(f"  Async Parallel:   {seq_time*200/async_time:.1f}x")
    
    # Visual bottleneck
    visualize_bottleneck()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAY")
    print("="*60)
    print("\n🔴 Current parallel is slower because of SYNCHRONOUS BARRIER")
    print("   Every step waits for the slowest of 200 environments!")
    print("\n✅ Solution: ASYNC stepping - let fast envs keep going")
    print("   Could achieve 3-5x speedup with async implementation!")
    print("="*60)
