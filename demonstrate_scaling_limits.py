#!/usr/bin/env python3
"""
Demonstrates why we can't achieve 100x speedup with 100 parallel environments.
This minimal example shows the fundamental limitations of Python multiprocessing.
"""

import time
import numpy as np
from multiprocessing import Process, Queue, Pipe
import multiprocessing as mp

class MockEnvironment:
    """Simulates a CAGE2 environment with realistic delays"""
    
    def __init__(self, env_id, slow_probability=0.05):
        self.env_id = env_id
        self.slow_probability = slow_probability
        self.state = np.random.randn(52)  # CAGE2 state size
        
    def step(self):
        """Simulate environment step with occasional slowness"""
        # Most steps are fast (1ms)
        base_time = 0.001
        
        # But some steps are slow (10ms) - simulates complex game states
        if np.random.random() < self.slow_probability:
            base_time = 0.01
            
        time.sleep(base_time)
        
        # Return new state and reward
        self.state = np.random.randn(52)
        reward = np.random.randn()
        return self.state, reward
    
    def get_true_state(self):
        """Simulate expensive true state computation"""
        time.sleep(0.002)  # 2ms for dictionary operations
        return {"env_id": self.env_id, "data": np.random.randn(100)}


def worker_sync(env_id, command_queue, result_queue):
    """Synchronous worker (current implementation)"""
    env = MockEnvironment(env_id)
    
    while True:
        cmd = command_queue.get()
        if cmd == "step":
            result = env.step()
            result_queue.put((env_id, result))
        elif cmd == "true_state":
            result = env.get_true_state()
            result_queue.put((env_id, result))
        elif cmd == "stop":
            break


def benchmark_synchronous(n_envs, n_steps):
    """Benchmark synchronous parallel stepping (current approach)"""
    print(f"\n📊 Synchronous Parallel Benchmark ({n_envs} envs)")
    print("-" * 50)
    
    # Setup
    command_queue = Queue()
    result_queue = Queue()
    workers = []
    
    for i in range(n_envs):
        p = Process(target=worker_sync, args=(i, command_queue, result_queue))
        p.start()
        workers.append(p)
    
    # Measure stepping time
    start = time.time()
    
    for step in range(n_steps):
        # Send step command to all (instant)
        for _ in range(n_envs):
            command_queue.put("step")
        
        # Wait for ALL results (synchronous barrier!)
        results = []
        for _ in range(n_envs):
            results.append(result_queue.get())
        
        # Everyone waits for the slowest!
        
    elapsed = time.time() - start
    
    # Cleanup
    for _ in range(n_envs):
        command_queue.put("stop")
    for p in workers:
        p.join()
    
    steps_per_sec = (n_steps * n_envs) / elapsed
    efficiency = (steps_per_sec / (n_steps * n_envs / (n_steps * 0.001))) * 100
    
    print(f"⏱️  Time: {elapsed:.2f}s")
    print(f"📈 Steps/sec: {steps_per_sec:.1f} total")
    print(f"📉 Per-env: {steps_per_sec/n_envs:.1f} steps/sec")
    print(f"⚡ Efficiency: {efficiency:.1f}% (vs ideal)")
    
    return steps_per_sec


def benchmark_single(n_steps):
    """Benchmark single environment (baseline)"""
    print(f"\n📊 Single Environment Benchmark")
    print("-" * 50)
    
    env = MockEnvironment(0, slow_probability=0)  # No slowness for baseline
    
    start = time.time()
    for _ in range(n_steps):
        env.step()
    elapsed = time.time() - start
    
    steps_per_sec = n_steps / elapsed
    
    print(f"⏱️  Time: {elapsed:.2f}s")
    print(f"📈 Steps/sec: {steps_per_sec:.1f}")
    
    return steps_per_sec


def analyze_scaling():
    """Analyze scaling efficiency"""
    print("=" * 60)
    print("🚀 Parallel Environment Scaling Analysis")
    print("=" * 60)
    
    n_steps = 100
    
    # Baseline
    single_speed = benchmark_single(n_steps)
    
    # Parallel benchmarks
    env_counts = [1, 10, 25, 50, 100]
    speeds = []
    
    for n_envs in env_counts:
        speed = benchmark_synchronous(n_envs, n_steps // n_envs)
        speeds.append(speed)
        time.sleep(0.5)  # Let system settle
    
    # Analysis
    print("\n" + "=" * 60)
    print("📊 Scaling Analysis Summary")
    print("=" * 60)
    
    print(f"\n{'Envs':>5} {'Total Speed':>12} {'Efficiency':>12} {'Speedup':>10}")
    print("-" * 40)
    
    for n_envs, speed in zip(env_counts, speeds):
        ideal_speed = single_speed * n_envs
        efficiency = (speed / ideal_speed) * 100
        speedup = speed / single_speed
        print(f"{n_envs:5d} {speed:11.1f}/s {efficiency:11.1f}% {speedup:9.1f}x")
    
    print("\n🔍 Key Insights:")
    print("1. Efficiency drops dramatically with more environments")
    print("2. Synchronous barriers mean waiting for the slowest")
    print("3. IPC overhead increases with more processes")
    print("4. 100 envs ≠ 100x speedup in practice!")
    
    print("\n💡 To achieve linear scaling, you need:")
    print("  ✓ Asynchronous stepping (no barriers)")
    print("  ✓ Shared memory (no serialization)")
    print("  ✓ Vectorized operations (single process)")
    print("  ✓ Professional frameworks (Ray/RLlib)")


if __name__ == "__main__":
    # Ensure clean multiprocessing
    mp.set_start_method('spawn', force=True)
    
    analyze_scaling()
    
    print("\n✅ Analysis complete!")
    print("This demonstrates why the current approach can't achieve 100x speedup.")
    print("See PARALLEL_SCALING_ANALYSIS.md for solutions.")
