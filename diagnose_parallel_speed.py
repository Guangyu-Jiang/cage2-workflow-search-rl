#!/usr/bin/env python3
"""
Diagnose why parallel environments are running slowly
"""

import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import time
import numpy as np
from pathlib import Path

# Import environment components
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

def test_single_environment():
    """Test speed of a single environment"""
    print("Testing single environment speed...")
    
    scenario_path = Path('/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml')
    
    # Create single environment
    cyborg = CybORG(str(scenario_path), 'sim', agents={'Red': B_lineAgent})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    # Time a single episode
    start = time.time()
    obs = env.reset()
    
    steps = 0
    done = False
    while not done and steps < 100:
        action = np.random.randint(0, 145)
        obs, reward, done, info = env.step(action)
        steps += 1
    
    episode_time = time.time() - start
    print(f"  Single episode: {episode_time:.2f}s for {steps} steps")
    print(f"  Steps per second: {steps/episode_time:.1f}")
    
    # Time multiple episodes
    print("\nTiming 10 episodes...")
    start = time.time()
    for ep in range(10):
        obs = env.reset()
        steps = 0
        done = False
        while not done and steps < 100:
            action = np.random.randint(0, 145)
            obs, reward, done, info = env.step(action)
            steps += 1
    
    total_time = time.time() - start
    print(f"  10 episodes: {total_time:.2f}s")
    print(f"  Episodes per second: {10/total_time:.2f}")
    
    return episode_time


def test_parallel_overhead():
    """Test overhead of parallel implementation"""
    print("\n" + "="*60)
    print("Testing parallel environment overhead...")
    
    from workflow_rl.parallel_env_shared_memory import ParallelEnvSharedMemory
    
    # Test with different numbers of environments
    for n_envs in [1, 2, 5, 10]:
        print(f"\nTesting with {n_envs} environments:")
        
        # Create environments
        start = time.time()
        envs = ParallelEnvSharedMemory(n_envs=n_envs, red_agent_type=B_lineAgent)
        creation_time = time.time() - start
        print(f"  Creation time: {creation_time:.2f}s")
        
        # Time a single episode per env
        start = time.time()
        obs = envs.reset()
        
        for step in range(100):
            actions = np.random.randint(0, 145, size=n_envs)
            obs, rewards, dones, infos = envs.step(actions)
            
            if np.any(dones):
                break
        
        episode_time = time.time() - start
        print(f"  Episode time: {episode_time:.2f}s")
        print(f"  Time per env: {episode_time/n_envs:.2f}s")
        print(f"  Overhead ratio: {(episode_time/n_envs) / (episode_time/max(1, n_envs)):.2f}x")
        
        # Clean up
        envs.close()
        
        if n_envs >= 10:
            break  # Don't test with too many


def test_step_breakdown():
    """Break down where time is spent in stepping"""
    print("\n" + "="*60)
    print("Analyzing step timing breakdown...")
    
    scenario_path = Path('/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml')
    cyborg = CybORG(str(scenario_path), 'sim', agents={'Red': B_lineAgent})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    obs = env.reset()
    
    # Time different parts
    n_steps = 100
    
    # Time action selection
    start = time.time()
    for _ in range(n_steps):
        action = np.random.randint(0, 145)
    action_time = time.time() - start
    
    # Time environment steps
    start = time.time()
    for _ in range(n_steps):
        action = np.random.randint(0, 145)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    step_time = time.time() - start
    
    print(f"  Action selection: {action_time*1000:.2f}ms for {n_steps} steps")
    print(f"  Environment step: {step_time*1000:.2f}ms for {n_steps} steps")
    print(f"  Step time each: {(step_time/n_steps)*1000:.2f}ms")
    print(f"  Steps per second: {n_steps/step_time:.1f}")


def main():
    print("="*60)
    print("PARALLEL ENVIRONMENT PERFORMANCE DIAGNOSIS")
    print("="*60)
    
    # Test single environment speed
    single_episode_time = test_single_environment()
    
    # Test parallel overhead
    test_parallel_overhead()
    
    # Test step breakdown
    test_step_breakdown()
    
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    print(f"Single episode takes: {single_episode_time:.2f}s")
    print(f"Expected parallel speed: {100/single_episode_time:.1f} episodes/sec")
    print("\nBut actual parallel speed is ~1.1 episodes/sec")
    print("This suggests a ~100x slowdown from parallelization overhead!")
    
    print("\nLikely causes:")
    print("1. Synchronous barrier - all envs wait for slowest")
    print("2. IPC overhead with queues")
    print("3. CybORG environment is inherently slow")
    print("4. Shared memory not being used effectively")


if __name__ == "__main__":
    main()
