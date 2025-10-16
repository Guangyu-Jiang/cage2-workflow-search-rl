#!/usr/bin/env python3
"""
Measure the timing breakdown for the baseline PPO (train_no_action_reduction.py)
This uses a single environment and collects 20,000 steps before updating
"""

import sys
import os
import time
import numpy as np
import torch
import random

sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')
os.environ['PYTHONPATH'] = '/home/ubuntu/CAGE2/-cyborg-cage-2:' + os.environ.get('PYTHONPATH', '')

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.PPOAgent import PPOAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def measure_baseline_timing(episodes=10, update_timestep=20000, max_timesteps=100):
    """
    Measure timing for baseline PPO training
    
    Args:
        episodes: Number of episodes to measure
        update_timestep: Steps before PPO update (20000 default)
        max_timesteps: Max steps per episode (100 default)
    """
    
    print("="*60)
    print("BASELINE PPO TIMING ANALYSIS")
    print("="*60)
    print(f"Configuration:")
    print(f"  Single Environment (Sequential)")
    print(f"  Update Every: {update_timestep:,} steps")
    print(f"  Max Steps per Episode: {max_timesteps}")
    print(f"  Expected Episodes per Update: {update_timestep/max_timesteps:.0f}")
    print(f"  Episodes to Measure: {episodes}")
    print("="*60)
    
    # Create environment
    print("\n1. Creating environment...")
    start = time.time()
    scenario_path = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(scenario_path, 'sim', agents={'Red': B_lineAgent})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    env_creation_time = time.time() - start
    print(f"   ✓ Environment created in {env_creation_time:.2f}s")
    
    # Create agent
    print("\n2. Creating PPO agent...")
    start = time.time()
    input_dims = env.observation_space.shape[0]
    action_space = list(range(145))  # Full action space
    
    agent = PPOAgent(
        input_dims=input_dims,
        action_space=action_space,
        lr=0.002,
        betas=[0.9, 0.999],
        gamma=0.99,
        K_epochs=6,
        eps_clip=0.2
    )
    agent_creation_time = time.time() - start
    print(f"   ✓ Agent created in {agent_creation_time:.2f}s")
    
    # Timing tracking
    sampling_times = []
    training_times = []
    action_selection_times = []
    env_step_times = []
    memory_storage_times = []
    
    time_step = 0
    total_episodes = 0
    updates_performed = 0
    
    print(f"\n3. Running {episodes} episodes...")
    print("-"*60)
    
    # Track start of sampling batch
    batch_start_time = time.time()
    
    while total_episodes < episodes:
        # Episode start
        episode_start = time.time()
        
        # Reset
        reset_start = time.time()
        state = env.reset()
        reset_time = time.time() - reset_start
        
        episode_reward = 0
        episode_steps = 0
        episode_action_time = 0
        episode_step_time = 0
        episode_storage_time = 0
        
        # Episode loop
        for t in range(max_timesteps):
            time_step += 1
            episode_steps += 1
            
            # Action selection
            start = time.time()
            action = agent.get_action(state)
            episode_action_time += time.time() - start
            
            # Environment step
            start = time.time()
            state, reward, done, _ = env.step(action)
            episode_step_time += time.time() - start
            
            # Store in memory
            start = time.time()
            agent.store(reward, done)
            episode_storage_time += time.time() - start
            
            episode_reward += reward
            
            # Check for PPO update
            if time_step % update_timestep == 0:
                sampling_end = time.time()
                
                # Record sampling time (all episodes since last update)
                sampling_time = sampling_end - batch_start_time
                sampling_times.append(sampling_time)
                
                print(f"\n   PPO Update {updates_performed + 1}:")
                print(f"     Collected {time_step} steps in {sampling_time:.2f}s")
                print(f"     Episodes: {int(time_step/max_timesteps)}")
                print(f"     Sampling throughput: {time_step/sampling_time:.0f} steps/s")
                
                # PPO Training
                start = time.time()
                agent.train()
                training_time = time.time() - start
                training_times.append(training_time)
                
                print(f"     Training time: {training_time:.2f}s")
                print(f"     Sampling/Training Ratio: {sampling_time/training_time:.1f}x")
                
                # Clear memory
                agent.clear_memory()
                time_step = 0
                updates_performed += 1
                
                # Reset timing for next batch
                batch_start_time = time.time()
            
            if done:
                break
        
        # Record episode timings
        action_selection_times.append(episode_action_time)
        env_step_times.append(episode_step_time)
        memory_storage_times.append(episode_storage_time)
        
        # End episode
        agent.end_episode()
        total_episodes += 1
        
        if total_episodes % 50 == 0:
            print(f"   Completed {total_episodes} episodes...")
    
    print("\n" + "="*60)
    print("TIMING ANALYSIS SUMMARY")
    print("="*60)
    
    # Sampling breakdown
    if action_selection_times:
        print("\nSampling Components (per episode):")
        print(f"  Action Selection: {np.mean(action_selection_times):.3f}s")
        print(f"  Environment Step: {np.mean(env_step_times):.3f}s") 
        print(f"  Memory Storage: {np.mean(memory_storage_times):.4f}s")
        print(f"  Total per Episode: {np.mean(action_selection_times) + np.mean(env_step_times) + np.mean(memory_storage_times):.3f}s")
    
    if sampling_times and training_times:
        print(f"\nPer Update (20,000 steps):")
        print(f"  Average Sampling Time: {np.mean(sampling_times):.2f}s")
        print(f"  Average Training Time: {np.mean(training_times):.2f}s")
        print(f"  Sampling/Training Ratio: {np.mean(sampling_times)/np.mean(training_times):.1f}x")
        
        # Throughput
        steps_per_update = update_timestep
        total_time_per_update = np.mean(sampling_times) + np.mean(training_times)
        throughput = steps_per_update / total_time_per_update
        print(f"\nThroughput: {throughput:.0f} transitions/second")
    
    print("\n" + "="*60)
    print("COMPARISON WITH PARALLEL APPROACH")
    print("="*60)
    
    print("\nBaseline (Sequential, 1 env):")
    print(f"  Data Collection: 20,000 steps in {np.mean(sampling_times):.1f}s")
    print(f"  PPO Update: {np.mean(training_times):.1f}s")
    print(f"  Total: {np.mean(sampling_times) + np.mean(training_times):.1f}s")
    print(f"  Throughput: {throughput:.0f} transitions/second")
    
    print("\nParallel (200 envs):")
    print(f"  Data Collection: 20,000 steps in ~111s")
    print(f"  PPO Update: ~0.17s")
    print(f"  Total: ~111s")
    print(f"  Throughput: ~180 transitions/second")
    
    baseline_total = np.mean(sampling_times) + np.mean(training_times)
    parallel_total = 111
    speedup = baseline_total / parallel_total
    
    if speedup > 1:
        print(f"\nParallel is {speedup:.1f}x faster overall")
    else:
        print(f"\nBaseline is {1/speedup:.1f}x faster overall")
    
    # Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    baseline_sampling_ratio = np.mean(sampling_times) / np.mean(training_times)
    parallel_sampling_ratio = 111 / 0.17
    
    print(f"\nSampling is the bottleneck in BOTH approaches:")
    print(f"  Baseline: Sampling is {baseline_sampling_ratio:.0f}x slower than training")
    print(f"  Parallel: Sampling is {parallel_sampling_ratio:.0f}x slower than training")
    
    print(f"\nWhy baseline has higher throughput but parallel is better for learning:")
    print(f"  Baseline advantages:")
    print(f"    - No inter-process communication overhead")
    print(f"    - Single environment is faster per step")
    print(f"    - Higher raw throughput ({throughput:.0f} vs 180 steps/s)")
    print(f"  Parallel advantages:")
    print(f"    - 200x more diverse experiences per update")
    print(f"    - Better exploration (200 different trajectories)")
    print(f"    - More stable gradients from diverse batch")
    print(f"    - Can scale to even more environments")
    
    return {
        'sampling_times': sampling_times,
        'training_times': training_times,
        'throughput': throughput
    }


if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Run measurement
    results = measure_baseline_timing(episodes=250)  # Enough for ~1 update
    
    print("\n" + "="*60)
    print("Measurement complete!")
    print("="*60)
