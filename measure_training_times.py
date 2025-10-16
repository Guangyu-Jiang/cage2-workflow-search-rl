#!/usr/bin/env python3
"""
Script to measure and analyze the time breakdown between:
1. Environment sampling (data collection)
2. Model training (PPO updates)
3. Other operations (compliance checking, logging, etc.)
"""

import sys
import os
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')
os.environ['PYTHONPATH'] = '/home/ubuntu/CAGE2/-cyborg-cage-2:' + os.environ.get('PYTHONPATH', '')

from CybORG.Agents import B_lineAgent
from workflow_rl.parallel_env_wrapper import ParallelEnvWrapper
from workflow_rl.parallel_order_conditioned_ppo import ParallelOrderConditionedPPO
from workflow_rl.order_based_workflow import OrderBasedWorkflow


class TimingAnalysis:
    def __init__(self):
        self.timings = defaultdict(list)
        
    def add_timing(self, category, duration):
        self.timings[category].append(duration)
        
    def print_summary(self):
        print("\n" + "="*60)
        print("TIMING ANALYSIS SUMMARY")
        print("="*60)
        
        total_time = sum(sum(times) for times in self.timings.values())
        
        for category, times in sorted(self.timings.items()):
            total = sum(times)
            avg = np.mean(times)
            percentage = (total / total_time * 100) if total_time > 0 else 0
            
            print(f"\n{category}:")
            print(f"  Total: {total:.2f}s ({percentage:.1f}%)")
            print(f"  Average: {avg:.3f}s")
            print(f"  Count: {len(times)}")
            if len(times) > 1:
                print(f"  Min: {min(times):.3f}s")
                print(f"  Max: {max(times):.3f}s")
        
        print(f"\nTotal Time: {total_time:.2f}s")


def measure_training_components(n_envs=200, n_steps=100, n_updates=5):
    """
    Measure the time for different components of training
    """
    print(f"="*60)
    print(f"MEASURING TRAINING COMPONENT TIMES")
    print(f"="*60)
    print(f"Configuration:")
    print(f"  Parallel Environments: {n_envs}")
    print(f"  Steps per Update: {n_steps}")
    print(f"  Number of Updates: {n_updates}")
    print(f"  Total Transitions: {n_envs * n_steps * n_updates:,}")
    print(f"="*60)
    
    timer = TimingAnalysis()
    
    # 1. Environment Creation
    print("\n1. Creating parallel environments...")
    start = time.time()
    envs = ParallelEnvWrapper(
        n_envs=n_envs,
        scenario_path='/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
        red_agent_type=B_lineAgent
    )
    timer.add_timing("Environment Creation", time.time() - start)
    print(f"   ✓ Created {n_envs} environments in {time.time() - start:.2f}s")
    
    # 2. Agent Creation
    print("\n2. Creating PPO agent...")
    start = time.time()
    workflow_manager = OrderBasedWorkflow()
    workflow_order = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
    
    agent = ParallelOrderConditionedPPO(
        input_dims=envs.observation_shape[0],
        n_envs=n_envs,
        workflow_order=workflow_order,
        workflow_manager=workflow_manager,
        alignment_lambda=30.0,
        update_steps=n_steps,
        K_epochs=4
    )
    timer.add_timing("Agent Creation", time.time() - start)
    print(f"   ✓ Created agent in {time.time() - start:.2f}s")
    
    # 3. Initial Reset
    print("\n3. Initial environment reset...")
    start = time.time()
    observations = envs.reset()
    timer.add_timing("Environment Reset", time.time() - start)
    print(f"   ✓ Reset in {time.time() - start:.2f}s")
    
    # 4. Sampling and Training Loop
    print(f"\n4. Running {n_updates} training cycles...")
    print("   Each cycle: sample {n_steps} steps, then PPO update")
    
    for update_idx in range(n_updates):
        print(f"\n   Cycle {update_idx + 1}/{n_updates}:")
        
        # Track sampling time for this cycle
        sampling_time = 0
        action_selection_time = 0
        env_step_time = 0
        reward_computation_time = 0
        buffer_storage_time = 0
        
        # Sampling phase
        for step in range(n_steps):
            # Get true states (for compliance)
            start = time.time()
            true_states = envs.get_true_states()
            timer.add_timing("Get True States", time.time() - start)
            
            # Select actions
            start = time.time()
            actions, log_probs, values = agent.get_actions(observations)
            action_selection_time += time.time() - start
            
            # Environment step
            start = time.time()
            observations, env_rewards, dones, infos = envs.step(actions)
            env_step_time += time.time() - start
            
            # Get new true states
            start = time.time()
            new_true_states = envs.get_true_states()
            timer.add_timing("Get True States", time.time() - start)
            
            # Compute alignment rewards
            start = time.time()
            alignment_rewards = agent.compute_alignment_rewards(
                actions, new_true_states, true_states, dones
            )
            total_rewards = env_rewards + alignment_rewards
            reward_computation_time += time.time() - start
            
            # Store in buffer
            start = time.time()
            agent.buffer.add(
                observations, actions, total_rewards, dones,
                log_probs.cpu().numpy(), values.cpu().numpy()
            )
            buffer_storage_time += time.time() - start
            
            # Update step count
            agent.step_count += 1
            
            # Handle episode resets
            for env_idx in range(n_envs):
                if dones[env_idx]:
                    agent.prev_true_states[env_idx] = None
                    agent.env_compliant_actions[env_idx] = 0
                    agent.env_total_fix_actions[env_idx] = 0
                    agent.env_fixed_types[env_idx] = set()
        
        # Record sampling times
        timer.add_timing("Action Selection", action_selection_time)
        timer.add_timing("Environment Step", env_step_time)
        timer.add_timing("Reward Computation", reward_computation_time)
        timer.add_timing("Buffer Storage", buffer_storage_time)
        
        total_sampling = action_selection_time + env_step_time + reward_computation_time + buffer_storage_time
        timer.add_timing("Total Sampling", total_sampling)
        
        print(f"      Sampling: {total_sampling:.2f}s")
        print(f"        - Action selection: {action_selection_time:.2f}s")
        print(f"        - Environment step: {env_step_time:.2f}s")
        print(f"        - Reward computation: {reward_computation_time:.2f}s")
        print(f"        - Buffer storage: {buffer_storage_time:.2f}s")
        
        # PPO Update phase
        print(f"      PPO Update:")
        start = time.time()
        agent.update()
        update_time = time.time() - start
        timer.add_timing("PPO Update", update_time)
        print(f"        - Time: {update_time:.2f}s")
        
        # Calculate ratio
        ratio = total_sampling / update_time if update_time > 0 else float('inf')
        print(f"      Sampling/Training Ratio: {ratio:.2f}x")
    
    # 5. Cleanup
    print("\n5. Cleaning up...")
    start = time.time()
    envs.close()
    timer.add_timing("Cleanup", time.time() - start)
    
    # Print summary
    timer.print_summary()
    
    # Additional analysis
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    sampling_total = sum(timer.timings["Total Sampling"])
    training_total = sum(timer.timings["PPO Update"])
    
    print(f"\nSampling vs Training:")
    print(f"  Total Sampling Time: {sampling_total:.2f}s")
    print(f"  Total Training Time: {training_total:.2f}s")
    print(f"  Ratio: {sampling_total/training_total:.2f}x")
    
    if sampling_total > training_total:
        print(f"\n⚠️  Sampling is the bottleneck ({sampling_total/training_total:.1f}x slower)")
        print("  Recommendations:")
        print("  - Consider using more parallel environments")
        print("  - Optimize environment step time")
        print("  - Use async environment stepping")
    else:
        print(f"\n⚠️  Training is the bottleneck ({training_total/sampling_total:.1f}x slower)")
        print("  Recommendations:")
        print("  - Reduce K_epochs in PPO")
        print("  - Use smaller batch sizes")
        print("  - Optimize neural network architecture")
    
    throughput = (n_envs * n_steps * n_updates) / (sampling_total + training_total)
    print(f"\nOverall Throughput: {throughput:.0f} transitions/second")
    
    return timer


def quick_benchmark():
    """Run a quick benchmark with smaller settings"""
    print("Running quick benchmark with reduced settings...")
    return measure_training_components(n_envs=50, n_steps=20, n_updates=3)


def full_benchmark():
    """Run full benchmark with production settings"""
    print("Running full benchmark with production settings...")
    return measure_training_components(n_envs=200, n_steps=100, n_updates=5)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Measure training time components')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Benchmark mode: quick (fast) or full (production settings)')
    args = parser.parse_args()
    
    if args.mode == 'quick':
        timer = quick_benchmark()
    else:
        timer = full_benchmark()
    
    print("\n" + "="*60)
    print("Benchmark completed!")
    print("="*60)
