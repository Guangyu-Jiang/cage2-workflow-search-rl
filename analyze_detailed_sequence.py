#!/usr/bin/env python3
"""
Analyze detailed action sequences with true network compromise states
"""

import torch
import numpy as np
import os
import inspect
from tabulate import tabulate

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.PPOAgent import PPOAgent

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'

def parse_true_state(true_table_str):
    """Parse the true state table to extract compromised hosts"""
    lines = true_table_str.split('\n')
    compromised_hosts = {}
    
    for line in lines:
        if '|' in line and 'Hostname' not in line and '---' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                hostname = parts[3]
                access = parts[6]
                if access not in ['None', '']:
                    compromised_hosts[hostname] = access
                    
    return compromised_hosts

def run_detailed_episode(model_path, red_agent_class, num_steps=30):
    """Run one episode with detailed logging of actions and network state"""
    
    # Create environment
    cyborg = CybORG(PATH, 'sim', agents={'Red': red_agent_class})
    env = ChallengeWrapper2(env=cyborg, agent_name="Blue")
    
    # Create agent
    action_space = [133, 134, 135, 139]  # restore enterprise and opserver
    action_space += [3, 4, 5, 9]  # analyse enterprise and opserver
    action_space += [16, 17, 18, 22]  # remove enterprise and opserer
    action_space += [11, 12, 13, 14]  # analyse user hosts
    action_space += [141, 142, 143, 144]  # restore user hosts
    action_space += [132]  # restore defender
    action_space += [2]  # analyse defender
    action_space += [15, 24, 25, 26, 27]  # remove defender and user hosts
    
    agent = PPOAgent(
        input_dims=52,
        action_space=action_space,
        restore=True,
        ckpt=model_path,
        deterministic=True,
        training=False
    )
    
    # Reset environment
    observation = env.reset()
    
    # Store episode data
    episode_log = []
    total_reward = 0
    
    print("\n" + "="*100)
    print(f"DETAILED EPISODE: Blue (PPO) vs {red_agent_class.__name__}")
    print("="*100)
    
    for step in range(num_steps):
        # Get blue action
        blue_action = agent.get_action(observation)
        
        # Step environment
        next_obs, reward, done, info = env.step(blue_action)
        
        # Get action names
        blue_action_name = str(cyborg.get_last_action('Blue'))
        red_action_name = str(cyborg.get_last_action('Red'))
        
        # Get true state
        true_state = cyborg.get_agent_state('True')
        true_table = true_obs_to_table(true_state, cyborg)
        compromised_hosts = parse_true_state(str(true_table))
        
        # Store step data
        step_data = {
            'step': step,
            'blue_action': blue_action_name,
            'red_action': red_action_name,
            'reward': reward,
            'compromised_hosts': compromised_hosts.copy()
        }
        episode_log.append(step_data)
        
        # Print step information
        print(f"\n{'='*80}")
        print(f"STEP {step}")
        print(f"{'='*80}")
        print(f"Blue Action: {blue_action_name}")
        print(f"Red Action:  {red_action_name}")
        print(f"Reward:      {reward:.2f}")
        print(f"Compromised Hosts:")
        if compromised_hosts:
            for host, access in compromised_hosts.items():
                print(f"  - {host}: {access}")
        else:
            print("  - None (network is clean)")
        
        total_reward += reward
        observation = next_obs
        
        if done:
            break
    
    print(f"\n{'='*80}")
    print(f"EPISODE SUMMARY")
    print(f"{'='*80}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Network State:")
    if compromised_hosts:
        for host, access in compromised_hosts.items():
            print(f"  - {host}: {access}")
    else:
        print("  - Network is clean")
    
    agent.end_episode()
    
    return episode_log, total_reward

def analyze_action_patterns(episode_log):
    """Analyze patterns in the action sequence"""
    
    print("\n" + "="*100)
    print("ACTION SEQUENCE ANALYSIS")
    print("="*100)
    
    # Count action types
    action_types = {
        'Analyse': 0,
        'Remove': 0,
        'Restore': 0,
        'Decoy': 0,
        'Monitor': 0,
        'Sleep': 0
    }
    
    for step_data in episode_log:
        blue_action = step_data['blue_action']
        for action_type in action_types:
            if action_type in blue_action:
                action_types[action_type] += 1
                break
    
    print("\nAction Type Distribution:")
    for action_type, count in action_types.items():
        percentage = (count / len(episode_log)) * 100
        print(f"  {action_type:10s}: {count:3d} ({percentage:5.1f}%)")
    
    # Analyze compromise progression
    print("\nCompromise Progression:")
    max_compromised = 0
    escalation_steps = []
    
    for i, step_data in enumerate(episode_log):
        num_compromised = len(step_data['compromised_hosts'])
        if num_compromised > max_compromised:
            max_compromised = num_compromised
            escalation_steps.append(i)
            print(f"  Step {i:2d}: Red compromised {step_data['compromised_hosts']}")
    
    # Analyze blue responses
    print("\nBlue Response Patterns:")
    for i in range(1, len(episode_log)):
        prev_compromised = episode_log[i-1]['compromised_hosts']
        curr_compromised = episode_log[i]['compromised_hosts']
        blue_action = episode_log[i]['blue_action']
        
        # Check if blue successfully cleaned a host
        if len(prev_compromised) > len(curr_compromised):
            cleaned = set(prev_compromised.keys()) - set(curr_compromised.keys())
            print(f"  Step {i:2d}: Blue cleaned {cleaned} using {blue_action}")
        
        # Check if blue prevented an escalation
        if 'Privileged' in str(prev_compromised.values()) and 'Restore' in blue_action:
            print(f"  Step {i:2d}: Blue restored privileged host using {blue_action}")

def create_summary_table(episode_log):
    """Create a summary table of the episode"""
    
    print("\n" + "="*100)
    print("EPISODE SUMMARY TABLE")
    print("="*100)
    
    table_data = []
    for step_data in episode_log[:20]:  # First 20 steps for readability
        compromised_str = ', '.join([f"{h}({a[0]})" for h, a in step_data['compromised_hosts'].items()])
        if not compromised_str:
            compromised_str = "Clean"
        
        # Truncate action names for readability
        blue_action = step_data['blue_action'][:30]
        red_action = step_data['red_action'][:30]
        
        table_data.append([
            step_data['step'],
            blue_action,
            red_action,
            f"{step_data['reward']:.1f}",
            compromised_str[:40]
        ])
    
    headers = ["Step", "Blue Action", "Red Action", "Reward", "Compromised"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    # Use the meander model against Meander agent for demonstration
    model_path = '/home/ubuntu/CAGE2/-cyborg-cage-2/Models/meander/model.pth'
    red_agent = RedMeanderAgent
    
    print("\n" + "#"*100)
    print("# DETAILED ACTION SEQUENCE WITH NETWORK COMPROMISE STATE")
    print("#"*100)
    
    # Run detailed episode
    episode_log, total_reward = run_detailed_episode(model_path, red_agent, num_steps=30)
    
    # Analyze patterns
    analyze_action_patterns(episode_log)
    
    # Create summary table
    create_summary_table(episode_log)
    
    # Save detailed log
    output_file = '/home/ubuntu/CAGE2/-cyborg-cage-2/detailed_episode_log.txt'
    with open(output_file, 'w') as f:
        f.write("DETAILED EPISODE LOG\n")
        f.write("="*80 + "\n")
        for step_data in episode_log:
            f.write(f"\nStep {step_data['step']}:\n")
            f.write(f"  Blue: {step_data['blue_action']}\n")
            f.write(f"  Red: {step_data['red_action']}\n")
            f.write(f"  Reward: {step_data['reward']}\n")
            f.write(f"  Compromised: {step_data['compromised_hosts']}\n")
        f.write(f"\nTotal Reward: {total_reward}\n")
    
    print(f"\nDetailed log saved to {output_file}")

if __name__ == "__main__":
    main()
