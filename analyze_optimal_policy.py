#!/usr/bin/env python3
"""
Analyze optimal policies and extract typical action sequences
"""

import torch
import numpy as np
import os
import inspect
from collections import defaultdict
import json

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent, SleepAgent
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.PPOAgent import PPOAgent

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'

def create_agent(model_path, action_space):
    """Create a PPO agent with loaded model"""
    agent = PPOAgent(
        input_dims=52, 
        action_space=action_space,
        restore=True,
        ckpt=model_path,
        deterministic=True,  # Use deterministic policy for analysis
        training=False
    )
    return agent

def analyze_episode(env, agent, red_agent_class, num_steps=100, verbose=True):
    """Run one episode and collect detailed action sequences"""
    
    # Reset environment with specific red agent
    cyborg = CybORG(PATH, 'sim', agents={'Red': red_agent_class})
    env = ChallengeWrapper2(env=cyborg, agent_name="Blue")
    
    observation = env.reset()
    
    episode_data = {
        'blue_actions': [],
        'red_actions': [],
        'rewards': [],
        'true_states': [],
        'observations': [],
        'action_names': [],
        'red_action_names': []
    }
    
    total_reward = 0
    
    for step in range(num_steps):
        # Get blue action
        blue_action = agent.get_action(observation)
        
        # Step environment
        next_obs, reward, done, info = env.step(blue_action)
        
        # Get action names
        blue_action_name = str(cyborg.get_last_action('Blue'))
        red_action_name = str(cyborg.get_last_action('Red'))
        
        # Get true state for analysis
        true_state = cyborg.get_agent_state('True')
        true_table = true_obs_to_table(true_state, cyborg)
        
        # Store data
        episode_data['blue_actions'].append(blue_action)
        episode_data['red_actions'].append(red_action_name)
        episode_data['action_names'].append(blue_action_name)
        episode_data['red_action_names'].append(red_action_name)
        episode_data['rewards'].append(reward)
        episode_data['observations'].append(observation.tolist())
        episode_data['true_states'].append(str(true_table))
        
        total_reward += reward
        observation = next_obs
        
        if verbose and step % 10 == 0:
            print(f"Step {step}: Blue: {blue_action_name[:30]}, Red: {red_action_name[:30]}, Reward: {reward:.2f}")
        
        if done:
            break
    
    episode_data['total_reward'] = total_reward
    agent.end_episode()
    
    return episode_data

def extract_action_patterns(episodes_data):
    """Extract common action patterns from multiple episodes"""
    
    patterns = {
        'blue_action_frequency': defaultdict(int),
        'red_action_frequency': defaultdict(int),
        'blue_sequences': [],
        'response_patterns': defaultdict(list),
        'milestone_sequences': []
    }
    
    for episode in episodes_data:
        # Count action frequencies
        for action in episode['action_names']:
            patterns['blue_action_frequency'][action] += 1
        for action in episode['red_action_names']:
            patterns['red_action_frequency'][action] += 1
            
        # Extract action sequences (first 20 steps)
        blue_seq = episode['action_names'][:20]
        patterns['blue_sequences'].append(blue_seq)
        
        # Extract response patterns (Blue action after specific Red action)
        for i in range(len(episode['red_action_names']) - 1):
            red_action = episode['red_action_names'][i]
            blue_response = episode['action_names'][i+1] if i+1 < len(episode['action_names']) else None
            if blue_response:
                patterns['response_patterns'][red_action].append(blue_response)
        
        # Extract milestone-like sequences (grouped actions)
        milestones = extract_milestones(episode['action_names'])
        patterns['milestone_sequences'].append(milestones)
    
    return patterns

def extract_milestones(action_sequence):
    """Group actions into milestone-like clusters"""
    milestones = []
    current_milestone = []
    current_type = None
    
    for action in action_sequence:
        # Determine action type
        if 'Decoy' in action:
            action_type = 'FORTIFY'
        elif 'Analyse' in action:
            action_type = 'SCAN'
        elif 'Remove' in action:
            action_type = 'CLEAN'
        elif 'Restore' in action:
            action_type = 'RESTORE'
        elif 'Monitor' in action or 'Sleep' in action:
            action_type = 'WAIT'
        else:
            action_type = 'OTHER'
        
        # Check if this is a new milestone
        if action_type != current_type and current_type is not None:
            if current_milestone:
                milestones.append({
                    'type': current_type,
                    'actions': current_milestone,
                    'count': len(current_milestone)
                })
            current_milestone = [action]
            current_type = action_type
        else:
            current_milestone.append(action)
            current_type = action_type
    
    # Add last milestone
    if current_milestone:
        milestones.append({
            'type': current_type,
            'actions': current_milestone,
            'count': len(current_milestone)
        })
    
    return milestones

def main():
    # Define action space (from train.py)
    action_space = [133, 134, 135, 139]  # restore enterprise and opserver
    action_space += [3, 4, 5, 9]  # analyse enterprise and opserver
    action_space += [16, 17, 18, 22]  # remove enterprise and opserer
    action_space += [11, 12, 13, 14]  # analyse user hosts
    action_space += [141, 142, 143, 144]  # restore user hosts
    action_space += [132]  # restore defender
    action_space += [2]  # analyse defender
    action_space += [15, 24, 25, 26, 27]  # remove defender and user hosts
    
    # Analyze both models
    models = {
        'bline': '/home/ubuntu/CAGE2/-cyborg-cage-2/Models/bline/model.pth',
        'meander': '/home/ubuntu/CAGE2/-cyborg-cage-2/Models/meander/model.pth'
    }
    
    red_agents = {
        'B_line': B_lineAgent,
        'Meander': RedMeanderAgent,
        'Sleep': SleepAgent
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name} model")
        print(f"{'='*60}")
        
        agent = create_agent(model_path, action_space)
        model_results = {}
        
        for red_name, red_class in red_agents.items():
            print(f"\n--- Against {red_name} agent ---")
            
            episodes_data = []
            
            # Run multiple episodes
            for episode_num in range(5):  # Run 5 episodes for each combination
                print(f"\nEpisode {episode_num + 1}:")
                
                # Create fresh environment for each episode
                cyborg = CybORG(PATH, 'sim', agents={'Red': red_class})
                env = ChallengeWrapper2(env=cyborg, agent_name="Blue")
                
                episode_data = analyze_episode(env, agent, red_class, num_steps=50, verbose=(episode_num == 0))
                episodes_data.append(episode_data)
                
                print(f"Total reward: {episode_data['total_reward']:.2f}")
            
            # Extract patterns
            patterns = extract_action_patterns(episodes_data)
            
            # Summary statistics
            avg_reward = np.mean([ep['total_reward'] for ep in episodes_data])
            std_reward = np.std([ep['total_reward'] for ep in episodes_data])
            
            model_results[red_name] = {
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'patterns': patterns,
                'episodes': episodes_data
            }
            
            print(f"\nSummary for {model_name} vs {red_name}:")
            print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
            
            # Print most common blue actions
            print("\nTop 5 Blue actions:")
            sorted_actions = sorted(patterns['blue_action_frequency'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            for action, count in sorted_actions:
                print(f"  {action[:40]}: {count}")
            
            # Print typical milestone sequence
            if patterns['milestone_sequences']:
                print("\nTypical milestone sequence (first episode):")
                for milestone in patterns['milestone_sequences'][0][:10]:
                    print(f"  {milestone['type']}: {milestone['count']} actions")
        
        results[model_name] = model_results
    
    # Save results
    output_file = '/home/ubuntu/CAGE2/-cyborg-cage-2/optimal_policy_analysis.json'
    with open(output_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for model_name, model_data in results.items():
            json_results[model_name] = {}
            for red_name, red_data in model_data.items():
                json_results[model_name][red_name] = {
                    'avg_reward': float(red_data['avg_reward']),
                    'std_reward': float(red_data['std_reward']),
                    'top_actions': dict(sorted(red_data['patterns']['blue_action_frequency'].items(), 
                                              key=lambda x: x[1], reverse=True)[:10])
                }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    main()
