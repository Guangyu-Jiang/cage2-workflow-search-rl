"""
Debug script to understand why compliance is always 100%
"""

import numpy as np
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

# Create environment
scenario_path = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'
cyborg = CybORG(scenario_path, 'sim', agents={'Red': B_lineAgent})
env = ChallengeWrapper2(env=cyborg, agent_name='Blue')

# Reset and get initial state
obs = env.reset()
true_state_initial = cyborg.get_agent_state('True')

print("Initial True State Keys:", true_state_initial.keys())
print("\nChecking initial compromise status:")

for host_name, host_info in true_state_initial.items():
    if host_name == 'success':
        continue
    
    compromised = host_info.get('System info', {}).get('Compromised', False)
    interface_comp = False
    if host_info.get('Interface'):
        interface_comp = host_info['Interface'][0].get('Compromised', False) if host_info['Interface'] else False
    
    if compromised or interface_comp:
        print(f"  {host_name}: COMPROMISED")
    else:
        print(f"  {host_name}: Clean")

# Take some actions and check for changes
print("\n" + "="*50)
print("Taking restore actions and checking for fixes...")
print("="*50)

# Try restore actions (132-144 are restore actions in CAGE2)
restore_actions = [132, 133, 134, 135, 139, 141, 142, 143, 144]

for i, action in enumerate(restore_actions[:5]):
    print(f"\nStep {i+1}: Taking action {action}")
    
    prev_state = cyborg.get_agent_state('True')
    obs, reward, done, info = env.step(action)
    curr_state = cyborg.get_agent_state('True')
    
    # Check for fixes
    fixed_hosts = []
    for host_name in curr_state:
        if host_name == 'success':
            continue
            
        if host_name in prev_state:
            # Previous compromise status
            prev_comp = prev_state[host_name].get('System info', {}).get('Compromised', False)
            prev_int = False
            if prev_state[host_name].get('Interface'):
                prev_int = prev_state[host_name]['Interface'][0].get('Compromised', False) if prev_state[host_name]['Interface'] else False
            was_compromised = prev_comp or prev_int
            
            # Current compromise status
            curr_comp = curr_state[host_name].get('System info', {}).get('Compromised', False)
            curr_int = False
            if curr_state[host_name].get('Interface'):
                curr_int = curr_state[host_name]['Interface'][0].get('Compromised', False) if curr_state[host_name]['Interface'] else False
            is_compromised = curr_comp or curr_int
            
            # Check if fixed
            if was_compromised and not is_compromised:
                fixed_hosts.append(host_name)
                print(f"  ✓ FIXED: {host_name}")
            elif is_compromised and not was_compromised:
                print(f"  ✗ COMPROMISED: {host_name}")
    
    if not fixed_hosts:
        print(f"  No fixes detected")
    
    print(f"  Reward: {reward}")

print("\n" + "="*50)
print("Summary:")
print("="*50)
print("If no fixes are detected, it means either:")
print("1. Hosts aren't getting compromised by Red")
print("2. Restore actions aren't working as expected")
print("3. The true state isn't showing compromise status correctly")
print("\nThis would explain why compliance is always 100% - ")
print("if no fix actions are detected, the default is 100% compliance.")
