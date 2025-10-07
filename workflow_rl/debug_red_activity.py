"""
Debug script to check Red agent activity over an episode
"""

import numpy as np
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

# Create environment
scenario_path = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'
cyborg = CybORG(scenario_path, 'sim', agents={'Red': B_lineAgent})
env = ChallengeWrapper2(env=cyborg, agent_name='Blue')

# Reset and run a full episode
obs = env.reset()
print("Running 100-step episode to observe Red agent activity...")
print("="*60)

compromised_hosts = set()
step_first_compromise = {}

for step in range(100):
    # Take a random action (or no-op)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    # Check true state
    true_state = cyborg.get_agent_state('True')
    
    for host_name, host_info in true_state.items():
        if host_name == 'success':
            continue
        
        # Check if compromised
        sys_comp = host_info.get('System info', {}).get('Compromised', False)
        int_comp = False
        if host_info.get('Interface'):
            int_comp = host_info['Interface'][0].get('Compromised', False) if host_info['Interface'] else False
        
        if (sys_comp or int_comp) and host_name not in compromised_hosts:
            compromised_hosts.add(host_name)
            step_first_compromise[host_name] = step + 1
            print(f"Step {step+1}: {host_name} COMPROMISED")
    
    # Print progress every 20 steps
    if (step + 1) % 20 == 0:
        print(f"Step {step+1}: Total compromised hosts: {len(compromised_hosts)}")

print("\n" + "="*60)
print("Episode Summary:")
print(f"Total hosts compromised: {len(compromised_hosts)}")
print("\nCompromise timeline:")
for host, step in sorted(step_first_compromise.items(), key=lambda x: x[1]):
    print(f"  Step {step}: {host}")

if len(compromised_hosts) == 0:
    print("\nNo hosts were compromised! Possible issues:")
    print("1. Red agent (B_lineAgent) may not be aggressive enough")
    print("2. Blue's random actions might be preventing compromises")
    print("3. The scenario might start with no initial compromises")
