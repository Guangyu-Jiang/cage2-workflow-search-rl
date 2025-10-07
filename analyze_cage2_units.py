#!/usr/bin/env python3
"""
Analyze CAGE2 environment to identify all units (hosts/servers) that can be:
1. Compromised by red agent
2. Fixed by blue agent (via analyze/remove/restore actions)
"""

import sys
import os
sys.path.insert(0, '/home/ubuntu/CAGE2/cage-challenge-2')
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import numpy as np
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
import inspect


def analyze_environment_structure():
    """Analyze the CAGE2 environment structure"""
    
    print("="*70)
    print("CAGE2 ENVIRONMENT STRUCTURE ANALYSIS")
    print("="*70)
    
    # Create environment
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim')
    
    # Wrap for challenge
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    # Get initial observation
    obs = env.reset()
    
    print(f"\n1. OBSERVATION SPACE:")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation size: {obs.shape[0]} dimensions")
    
    # Analyze observation structure
    # In BlueTableWrapper, observation is structured as:
    # Each host has 4 values: [activity, access_level, compromise_flag, extra]
    num_features_per_host = 4
    num_hosts = obs.shape[0] // num_features_per_host
    print(f"   Number of hosts tracked: {num_hosts}")
    
    print(f"\n2. ACTION SPACE:")
    print(f"   Action space size: {env.action_space.n}")
    
    # Get the actual action meanings
    # From the wrapper chain, we need to understand the action mapping
    inner_env = env.env  # Get to OpenAIGymWrapper
    if hasattr(inner_env, 'env'):
        inner_env = inner_env.env  # Get to EnumActionWrapper
        if hasattr(inner_env, 'possible_actions'):
            print(f"   Number of possible actions: {len(inner_env.possible_actions)}")
            
            # Categorize actions
            action_types = {
                'Sleep': [],
                'Monitor': [],
                'Analyse': [],
                'Remove': [],
                'Restore': [],
                'Decoy': [],
                'Other': []
            }
            
            for i, action in enumerate(inner_env.possible_actions):
                action_str = str(action)
                if 'Sleep' in action_str:
                    action_types['Sleep'].append(i)
                elif 'Monitor' in action_str:
                    action_types['Monitor'].append(i)
                elif 'Analyse' in action_str:
                    action_types['Analyse'].append(i)
                elif 'Remove' in action_str:
                    action_types['Remove'].append(i)
                elif 'Restore' in action_str:
                    action_types['Restore'].append(i)
                elif 'Decoy' in action_str:
                    action_types['Decoy'].append(i)
                else:
                    action_types['Other'].append(i)
            
            print("\n   Action breakdown:")
            for action_type, indices in action_types.items():
                if indices:
                    print(f"   - {action_type}: {len(indices)} actions (indices {indices[:5]}...)")
    
    return env


def identify_hosts_from_actions():
    """Identify hosts based on action space analysis"""
    
    print("\n" + "="*70)
    print("HOST IDENTIFICATION FROM ACTION SPACE")
    print("="*70)
    
    # Create environment
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim')
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    # Get to the EnumActionWrapper to see actual actions
    inner_env = env.env.env  # Navigate through wrapper chain
    
    if hasattr(inner_env, 'possible_actions'):
        # Extract unique hosts from actions
        hosts = set()
        host_actions = {}
        
        for i, action in enumerate(inner_env.possible_actions):
            action_str = str(action)
            
            # Parse hostname from action string
            # Actions typically look like: "Analyse User0", "Remove Enterprise1", etc.
            parts = action_str.split()
            if len(parts) >= 2:
                action_type = parts[0]
                if len(parts) == 2:
                    hostname = parts[1]
                elif len(parts) == 3 and parts[1] in ['User', 'Enterprise']:
                    hostname = f"{parts[1]} {parts[2]}"
                else:
                    hostname = ' '.join(parts[1:])
                
                # Clean up hostname
                hostname = hostname.replace(':', '').strip()
                
                if hostname and hostname not in ['', 'None']:
                    hosts.add(hostname)
                    
                    if hostname not in host_actions:
                        host_actions[hostname] = {'indices': [], 'types': set()}
                    
                    host_actions[hostname]['indices'].append(i)
                    host_actions[hostname]['types'].add(action_type)
        
        print(f"\nTotal unique hosts found: {len(hosts)}")
        print("\nHosts and their available actions:")
        
        # Sort hosts by type for better display
        sorted_hosts = sorted(hosts, key=lambda x: (
            0 if 'Defender' in x else
            1 if 'Op' in x and 'Server' in x else
            2 if 'Op' in x else
            3 if 'Enterprise' in x else
            4 if 'User' in x else
            5
        ))
        
        for host in sorted_hosts:
            if host in host_actions:
                actions = host_actions[host]
                print(f"\n  {host}:")
                print(f"    Action types: {', '.join(sorted(actions['types']))}")
                print(f"    Action indices: {actions['indices'][:5]}..." if len(actions['indices']) > 5 else f"    Action indices: {actions['indices']}")
    
    return hosts, host_actions


def analyze_true_state_structure():
    """Analyze the true state to understand all compromisable units"""
    
    print("\n" + "="*70)
    print("TRUE STATE STRUCTURE ANALYSIS")
    print("="*70)
    
    # Create environment
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim')
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    # Reset and get true state
    obs = env.reset()
    
    # Access true state through the cyborg object
    if hasattr(cyborg, 'get_agent_state'):
        true_state = cyborg.get_agent_state('True')
        
        print("\nTrue state structure:")
        if isinstance(true_state, dict):
            for key in true_state.keys():
                print(f"  - {key}")
                if key == 'hosts' and isinstance(true_state[key], dict):
                    print(f"    Number of hosts: {len(true_state[key])}")
                    print("    Host names:")
                    for hostname in sorted(true_state[key].keys()):
                        host_info = true_state[key][hostname]
                        # Check if host can be compromised
                        compromisable = "Yes" if not hostname.startswith('router') else "No"
                        print(f"      * {hostname}: Compromisable: {compromisable}")
    
    # Try alternative method - check observation wrapper
    print("\n" + "="*70)
    print("HOSTS FROM OBSERVATION WRAPPER")
    print("="*70)
    
    # The BlueTableWrapper tracks specific hosts
    # Based on the code analysis, these are the hosts:
    hostnames = [
        'Defender',
        'Enterprise0', 'Enterprise1', 'Enterprise2',
        'Op_Host0', 'Op_Host1', 'Op_Host2',
        'Op_Server0',
        'User0', 'User1', 'User2', 'User3', 'User4'
    ]
    
    print(f"\nHosts tracked in observations: {len(hostnames)}")
    for i, host in enumerate(hostnames):
        print(f"  {i:2d}. {host}")
    
    return hostnames


def test_compromise_and_fix():
    """Test which hosts can actually be compromised and fixed"""
    
    print("\n" + "="*70)
    print("TESTING COMPROMISE AND FIX CAPABILITIES")
    print("="*70)
    
    # Create environment with red agent
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    # Run for several steps to see what gets compromised
    obs = env.reset()
    
    print("\nRunning simulation to observe compromises...")
    
    compromised_hosts = set()
    fixed_hosts = set()
    
    for step in range(30):
        # Blue agent sleeps to let red act
        obs, reward, done, info = env.step(0)  # Sleep action
        
        # Check observation for compromises
        # Observation structure: each host has 4 features
        # Index 2 (every 4th starting from 2) indicates compromise
        for i in range(2, len(obs), 4):
            if obs[i] > 0:  # Compromise detected
                host_idx = i // 4
                compromised_hosts.add(host_idx)
        
        if step % 10 == 0:
            print(f"  Step {step}: {len(compromised_hosts)} hosts compromised")
    
    print(f"\nTotal hosts that got compromised: {len(compromised_hosts)}")
    print(f"Host indices compromised: {sorted(compromised_hosts)}")
    
    # Map indices to hostnames (based on BlueTableWrapper order)
    hostnames = [
        'Defender',
        'Enterprise0', 'Enterprise1', 'Enterprise2',
        'Op_Host0', 'Op_Host1', 'Op_Host2',
        'Op_Server0',
        'User0', 'User1', 'User2', 'User3', 'User4'
    ]
    
    print("\nCompromised hosts by name:")
    for idx in sorted(compromised_hosts):
        if idx < len(hostnames):
            print(f"  - {hostnames[idx]}")


def analyze_action_mapping():
    """Analyze the detailed action mapping for each host"""
    
    print("\n" + "="*70)
    print("DETAILED ACTION MAPPING ANALYSIS")
    print("="*70)
    
    # Based on the code analysis, here's the action mapping
    # From train.py and PPOAgent.py
    
    action_mapping = {
        'Defender': {
            'analyze': 2,
            'remove': 15,
            'restore': 132
        },
        'Enterprise0': {
            'analyze': 3,
            'remove': 16,
            'restore': 133,
            'decoy': 1000  # Virtual action ID
        },
        'Enterprise1': {
            'analyze': 4,
            'remove': 17,
            'restore': 134,
            'decoy': 1001
        },
        'Enterprise2': {
            'analyze': 5,
            'remove': 18,
            'restore': 135,
            'decoy': 1002
        },
        'Op_Server0': {
            'analyze': 9,
            'remove': 22,
            'restore': 139,
            'decoy': 1008
        },
        'User0': {
            'analyze': 11,
            'remove': 24,
            'restore': 141,
            'decoy': 1003
        },
        'User1': {
            'analyze': 12,
            'remove': 25,
            'restore': 142,
            'decoy': 1004
        },
        'User2': {
            'analyze': 13,
            'remove': 26,
            'restore': 143,
            'decoy': 1005
        },
        'User3': {
            'analyze': 14,
            'remove': 27,
            'restore': 144,
            'decoy': 1006
        },
        'User4': {
            'analyze': 14,  # Shares with User3
            'remove': 27,
            'restore': 144,
            'decoy': 1006
        },
        'Op_Host0': {
            'analyze': 9,  # Shares with Op_Server0
            'remove': 22,
            'restore': 139,
            'decoy': 1007
        },
        'Op_Host1': {
            'analyze': 9,
            'remove': 22,
            'restore': 139,
            'decoy': 1007
        },
        'Op_Host2': {
            'analyze': 9,
            'remove': 22,
            'restore': 139,
            'decoy': 1007
        }
    }
    
    print("\nHosts that can be individually fixed:")
    individual_hosts = []
    shared_action_groups = {}
    
    for host, actions in action_mapping.items():
        key = (actions.get('analyze'), actions.get('remove'), actions.get('restore'))
        if key not in shared_action_groups:
            shared_action_groups[key] = []
        shared_action_groups[key].append(host)
    
    print("\nAction groups (hosts that share same action IDs):")
    for i, (action_ids, hosts) in enumerate(shared_action_groups.items()):
        print(f"\n  Group {i+1}: {hosts}")
        print(f"    Analyze: {action_ids[0]}, Remove: {action_ids[1]}, Restore: {action_ids[2]}")
    
    print(f"\nTotal distinct units: {len(action_mapping)}")
    print(f"Total action groups: {len(shared_action_groups)}")
    
    return action_mapping, shared_action_groups


def main():
    """Main analysis function"""
    
    print("\n" + "="*70)
    print("COMPLETE CAGE2 UNIT ANALYSIS")
    print("="*70)
    
    # 1. Analyze environment structure
    env = analyze_environment_structure()
    
    # 2. Identify hosts from actions
    hosts, host_actions = identify_hosts_from_actions()
    
    # 3. Analyze true state structure
    hostnames = analyze_true_state_structure()
    
    # 4. Test compromise and fix
    test_compromise_and_fix()
    
    # 5. Analyze action mapping
    action_mapping, shared_groups = analyze_action_mapping()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: UNITS FOR PRIORITY ORDERING")
    print("="*70)
    
    print("\nUnits that can be compromised and need individual priority:")
    
    # Distinct action groups (units that can be distinguished)
    distinct_units = [
        'Defender',
        'Enterprise0', 'Enterprise1', 'Enterprise2',
        'Op_Server0',
        'Op_Hosts (Op_Host0/1/2 grouped)',  # These share actions
        'User0', 'User1', 'User2', 
        'User3+User4 (grouped)'  # These share actions
    ]
    
    print(f"\nDistinct fixable units: {len(distinct_units)}")
    for i, unit in enumerate(distinct_units):
        print(f"  {i+1:2d}. {unit}")
    
    print("\nNOTE: Some hosts share action IDs and cannot be distinguished:")
    print("  - Op_Host0, Op_Host1, Op_Host2 share same actions")
    print("  - User3 and User4 share same actions")
    
    print(f"\nFor workflow ordering, we have approximately 9-10 distinct priority levels")
    print("This gives us 9! = 362,880 possible orderings (sufficient for search)")


if __name__ == "__main__":
    main()
