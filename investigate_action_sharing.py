#!/usr/bin/env python3
"""
Investigate why certain hosts share action IDs in CAGE2
- Why do Op_Server0 and Op_Hosts share actions?
- Why do User3 and User4 share actions?
"""

import sys
import os
sys.path.insert(0, '/home/ubuntu/CAGE2/cage-challenge-2')
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import numpy as np
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Wrappers.EnumActionWrapper import EnumActionWrapper
from Wrappers.BlueTableWrapper import BlueTableWrapper
import inspect


def analyze_enum_wrapper():
    """Analyze how EnumActionWrapper maps actions"""
    
    print("="*70)
    print("ANALYZING ENUM ACTION WRAPPER")
    print("="*70)
    
    # Create environment
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim')
    
    # Apply wrappers step by step
    blue_table = BlueTableWrapper(env=cyborg, agent='Blue')
    enum_wrapper = EnumActionWrapper(env=blue_table)
    
    print("\nTotal possible actions:", len(enum_wrapper.possible_actions))
    
    # Group actions by type and hostname
    action_groups = {}
    
    for i, action in enumerate(enum_wrapper.possible_actions):
        action_str = str(action)
        
        # Parse action
        if 'Sleep' in action_str:
            continue
        
        # Extract action type and hostname
        parts = action_str.split()
        if len(parts) >= 2:
            action_type = parts[0]
            hostname = ' '.join(parts[1:]) if len(parts) > 1 else ''
            
            # Clean hostname
            hostname = hostname.replace('hostname=', '').replace("'", "").replace(',', '').strip()
            
            if hostname:
                key = (action_type, hostname)
                if key not in action_groups:
                    action_groups[key] = []
                action_groups[key].append(i)
    
    # Find duplicates - hosts that appear with same action type
    print("\nAnalyzing action assignments:")
    
    for action_type in ['Analyse', 'Remove', 'Restore']:
        print(f"\n{action_type} actions:")
        type_actions = [(host, indices) for (act, host), indices in action_groups.items() if act == action_type]
        type_actions.sort(key=lambda x: x[0])
        
        for host, indices in type_actions:
            print(f"  {host:15s}: Action index {indices[0] if indices else 'None'}")
    
    return enum_wrapper


def check_network_topology():
    """Check if network topology explains the sharing"""
    
    print("\n" + "="*70)
    print("CHECKING NETWORK TOPOLOGY")
    print("="*70)
    
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim')
    
    # Get true state to see network structure
    true_state = cyborg.get_agent_state('True')
    
    print("\nSubnet assignments:")
    
    # Operational subnet
    print("\nOperational Subnet:")
    for host in ['Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2']:
        if host in true_state:
            interfaces = true_state[host].get('Interface', [])
            for iface in interfaces:
                if 'IP Address' in iface and 'Subnet' in iface:
                    print(f"  {host}: IP={iface['IP Address']}, Subnet={iface['Subnet']}")
                    break
    
    # User subnet
    print("\nUser Subnet:")
    for i in range(5):
        host = f'User{i}'
        if host in true_state:
            interfaces = true_state[host].get('Interface', [])
            for iface in interfaces:
                if 'IP Address' in iface and 'Subnet' in iface:
                    print(f"  {host}: IP={iface['IP Address']}, Subnet={iface['Subnet']}")
                    break
    
    print("\nHypothesis: Hosts in same subnet might share actions?")
    print("But that doesn't explain why User0, User1, User2 are different...")


def check_wrapper_code():
    """Check the wrapper source code for clues"""
    
    print("\n" + "="*70)
    print("CHECKING WRAPPER SOURCE CODE")
    print("="*70)
    
    # Read EnumActionWrapper
    wrapper_path = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Agents/Wrappers/EnumActionWrapper.py'
    
    print("\nChecking EnumActionWrapper.py for action generation logic...")
    
    with open(wrapper_path, 'r') as f:
        lines = f.readlines()
    
    # Look for where actions are created
    in_init = False
    relevant_lines = []
    
    for i, line in enumerate(lines):
        if 'def __init__' in line:
            in_init = True
        elif in_init and 'def ' in line:
            in_init = False
        
        if in_init and ('possible_actions' in line or 'append' in line or 'for' in line):
            relevant_lines.append((i+1, line.rstrip()))
    
    if relevant_lines:
        print("\nRelevant lines from __init__:")
        for line_no, line in relevant_lines[:20]:
            print(f"  Line {line_no}: {line}")


def check_action_space_definition():
    """Check how action space is defined in the scenario"""
    
    print("\n" + "="*70)
    print("CHECKING ACTION SPACE DEFINITION")
    print("="*70)
    
    # Check if there's a specific configuration limiting actions
    path = str(inspect.getfile(CybORG))
    scenario_path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    
    print(f"\nChecking {scenario_path} for action definitions...")
    
    with open(scenario_path, 'r') as f:
        lines = f.readlines()
    
    # Look for action-related configuration
    for i, line in enumerate(lines):
        if 'Actions:' in line or 'action' in line.lower():
            print(f"Line {i+1}: {line.rstrip()}")
            # Print context
            for j in range(max(0, i-2), min(len(lines), i+3)):
                if j != i:
                    print(f"       {lines[j].rstrip()}")


def test_actual_actions():
    """Test if we can actually distinguish these hosts in practice"""
    
    print("\n" + "="*70)
    print("TESTING ACTUAL ACTION EXECUTION")
    print("="*70)
    
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim')
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    print("\nTesting if actions actually affect different hosts:")
    
    # Get the EnumActionWrapper
    enum_wrapper = env.env.env  # Navigate through wrappers
    
    # Test Op_Server0 vs Op_Hosts
    print("\n1. Testing Operational hosts:")
    print("   Action 9 (Analyse for Op_Server0/Op_Hosts):")
    
    obs = env.reset()
    
    # Try action 9 (should be Analyse for operational hosts)
    obs, reward, done, info = env.step(9)
    
    # Check which host was actually analyzed
    # This would require checking the internal state
    true_state = cyborg.get_agent_state('Blue')
    if 'last_action' in true_state:
        print(f"   Last action: {true_state['last_action']}")
    
    print("\n2. Testing User3 vs User4:")
    print("   Action 14 (Analyse for User3/User4):")
    
    obs = env.reset()
    obs, reward, done, info = env.step(14)
    
    true_state = cyborg.get_agent_state('Blue')
    if 'last_action' in true_state:
        print(f"   Last action: {true_state['last_action']}")


def analyze_blue_action_space():
    """Analyze the Blue agent's action space from the agent itself"""
    
    print("\n" + "="*70)
    print("ANALYZING BLUE AGENT ACTION SPACE")
    print("="*70)
    
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim')
    
    # Get blue agent's action space
    blue_action_space = cyborg.get_action_space('Blue')
    
    print(f"\nBlue agent action space: {type(blue_action_space)}")
    
    if isinstance(blue_action_space, dict):
        print(f"Number of action types: {len(blue_action_space)}")
        
        for action_type, params in blue_action_space.items():
            if 'Analyse' in str(action_type) or 'Remove' in str(action_type) or 'Restore' in str(action_type):
                print(f"\n{action_type}:")
                if isinstance(params, dict):
                    for key, value in params.items():
                        if 'hostname' in str(key).lower():
                            print(f"  {key}: {value}")


def main():
    """Main investigation"""
    
    print("\n" + "="*70)
    print("INVESTIGATING WHY HOSTS SHARE ACTION IDS")
    print("="*70)
    
    # 1. Analyze enum wrapper
    enum_wrapper = analyze_enum_wrapper()
    
    # 2. Check network topology
    check_network_topology()
    
    # 3. Check wrapper code
    check_wrapper_code()
    
    # 4. Check action space definition
    check_action_space_definition()
    
    # 5. Test actual actions
    test_actual_actions()
    
    # 6. Analyze blue action space
    analyze_blue_action_space()
    
    print("\n" + "="*70)
    print("HYPOTHESIS SUMMARY")
    print("="*70)
    
    print("""
    LIKELY EXPLANATION:
    
    1. The action space is defined at the SUBNET level for some actions
       - Operational subnet: All hosts share actions (by design)
       - User subnet: Mixed (some individual, some shared)
    
    2. This might be intentional to:
       - Simplify the action space (reduce from 13*3 to fewer actions)
       - Reflect that Blue doesn't always know exactly which host to target
       - Model realistic defensive scenarios where you protect subnets
    
    3. User3 and User4 sharing actions might be:
       - A simplification (5 users is a lot)
       - They might be on the same physical machine
       - Or connected through same gateway
    
    4. Op_Server0 sharing with Op_Hosts is strange because:
       - Servers are usually more critical than hosts
       - But in this scenario, they're treated as one unit
       - This forces Blue to protect the entire operational subnet together
    """)


if __name__ == "__main__":
    main()
