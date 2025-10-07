#!/usr/bin/env python3
"""
Final confirmation of unit types in CAGE2
Checking which units are actually actionable vs which exist
"""

import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/cage-challenge-2')
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import inspect
from CybORG import CybORG
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

# Create environment
path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
cyborg = CybORG(path, 'sim')
env = ChallengeWrapper2(env=cyborg, agent_name='Blue')

# Get the enum wrapper to see all possible actions
enum_wrapper = env.env.env

# Categorize all actions by unit type
unit_types = {
    'defender': [],
    'enterprise': [],
    'op_server': [],
    'op_host': [],
    'user': []
}

# Analyze each action
for i, action in enumerate(enum_wrapper.possible_actions):
    action_str = str(action)
    
    # Skip non-host actions
    if any(x in action_str for x in ['Sleep', 'Monitor', 'Decoy']):
        continue
        
    # Only look at Analyse, Remove, Restore
    if not any(x in action_str for x in ['Analyse', 'Remove', 'Restore']):
        continue
    
    # Categorize by hostname
    if 'Defender' in action_str:
        unit_types['defender'].append((i, action_str))
    elif 'Enterprise' in action_str:
        unit_types['enterprise'].append((i, action_str))
    elif 'Op_Server' in action_str:
        unit_types['op_server'].append((i, action_str))
    elif 'Op_Host' in action_str:
        unit_types['op_host'].append((i, action_str))
    elif 'User' in action_str:
        unit_types['user'].append((i, action_str))

print("="*70)
print("UNIT TYPES IN CAGE2 - FINAL CONFIRMATION")
print("="*70)

# Display results
for unit_type, actions in unit_types.items():
    print(f"\n{unit_type.upper()} TYPE:")
    if actions:
        # Extract unique hosts
        hosts = set()
        for _, action_str in actions:
            if 'Analyse' in action_str:
                host = action_str.replace('Analyse ', '')
                hosts.add(host)
        
        print(f"  Units: {sorted(hosts)}")
        print(f"  Count: {len(hosts)} units")
        
        # Show action indices for first host
        if hosts:
            first_host = sorted(hosts)[0]
            print(f"  Example actions for {first_host}:")
            for idx, act in actions:
                if first_host in act:
                    print(f"    {idx}: {act}")
    else:
        print("  No actionable units")

# Check what the training script actually uses
print("\n" + "="*70)
print("WHAT THE TRAINING SCRIPT ACTUALLY DEFENDS")
print("="*70)

# From README.md and train.py
training_units = {
    'defender': ['Defender'],
    'enterprise': ['Enterprise0', 'Enterprise1', 'Enterprise2'],
    'op_server': ['Op_Server0'],
    'op_host': [],  # Not defended in training
    'user': ['User1', 'User2', 'User3', 'User4']  # User0 excluded
}

print("\nUnits defended in training:")
for unit_type, units in training_units.items():
    if units:
        print(f"  {unit_type.upper()}: {units}")
    else:
        print(f"  {unit_type.upper()}: [Not defended]")

# Final summary
print("\n" + "="*70)
print("FINAL UNIT TYPE COUNT")
print("="*70)

defended_types = [t for t, u in training_units.items() if u]
print(f"\nUnit types that are actually defended: {len(defended_types)}")
for t in defended_types:
    print(f"  - {t}")

all_types = [t for t, u in unit_types.items() if u]
print(f"\nUnit types that exist in environment: {len(all_types)}")
for t in all_types:
    print(f"  - {t}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

print("""
Based on the training approach in README.md:

DEFENDED UNIT TYPES (4 types):
1. DEFENDER (1 unit: Defender)
2. ENTERPRISE (3 units: Enterprise0, Enterprise1, Enterprise2)
3. OP_SERVER (1 unit: Op_Server0)
4. USER (4 units: User1, User2, User3, User4)

NOT DEFENDED:
- OP_HOST type (Op_Host0, Op_Host1, Op_Host2) - ignored in training
- User0 - explicitly excluded ("cant do actions on user0")

So for workflow design based on unit TYPES:
- If including all types: 5! = 120 possible type orderings
- If only defended types: 4! = 24 possible type orderings
""")
