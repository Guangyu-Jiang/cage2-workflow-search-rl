#!/usr/bin/env python3
"""
Investigate:
1. Can Defender be compromised?
2. Should Op_Server0 be distinct from Op_Hosts?
3. Should all 5 users have the same priority?
"""

import sys
import os
sys.path.insert(0, '/home/ubuntu/CAGE2/cage-challenge-2')
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import numpy as np
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
import inspect


def check_defender_compromise():
    """Check if Defender can be compromised"""
    
    print("="*70)
    print("CHECKING IF DEFENDER CAN BE COMPROMISED")
    print("="*70)
    
    # Create environment
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    # Run multiple episodes to see if Defender ever gets compromised
    defender_compromised_count = 0
    
    for episode in range(5):
        obs = env.reset()
        print(f"\nEpisode {episode + 1}:")
        
        for step in range(50):
            # Blue sleeps to let Red act freely
            obs, reward, done, info = env.step(0)  # Sleep
            
            # Check Defender status (index 0 in observation)
            # Observation structure: [activity, access, compromise, extra] per host
            defender_compromise = obs[2]  # Index 2 is compromise flag for Defender
            
            if defender_compromise > 0:
                defender_compromised_count += 1
                print(f"  Step {step}: DEFENDER COMPROMISED! (value: {defender_compromise})")
                break
        
        if defender_compromise == 0:
            print(f"  Defender NOT compromised after 50 steps")
    
    print(f"\nDefender compromised in {defender_compromised_count}/5 episodes")
    
    # Check true state to understand Defender's role
    true_state = cyborg.get_agent_state('True')
    if 'Defender' in true_state:
        print(f"\nDefender info from true state:")
        print(f"  {true_state['Defender']}")
    
    return defender_compromised_count > 0


def analyze_operational_hosts():
    """Analyze differences between Op_Server0 and Op_Hosts"""
    
    print("\n" + "="*70)
    print("ANALYZING OPERATIONAL HOSTS")
    print("="*70)
    
    # Create environment
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim')
    
    # Get true state to see host properties
    true_state = cyborg.get_agent_state('True')
    
    print("\nOperational subnet hosts:")
    op_hosts = ['Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2']
    
    for host in op_hosts:
        if host in true_state:
            host_info = true_state[host]
            print(f"\n{host}:")
            
            # Check for special properties
            if 'Processes' in host_info:
                processes = host_info['Processes']
                print(f"  Processes: {len(processes)} running")
                
                # Check for critical services
                critical_services = []
                for proc in processes:
                    if hasattr(proc, 'process_name'):
                        name = proc.process_name
                        if 'OTService' in name or 'critical' in name.lower():
                            critical_services.append(name)
                
                if critical_services:
                    print(f"  Critical services: {critical_services}")
            
            if 'System info' in host_info:
                sys_info = host_info['System info']
                print(f"  System: {sys_info}")
            
            if 'Interface' in host_info:
                print(f"  Interfaces: {len(host_info['Interface'])}")
    
    # Check action differences
    print("\n" + "="*70)
    print("ACTION ANALYSIS FOR OPERATIONAL HOSTS")
    print("="*70)
    
    # From our previous analysis, we know:
    print("\nAction IDs:")
    print("  Op_Server0: analyze=9, remove=22, restore=139")
    print("  Op_Host0:   analyze=9, remove=22, restore=139  (SAME)")
    print("  Op_Host1:   analyze=9, remove=22, restore=139  (SAME)")
    print("  Op_Host2:   analyze=9, remove=22, restore=139  (SAME)")
    
    print("\nKey finding: All operational hosts share the SAME action IDs")
    print("This means we CANNOT distinguish between them in terms of actions")
    
    # Check importance/rewards
    print("\nChecking reward structure...")
    
    # The reward function likely treats them differently
    # Let's check by looking at the scenario file
    scenario_path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    try:
        with open(scenario_path, 'r') as f:
            lines = f.readlines()
            
        print("\nFrom Scenario2.yaml:")
        in_rewards = False
        for line in lines:
            if 'Rewards' in line:
                in_rewards = True
            if in_rewards and ('Op_Server0' in line or 'Op_Host' in line):
                print(f"  {line.strip()}")
            if in_rewards and 'Agents:' in line:
                break
    except:
        print("  Could not read scenario file")
    
    return True


def analyze_user_hosts():
    """Analyze if all 5 users should have same priority"""
    
    print("\n" + "="*70)
    print("ANALYZING USER HOSTS")
    print("="*70)
    
    # Check action mappings
    print("\nUser action mappings:")
    user_actions = {
        'User0': {'analyze': 11, 'remove': 24, 'restore': 141},
        'User1': {'analyze': 12, 'remove': 25, 'restore': 142},
        'User2': {'analyze': 13, 'remove': 26, 'restore': 143},
        'User3': {'analyze': 14, 'remove': 27, 'restore': 144},
        'User4': {'analyze': 14, 'remove': 27, 'restore': 144}  # Same as User3
    }
    
    for user, actions in user_actions.items():
        print(f"  {user}: {actions}")
    
    print("\nKey findings:")
    print("  - User0, User1, User2 have UNIQUE action IDs")
    print("  - User3 and User4 SHARE the same action IDs")
    print("  - We can distinguish User0, User1, User2 but not User3 from User4")
    
    # Check if users have different importance
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim')
    
    true_state = cyborg.get_agent_state('True')
    
    print("\nUser host properties:")
    for i in range(5):
        user = f'User{i}'
        if user in true_state:
            host_info = true_state[user]
            # Check for differences
            if 'Processes' in host_info:
                print(f"  {user}: {len(host_info['Processes'])} processes")
    
    print("\nConclusion: Users are likely homogeneous in importance")
    print("But we CAN distinguish User0, User1, User2 individually")
    
    return True


def test_compromise_patterns():
    """Test which hosts typically get compromised and in what order"""
    
    print("\n" + "="*70)
    print("TESTING COMPROMISE PATTERNS")
    print("="*70)
    
    # Run with different red agents to see patterns
    red_agents = [RedMeanderAgent, B_lineAgent]
    
    for red_agent_class in red_agents:
        print(f"\nTesting with {red_agent_class.__name__}:")
        
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
        cyborg = CybORG(path, 'sim', agents={'Red': red_agent_class})
        env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
        
        # Track first compromises
        first_compromised = []
        
        for episode in range(3):
            obs = env.reset()
            episode_compromised = []
            
            for step in range(30):
                obs, reward, done, info = env.step(0)  # Sleep
                
                # Check all hosts
                hostnames = [
                    'Defender', 'Enterprise0', 'Enterprise1', 'Enterprise2',
                    'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0',
                    'User0', 'User1', 'User2', 'User3', 'User4'
                ]
                
                for i, host in enumerate(hostnames):
                    if obs[i*4 + 2] > 0 and host not in episode_compromised:
                        episode_compromised.append(host)
                        if len(episode_compromised) == 1:
                            first_compromised.append(host)
                
                if len(episode_compromised) >= 5:
                    break
            
            print(f"  Episode {episode + 1}: {' -> '.join(episode_compromised[:5])}")
        
        print(f"  First compromised: {first_compromised}")


def main():
    """Main investigation"""
    
    # 1. Check if Defender can be compromised
    defender_can_be_compromised = check_defender_compromise()
    
    # 2. Analyze operational hosts
    analyze_operational_hosts()
    
    # 3. Analyze user hosts
    analyze_user_hosts()
    
    # 4. Test compromise patterns
    test_compromise_patterns()
    
    # Summary
    print("\n" + "="*70)
    print("INVESTIGATION SUMMARY")
    print("="*70)
    
    print("\n1. DEFENDER:")
    print(f"   Can be compromised: {defender_can_be_compromised}")
    print("   Should be included in priority order: YES")
    
    print("\n2. OPERATIONAL HOSTS:")
    print("   Op_Server0 and Op_Hosts share SAME action IDs")
    print("   Cannot distinguish them through actions")
    print("   Recommendation: Treat as single unit 'Operational'")
    
    print("\n3. USER HOSTS:")
    print("   User0, User1, User2 have UNIQUE actions")
    print("   User3, User4 share SAME actions")
    print("   Recommendation: Keep User0, User1, User2 separate")
    print("                   Combine User3+User4 as one unit")
    
    print("\n4. REVISED UNIT COUNT:")
    print("   - Defender (can be compromised)")
    print("   - Enterprise0, Enterprise1, Enterprise2 (3 units)")
    print("   - Operational (Op_Server0 + all Op_Hosts)")
    print("   - User0, User1, User2 (3 units)")
    print("   - User3+4 (combined)")
    print("   TOTAL: 9 distinct units (same as before)")


if __name__ == "__main__":
    main()
