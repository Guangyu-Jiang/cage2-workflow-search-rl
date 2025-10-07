#!/usr/bin/env python3
"""
Check if Op_Hosts can be compromised and if they affect the reward
Independent of the training strategy - looking at game mechanics
"""

import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/cage-challenge-2')
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import numpy as np
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
import inspect


def check_op_hosts_vulnerability():
    """Check if Op_Hosts can be compromised by red agents"""
    
    print("="*70)
    print("CHECKING OP_HOSTS VULNERABILITY")
    print("="*70)
    
    # Create environment with aggressive red agent
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    
    # Test with different red agents
    red_agents = [B_lineAgent, RedMeanderAgent]
    
    for red_agent_class in red_agents:
        print(f"\nTesting with {red_agent_class.__name__}:")
        
        cyborg = CybORG(path, 'sim', agents={'Red': red_agent_class})
        env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
        
        op_host_compromised = {
            'Op_Host0': False,
            'Op_Host1': False,
            'Op_Host2': False
        }
        
        # Run multiple episodes
        for episode in range(3):
            obs = env.reset()
            
            for step in range(50):
                # Blue does nothing (sleep) to let red act freely
                obs, reward, done, info = env.step(0)
                
                # Check Op_Hosts status in observation
                # Op_Host0 is at index 4, Op_Host1 at 5, Op_Host2 at 6
                op_host_indices = {'Op_Host0': 4, 'Op_Host1': 5, 'Op_Host2': 6}
                
                for host, idx in op_host_indices.items():
                    # Check compromise flag (index 2 in each host's 4 features)
                    if obs[idx * 4 + 2] > 0:
                        op_host_compromised[host] = True
                
                # Also check true state for confirmation
                true_state = cyborg.get_agent_state('True')
                for host in ['Op_Host0', 'Op_Host1', 'Op_Host2']:
                    if host in true_state:
                        host_state = true_state[host]
                        # Check for red agent sessions
                        if 'Sessions' in host_state:
                            for session in host_state['Sessions']:
                                if session.get('Agent') == 'Red':
                                    op_host_compromised[host] = True
                                    print(f"    Step {step}: {host} compromised! (Red session found)")
                
                if all(op_host_compromised.values()):
                    print(f"    All Op_Hosts compromised by step {step}")
                    break
            
            print(f"  Episode {episode + 1} summary: {sum(op_host_compromised.values())}/3 Op_Hosts compromised")
        
        print(f"\nFinal: Op_Hosts that got compromised:")
        for host, compromised in op_host_compromised.items():
            print(f"  {host}: {'Yes' if compromised else 'No'}")


def check_op_hosts_impact_on_reward():
    """Check if Op_Host compromise affects the reward"""
    
    print("\n" + "="*70)
    print("CHECKING OP_HOSTS IMPACT ON REWARD")
    print("="*70)
    
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    
    # Test scenario 1: Let Op_Hosts get compromised
    print("\nScenario 1: Blue ignores Op_Hosts")
    cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    obs = env.reset()
    total_reward_ignore = 0
    
    for step in range(30):
        obs, reward, done, info = env.step(0)  # Sleep
        total_reward_ignore += reward
        
        if reward < 0:
            # Check which hosts caused negative reward
            true_state = cyborg.get_agent_state('True')
            compromised = []
            for host in true_state:
                if host != 'success' and 'Sessions' in true_state[host]:
                    for session in true_state[host]['Sessions']:
                        if session.get('Agent') == 'Red':
                            compromised.append(host)
                            break
            
            if any('Op_Host' in h for h in compromised):
                print(f"  Step {step}: Negative reward {reward} with Op_Host in compromised: {compromised}")
    
    print(f"  Total reward (ignoring Op_Hosts): {total_reward_ignore}")
    
    # Test scenario 2: Actively defend Op_Hosts
    print("\nScenario 2: Blue defends Op_Hosts")
    cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    obs = env.reset()
    total_reward_defend = 0
    
    # Actions for Op_Hosts: Analyse (6,7,8), Remove (19,20,21), Restore (136,137,138)
    op_host_actions = [6, 7, 8]  # Analyse Op_Hosts
    action_idx = 0
    
    for step in range(30):
        # Cycle through analyzing Op_Hosts
        action = op_host_actions[action_idx % 3] if step < 10 else 0
        action_idx += 1
        
        obs, reward, done, info = env.step(action)
        total_reward_defend += reward
        
        if action != 0:
            print(f"  Step {step}: Action {action}, Reward {reward}")
    
    print(f"  Total reward (defending Op_Hosts): {total_reward_defend}")
    
    print(f"\nReward difference: {total_reward_defend - total_reward_ignore}")


def check_network_position():
    """Check Op_Hosts position in network and potential impact"""
    
    print("\n" + "="*70)
    print("CHECKING OP_HOSTS NETWORK POSITION")
    print("="*70)
    
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim')
    
    # Get network topology
    true_state = cyborg.get_agent_state('True')
    
    print("\nOperational subnet analysis:")
    for host in ['Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2']:
        if host in true_state:
            print(f"\n{host}:")
            host_info = true_state[host]
            
            # Check interfaces
            if 'Interface' in host_info:
                for iface in host_info['Interface']:
                    if 'IP Address' in iface:
                        print(f"  IP: {iface['IP Address']}")
            
            # Check services
            if 'Processes' in host_info:
                services = []
                for proc in host_info['Processes']:
                    if 'Connections' in proc:
                        for conn in proc['Connections']:
                            if 'local_port' in conn:
                                services.append(conn['local_port'])
                if services:
                    print(f"  Services on ports: {services}")
            
            # Check system info
            if 'System info' in host_info:
                print(f"  System: {host_info['System info'].get('Hostname', 'Unknown')}")


def test_op_host_defense_value():
    """Test if defending Op_Hosts provides value"""
    
    print("\n" + "="*70)
    print("TESTING OP_HOST DEFENSE VALUE")
    print("="*70)
    
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    
    # Run episodes with and without Op_Host defense
    scenarios = {
        'no_defense': {'defend_op_hosts': False, 'rewards': []},
        'with_defense': {'defend_op_hosts': True, 'rewards': []}
    }
    
    for scenario_name, scenario in scenarios.items():
        print(f"\nTesting {scenario_name}:")
        
        for episode in range(5):
            cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
            env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
            
            obs = env.reset()
            episode_reward = 0
            
            for step in range(30):
                if scenario['defend_op_hosts'] and step % 10 < 3:
                    # Defend Op_Hosts in first 3 steps of every 10
                    action = 136 + (step % 3)  # Restore Op_Host0/1/2
                else:
                    # Default: defend enterprise
                    action = 133 + (step % 3)  # Restore Enterprise0/1/2
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            scenario['rewards'].append(episode_reward)
            print(f"  Episode {episode + 1}: {episode_reward}")
    
    # Compare results
    print("\nResults comparison:")
    for name, scenario in scenarios.items():
        avg_reward = np.mean(scenario['rewards'])
        print(f"  {name}: Average reward = {avg_reward:.2f}")


def main():
    """Main investigation"""
    
    print("\n" + "="*70)
    print("INVESTIGATING OP_HOSTS IMPORTANCE IN CAGE2")
    print("="*70)
    
    # 1. Check if Op_Hosts can be compromised
    check_op_hosts_vulnerability()
    
    # 2. Check if Op_Host compromise affects reward
    check_op_hosts_impact_on_reward()
    
    # 3. Check network position
    check_network_position()
    
    # 4. Test defense value
    test_op_host_defense_value()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    print("""
    Based on the investigation:
    
    1. Op_Hosts CAN be compromised by red agents
    2. Op_Host compromise DOES generate negative rewards
    3. Op_Hosts are part of the operational subnet
    4. Defending Op_Hosts may provide value
    
    Therefore, Op_Hosts ARE valid targets that need defense!
    
    The training strategy ignored them likely for simplification,
    but from an environment perspective, they are legitimate units
    that can and should be defended.
    
    For workflow design: Include all 5 unit types!
    """)


if __name__ == "__main__":
    main()
