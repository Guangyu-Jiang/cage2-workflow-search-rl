#!/usr/bin/env python3
"""
Sequential Training for Workflow-Conditioned RL using single environment
Collects 100 episodes sequentially before each PPO update
"""

import os
import sys
# Add parent directory to path for imports
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import numpy as np
import torch
import json
import csv
import time
import argparse
from typing import List, Tuple, Dict
from datetime import datetime
from pathlib import Path

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

from workflow_rl.order_based_workflow import OrderBasedWorkflow
from workflow_rl.gp_ucb_order_search import GPUCBOrderSearch
from workflow_rl.sequential_order_conditioned_ppo_simple import SimpleSequentialPPO


class SequentialWorkflowTrainer:
    """Single environment trainer that collects episodes sequentially"""
    
    def __init__(self,
                 scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                 total_episode_budget: int = 100000,
                 episodes_per_update: int = 100,  # Collect 100 episodes before update
                 max_episodes_per_workflow: int = 5000,  # Max episodes per workflow
                 alignment_lambda: float = 30.0,
                 compliance_threshold: float = 0.95,
                 red_agent_type=B_lineAgent,
                 max_steps: int = 100,
                 checkpoint_dir: str = None):
        """
        Initialize sequential trainer with single environment
        
        Args:
            episodes_per_update: Number of episodes to collect before PPO update
            max_episodes_per_workflow: Maximum episodes to train per workflow
        """
        self.scenario_path = Path(scenario_path)
        self.total_episode_budget = total_episode_budget
        self.episodes_per_update = episodes_per_update
        self.max_episodes_per_workflow = max_episodes_per_workflow
        self.alignment_lambda = alignment_lambda
        self.compliance_threshold = compliance_threshold
        self.red_agent_type = red_agent_type
        self.max_steps = max_steps
        
        # Track total episodes used
        self.total_episodes_used = 0
        
        # Create checkpoint directory
        if checkpoint_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pid = os.getpid()
            self.experiment_name = f"exp_{timestamp}_{pid}_sequential"
            self.checkpoint_dir = Path('logs') / self.experiment_name
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.experiment_name = self.checkpoint_dir.name
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Experiment directory: {self.checkpoint_dir}")
        
        # Initialize workflow components
        self.workflow_manager = OrderBasedWorkflow()
        all_workflows = self.workflow_manager.get_canonical_workflows()
        
        # Initialize GP search
        self.gp_search = GPUCBOrderSearch(beta=2.0)
        
        # Get environment dimensions
        self._setup_dimensions()
        
        # Logging setup
        self._init_consolidated_logging()
        self._init_gp_sampling_log()
        self._save_experiment_config()
        
        print(f"\n🎯 Sequential Training Configuration:")
        print(f"  - Single environment (sequential collection)")
        print(f"  - Episodes per update: {self.episodes_per_update}")
        print(f"  - Total episode budget: {self.total_episode_budget}")
        print(f"  - Max episodes per workflow: {self.max_episodes_per_workflow}")
        print(f"  - Alignment lambda: {self.alignment_lambda}")
        print(f"  - Compliance threshold: {self.compliance_threshold:.0%}")
    
    def _setup_dimensions(self):
        """Get environment dimensions"""
        cyborg = CybORG(str(self.scenario_path), 'sim',
                       agents={'Red': self.red_agent_type})
        env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
        
        obs = env.reset()
        self.obs_dim = obs.shape[0]
        # ChallengeWrapper2.get_action_space returns an int (the number of actions)
        self.action_dim = env.get_action_space('Blue')
        
        print(f"Environment: obs_dim={self.obs_dim}, action_dim={self.action_dim}")
    
    def _init_consolidated_logging(self):
        """Initialize CSV logging"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = self.checkpoint_dir / f"training_log_{timestamp}.csv"
        
        self.consolidated_log_file = open(log_filename, 'w', newline='')
        self.consolidated_csv_writer = csv.writer(self.consolidated_log_file)
        
        # Write header
        self.consolidated_csv_writer.writerow([
            'Workflow_ID', 'Workflow_Order', 'Type', 'Episode', 'Total_Episodes',
            'Env_Reward', 'Total_Reward', 'Alignment_Bonus',
            'Compliance', 'Fixes', 'Steps', 'Success', 'Eval_Reward'
        ])
        self.consolidated_log_file.flush()
        
        print(f"📊 Training log: {log_filename}")
    
    def _init_gp_sampling_log(self):
        """Initialize GP-UCB sampling log"""
        gp_log_filename = self.checkpoint_dir / "gp_sampling_log.csv"
        
        self.gp_sampling_file = open(gp_log_filename, 'w', newline='')
        self.gp_sampling_writer = csv.writer(self.gp_sampling_file)
        
        # Write header
        self.gp_sampling_writer.writerow([
            'Iteration', 'Selected_Workflow', 'UCB_Score',
            'Top1_Workflow', 'Top1_UCB', 'Top1_Mean', 'Top1_Std',
            'Top2_Workflow', 'Top2_UCB', 'Top2_Mean', 'Top2_Std',
            'Top3_Workflow', 'Top3_UCB', 'Top3_Mean', 'Top3_Std',
            'Selection_Method', 'Exploitation_Value', 'Exploration_Bonus'
        ])
        self.gp_sampling_file.flush()
        
        print(f"📈 GP-UCB sampling log: {gp_log_filename}")
    
    def _save_experiment_config(self):
        """Save experiment configuration"""
        config = {
            'experiment_name': self.experiment_name,
            'pid': os.getpid(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mode': 'sequential',
            'environment': {
                'n_envs': 1,
                'episodes_per_update': self.episodes_per_update,
                'max_steps': self.max_steps,
                'red_agent_type': self.red_agent_type.__name__,
                'scenario': str(self.scenario_path)
            },
            'training': {
                'total_episode_budget': self.total_episode_budget,
                'max_episodes_per_workflow': self.max_episodes_per_workflow,
                'compliance_threshold': self.compliance_threshold
            },
            'rewards': {
                'alignment_lambda': self.alignment_lambda
            },
            'search': {
                'gp_beta': self.gp_search.beta if hasattr(self.gp_search, 'beta') else 2.0
            }
        }
        
        config_file = self.checkpoint_dir / 'experiment_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"⚙️ Experiment config: {config_file}")
    
    def train_workflow(self, workflow_id: int, workflow_str: str,
                      workflow_order: List[str], inherited_agent=None) -> tuple:
        """Train a specific workflow using sequential episode collection"""
        print(f"\n{'='*60}")
        print(f"🎯 Training Workflow {workflow_id}: {workflow_str}")
        print(f"  Episodes used so far: {self.total_episodes_used}/{self.total_episode_budget}")
        print(f"{'='*60}")
        
        # Check budget
        if self.total_episodes_used >= self.total_episode_budget:
            print(f"❌ Episode budget exhausted ({self.total_episodes_used}/{self.total_episode_budget})")
            return None, 0
        
        # Create single environment
        print(f"🔧 Creating single environment...")
        cyborg = CybORG(str(self.scenario_path), 'sim',
                       agents={'Red': self.red_agent_type})
        env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
        
        # Initialize or inherit agent
        if inherited_agent is not None:
            print("♻️ Inheriting from previous workflow's agent")
            agent = inherited_agent
            agent.set_workflow(workflow_order)
        else:
            print("🆕 Creating new PPO agent")
            agent = SimpleSequentialPPO(
                input_dims=self.obs_dim,
                lr=0.002,
                gamma=0.99,
                K_epochs=4,
                eps_clip=0.2,
                workflow_order=workflow_order,
                workflow_manager=self.workflow_manager,
                alignment_lambda=self.alignment_lambda,
                episodes_per_update=self.episodes_per_update
            )
        
        # Training metrics
        all_episode_rewards = []
        all_episode_total_rewards = []
        all_episode_compliances = []
        all_episode_fixes = []
        
        # Timing metrics
        sampling_times = []
        update_times = []
        update_count = 0
        
        # Calculate max episodes for this workflow
        remaining_budget = self.total_episode_budget - self.total_episodes_used
        max_episodes = min(self.max_episodes_per_workflow, remaining_budget)
        
        if max_episodes <= 0:
            print(f"❌ Not enough budget for this workflow")
            return None, 0
        
        print(f"📊 Training for up to {max_episodes} episodes")
        print(f"   Collecting {self.episodes_per_update} episodes per update")
        
        # Training loop
        start_time = time.time()
        episodes_completed = 0
        compliance_achieved = False
        
        while episodes_completed < max_episodes and not compliance_achieved:
            # Sampling phase - collect episodes_per_update episodes
            sampling_start = time.time()
            batch_episode_rewards = []
            batch_episode_total_rewards = []
            batch_episode_compliances = []
            batch_episode_fixes = []
            
            print(f"\n📦 Collecting batch of {self.episodes_per_update} episodes...")
            
            for ep in range(self.episodes_per_update):
                if episodes_completed >= max_episodes:
                    break
                
                # Reset environment
                obs = env.reset()
                episode_reward = 0
                episode_total_reward = 0
                episode_steps = 0
                
                # Episode loop
                done = False
                while not done and episode_steps < self.max_steps:
                    # Get action from agent
                    action = agent.get_action(obs)
                    
                    # Step environment
                    next_obs, reward, done, info = env.step(action)
                    
                    # Get true state for alignment reward
                    true_state = env.get_agent_state('Blue')
                    
                    # Calculate alignment reward
                    alignment_bonus = agent.compute_alignment_reward(action, true_state)
                    total_reward = reward + alignment_bonus
                    
                    # Store reward and terminal flag in memory
                    # (states, actions, log_probs are stored in act method)
                    agent.memory.rewards.append(total_reward)
                    agent.memory.is_terminals.append(done)
                    
                    # Update counters
                    episode_reward += reward
                    episode_total_reward += total_reward
                    episode_steps += 1
                    
                    obs = next_obs
                
                # Calculate episode metrics
                compliance = agent.get_compliance_rate()
                fixes = agent.total_fix_actions  # Get total fixes
                
                # Store episode data
                batch_episode_rewards.append(episode_reward)
                batch_episode_total_rewards.append(episode_total_reward)
                batch_episode_compliances.append(compliance)
                batch_episode_fixes.append(fixes)
                
                episodes_completed += 1
                self.total_episodes_used += 1
                
                # Log episode
                self.consolidated_csv_writer.writerow([
                    workflow_id, workflow_str, 'episode',
                    episodes_completed, self.total_episodes_used,
                    f"{episode_reward:.2f}",
                    f"{episode_total_reward:.2f}",
                    f"{total_reward - episode_reward:.2f}",  # alignment bonus
                    f"{compliance:.4f}",
                    fixes, episode_steps, '', ''
                ])
                
                # Progress indicator
                if (ep + 1) % 20 == 0:
                    print(f"  Collected {ep + 1}/{self.episodes_per_update} episodes")
            
            sampling_time = time.time() - sampling_start
            sampling_times.append(sampling_time)
            
            # PPO Update phase - update after collecting episodes_per_update episodes
            if len(batch_episode_rewards) >= self.episodes_per_update:
                print(f"\n🔄 Performing PPO update {update_count + 1}...")
                update_start = time.time()
                agent.update()
                update_time = time.time() - update_start
                update_times.append(update_time)
                update_count += 1
                
                # Calculate batch statistics
                avg_reward = np.mean(batch_episode_rewards)
                avg_total_reward = np.mean(batch_episode_total_rewards)
                avg_compliance = np.mean(batch_episode_compliances)
                avg_fixes = np.mean(batch_episode_fixes)
                
                # Store for overall tracking
                all_episode_rewards.extend(batch_episode_rewards)
                all_episode_total_rewards.extend(batch_episode_total_rewards)
                all_episode_compliances.extend(batch_episode_compliances)
                all_episode_fixes.extend(batch_episode_fixes)
                
                # Calculate timing stats
                avg_sampling_time = np.mean(sampling_times)
                avg_update_time = np.mean(update_times)
                update_ratio = avg_update_time / (avg_update_time + avg_sampling_time) * 100
                
                # Progress report
                elapsed = time.time() - start_time
                eps_per_sec = episodes_completed / elapsed if elapsed > 0 else 0
                
                print(f"\n⚡ Update {update_count}:")
                print(f"  Episodes: {episodes_completed}/{max_episodes} ({eps_per_sec:.1f} eps/sec)")
                print(f"  Batch Performance:")
                print(f"    Env Reward: {avg_reward:.2f}")
                print(f"    Total Reward: {avg_total_reward:.2f}")
                print(f"    Compliance: {avg_compliance:.4f}")
                print(f"    Fixes/Episode: {avg_fixes:.1f}")
                print(f"  ⏱️ Timing:")
                print(f"    Sampling: {sampling_time:.2f}s ({self.episodes_per_update / sampling_time:.1f} eps/sec)")
                print(f"    PPO Update: {update_time:.2f}s")
                print(f"    Update takes {update_ratio:.1f}% of total time")
                
                # Log summary
                self.consolidated_csv_writer.writerow([
                    workflow_id, workflow_str, 'summary',
                    '', episodes_completed,
                    f"{avg_reward:.2f}",
                    f"{avg_total_reward:.2f}",
                    f"{avg_total_reward - avg_reward:.2f}",
                    f"{avg_compliance:.4f}",
                    f"{avg_fixes:.2f}",
                    '', '', ''
                ])
                self.consolidated_log_file.flush()
                
                # Check for compliance threshold
                if avg_compliance >= self.compliance_threshold:
                    print(f"\n🎉 Compliance threshold {self.compliance_threshold:.0%} achieved!")
                    compliance_achieved = True
        
        # Calculate final eval reward (average of last 100 episodes or all if less)
        eval_episodes = min(100, len(all_episode_rewards))
        eval_reward = np.mean(all_episode_rewards[-eval_episodes:]) if all_episode_rewards else -1000
        
        # Log workflow complete
        self.consolidated_csv_writer.writerow([
            workflow_id, workflow_str, 'workflow_complete',
            '', episodes_completed,
            '', '', '',
            f"{np.mean(all_episode_compliances[-eval_episodes:]):.4f}" if all_episode_compliances else '0',
            '', '', compliance_achieved, f"{eval_reward:.2f}"
        ])
        self.consolidated_log_file.flush()
        
        # Final timing report
        total_time = time.time() - start_time
        if update_times:
            total_sampling = sum(sampling_times)
            total_updating = sum(update_times)
            print(f"\n📊 Workflow {workflow_id} Training Complete:")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Episodes completed: {episodes_completed}")
            print(f"  Updates performed: {len(update_times)}")
            print(f"  Time breakdown:")
            print(f"    Sampling: {total_sampling:.1f}s ({total_sampling/total_time*100:.1f}%)")
            print(f"    PPO Updates: {total_updating:.1f}s ({total_updating/total_time*100:.1f}%)")
            print(f"    Other: {total_time - total_sampling - total_updating:.1f}s")
            print(f"  Performance: {episodes_completed/total_time:.1f} episodes/sec")
        
        return agent, eval_reward
    
    def run_workflow_search(self):
        """Run GP-UCB guided workflow search"""
        print("\n" + "="*60)
        print("🚀 STARTING SEQUENTIAL WORKFLOW SEARCH")
        print("="*60)
        
        total_start_time = time.time()
        current_agent = None
        workflow_count = 0
        
        while self.total_episodes_used < self.total_episode_budget:
            workflow_count += 1
            
            # Get candidate orders - convert dict values to list
            canonical_workflows = self.workflow_manager.get_canonical_workflows()
            candidate_orders = list(canonical_workflows.values())
            
            # Sample next workflow
            workflow_order, ucb_score, info = self.gp_search.select_next_order(
                candidate_orders, self.workflow_manager
            )
            # Use order index as workflow ID
            workflow_id = self.workflow_manager.get_order_index(workflow_order)
            workflow_str = ' → '.join(workflow_order)
            
            print(f"\n{'='*60}")
            print(f"🎲 Workflow {workflow_count}: Selected ID {workflow_id}")
            print(f"   Order: {workflow_str}")
            print(f"   UCB Score: {ucb_score:.4f}")
            print(f"   Episodes remaining: {self.total_episode_budget - self.total_episodes_used}")
            
            # Log GP sampling decision
            log_entry = [workflow_count, workflow_id, f"{ucb_score:.4f}"]
            
            # Add placeholder data for top-3 (simplified for sequential)
            for i in range(3):
                log_entry.extend([workflow_id, f"{ucb_score:.4f}", "0.0", "0.0"])
            
            # Determine selection method
            selection_method = info.get('method', 'Random')
            exploitation = info.get('exploitation', 0.0)
            exploration = info.get('exploration', ucb_score)
            
            log_entry.extend([selection_method, f"{exploitation:.4f}", f"{exploration:.4f}"])
            self.gp_sampling_writer.writerow(log_entry)
            self.gp_sampling_file.flush()
            
            # Train workflow
            current_agent, eval_reward = self.train_workflow(
                workflow_id, workflow_str, workflow_order, current_agent
            )
            
            if current_agent is None:
                print("❌ Training failed or budget exhausted")
                break
            
            # Update GP model
            self.gp_search.add_observation(workflow_order, eval_reward)
            
            print(f"\n📈 GP model updated with reward: {eval_reward:.2f}")
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"workflow_{workflow_id}_agent.pt"
            torch.save(current_agent.policy.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Final summary
        total_time = time.time() - total_start_time
        print("\n" + "="*60)
        print("✅ SEQUENTIAL WORKFLOW SEARCH COMPLETE")
        print(f"   Total workflows trained: {workflow_count}")
        print(f"   Total episodes used: {self.total_episodes_used}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average speed: {self.total_episodes_used/total_time:.1f} episodes/sec")
        print("="*60)
        
        # Close log files
        self.consolidated_log_file.close()
        self.gp_sampling_file.close()
        
        return current_agent


def main():
    parser = argparse.ArgumentParser(description='Sequential Workflow Search Training')
    
    # Environment settings
    parser.add_argument('--scenario', type=str, 
                       default='/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                       help='Scenario configuration file')
    parser.add_argument('--red-agent', type=str, default='B_lineAgent',
                       choices=['B_lineAgent', 'RedMeanderAgent', 'SleepAgent'],
                       help='Type of red agent')
    
    # Training settings
    parser.add_argument('--total-episodes', type=int, default=100000,
                       help='Total episode budget across all workflows')
    parser.add_argument('--episodes-per-update', type=int, default=100,
                       help='Episodes to collect before each PPO update')
    parser.add_argument('--max-episodes', type=int, default=5000,
                       help='Max episodes per workflow')
    
    # Reward settings
    parser.add_argument('--alignment-lambda', type=float, default=30.0,
                       help='Weight for alignment bonus')
    parser.add_argument('--compliance-threshold', type=float, default=0.95,
                       help='Compliance rate for early stopping')
    
    # Checkpoint settings
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory for saving checkpoints')
    
    args = parser.parse_args()
    
    # Map red agent string to class
    red_agent_map = {
        'B_lineAgent': B_lineAgent,
        'RedMeanderAgent': RedMeanderAgent,
        'SleepAgent': SleepAgent
    }
    red_agent_class = red_agent_map[args.red_agent]
    
    # Create trainer
    trainer = SequentialWorkflowTrainer(
        scenario_path=args.scenario,
        total_episode_budget=args.total_episodes,
        episodes_per_update=args.episodes_per_update,
        max_episodes_per_workflow=args.max_episodes,
        alignment_lambda=args.alignment_lambda,
        compliance_threshold=args.compliance_threshold,
        red_agent_type=red_agent_class,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Run training
    trainer.run_workflow_search()


if __name__ == "__main__":
    main()
