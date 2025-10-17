#!/usr/bin/env python3
"""
Fast version of parallel workflow training with optimizations
Key changes:
- K_epochs kept at 4 for stability
- Vectorized environments instead of shared memory (2.2x faster)
- Batch logging to reduce I/O overhead
- Option to use GPU if available
"""

import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import os
import argparse
import json
import csv
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Agents.Wrappers import EnterpriseMAE

# Import Red Agents
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Simulator.Actions.ComplexActions.BLineActions import B_lineAgent
from agents.hierachy_agents.loadagents import BlueHierAgent

# Import our PPO and components
from workflow_rl.workflow_manager import WorkflowManager
from workflow_rl.gp_workflow_search import GPWorkflowSearch
from workflow_rl.parallel_order_conditioned_ppo import OrderConditionedPPO
from workflow_rl.parallel_env_vectorized import VectorizedCAGE2Envs
from workflow_rl.parallel_env_shared_memory_optimized import ParallelEnvSharedMemoryOptimized


class FastParallelWorkflowTrainer:
    def __init__(self, 
                 n_envs: int = 200,  # More envs for faster data collection
                 scenario_path: str = 'Scenario2.yaml',
                 n_workflows: int = 20,
                 total_episode_budget: int = 100000,
                 max_train_episodes_per_env: int = 50,  # Reduced from 100
                 alignment_lambda: float = 0.01,
                 compliance_threshold: float = 0.95,
                 red_agent_type=B_lineAgent,
                 max_steps: int = 100,
                 update_every_steps: int = 50,  # More frequent updates
                 checkpoint_dir: str = None,
                 use_gpu: bool = True):
        """
        Initialize fast parallel trainer
        """
        self.n_envs = n_envs
        self.scenario_path = Path(scenario_path)
        self.n_workflows = n_workflows
        self.total_episode_budget = total_episode_budget
        self.max_train_episodes_per_env = max_train_episodes_per_env
        self.alignment_lambda = alignment_lambda
        self.compliance_threshold = compliance_threshold
        self.red_agent_type = red_agent_type
        self.max_steps = max_steps
        self.update_every_steps = update_every_steps
        
        # Track total episodes used
        self.total_episodes_used = 0
        
        # Check GPU availability
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print(f"🚀 Using GPU acceleration: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Using CPU (consider enabling GPU for faster training)")
        
        # Create checkpoint directory
        if checkpoint_dir is None:
            # Create experiment directory with timestamp and PID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pid = os.getpid()
            self.experiment_name = f"exp_{timestamp}_{pid}_fast"
            self.checkpoint_dir = Path('logs') / self.experiment_name
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.experiment_name = self.checkpoint_dir.name
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Experiment directory: {self.checkpoint_dir}")
        
        # Initialize workflow components
        self.workflow_manager = WorkflowManager()
        all_workflows = self.workflow_manager.get_all_workflows()
        
        # Initialize GP search
        self.gp_search = GPWorkflowSearch(
            workflow_ids=list(range(len(all_workflows))),
            beta=2.0
        )
        
        # Get dimensions for environment setup
        self._setup_dimensions()
        
        # Logging setup - with batching
        self.log_buffer = []
        self.log_batch_size = 100  # Write every 100 entries
        self._init_consolidated_logging()
        self._init_gp_sampling_log()
        
        # Save experiment configuration
        self._save_experiment_config()
        
        print(f"\n🎯 Fast Training Configuration:")
        print(f"  - Parallel environments: {self.n_envs}")
        print(f"  - Total episode budget: {self.total_episode_budget}")
        print(f"  - Max episodes per workflow: {self.max_train_episodes_per_env}")
        print(f"  - Update frequency: every {self.update_every_steps} steps")
        print(f"  - PPO epochs: 4 (stable convergence)")
        print(f"  - Environment: Vectorized (2.2x faster)")
        print(f"  - Batch logging: {self.log_batch_size} entries")
        print(f"  - Device: {self.device}")
    
    def _setup_dimensions(self):
        """Setup environment dimensions"""
        env = CybORG(
            scenario_generator=EnterpriseScenarioGenerator(
                blue_agent_class=BlueHierAgent,
                green_agent_class=EnterpriseGreenAgent,
                red_agent_class=self.red_agent_type,
                steps=self.max_steps
            ),
            seed=42
        )
        env = EnterpriseMAE(env, pad_spaces=False)
        
        obs = env.reset()
        self.obs_dim = obs.shape[0]
        self.action_dim = env.action_space.n
        
        print(f"Environment: obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        env.close()
    
    def _init_consolidated_logging(self):
        """Initialize single CSV file for all training logs"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = self.checkpoint_dir / f"training_log_{timestamp}.csv"
        
        self.consolidated_log_file = open(log_filename, 'w', newline='')
        self.consolidated_csv_writer = csv.writer(self.consolidated_log_file)
        
        # Write header
        self.consolidated_csv_writer.writerow([
            'Workflow_ID', 'Workflow_Order', 'Type', 'Episode', 'Total_Episodes', 
            'Env_ID', 'Env_Reward', 'Total_Reward', 'Alignment_Bonus', 
            'Compliance', 'Fixes', 'Steps', 'Success', 'Eval_Reward'
        ])
        self.consolidated_log_file.flush()
        
        print(f"📊 Consolidated training log: {log_filename}")
    
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
    
    def _write_log_buffer(self):
        """Write buffered logs to file"""
        if self.log_buffer:
            self.consolidated_csv_writer.writerows(self.log_buffer)
            self.consolidated_log_file.flush()
            self.log_buffer = []
    
    def _add_log_entry(self, entry):
        """Add entry to log buffer and write if full"""
        self.log_buffer.append(entry)
        if len(self.log_buffer) >= self.log_batch_size:
            self._write_log_buffer()
    
    def _save_experiment_config(self):
        """Save experiment configuration"""
        config = {
            'experiment_name': self.experiment_name,
            'pid': os.getpid(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'optimizations': {
                'environment_type': 'VectorizedCAGE2Envs',
                'k_epochs': 4,
                'batch_logging': True,
                'log_batch_size': self.log_batch_size,
                'device': str(self.device),
                'update_frequency': self.update_every_steps
            },
            'environment': {
                'n_envs': self.n_envs,
                'max_steps': self.max_steps,
                'red_agent_type': self.red_agent_type.__name__,
                'scenario': str(self.scenario_path)
            },
            'training': {
                'total_episode_budget': self.total_episode_budget,
                'max_train_episodes_per_env': self.max_train_episodes_per_env,
                'compliance_threshold': self.compliance_threshold,
                'update_every_steps': self.update_every_steps
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
                      inherited_agent=None) -> tuple:
        """Train a specific workflow with fast settings"""
        print(f"\n{'='*60}")
        print(f"🎯 Training Workflow {workflow_id}: {workflow_str}")
        print(f"  Episodes used so far: {self.total_episodes_used}/{self.total_episode_budget}")
        print(f"{'='*60}")
        
        # Check budget
        if self.total_episodes_used >= self.total_episode_budget:
            print(f"❌ Episode budget exhausted ({self.total_episodes_used}/{self.total_episode_budget})")
            return None, 0
        
        # Create vectorized environments (2.2x faster than shared memory)
        print(f"🚀 Creating {self.n_envs} vectorized environments...")
        envs = VectorizedCAGE2Envs(
            n_envs=self.n_envs,
            scenario_path=self.scenario_path,
            red_agent_type=self.red_agent_type
        )
        
        # Initialize or inherit agent with optimized settings
        if inherited_agent is not None:
            print("♻️ Inheriting from previous workflow's agent")
            agent = inherited_agent
            agent.set_workflow(workflow_id)
        else:
            print("🆕 Creating new PPO agent with optimized settings")
            agent = OrderConditionedPPO(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                n_envs=self.n_envs,
                workflow_dim=len(self.workflow_manager.get_all_workflows()),
                alignment_lambda=self.alignment_lambda,
                update_steps=self.update_every_steps,
                K_epochs=4,  # Keep at 4 for better convergence
                eps_clip=0.2,
                gamma=0.99,
                lr=0.002,
                device=self.device  # Use GPU if available
            )
            agent.set_workflow(workflow_id)
        
        # Training metrics
        episode_rewards = [[] for _ in range(self.n_envs)]
        episode_total_rewards = [[] for _ in range(self.n_envs)]
        episode_alignment_bonuses = [[] for _ in range(self.n_envs)]
        episode_compliances = [[] for _ in range(self.n_envs)]
        episode_fix_counts = [[] for _ in range(self.n_envs)]
        episode_counts = np.zeros(self.n_envs)
        episode_dones = [True] * self.n_envs
        
        # Timing metrics
        update_times = []
        sampling_times = []
        update_count = 0
        
        # Training loop
        start_time = time.time()
        states = envs.reset()
        total_steps = 0
        env_rewards = np.zeros(self.n_envs)
        env_total_rewards = np.zeros(self.n_envs)
        env_alignment_bonuses = np.zeros(self.n_envs)
        compliance_achieved = False
        sampling_start = time.time()
        
        # Calculate episodes we can afford
        remaining_budget = self.total_episode_budget - self.total_episodes_used
        max_episodes_this_workflow = min(
            self.max_train_episodes_per_env,
            remaining_budget // self.n_envs
        )
        
        if max_episodes_this_workflow <= 0:
            print(f"❌ Not enough budget for this workflow")
            envs.close()
            return None, 0
        
        print(f"📊 Training for up to {max_episodes_this_workflow} episodes per env")
        print(f"⚡ Using optimized settings: K_epochs=4, vectorized envs, batch logging")
        
        while not compliance_achieved:
            # Check if any env exceeded episode limit
            if np.any(episode_counts >= max_episodes_this_workflow):
                print(f"\n✅ Reached episode limit for workflow {workflow_id}")
                break
            
            # Get actions
            actions, total_rewards = agent.get_actions_and_rewards(states)
            
            # Step environments (vectorized - much faster!)
            next_states, rewards, dones, infos = envs.step(actions)
            
            # Track rewards
            env_rewards += rewards
            env_total_rewards += total_rewards
            env_alignment_bonuses += (total_rewards - rewards)
            
            # Process episode endings
            for env_idx in range(self.n_envs):
                if dones[env_idx]:
                    if episode_counts[env_idx] < max_episodes_this_workflow:
                        episode_counts[env_idx] += 1
                        episode_dones[env_idx] = True
                        
                        # Calculate metrics
                        compliance = agent.get_compliance_rates()[env_idx]
                        fixes = len(agent.get_fixed_hosts())
                        
                        # Store episode data
                        episode_rewards[env_idx].append(env_rewards[env_idx])
                        episode_total_rewards[env_idx].append(env_total_rewards[env_idx])
                        episode_alignment_bonuses[env_idx].append(env_alignment_bonuses[env_idx])
                        episode_compliances[env_idx].append(compliance)
                        episode_fix_counts[env_idx].append(fixes)
                        
                        # Add to log buffer (batch writing)
                        self._add_log_entry([
                            workflow_id, workflow_str, 'episode',
                            int(episode_counts[env_idx]), '',
                            env_idx,
                            f"{env_rewards[env_idx]:.2f}",
                            f"{env_total_rewards[env_idx]:.2f}",
                            f"{env_alignment_bonuses[env_idx]:.2f}",
                            f"{compliance:.4f}",
                            fixes, self.max_steps, '', ''
                        ])
                        
                        # Reset metrics
                        env_rewards[env_idx] = 0
                        env_total_rewards[env_idx] = 0
                        env_alignment_bonuses[env_idx] = 0
                        agent.reset_episode_compliance(env_idx)
                else:
                    episode_dones[env_idx] = False
            
            # PPO update if needed
            if agent.should_update():
                # Measure sampling time
                sampling_time = time.time() - sampling_start
                sampling_times.append(sampling_time)
                
                # Measure update time
                update_start = time.time()
                agent.update()
                update_time = time.time() - update_start
                update_times.append(update_time)
                update_count += 1
                
                total_episodes = np.sum(episode_counts)
                
                # Progress report
                if len(episode_rewards[0]) > 0:
                    # Get last episode stats
                    last_env_rewards = [episode_rewards[i][-1] for i in range(self.n_envs) 
                                       if len(episode_rewards[i]) > 0]
                    last_compliances = [episode_compliances[i][-1] for i in range(self.n_envs)
                                      if len(episode_compliances[i]) > 0]
                    
                    if last_env_rewards:
                        avg_env_reward = np.mean(last_env_rewards)
                        avg_compliance = np.mean(last_compliances)
                        
                        # Check for early stopping (only on latest episode)
                        if avg_compliance >= self.compliance_threshold:
                            print(f"\n🎉 Compliance threshold {self.compliance_threshold:.2%} achieved!")
                            print(f"   Latest avg compliance: {avg_compliance:.4f}")
                            compliance_achieved = True
                        
                        # Progress update with timing
                        elapsed = time.time() - start_time
                        eps_per_sec = total_episodes / elapsed if elapsed > 0 else 0
                        
                        # Timing statistics
                        avg_update_time = np.mean(update_times) if update_times else 0
                        avg_sampling_time = np.mean(sampling_times) if sampling_times else 0
                        update_ratio = avg_update_time / (avg_update_time + avg_sampling_time) * 100 if (avg_update_time + avg_sampling_time) > 0 else 0
                        
                        print(f"\n⚡ Update {update_count}: "
                              f"{int(total_episodes)} episodes "
                              f"({eps_per_sec:.1f} eps/sec)")
                        print(f"   Env Reward: {avg_env_reward:.2f}, "
                              f"Compliance: {avg_compliance:.4f}")
                        print(f"   ⏱️ Timing: Sampling={sampling_time:.2f}s, Update={update_time:.2f}s")
                        print(f"   📊 Average: Sampling={avg_sampling_time:.2f}s, Update={avg_update_time:.2f}s (PPO takes {update_ratio:.1f}% of time)")
                
                # Reset sampling timer
                sampling_start = time.time()
            
            states = next_states
            total_steps += 1
        
        # Calculate final eval reward from last episode
        eval_rewards = []
        for env_idx in range(self.n_envs):
            if len(episode_rewards[env_idx]) > 0:
                eval_rewards.append(episode_rewards[env_idx][-1])
        
        eval_reward = np.mean(eval_rewards) if eval_rewards else -1000
        
        # Calculate episodes used
        episodes_used = int(np.sum(episode_counts))
        self.total_episodes_used += episodes_used
        
        # Write final summary
        self._add_log_entry([
            workflow_id, workflow_str, 'workflow_complete',
            '', episodes_used, '', '', '', '', 
            f"{np.mean([c[-1] for c in episode_compliances if c]):.4f}",
            '', '', compliance_achieved, f"{eval_reward:.2f}"
        ])
        
        # Flush any remaining logs
        self._write_log_buffer()
        
        # Close environments
        envs.close()
        
        # Report timing
        elapsed = time.time() - start_time
        print(f"\n⏱️ Workflow training completed in {elapsed:.1f}s")
        print(f"   Episodes: {episodes_used} ({episodes_used/elapsed:.1f} eps/sec)")
        print(f"   Final reward: {eval_reward:.2f}")
        
        if update_times:
            total_update_time = sum(update_times)
            total_sampling_time = sum(sampling_times)
            print(f"\n📊 Timing Breakdown:")
            print(f"   Total sampling time: {total_sampling_time:.1f}s ({total_sampling_time/elapsed*100:.1f}%)")
            print(f"   Total PPO update time: {total_update_time:.1f}s ({total_update_time/elapsed*100:.1f}%)")
            print(f"   Other (logging, etc): {elapsed - total_sampling_time - total_update_time:.1f}s")
            print(f"   Updates performed: {len(update_times)}")
            print(f"   Avg time per update: {np.mean(update_times):.2f}s")
        
        return agent, eval_reward
    
    def run_workflow_search(self):
        """Run GP-UCB guided workflow search with optimizations"""
        print("\n" + "="*60)
        print("🚀 STARTING FAST WORKFLOW SEARCH")
        print("="*60)
        
        total_start_time = time.time()
        current_agent = None
        workflow_count = 0
        
        while self.total_episodes_used < self.total_episode_budget:
            workflow_count += 1
            
            # Sample next workflow
            workflow_id, ucb_score, top_k = self.gp_search.select_workflow()
            workflow_str = self.workflow_manager.get_workflow_string(workflow_id)
            
            print(f"\n{'='*60}")
            print(f"🎲 Workflow {workflow_count}: Selected ID {workflow_id}")
            print(f"   Order: {workflow_str}")
            print(f"   UCB Score: {ucb_score:.4f}")
            print(f"   Episodes remaining: {self.total_episode_budget - self.total_episodes_used}")
            
            # Log GP sampling decision
            log_entry = [workflow_count, workflow_id, f"{ucb_score:.4f}"]
            for i, (wid, score, mean, std) in enumerate(top_k[:3]):
                log_entry.extend([wid, f"{score:.4f}", f"{mean:.4f}", f"{std:.4f}"])
            
            # Determine selection method
            if hasattr(self.gp_search, 'gp') and self.gp_search.gp is not None:
                mean, std = self.gp_search.gp.predict([[workflow_id]], return_std=True)
                exploitation = mean[0]
                exploration = self.gp_search.beta * std[0]
                selection_method = 'GP-UCB'
            else:
                exploitation = 0.0
                exploration = ucb_score
                selection_method = 'Random'
            
            log_entry.extend([selection_method, f"{exploitation:.4f}", f"{exploration:.4f}"])
            self.gp_sampling_writer.writerow(log_entry)
            self.gp_sampling_file.flush()
            
            # Train workflow
            current_agent, eval_reward = self.train_workflow(
                workflow_id, workflow_str, current_agent
            )
            
            if current_agent is None:
                print("❌ Training failed or budget exhausted")
                break
            
            # Update GP model
            self.gp_search.update(workflow_id, eval_reward)
            
            print(f"\n📈 GP model updated with reward: {eval_reward:.2f}")
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"workflow_{workflow_id}_agent.pt"
            torch.save(current_agent.policy.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Final summary
        total_time = time.time() - total_start_time
        print("\n" + "="*60)
        print("✅ WORKFLOW SEARCH COMPLETE")
        print(f"   Total workflows trained: {workflow_count}")
        print(f"   Total episodes used: {self.total_episodes_used}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average speed: {self.total_episodes_used/total_time:.1f} episodes/sec")
        print("="*60)
        
        # Ensure all logs are written
        self._write_log_buffer()
        
        # Close log files
        self.consolidated_log_file.close()
        self.gp_sampling_file.close()
        
        return current_agent


def main():
    parser = argparse.ArgumentParser(description='Fast Parallel Workflow Search Training')
    
    # Environment settings
    parser.add_argument('--n-envs', type=int, default=200,
                       help='Number of parallel environments')
    parser.add_argument('--scenario', type=str, default='Scenario2.yaml',
                       help='Scenario configuration file')
    parser.add_argument('--red-agent', type=str, default='B_lineAgent',
                       choices=['B_lineAgent', 'RedMeanderAgent', 'SleepAgent'],
                       help='Type of red agent')
    
    # Training settings
    parser.add_argument('--total-episodes', type=int, default=100000,
                       help='Total episode budget across all workflows')
    parser.add_argument('--max-episodes', type=int, default=50,
                       help='Max episodes per environment per workflow')
    parser.add_argument('--update-steps', type=int, default=50,
                       help='PPO update frequency (steps)')
    
    # Reward settings
    parser.add_argument('--alignment-lambda', type=float, default=0.01,
                       help='Weight for alignment bonus')
    parser.add_argument('--compliance-threshold', type=float, default=0.95,
                       help='Compliance rate for early stopping')
    
    # GPU settings
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU even if available')
    
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
    trainer = FastParallelWorkflowTrainer(
        n_envs=args.n_envs,
        scenario_path=args.scenario,
        total_episode_budget=args.total_episodes,
        max_train_episodes_per_env=args.max_episodes,
        alignment_lambda=args.alignment_lambda,
        compliance_threshold=args.compliance_threshold,
        red_agent_type=red_agent_class,
        update_every_steps=args.update_steps,
        checkpoint_dir=args.checkpoint_dir,
        use_gpu=not args.no_gpu
    )
    
    # Run training
    trainer.run_workflow_search()


if __name__ == "__main__":
    main()
