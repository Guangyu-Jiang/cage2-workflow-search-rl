"""
Parallel Training for Workflow-Conditioned RL using multiple environments
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

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

from workflow_rl.order_based_workflow import OrderBasedWorkflow
from workflow_rl.gp_ucb_order_search import GPUCBOrderSearch
from workflow_rl.parallel_order_conditioned_ppo import ParallelOrderConditionedPPO, device
from workflow_rl.parallel_env_wrapper import ParallelEnvWrapper
from workflow_rl.parallel_env_shared_memory import ParallelEnvSharedMemory
from workflow_rl.parallel_env_shared_memory_optimized import ParallelEnvSharedMemoryOptimized


class ParallelWorkflowRLTrainer:
    """Main trainer for workflow search-based RL with parallel environments"""
    
    def __init__(self, 
                 n_envs: int = 100,
                 total_episode_budget: int = 100000,  # Total episodes across all workflows
                 max_train_episodes_per_env: int = 100,  # Max episodes per workflow before giving up
                 max_steps: int = 100,
                 scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                 red_agent_type=RedMeanderAgent,
                 alignment_lambda: float = 30.0,  # Increased for stricter compliance
                 gp_beta: float = 2.0,
                 checkpoint_dir: str = 'checkpoints',
                 compliance_threshold: float = 0.95,  # Must achieve this before evaluation
                 n_eval_episodes: int = 20,  # Episodes for evaluation (not used anymore)
                 update_every_steps: int = 100):  # Update every 100 steps (full episode) = 10000 transitions with 100 envs
        """
        Initialize parallel trainer
        
        Training Strategy:
        1. Train PPO with alignment rewards until compliance >= 95%
        2. Use last episode rewards for GP-UCB (no separate evaluation)
        3. Only compliant workflows contribute to GP-UCB
        4. Continue exploring workflows until episode budget is exhausted
        
        Args:
            n_envs: Number of parallel environments
            total_episode_budget: Total episodes allowed across all workflows
            max_train_episodes_per_env: Max episodes per workflow before giving up
            compliance_threshold: Required compliance rate
        """
        
        self.n_envs = n_envs
        self.total_episode_budget = total_episode_budget
        self.max_train_episodes_per_env = max_train_episodes_per_env
        self.max_steps = max_steps
        self.scenario_path = scenario_path
        self.red_agent_type = red_agent_type
        self.alignment_lambda = alignment_lambda
        self.compliance_threshold = compliance_threshold
        self.n_eval_episodes = n_eval_episodes
        self.update_every_steps = update_every_steps
        
        # Track total episodes used across all workflows
        self.total_episodes_used = 0
        
        # Create experiment-specific directory with timestamp and PID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pid = os.getpid()
        self.experiment_name = f"exp_{timestamp}_{pid}"
        # Create logs directory at the same level as checkpoint_dir
        self.checkpoint_dir = os.path.join("logs", self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"Starting experiment with PID: {pid}")
        
        # Store base directory for reference (for backwards compatibility)
        self.base_checkpoint_dir = checkpoint_dir
        
        # Initialize workflow manager
        self.workflow_manager = OrderBasedWorkflow()
        
        # Initialize GP-UCB search
        self.gp_search = GPUCBOrderSearch(
            beta=gp_beta
        )
        
        # Training history
        self.training_history = []
        
        # Store the shared policy across workflows (for policy inheritance)
        self.shared_agent = None
        
        # Initialize consolidated CSV logging
        self.consolidated_log_file = None
        self.consolidated_csv_writer = None
        self._init_consolidated_logging()
        
        # Initialize GP-UCB sampling CSV logging
        self.gp_sampling_file = None
        self.gp_sampling_writer = None
        self._init_gp_sampling_log()
    
    def _init_consolidated_logging(self):
        """Initialize the consolidated CSV log file"""
        log_filename = os.path.join(self.checkpoint_dir, "training_log.csv")
        self.consolidated_log_file = open(log_filename, 'w', newline='')
        self.consolidated_csv_writer = csv.writer(self.consolidated_log_file)
        # Write header
        self.consolidated_csv_writer.writerow([
            'Workflow_ID', 'Workflow_Order', 'Type', 'Episode', 'Total_Episodes', 
            'Env_ID', 'Env_Reward', 'Total_Reward', 'Alignment_Bonus', 
            'Compliance', 'Fixes', 'Steps', 'Success', 'Eval_Reward'
        ])
        self.consolidated_log_file.flush()
        
        # Save experiment configuration
        self._save_experiment_config()
        
        print(f"Experiment directory: {self.checkpoint_dir}")
        print(f"Training log: {log_filename}")
    
    def _init_gp_sampling_log(self):
        """Initialize the GP-UCB sampling CSV log file"""
        gp_log_filename = os.path.join(self.checkpoint_dir, "gp_sampling_log.csv")
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
        print(f"GP-UCB sampling log: {gp_log_filename}")
    
    def _save_experiment_config(self):
        """Save experiment configuration to JSON file"""
        config = {
            'experiment_name': self.experiment_name,
            'pid': os.getpid(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
        
        config_file = os.path.join(self.checkpoint_dir, 'experiment_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Experiment config: {config_file}")
    
    def _log_gp_sampling(self, iteration: int, selected_workflow: List[str], 
                        ucb_score: float, info: Dict):
        """Log GP-UCB sampling decision to CSV"""
        row = [iteration, ' → '.join(selected_workflow), f"{ucb_score:.3f}"]
        
        # Get top 3 candidates from info
        top_candidates = info.get('top_3_candidates', [])
        
        # Fill in top 3 choices
        for i in range(3):
            if i < len(top_candidates):
                candidate = top_candidates[i]
                row.extend([
                    ' → '.join(candidate['order']) if isinstance(candidate['order'], list) else candidate['order'],
                    f"{candidate['ucb']:.3f}",
                    f"{candidate['mean']:.3f}",
                    f"{candidate['std']:.3f}"
                ])
            else:
                row.extend(['', '', '', ''])  # Empty if less than 3 candidates
        
        # Add selection method and other info
        selection_method = info.get('selection_method', 'gp-ucb')
        exploitation_value = info.get('exploitation_value', '')
        exploration_bonus = info.get('exploration_bonus', '')
        
        row.extend([
            selection_method,
            f"{exploitation_value:.3f}" if exploitation_value != '' else '',
            f"{exploration_bonus:.3f}" if exploration_bonus != '' else ''
        ])
        
        self.gp_sampling_writer.writerow(row)
        self.gp_sampling_file.flush()
        
    def train_workflow_parallel(self, workflow_order: List[str], workflow_vector: np.ndarray,
                               workflow_id: int) -> Tuple[float, float, int, bool]:
        """
        Train PPO agent until compliance threshold is met, then evaluate
        
        Strategy:
        1. Train with alignment rewards until compliance >= threshold
        2. Evaluate on pure environment reward (no alignment)
        3. Return evaluation reward for GP-UCB
        
        Args:
            workflow_order: Priority order of unit types
            workflow_vector: Workflow embedding vector  
            workflow_id: ID for saving checkpoint
            
        Returns:
            (evaluation_env_reward, final_compliance, total_episodes_trained, success)
        """
        
        print(f"\n{'='*60}")
        print(f"Training with workflow: {' → '.join(workflow_order)}")
        print(f"Goal: Train until compliance >= {self.compliance_threshold:.1%}")
        print(f"Using {self.n_envs} parallel environments")
        print(f"{'='*60}")
        
        workflow_str = ' → '.join(workflow_order)
        
        # Create parallel environments with optimized shared memory
        # Using dedicated pipes and caching for 3.4x speedup with true states!
        envs = ParallelEnvSharedMemoryOptimized(
            n_envs=self.n_envs,
            scenario_path=self.scenario_path,
            red_agent_type=self.red_agent_type,
            sparse_true_states=False  # Keep getting states as before for now
        )
        
        # Create or reuse PPO agent (policy inheritance across workflows)
        if self.shared_agent is None:
            # First workflow - create new agent
            print("  Creating new agent (first workflow)")
            agent = ParallelOrderConditionedPPO(
                input_dims=envs.observation_shape[0],
                n_envs=self.n_envs,
                workflow_order=workflow_order,
                workflow_manager=self.workflow_manager,
                alignment_lambda=self.alignment_lambda,
                update_steps=self.update_every_steps,
                K_epochs=4,
                eps_clip=0.2,
                gamma=0.99,
                lr=0.002
            )
        else:
            # Subsequent workflows - inherit policy from previous workflow
            print("  Inheriting policy from previous workflow")
            agent = ParallelOrderConditionedPPO(
                input_dims=envs.observation_shape[0],
                n_envs=self.n_envs,
                workflow_order=workflow_order,
                workflow_manager=self.workflow_manager,
                alignment_lambda=self.alignment_lambda,
                update_steps=self.update_every_steps,
                K_epochs=4,
                eps_clip=0.2,
                gamma=0.99,
                lr=0.002
            )
            # Load the shared policy weights
            agent.policy.load_state_dict(self.shared_agent.policy.state_dict())
            agent.policy_old.load_state_dict(self.shared_agent.policy_old.state_dict())
            print("  Policy weights loaded successfully")
        
        # Training metrics
        episode_rewards = [[] for _ in range(self.n_envs)]  # Pure env rewards
        episode_total_rewards = [[] for _ in range(self.n_envs)]  # Env + alignment rewards
        episode_alignment_bonuses = [[] for _ in range(self.n_envs)]  # Track actual alignment bonuses
        episode_compliances = [[] for _ in range(self.n_envs)]
        episode_fix_counts = [[] for _ in range(self.n_envs)]  # Fixes per episode
        episode_counts = np.zeros(self.n_envs, dtype=int)
        episode_dones = np.zeros(self.n_envs, dtype=bool)
        
        # Timing metrics
        update_times = []
        sampling_times = []
        update_count = 0
        sampling_start = time.time()
        
        # Episode tracking
        current_episode_rewards = np.zeros(self.n_envs)  # Pure env rewards
        current_episode_total_rewards = np.zeros(self.n_envs)  # Env + alignment
        current_episode_alignment_bonus = np.zeros(self.n_envs)  # Track episode-end alignment bonus
        current_episode_steps = np.zeros(self.n_envs, dtype=int)
        
        
        # Track cumulative fix actions across all episodes
        cumulative_fix_actions = np.zeros(self.n_envs)
        cumulative_compliant_actions = np.zeros(self.n_envs)
        
        # Reset all environments
        observations = envs.reset()
        
        # Training loop - continue until compliance threshold is met
        total_steps = 0
        compliance_achieved = False
        max_episodes_reached = False
        
        while np.min(episode_counts) < self.max_train_episodes_per_env:
            # Get current true states before action
            true_states = envs.get_true_states()
            
            # Get actions WITHOUT storing yet (we don't have rewards)
            actions, log_probs, values = agent.get_actions(observations)
            
            # Step environments to get rewards
            observations, env_rewards, dones, infos = envs.step(actions)
            
            # Get new true states after action
            new_true_states = envs.get_true_states()
            
            # Compute alignment rewards based on compliance delta
            alignment_rewards = agent.compute_alignment_rewards(
                actions, new_true_states, true_states, dones
            )
            
            # Track alignment contributions per episode for logging
            current_episode_alignment_bonus += alignment_rewards
            
            # Combine rewards
            total_rewards = env_rewards + alignment_rewards
            
            # Store in buffer
            agent.buffer.add(
                observations, actions, total_rewards, dones,
                log_probs.cpu().numpy(), values.cpu().numpy()
            )
            
            # Update agent's internal state
            agent.prev_true_states = new_true_states.copy()
            agent.step_count += 1
            
            # Reset compliance tracking for done environments
            for env_idx in range(self.n_envs):
                if dones[env_idx]:
                    agent.prev_true_states[env_idx] = None
                    agent.env_compliant_actions[env_idx] = 0
                    agent.env_total_fix_actions[env_idx] = 0
                    agent.env_fixed_types[env_idx] = set()  # Reset fixed types
            
            # Track episode progress
            # Note: We track both pure env rewards and total rewards (env + alignment)
            current_episode_rewards += env_rewards  # Pure environment rewards
            current_episode_total_rewards += total_rewards  # Total with distributed alignment
            
            current_episode_steps += 1
            
            # Handle episode completions
            for env_idx in range(self.n_envs):
                # Check if episode is done (either by environment or max steps)
                if dones[env_idx] or current_episode_steps[env_idx] >= self.max_steps:
                    if episode_counts[env_idx] < self.max_train_episodes_per_env:
                        # Record episode stats
                        episode_rewards[env_idx].append(current_episode_rewards[env_idx])
                        episode_total_rewards[env_idx].append(current_episode_total_rewards[env_idx])
                        episode_alignment_bonuses[env_idx].append(current_episode_alignment_bonus[env_idx])
                        
                        # Calculate compliance for this episode
                        compliance = agent.get_compliance_rates()[env_idx]
                        episode_compliances[env_idx].append(compliance)
                        
                        # Record fixes for this episode
                        fixes_this_episode = agent.env_total_fix_actions[env_idx]
                        episode_fix_counts[env_idx].append(fixes_this_episode)
                        
                        # Store cumulative counts before reset
                        cumulative_fix_actions[env_idx] += fixes_this_episode
                        cumulative_compliant_actions[env_idx] += agent.env_compliant_actions[env_idx]
                        
                        # Increment episode count
                        episode_counts[env_idx] += 1
                        
                        # Log episode to consolidated CSV file
                        self.consolidated_csv_writer.writerow([
                            workflow_id,  # Workflow ID
                            workflow_str,  # Workflow order string
                            'episode',  # Type
                            episode_counts[env_idx],  # Episode number for this env
                            '',  # Total_Episodes (only for summary rows)
                            env_idx,  # Environment ID
                            f"{current_episode_rewards[env_idx]:.2f}",  # Env reward
                            f"{current_episode_total_rewards[env_idx]:.2f}",  # Total reward
                            f"{current_episode_alignment_bonus[env_idx]:.2f}",  # Alignment bonus
                            f"{compliance:.4f}",  # Compliance rate
                            fixes_this_episode,  # Number of fixes
                            current_episode_steps[env_idx],  # Steps taken
                            '',  # Success (only for summary rows)
                            ''  # Eval_Reward (only for summary rows)
                        ])
                        self.consolidated_log_file.flush()  # Ensure data is written immediately
                        
                        # Reset episode tracking
                        current_episode_rewards[env_idx] = 0
                        current_episode_total_rewards[env_idx] = 0
                        current_episode_alignment_bonus[env_idx] = 0
                        current_episode_steps[env_idx] = 0
                        episode_dones[env_idx] = True
                        
                        # Reset compliance tracking for next episode
                        agent.reset_episode_compliance(env_idx)
                else:
                    episode_dones[env_idx] = False
            
            # PPO update if needed (happens every 100 steps with 100 envs)
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
                
                # Progress report after every update
                total_episodes = np.sum(episode_counts)
                
                # Calculate average rewards and compliance from the last completed episodes
                all_env_rewards = []
                all_total_rewards = []
                all_alignment_bonuses = []
                all_compliances = []
                all_fixes_per_episode = []
                
                for env_idx in range(self.n_envs):
                    if len(episode_rewards[env_idx]) > 0:
                        # Get last episode from each env
                        all_env_rewards.append(episode_rewards[env_idx][-1])
                        all_total_rewards.append(episode_total_rewards[env_idx][-1])
                        all_alignment_bonuses.append(episode_alignment_bonuses[env_idx][-1])
                        all_compliances.append(episode_compliances[env_idx][-1])
                        all_fixes_per_episode.append(episode_fix_counts[env_idx][-1])
                
                if all_env_rewards:
                    avg_env_reward = np.mean(all_env_rewards)
                    avg_total_reward = np.mean(all_total_rewards)
                    avg_alignment_bonus = np.mean(all_alignment_bonuses)
                    avg_compliance = np.mean(all_compliances)
                    avg_fixes_per_episode = np.mean(all_fixes_per_episode) if all_fixes_per_episode else 0
                    
                    # Log summary to consolidated CSV
                    self.consolidated_csv_writer.writerow([
                        workflow_id,  # Workflow ID
                        workflow_str,  # Workflow order string
                        'summary',  # Type
                        '',  # Episode (only for episode rows)
                        int(total_episodes),  # Total episodes
                        '',  # Env_ID (only for episode rows)
                        f"{avg_env_reward:.2f}",  # Avg env reward
                        f"{avg_total_reward:.2f}",  # Avg total reward
                        f"{avg_alignment_bonus:.2f}",  # Avg alignment bonus
                        f"{avg_compliance:.4f}",  # Avg compliance
                        f"{avg_fixes_per_episode:.2f}",  # Avg fixes per episode
                        '',  # Steps (only for episode rows)
                        '',  # Success (filled later)
                        ''  # Eval_Reward (filled later)
                    ])
                    self.consolidated_log_file.flush()
                    
                    # Timing statistics
                    avg_update_time = np.mean(update_times) if update_times else 0
                    avg_sampling_time = np.mean(sampling_times) if sampling_times else 0
                    update_ratio = avg_update_time / (avg_update_time + avg_sampling_time) * 100 if (avg_update_time + avg_sampling_time) > 0 else 0
                    
                    print(f"\n  Update {update_count}: "
                          f"Episodes: {int(total_episodes)} total")
                    print(f"    Env Reward/Episode: {avg_env_reward:.2f}")
                    print(f"    Total Reward/Episode: {avg_total_reward:.2f}")
                    print(f"    Alignment Bonus (episode-end): {avg_alignment_bonus:+.2f}")
                    print(f"    Compliance: {avg_compliance:.2%}")
                    print(f"    Avg Fixes/Episode: {avg_fixes_per_episode:.1f}")
                    print(f"    ⏱️ Timing: Sampling={sampling_time:.2f}s, Update={update_time:.2f}s")
                    print(f"    📊 Average: Sampling={avg_sampling_time:.2f}s, Update={avg_update_time:.2f}s (PPO takes {update_ratio:.1f}% of time)")
                
                # Reset sampling timer
                sampling_start = time.time()
            
            total_steps += 1
            
            # Compliance check - can stop immediately when threshold is achieved (no min episodes)
            # Calculate latest compliance for all environments
            latest_compliances = []
            # Use cumulative fixes for compliance check
            total_cumulative_fixes = np.sum(cumulative_fix_actions)
            current_episode_fixes = np.sum(agent.env_total_fix_actions)
            total_fixes = total_cumulative_fixes + current_episode_fixes
            
            for env_idx in range(self.n_envs):
                if len(episode_compliances[env_idx]) >= 1:
                    # Use only the most recent episode's compliance
                    latest = episode_compliances[env_idx][-1]
                    latest_compliances.append(latest)
            
            if latest_compliances:
                avg_latest_compliance = np.mean(latest_compliances)
                min_episodes = np.min(episode_counts)
                
                # Stop when latest compliance threshold is achieved with meaningful fixes
                if avg_latest_compliance >= self.compliance_threshold and total_fixes >= 10:
                    print(f"\n  ✓ Compliance threshold achieved!")
                    print(f"    Episodes trained: {min_episodes} per env")
                    print(f"    Latest compliance: {avg_latest_compliance:.2%}")
                    print(f"    Total fixes detected: {int(total_fixes)}")
                    compliance_achieved = True
                    break
                elif avg_latest_compliance >= self.compliance_threshold and total_fixes < 10:
                    # High compliance but no fixes - keep training
                    print(f"  High compliance ({avg_latest_compliance:.2%}) but only "
                          f"{int(total_fixes)} fixes detected - continuing training")
        
        # Check if max episodes reached without achieving compliance
        if not compliance_achieved:
            max_episodes_reached = True
            print(f"\n  ✗ Max episodes reached without achieving compliance threshold")
            print(f"    Episodes trained: {int(np.mean(episode_counts))} per env")
        
        # Close environments
        envs.close()
        
        # Report final timing breakdown if we have timing data
        if update_times:
            total_time = time.time() - sampling_start + sum(sampling_times) + sum(update_times)
            total_update_time = sum(update_times)
            total_sampling_time = sum(sampling_times)
            print(f"\n📊 Timing Breakdown for Workflow {workflow_id}:")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   Total sampling time: {total_sampling_time:.1f}s ({total_sampling_time/total_time*100:.1f}%)")
            print(f"   Total PPO update time: {total_update_time:.1f}s ({total_update_time/total_time*100:.1f}%)")
            print(f"   Other (logging, etc): {total_time - total_sampling_time - total_update_time:.1f}s")
            print(f"   Updates performed: {len(update_times)}")
            print(f"   Avg time per update: {np.mean(update_times):.2f}s")
        
        # Calculate training stats
        all_final_compliances = []
        for env_idx in range(self.n_envs):
            if len(episode_compliances[env_idx]) > 0:
                final_compliances = episode_compliances[env_idx][-5:]
                all_final_compliances.extend(final_compliances)
        
        final_training_compliance = np.mean(all_final_compliances) if all_final_compliances else 0
        total_episodes = np.sum(episode_counts)
        
        # Use training environment rewards directly (no separate evaluation needed!)
        # We already have rewards from 200 parallel environments
        if compliance_achieved:
            # Calculate average environment reward from last episode only
            last_episode_rewards = []
            for env_idx in range(self.n_envs):
                if len(episode_rewards[env_idx]) > 0:
                    # Take only the last episode from each environment
                    last_episode_rewards.append(episode_rewards[env_idx][-1])
            
            # Average reward across last episodes from all environments
            eval_reward = np.mean(last_episode_rewards) if last_episode_rewards else -1000.0
            eval_compliance = final_training_compliance
            
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETE - Using Training Rewards")
            print(f"{'='*60}")
            print(f"  Average Environment Reward (last episode × {self.n_envs} envs): {eval_reward:.2f}")
            print(f"  Final Compliance: {eval_compliance:.2%}")
            print(f"  → This reward will be used for GP-UCB")
            print(f"{'='*60}")
            
            # Save trained agent
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"workflow_{workflow_id}_compliant_agent.pth"
            )
            agent.save(checkpoint_path)
            
            # Store agent for policy inheritance in next workflow
            self.shared_agent = agent
            print(f"  Policy saved for next workflow inheritance")
        else:
            # Training failed - use penalty
            eval_reward = -1000.0
            eval_compliance = final_training_compliance
            
            print(f"\n  ✗ Training failed - compliance threshold not achieved")
            print(f"  Final compliance: {final_training_compliance:.2%}")
            print(f"  Assigning penalty reward: {eval_reward:.2f}")
            # Still store the agent for potential learning transfer
            self.shared_agent = agent
            print(f"  Policy still saved for next workflow (partial learning may help)")
        
        # Log workflow completion summary
        self.consolidated_csv_writer.writerow([
            workflow_id,  # Workflow ID
            workflow_str,  # Workflow order string
            'workflow_complete',  # Type
            '',  # Episode
            int(total_episodes),  # Total episodes trained
            '',  # Env_ID
            '',  # Env reward
            '',  # Total reward
            '',  # Alignment bonus
            f"{final_training_compliance:.4f}",  # Final compliance
            '',  # Fixes
            '',  # Steps
            'Yes' if compliance_achieved else 'No',  # Success
            f"{eval_reward:.2f}"  # Eval_Reward
        ])
        self.consolidated_log_file.flush()
        
        # Return actual episodes used (per env), not average
        episodes_used = int(np.min(episode_counts))  # Min ensures we're counting actual completed episodes
        return eval_reward, final_training_compliance, episodes_used, compliance_achieved
    
    # REMOVED: No longer need separate evaluation - we use training rewards directly
    # We already have rewards from 200 parallel environments during training
    # This provides a much better estimate than a separate evaluation phase
    
    def run_workflow_search(self):
        """
        Main training loop for workflow search
        
        Strategy:
        1. GP-UCB selects next workflow to explore
        2. Train PPO with alignment rewards until compliance >= 95%
        3. Use average training environment reward as GP-UCB observation
           (No separate evaluation - we have 200 parallel environments!)
        4. Continue until episode budget is exhausted
        """
        
        print(f"\n{'='*60}")
        print(f"Compliance-Gated Workflow Search")
        print(f"{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Directory: {self.checkpoint_dir}")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Red Agent: {self.red_agent_type.__name__}")
        print(f"  Parallel Environments: {self.n_envs}")
        print(f"  Episode Budget: {self.total_episode_budget}")
        print(f"  Max Episodes per Workflow: {self.max_train_episodes_per_env}")
        print(f"  Compliance Threshold: {self.compliance_threshold:.1%}")
        print(f"  Alignment Lambda: {self.alignment_lambda}")
        print(f"  Shared Memory: Enabled (17x speedup!)")
        print(f"\nTraining Strategy:")
        print(f"  1. Train with alignment rewards until compliance >= {self.compliance_threshold:.1%}")
        print(f"  2. Use training environment rewards for GP-UCB")
        print(f"  3. Continue until {self.total_episode_budget} episodes are used")
        print(f"{'='*60}")
        
        # Main search loop - continue until budget exhausted
        iteration = 0
        while self.total_episodes_used < self.total_episode_budget:
            print(f"\n{'='*50}")
            print(f"Iteration {iteration + 1}")
            print(f"Episode Budget: {self.total_episodes_used}/{self.total_episode_budget} used")
            print(f"{'='*50}")
            
            # 1. Select next workflow using GP-UCB
            # Evaluate ALL possible permutations (5! = 120 workflows)
            from itertools import permutations as iter_permutations
            
            unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
            all_permutations = list(iter_permutations(unit_types))
            
            # Convert to list of lists
            candidate_orders = [list(perm) for perm in all_permutations]
            
            print(f"  Evaluating UCB for ALL {len(candidate_orders)} possible workflows...")
            
            # GP-UCB evaluates all 120 candidates and picks highest UCB
            workflow_order, ucb_score, info = self.gp_search.select_next_order(
                candidate_orders, self.workflow_manager
            )
            
            # Print GP-UCB selection details
            print("\n" + "-"*50)
            print("GP-UCB Selection Details:")
            print(f"  Selected: {' → '.join(workflow_order)}")
            print(f"  UCB Score: {ucb_score:.3f}")
            
            if 'selection_method' in info:
                # Initial selection (random or diversity)
                print(f"  Method: {info['selection_method']}")
                print(f"  Reason: {info['reason']}")
            else:
                # GP-UCB selection
                print(f"  Mean Reward: {info['mean']:.2f}")
                print(f"  Uncertainty (std): {info['std']:.3f}")
                print(f"  Exploration Bonus: {info['exploration_bonus']:.3f}")
                print(f"  Exploitation Value: {info['exploitation_value']:.2f}")
                print(f"  Selection Type: {info['selection_reason']}")
                print(f"  Previous Visits: {info['visit_count']}")
                print(f"  Closest Known: {info['closest_observed']} (dist={info['closest_distance']:.2f}, reward={info['closest_reward']:.2f})")
                
                # Show top 3 candidates
                print("\n  Top 3 Candidates:")
                for i, candidate in enumerate(info.get('top_3_candidates', []), 1):
                    print(f"    {i}. {candidate['order']}")
                    print(f"       UCB={candidate['ucb']:.3f}, Mean={candidate['mean']:.2f}, Std={candidate['std']:.3f}")
            
            print("-"*50)
            
            # Log GP-UCB sampling decision
            self._log_gp_sampling(iteration, workflow_order, ucb_score, info)
            
            # Check if we have budget remaining (allow at least 1 episode)
            if self.total_episodes_used >= self.total_episode_budget:
                print(f"\n✗ Episode budget exhausted ({self.total_episodes_used}/{self.total_episode_budget})")
                print(f"   Stopping workflow search...")
                break
            
            # Calculate remaining budget for this workflow
            remaining_budget = min(
                self.max_train_episodes_per_env,
                self.total_episode_budget - self.total_episodes_used
            )
            
            if remaining_budget <= 0:
                print(f"\n✗ No remaining budget for new workflow")
                break
            
            # Convert to workflow vector (one-hot encoding)
            workflow_vector = self.workflow_manager.order_to_onehot(workflow_order)
            
            # 2. Train PPO with this workflow until compliance threshold is met
            eval_reward, train_compliance, episodes_used, success = self.train_workflow_parallel(
                workflow_order, workflow_vector, iteration
            )
            
            # Update total episodes used
            self.total_episodes_used += episodes_used
            print(f"\n📊 Episodes used this workflow: {episodes_used}")
            print(f"   Total episodes used: {self.total_episodes_used}/{self.total_episode_budget}")
            
            # 3. Update GP-UCB ONLY if compliance threshold was achieved
            if success:
                self.gp_search.add_observation(
                    workflow_order, eval_reward
                )
                print(f"\n✓ GP-UCB updated with training reward: {eval_reward:.2f}")
            else:
                # Do NOT record failed samples - skip this workflow
                print(f"\n✗ Compliance not achieved - sample NOT recorded in GP-UCB")
                print(f"   (This workflow will not influence the search)")
            
            # 4. Record results
            result = {
                'iteration': iteration,
                'workflow': workflow_order,
                'eval_reward': eval_reward,  # Pure environment reward from evaluation
                'train_compliance': train_compliance,
                'episodes_trained': episodes_used,
                'total_episodes_used': self.total_episodes_used,
                'success': success,
                'n_envs': self.n_envs
            }
            self.training_history.append(result)
            
            # 5. Print iteration summary
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1} SUMMARY")
            print(f"{'='*60}")
            print(f"  Workflow: {' → '.join(workflow_order)}")
            print(f"  Training Episodes: {episodes_used} per env")
            print(f"  Final Compliance: {train_compliance:.2%}")
            print(f"  Success: {'✓ Yes' if success else '✗ No'}")
            if success:
                print(f"  Training Reward (for GP-UCB): {eval_reward:.2f}")
            else:
                print(f"  Penalty (for GP-UCB): {eval_reward:.2f}")
            print(f"{'='*60}")
            
            # 6. Print best so far
            if self.gp_search.observed_orders:
                best_idx = np.argmax(self.gp_search.observed_rewards)
                best_workflow = self.gp_search.observed_orders[best_idx]
                best_reward = self.gp_search.observed_rewards[best_idx]
                print(f"\nBest workflow so far: {' → '.join(best_workflow)}")
                print(f"Best reward: {best_reward:.2f}")
            
            # 7. Save progress
            self.save_results()
            
            # Increment iteration counter
            iteration += 1
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total Episodes Used: {self.total_episodes_used}/{self.total_episode_budget}")
        print(f"Total Workflows Explored: {iteration}")
        print(f"{'='*60}")
        
        # Close consolidated log file
        if self.consolidated_log_file:
            self.consolidated_log_file.close()
            print(f"  Consolidated log saved")
        
        # Close GP sampling log file
        if self.gp_sampling_file:
            self.gp_sampling_file.close()
            print(f"  GP sampling log saved")
        
        # Final summary
        self.print_summary()
    
    def save_results(self):
        """Save training history to file"""
        results_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        
        with open(results_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Also save a summary file with best workflows
        self._save_summary()
        
        print(f"  Results saved to {results_path}")
    
    def _save_summary(self):
        """Save a human-readable summary of the experiment"""
        summary_path = os.path.join(self.checkpoint_dir, 'summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            # Configuration summary
            f.write("Configuration:\n")
            f.write(f"  Red Agent: {self.red_agent_type.__name__}\n")
            f.write(f"  Parallel Envs: {self.n_envs}\n")
            f.write(f"  Workflows Explored: {len(self.training_history)}\n")
            f.write(f"  Compliance Threshold: {self.compliance_threshold:.1%}\n")
            f.write(f"  Alignment Lambda: {self.alignment_lambda}\n")
            f.write("\n")
            
            # Results summary
            successful = [r for r in self.training_history if r['success']]
            f.write("Results:\n")
            f.write(f"  Total Workflows: {len(self.training_history)}\n")
            f.write(f"  Successful: {len(successful)}\n")
            f.write(f"  Success Rate: {len(successful)/len(self.training_history):.1%}\n")
            f.write("\n")
            
            # Best workflows
            if successful:
                sorted_successful = sorted(successful, key=lambda x: x['eval_reward'], reverse=True)
                f.write("Top 5 Successful Workflows:\n")
                for i, result in enumerate(sorted_successful[:5], 1):
                    f.write(f"\n  {i}. {' → '.join(result['workflow'])}\n")
                    f.write(f"     Eval Reward: {result['eval_reward']:.2f}\n")
                    f.write(f"     Compliance: {result['train_compliance']:.2%}\n")
                    f.write(f"     Episodes: {result['avg_episodes_trained']}\n")
            else:
                f.write("No successful workflows found.\n")
            
            # Best overall (from GP-UCB)
            if self.gp_search.observed_orders:
                best_idx = np.argmax(self.gp_search.observed_rewards)
                best_workflow = self.gp_search.observed_orders[best_idx]
                best_reward = self.gp_search.observed_rewards[best_idx]
                f.write(f"\nBest Workflow (GP-UCB):\n")
                f.write(f"  {' → '.join(best_workflow)}\n")
                f.write(f"  Reward: {best_reward:.2f}\n")
    
    def print_summary(self):
        """Print final summary of training"""
        print("\nFinal Summary:")
        print("-" * 40)
        
        # Find best workflow
        if self.gp_search.observed_orders:
            best_idx = np.argmax(self.gp_search.observed_rewards)
            best_workflow = self.gp_search.observed_orders[best_idx]
            best_reward = self.gp_search.observed_rewards[best_idx]
            print(f"Best Workflow: {' → '.join(best_workflow)}")
            print(f"Best Reward: {best_reward:.2f}")
        
        # Print top 5 successful workflows
        print("\nTop 5 Workflows (successful only):")
        successful = [r for r in self.training_history if r['success']]
        sorted_history = sorted(successful, 
                               key=lambda x: x['eval_reward'], 
                               reverse=True)[:5]
        
        if sorted_history:
            for i, result in enumerate(sorted_history):
                print(f"{i+1}. {' → '.join(result['workflow'])}")
                print(f"   Eval Reward: {result['eval_reward']:.2f}, "
                      f"Compliance: {result['train_compliance']:.2%}")
        else:
            print("  No successful workflows yet")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Compliance-Gated Workflow Search for CAGE2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment Configuration
    env_group = parser.add_argument_group('Environment Configuration')
    env_group.add_argument('--n-envs', type=int, default=100,
                          help='Number of parallel environments')
    env_group.add_argument('--total-episodes', type=int, default=100000,
                          help='Total episode budget across all workflows')
    env_group.add_argument('--max-episodes', type=int, default=100,
                          help='Max episodes per workflow before giving up')
    env_group.add_argument('--max-steps', type=int, default=100,
                          help='Max steps per episode')
    env_group.add_argument('--red-agent', type=str, default='meander',
                          choices=['meander', 'bline', 'sleep'],
                          help='Red agent type: meander (aggressive), bline (moderate), sleep (none)')
    
    # Learning Configuration
    learn_group = parser.add_argument_group('Learning Configuration')
    learn_group.add_argument('--alignment-lambda', type=float, default=30.0,
                            help='Compliance reward strength (higher = stricter)')
    learn_group.add_argument('--compliance-threshold', type=float, default=0.95,
                            help='Required compliance before evaluation (0.0-1.0)')
    learn_group.add_argument('--update-steps', type=int, default=100,
                            help='PPO update frequency (steps)')
    
    # Search Configuration
    search_group = parser.add_argument_group('Search Configuration')
    search_group.add_argument('--gp-beta', type=float, default=2.0,
                             help='GP-UCB exploration parameter')
    search_group.add_argument('--n-eval-episodes', type=int, default=20,
                             help='Episodes for final evaluation')
    
    # Output Configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--checkpoint-dir', type=str, default='compliance_checkpoints',
                             help='Directory for logs and checkpoints')
    output_group.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main entry point for parallel training"""
    
    # Parse command-line arguments
    args = parse_args()
    
    # Map red agent string to class
    red_agent_map = {
        'meander': RedMeanderAgent,
        'bline': B_lineAgent,
        'sleep': SleepAgent
    }
    RED_AGENT_TYPE = red_agent_map[args.red_agent]
    
    # ========== HYPERPARAMETERS ==========
    
    # Environment Configuration
    N_ENVS = args.n_envs
    TOTAL_EPISODE_BUDGET = args.total_episodes
    MAX_TRAIN_EPISODES_PER_ENV = args.max_episodes
    MAX_STEPS = args.max_steps
    
    # Learning Configuration
    ALIGNMENT_LAMBDA = args.alignment_lambda
    COMPLIANCE_THRESHOLD = args.compliance_threshold
    UPDATE_EVERY_STEPS = args.update_steps
    
    # Search Configuration
    GP_BETA = args.gp_beta
    N_EVAL_EPISODES = args.n_eval_episodes
    
    # Output Configuration
    CHECKPOINT_DIR = args.checkpoint_dir
    SEED = args.seed
    
    # =====================================
    
    print(f"\n{'='*60}")
    print(f"Configuration")
    print(f"{'='*60}")
    print(f"Red Agent: {args.red_agent} ({RED_AGENT_TYPE.__name__})")
    print(f"Parallel Envs: {N_ENVS}")
    print(f"Episode Budget: {TOTAL_EPISODE_BUDGET}")
    print(f"Max Episodes/Workflow: {MAX_TRAIN_EPISODES_PER_ENV}")
    print(f"Alignment Lambda: {ALIGNMENT_LAMBDA}")
    print(f"Compliance Threshold: {COMPLIANCE_THRESHOLD:.1%}")
    print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"Random Seed: {SEED}")
    print(f"{'='*60}\n")
    
    # Set seeds for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Create trainer with parallel environments
    trainer = ParallelWorkflowRLTrainer(
        n_envs=N_ENVS,
        total_episode_budget=TOTAL_EPISODE_BUDGET,
        max_train_episodes_per_env=MAX_TRAIN_EPISODES_PER_ENV,
        max_steps=MAX_STEPS,
        red_agent_type=RED_AGENT_TYPE,
        alignment_lambda=ALIGNMENT_LAMBDA,
        gp_beta=GP_BETA,
        compliance_threshold=COMPLIANCE_THRESHOLD,
        n_eval_episodes=N_EVAL_EPISODES,
        update_every_steps=UPDATE_EVERY_STEPS,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    # Run workflow search
    trainer.run_workflow_search()


if __name__ == "__main__":
    main()
