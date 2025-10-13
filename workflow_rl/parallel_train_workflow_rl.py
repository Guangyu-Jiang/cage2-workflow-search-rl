"""
Parallel Training for Workflow-Conditioned RL using multiple environments
"""

import os
import numpy as np
import torch
import json
from typing import List, Tuple, Dict
from datetime import datetime

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

from workflow_rl.order_based_workflow import OrderBasedWorkflow
from workflow_rl.gp_ucb_order_search import GPUCBOrderSearch
from workflow_rl.parallel_order_conditioned_ppo import ParallelOrderConditionedPPO, device
from workflow_rl.parallel_env_wrapper import ParallelEnvWrapper


class ParallelWorkflowRLTrainer:
    """Main trainer for workflow search-based RL with parallel environments"""
    
    def __init__(self, 
                 n_envs: int = 25,
                 n_workflows: int = 20,
                 max_train_episodes_per_env: int = 1000,  # Max episodes before giving up
                 max_steps: int = 100,
                 scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                 red_agent_type=RedMeanderAgent,
                 alignment_lambda: float = 30.0,  # Increased for stricter compliance
                 gp_beta: float = 2.0,
                 checkpoint_dir: str = 'checkpoints',
                 compliance_threshold: float = 0.95,  # Must achieve this before evaluation
                 min_episodes: int = 25,  # Minimum episodes before checking compliance
                 n_eval_episodes: int = 20,  # Episodes for evaluation
                 update_every_steps: int = 100):  # Update every 100 steps (full episode) = 2500 transitions with 25 envs
        """
        Initialize parallel trainer
        
        Training Strategy:
        1. Train PPO with alignment rewards until compliance >= 95%
        2. Evaluate on pure environment reward (without alignment)
        3. Use evaluation reward as GP-UCB observation
        
        Args:
            n_envs: Number of parallel environments
            max_train_episodes_per_env: Max episodes to train before giving up
            compliance_threshold: Required compliance rate before evaluation
            n_eval_episodes: Episodes for final evaluation
        """
        
        self.n_envs = n_envs
        self.n_workflows = n_workflows
        self.max_train_episodes_per_env = max_train_episodes_per_env
        self.max_steps = max_steps
        self.scenario_path = scenario_path
        self.red_agent_type = red_agent_type
        self.alignment_lambda = alignment_lambda
        self.compliance_threshold = compliance_threshold
        self.min_episodes = min_episodes
        self.n_eval_episodes = n_eval_episodes
        self.update_every_steps = update_every_steps
        
        # Create checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize workflow manager
        self.workflow_manager = OrderBasedWorkflow()
        
        # Initialize GP-UCB search
        self.gp_search = GPUCBOrderSearch(
            beta=gp_beta
        )
        
        # Training history
        self.training_history = []
        
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
        
        # Create parallel environments
        envs = ParallelEnvWrapper(
            n_envs=self.n_envs,
            scenario_path=self.scenario_path,
            red_agent_type=self.red_agent_type
        )
        
        # Create parallel PPO agent
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
        
        # Training metrics
        episode_rewards = [[] for _ in range(self.n_envs)]  # Pure env rewards
        episode_total_rewards = [[] for _ in range(self.n_envs)]  # Env + alignment rewards
        episode_alignment_bonuses = [[] for _ in range(self.n_envs)]  # Track actual alignment bonuses
        episode_compliances = [[] for _ in range(self.n_envs)]
        episode_fix_counts = [[] for _ in range(self.n_envs)]  # Fixes per episode
        episode_counts = np.zeros(self.n_envs, dtype=int)
        episode_dones = np.zeros(self.n_envs, dtype=bool)
        
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
            
            # PPO update if needed (happens every 25 episodes with 25 envs)
            if agent.should_update():
                agent.update()
                
                # Progress report after every update (every 25 episodes)
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
                    
                    print(f"\n  Update {total_steps // self.update_every_steps}: "
                          f"Episodes: {int(total_episodes)} total")
                    print(f"    Env Reward/Episode: {avg_env_reward:.2f}")
                    print(f"    Total Reward/Episode: {avg_total_reward:.2f}")
                    print(f"    Alignment Bonus (episode-end): {avg_alignment_bonus:+.2f}")
                    print(f"    Compliance: {avg_compliance:.2%}")
                    print(f"    Avg Fixes/Episode: {avg_fixes_per_episode:.1f}")
            
            total_steps += 1
            
            # Compliance check - stop when threshold is achieved
            min_episodes = np.min(episode_counts)
            if min_episodes >= self.min_episodes:
                # Calculate recent compliance for all environments
                recent_compliances = []
                # Use cumulative fixes for compliance check
                total_cumulative_fixes = np.sum(cumulative_fix_actions)
                current_episode_fixes = np.sum(agent.env_total_fix_actions)
                total_fixes = total_cumulative_fixes + current_episode_fixes
                
                for env_idx in range(self.n_envs):
                    if len(episode_compliances[env_idx]) >= 5:
                        recent = np.mean(episode_compliances[env_idx][-5:])
                        recent_compliances.append(recent)
                
                if recent_compliances:
                    avg_recent_compliance = np.mean(recent_compliances)
                    
                    # Stop when compliance threshold is achieved with meaningful fixes
                    if avg_recent_compliance >= self.compliance_threshold and total_fixes >= 10:
                        print(f"\n  ✓ Compliance threshold achieved!")
                        print(f"    Episodes trained: {min_episodes} per env")
                        print(f"    Compliance: {avg_recent_compliance:.2%}")
                        print(f"    Total fixes detected: {int(total_fixes)}")
                        compliance_achieved = True
                        break
                    elif avg_recent_compliance >= self.compliance_threshold and total_fixes < 10:
                        # High compliance but no fixes - keep training
                        print(f"  High compliance ({avg_recent_compliance:.2%}) but only "
                              f"{int(total_fixes)} fixes detected - continuing training")
        
        # Check if max episodes reached without achieving compliance
        if not compliance_achieved:
            max_episodes_reached = True
            print(f"\n  ✗ Max episodes reached without achieving compliance threshold")
            print(f"    Episodes trained: {int(np.mean(episode_counts))} per env")
        
        # Close environments
        envs.close()
        
        # Calculate training stats
        all_final_compliances = []
        for env_idx in range(self.n_envs):
            if len(episode_compliances[env_idx]) > 0:
                final_compliances = episode_compliances[env_idx][-5:]
                all_final_compliances.extend(final_compliances)
        
        final_training_compliance = np.mean(all_final_compliances) if all_final_compliances else 0
        total_episodes = np.sum(episode_counts)
        
        # Evaluate ONLY if compliance threshold was achieved
        eval_reward = -1000.0  # Default penalty for failed training
        eval_compliance = 0.0
        
        if compliance_achieved:
            # Save trained agent
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"workflow_{workflow_id}_compliant_agent.pth"
            )
            agent.save(checkpoint_path)
            
            print(f"\n{'='*60}")
            print(f"EVALUATION PHASE")
            print(f"{'='*60}")
            print(f"Evaluating agent on PURE ENVIRONMENT REWARD (no alignment)")
            print(f"Running {self.n_eval_episodes} episodes per environment...")
            
            # Evaluate on pure environment reward (no alignment rewards)
            eval_reward, eval_compliance = self.evaluate_pure_performance_parallel(
                agent, workflow_order, n_eval_episodes=self.n_eval_episodes
            )
            
            print(f"\nEvaluation Results:")
            print(f"  Environment Reward: {eval_reward:.2f}")
            print(f"  Compliance (eval): {eval_compliance:.2%}")
            print(f"  → This reward will be used for GP-UCB")
            print(f"{'='*60}")
        else:
            print(f"\n  ✗ Training failed - compliance threshold not achieved")
            print(f"  Final compliance: {final_training_compliance:.2%}")
            print(f"  Assigning penalty reward: {eval_reward:.2f}")
        
        return eval_reward, final_training_compliance, int(np.mean(episode_counts)), compliance_achieved
    
    def evaluate_pure_performance_parallel(self, agent: ParallelOrderConditionedPPO,
                                          workflow_order: List[str],
                                          n_eval_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate agent on pure environment reward (no alignment rewards)
        
        Args:
            agent: Trained PPO agent
            workflow_order: Workflow being evaluated
            n_eval_episodes: Episodes per environment for evaluation
            
        Returns:
            (average_env_reward, average_compliance_rate)
        """
        # Create evaluation environments
        n_eval_envs = min(self.n_envs, 10)
        envs = ParallelEnvWrapper(
            n_envs=n_eval_envs,  # Use fewer envs for evaluation
            scenario_path=self.scenario_path,
            red_agent_type=self.red_agent_type
        )
        
        # Reset agent's compliance tracking for evaluation environments
        agent.env_compliant_actions = np.zeros(n_eval_envs)
        agent.env_total_fix_actions = np.zeros(n_eval_envs)
        agent.prev_true_states = [None] * n_eval_envs
        
        eval_rewards = [[] for _ in range(n_eval_envs)]
        eval_compliances = [[] for _ in range(n_eval_envs)]
        episode_counts = np.zeros(n_eval_envs, dtype=int)
        
        current_rewards = np.zeros(n_eval_envs)
        current_steps = np.zeros(n_eval_envs, dtype=int)
        
        observations = envs.reset()
        
        while np.min(episode_counts) < n_eval_episodes:
            # Get actions (deterministic for evaluation)
            actions, _, _ = agent.get_actions(observations, deterministic=True)
            
            # Step environments
            observations, env_rewards, dones, infos = envs.step(actions)
            
            # Track rewards (pure environment rewards only)
            current_rewards += env_rewards
            current_steps += 1
            
            # Handle episode completions
            for env_idx in range(n_eval_envs):
                if dones[env_idx] or current_steps[env_idx] >= self.max_steps:
                    if episode_counts[env_idx] < n_eval_episodes:
                        eval_rewards[env_idx].append(current_rewards[env_idx])
                        
                        # Get compliance (for reporting, not reward)
                        compliance = agent.get_compliance_rates()[env_idx]
                        eval_compliances[env_idx].append(compliance)
                        
                        episode_counts[env_idx] += 1
                        current_rewards[env_idx] = 0
                        current_steps[env_idx] = 0
        
        envs.close()
        
        # Calculate averages
        all_rewards = []
        all_compliances = []
        for env_idx in range(n_eval_envs):
            all_rewards.extend(eval_rewards[env_idx])
            all_compliances.extend(eval_compliances[env_idx])
        
        return np.mean(all_rewards), np.mean(all_compliances)
    
    def run_workflow_search(self):
        """
        Main training loop for workflow search
        
        Strategy:
        1. GP-UCB selects next workflow to explore
        2. Train PPO with alignment rewards until compliance >= 95%
        3. Evaluate on pure environment reward (no alignment)
        4. Use evaluation reward as GP-UCB observation
        5. Repeat for N workflows
        """
        
        print(f"\n{'='*60}")
        print(f"Compliance-Gated Workflow Search")
        print(f"{'='*60}")
        print(f"Number of parallel environments: {self.n_envs}")
        print(f"Number of workflows to explore: {self.n_workflows}")
        print(f"Max episodes per environment: {self.max_train_episodes_per_env}")
        print(f"Compliance threshold: {self.compliance_threshold:.1%}")
        print(f"Evaluation episodes: {self.n_eval_episodes} per env")
        print(f"Alignment lambda: {self.alignment_lambda}")
        print(f"\nTraining Strategy:")
        print(f"  1. Train with alignment rewards until compliance >= {self.compliance_threshold:.1%}")
        print(f"  2. Evaluate on pure environment reward")
        print(f"  3. Use evaluation reward for GP-UCB")
        print(f"{'='*60}")
        
        # Main search loop
        for iteration in range(self.n_workflows):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration + 1}/{self.n_workflows}")
            print(f"{'='*50}")
            
            # 1. Select next workflow using GP-UCB
            # Generate candidates if needed
            if iteration == 0:
                # Initialize with some diverse candidates
                candidate_orders = [
                    ['defender', 'enterprise', 'op_server', 'op_host', 'user'],
                    ['op_server', 'defender', 'enterprise', 'op_host', 'user'],
                    ['enterprise', 'op_server', 'defender', 'user', 'op_host'],
                    ['user', 'op_host', 'op_server', 'enterprise', 'defender'],
                    ['op_host', 'user', 'defender', 'enterprise', 'op_server'],
                ]
            else:
                # Use random permutations as candidates
                unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
                candidate_orders = []
                for _ in range(10):
                    perm = unit_types.copy()
                    np.random.shuffle(perm)
                    candidate_orders.append(perm)
            
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
            
            # Convert to workflow vector (one-hot encoding)
            workflow_vector = self.workflow_manager.order_to_onehot(workflow_order)
            
            # 2. Train PPO with this workflow until compliance threshold is met
            eval_reward, train_compliance, avg_episodes, success = self.train_workflow_parallel(
                workflow_order, workflow_vector, iteration
            )
            
            # 3. Update GP-UCB ONLY if compliance threshold was achieved
            if success:
                self.gp_search.add_observation(
                    workflow_order, eval_reward
                )
                print(f"\n✓ GP-UCB updated with evaluation reward: {eval_reward:.2f}")
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
                'avg_episodes_trained': avg_episodes,
                'success': success,
                'n_envs': self.n_envs
            }
            self.training_history.append(result)
            
            # 5. Print iteration summary
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1} SUMMARY")
            print(f"{'='*60}")
            print(f"  Workflow: {' → '.join(workflow_order)}")
            print(f"  Training Episodes: {avg_episodes} per env")
            print(f"  Final Compliance: {train_compliance:.2%}")
            print(f"  Success: {'✓ Yes' if success else '✗ No'}")
            if success:
                print(f"  Evaluation Reward (for GP-UCB): {eval_reward:.2f}")
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
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        
        # Final summary
        self.print_summary()
    
    def save_results(self):
        """Save training history to file"""
        results_path = os.path.join(self.checkpoint_dir, 'parallel_training_history.json')
        
        with open(results_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"  Results saved to {results_path}")
    
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


def main():
    """Main entry point for parallel training"""
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create trainer with parallel environments
    trainer = ParallelWorkflowRLTrainer(
        n_envs=25,  # 25 parallel environments
        n_workflows=20,
        max_train_episodes_per_env=1000,  # Max episodes to train before giving up
        max_steps=100,
        alignment_lambda=30.0,  # Increased for stricter compliance (was 10.0)
        gp_beta=2.0,
        compliance_threshold=0.95,  # Must achieve 95% before evaluation
        min_episodes=25,  # Min episodes before checking compliance
        n_eval_episodes=20,  # Episodes for final evaluation
        update_every_steps=100,  # Update every 2500 transitions (100*25)
        checkpoint_dir='compliance_checkpoints'
    )
    
    # Run workflow search
    trainer.run_workflow_search()


if __name__ == "__main__":
    main()
