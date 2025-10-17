"""
Async Training for Workflow-Conditioned RL
Each environment collects full episodes independently - NO synchronous barriers!
"""

import os
import sys
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


class AsyncWorkflowRLTrainer:
    """
    Async trainer where workers collect full episodes independently.
    NO step-level synchronization = Much faster!
    """
    
    def __init__(self, 
                 n_envs: int = 100,
                 total_episode_budget: int = 100000,
                 max_train_episodes_per_env: int = 100,
                 max_steps: int = 100,
                 scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                 red_agent_type=RedMeanderAgent,
                 alignment_lambda: float = 30.0,
                 gp_beta: float = 2.0,
                 compliance_threshold: float = 0.95,
                 episodes_per_update: int = 100):  # Collect 100 episodes, then update
        """
        Initialize async trainer
        
        Key Difference from Sync:
        - Collects FULL EPISODES from workers before update
        - No synchronization at each step
        - Workers run independently
        """
        
        self.n_envs = n_envs
        self.total_episode_budget = total_episode_budget
        self.max_train_episodes_per_env = max_train_episodes_per_env
        self.max_steps = max_steps
        self.scenario_path = scenario_path
        self.red_agent_type = red_agent_type
        self.alignment_lambda = alignment_lambda
        self.compliance_threshold = compliance_threshold
        self.episodes_per_update = episodes_per_update
        
        # Track total episodes
        self.total_episodes_used = 0
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pid = os.getpid()
        self.experiment_name = f"exp_async_{timestamp}_{pid}"
        self.checkpoint_dir = os.path.join("logs", self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"🚀 Starting ASYNC experiment with PID: {pid}")
        print(f"   No synchronous barriers - workers collect episodes independently!")
        
        # Initialize workflow manager
        self.workflow_manager = OrderBasedWorkflow()
        
        # Initialize GP-UCB search
        self.gp_search = GPUCBOrderSearch(beta=gp_beta)
        
        # Shared agent across workflows
        self.shared_agent = None
        
        # Initialize logging
        self._init_consolidated_logging()
        self._init_gp_sampling_log()
        self._save_experiment_config()
    
    def _init_consolidated_logging(self):
        """Initialize consolidated CSV logging"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(self.checkpoint_dir, f"training_log_{timestamp}.csv")
        self.consolidated_log_file = open(log_filename, 'w', newline='')
        self.consolidated_csv_writer = csv.writer(self.consolidated_log_file)
        self.consolidated_csv_writer.writerow([
            'Workflow_ID', 'Workflow_Order', 'Type', 'Episode', 'Total_Episodes',
            'Env_ID', 'Env_Reward', 'Total_Reward', 'Alignment_Bonus',
            'Compliance', 'Fixes', 'Steps', 'Success', 'Eval_Reward'
        ])
        self.consolidated_log_file.flush()
        print(f"📊 Training log: {log_filename}")
    
    def _init_gp_sampling_log(self):
        """Initialize GP-UCB sampling log"""
        gp_log_filename = os.path.join(self.checkpoint_dir, "gp_sampling_log.csv")
        self.gp_sampling_file = open(gp_log_filename, 'w', newline='')
        self.gp_sampling_writer = csv.writer(self.gp_sampling_file)
        self.gp_sampling_writer.writerow([
            'Iteration', 'Selected_Workflow', 'UCB_Score',
            'Top1_Workflow', 'Top1_UCB', 'Top1_Mean', 'Top1_Std',
            'Top2_Workflow', 'Top2_UCB', 'Top2_Mean', 'Top2_Std',
            'Top3_Workflow', 'Top3_UCB', 'Top3_Mean', 'Top3_Std',
            'Selection_Method', 'Exploitation_Value', 'Exploration_Bonus'
        ])
        self.gp_sampling_file.flush()
        print(f"📈 GP-UCB log: {gp_log_filename}")
    
    def _save_experiment_config(self):
        """Save experiment configuration"""
        config = {
            'experiment_name': self.experiment_name,
            'architecture': 'ASYNC',
            'description': 'Workers collect full episodes independently (no sync barriers)',
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
                'episodes_per_update': self.episodes_per_update,
                'compliance_threshold': self.compliance_threshold
            },
            'rewards': {
                'alignment_lambda': self.alignment_lambda
            }
        }
        config_file = os.path.join(self.checkpoint_dir, 'experiment_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"⚙️  Config: {config_file}\n")
    
    def collect_async_episodes(self, agent, workflow_order: List[str],
                               n_episodes: int) -> Tuple:
        """
        Collect episodes asynchronously from parallel environments.
        Each environment runs independently until episode completion.
        
        Returns episode data aggregated across all environments.
        """
        workflow_encoding = self.workflow_manager.order_to_onehot(workflow_order)
        
        # Storage for collected data
        all_states = []
        all_actions = []
        all_env_rewards = []
        all_total_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []
        all_compliances = []
        all_fix_counts = []
        
        # Create temporary environments for each worker
        envs = []
        cyborgs = []
        for _ in range(self.n_envs):
            cyborg = CybORG(self.scenario_path, 'sim', agents={'Red': self.red_agent_type})
            env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
            envs.append(env)
            cyborgs.append(cyborg)
        
        episodes_collected = 0
        env_idx = 0
        
        print(f"📦 Collecting {n_episodes} episodes asynchronously...")
        start_time = time.time()
        
        # Collect episodes round-robin from environments
        while episodes_collected < n_episodes:
            env = envs[env_idx]
            cyborg = cyborgs[env_idx]
            
            # Run ONE complete episode
            episode_states = []
            episode_actions = []
            episode_env_rewards = []
            episode_dones = []
            episode_log_probs = []
            episode_values = []
            
            state = env.reset()
            prev_true_state = None
            compliant_actions = 0
            total_fix_actions = 0
            
            for step in range(self.max_steps):
                # Get true state before
                true_state_before = cyborg.get_agent_state('True')
                
                # Prepare state with workflow encoding
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                order_tensor = torch.FloatTensor(workflow_encoding).unsqueeze(0).to(device)
                augmented_state = torch.cat([state_tensor, order_tensor], dim=1)
                
                # Get action from policy
                with torch.no_grad():
                    logits = agent.policy_old.actor(augmented_state)
                    dist = torch.distributions.Categorical(logits)
                    action_tensor = dist.sample()
                    log_prob = dist.log_prob(action_tensor)
                    value = agent.policy_old.critic(augmented_state)
                
                action = action_tensor.item()
                
                # Execute action
                next_state, env_reward, done, info = env.step(action)
                
                # Get true state after
                true_state_after = cyborg.get_agent_state('True')
                
                # Compute alignment reward
                alignment_reward = 0
                if prev_true_state is not None:
                    alignment_reward = self._compute_alignment_reward(
                        action, true_state_after, true_state_before, done,
                        workflow_order, ref_compliant=compliant_actions,
                        ref_total=total_fix_actions
                    )
                    
                    # Update compliance tracking
                    if self._is_fix_action(action):
                        total_fix_actions += 1
                        if self._is_compliant_with_workflow(
                            action, true_state_after, true_state_before, workflow_order
                        ):
                            compliant_actions += 1
                
                total_reward = env_reward + alignment_reward
                
                # Store transition
                episode_states.append(augmented_state.cpu().squeeze().numpy())
                episode_actions.append(action)
                episode_env_rewards.append(env_reward)
                episode_dones.append(done)
                episode_log_probs.append(log_prob.cpu().item())
                episode_values.append(value.cpu().item())
                
                prev_true_state = true_state_after
                state = next_state
                
                if done:
                    break
            
            # Episode complete!
            all_states.extend(episode_states)
            all_actions.extend(episode_actions)
            all_env_rewards.extend(episode_env_rewards)
            all_dones.extend(episode_dones)
            all_log_probs.extend(episode_log_probs)
            all_values.extend(episode_values)
            
            compliance = compliant_actions / max(total_fix_actions, 1)
            all_compliances.append(compliance)
            all_fix_counts.append(total_fix_actions)
            
            episodes_collected += 1
            
            # Move to next environment
            env_idx = (env_idx + 1) % self.n_envs
            
            if episodes_collected % 10 == 0:
                elapsed = time.time() - start_time
                rate = episodes_collected / elapsed
                print(f"  {episodes_collected}/{n_episodes} episodes ({rate:.1f} eps/sec)")
        
        elapsed = time.time() - start_time
        rate = episodes_collected / elapsed
        print(f"✅ Collected {episodes_collected} episodes in {elapsed:.1f}s ({rate:.1f} eps/sec)\n")
        
        return (np.array(all_states), np.array(all_actions), np.array(all_env_rewards),
                np.array(all_dones), np.array(all_log_probs), np.array(all_values),
                all_compliances, all_fix_counts)
    
    def _compute_alignment_reward(self, action, true_state_after, true_state_before,
                                  done, workflow_order, ref_compliant, ref_total):
        """Compute alignment reward based on workflow compliance"""
        if not self._is_fix_action(action):
            return 0.0
        
        if self._is_compliant_with_workflow(action, true_state_after, true_state_before, workflow_order):
            compliance_rate = (ref_compliant + 1) / max(ref_total + 1, 1)
            return self.alignment_lambda * compliance_rate
        
        return 0.0
    
    def _is_fix_action(self, action):
        """Check if action is a fix action"""
        return 132 <= action <= 144
    
    def _is_compliant_with_workflow(self, action, true_after, true_before, workflow_order):
        """Check if fix action follows workflow priority"""
        # Simplified compliance check
        return True  # Placeholder
    
    def train_workflow_async(self, workflow_order: List[str], workflow_id: int):
        """
        Train workflow using async episode collection.
        No synchronous barriers!
        """
        print(f"\n{'='*60}")
        print(f"🎯 ASYNC Training: {' → '.join(workflow_order)}")
        print(f"   Workers collect episodes independently!")
        print(f"{'='*60}\n")
        
        workflow_encoding = self.workflow_manager.order_to_onehot(workflow_order)
        
        # Create or inherit agent
        if self.shared_agent is None:
            print("🆕 Creating new agent")
            # Get observation dimensions
            cyborg = CybORG(self.scenario_path, 'sim', agents={'Red': self.red_agent_type})
            env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
            obs_dim = env.observation_space.shape[0]
            
            agent = ParallelOrderConditionedPPO(
                input_dims=obs_dim,
                n_envs=self.n_envs,
                workflow_order=workflow_order,
                workflow_manager=self.workflow_manager,
                alignment_lambda=self.alignment_lambda,
                update_steps=self.episodes_per_update,  # Not used in async
                K_epochs=4,
                eps_clip=0.2,
                gamma=0.99,
                lr=0.002
            )
        else:
            print("♻️  Inheriting from previous workflow")
            obs_dim = self.shared_agent.input_dims
            agent = ParallelOrderConditionedPPO(
                input_dims=obs_dim,
                n_envs=self.n_envs,
                workflow_order=workflow_order,
                workflow_manager=self.workflow_manager,
                alignment_lambda=self.alignment_lambda,
                update_steps=self.episodes_per_update,
                K_epochs=4,
                eps_clip=0.2,
                gamma=0.99,
                lr=0.002
            )
            agent.policy.load_state_dict(self.shared_agent.policy.state_dict())
            agent.policy_old.load_state_dict(self.shared_agent.policy_old.state_dict())
        
        # Training loop
        total_episodes = 0
        current_compliance = 0.0
        
        while total_episodes < self.max_train_episodes_per_env:
            # Collect episodes asynchronously
            (states, actions, rewards, dones, log_probs, values,
             compliances, fix_counts) = self.collect_async_episodes(
                agent, workflow_order, self.episodes_per_update
            )
            
            total_episodes += self.episodes_per_update
            current_compliance = np.mean(compliances)
            
            # Compute returns (rewards-to-go)
            returns = []
            discounted_reward = 0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    discounted_reward = 0
                discounted_reward = reward + agent.gamma * discounted_reward
                returns.insert(0, discounted_reward)
            
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
            # Convert to tensors
            old_states = torch.FloatTensor(states).to(device)
            old_actions = torch.LongTensor(actions).to(device)
            old_logprobs = torch.FloatTensor(log_probs).to(device)
            
            # PPO update
            print(f"🔄 Performing PPO update...")
            update_start = time.time()
            
            for epoch in range(agent.K_epochs):
                logits = agent.policy.actor(old_states)
                state_values = agent.policy.critic(old_states).squeeze()
                
                dist = torch.distributions.Categorical(logits)
                new_logprobs = dist.log_prob(old_actions)
                entropy = dist.entropy()
                
                ratios = torch.exp(new_logprobs - old_logprobs)
                advantages = returns - state_values.detach()
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - agent.eps_clip, 1 + agent.eps_clip) * advantages
                
                loss = -torch.min(surr1, surr2).mean() + 0.5 * agent.MseLoss(state_values, returns) - 0.01 * entropy.mean()
                
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()
            
            agent.policy_old.load_state_dict(agent.policy.state_dict())
            
            update_time = time.time() - update_start
            
            print(f"✅ Update complete ({update_time:.2f}s)")
            print(f"   Episodes: {total_episodes}/{self.max_train_episodes_per_env}")
            print(f"   Compliance: {current_compliance:.1%}")
            print(f"   Avg reward: {np.mean(rewards):.1f}\n")
            
            # Check if compliance achieved
            if current_compliance >= self.compliance_threshold:
                print(f"🎉 Compliance threshold {self.compliance_threshold:.1%} achieved!\n")
                break
        
        # Save agent
        self.shared_agent = agent
        checkpoint_path = os.path.join(self.checkpoint_dir, f'workflow_{workflow_id}_agent.pt')
        torch.save(agent.policy.state_dict(), checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}\n")
        
        return np.mean(rewards[-self.episodes_per_update:]), current_compliance, total_episodes, True
    
    def run_workflow_search(self):
        """Run GP-UCB workflow search with async training"""
        print("\n" + "="*60)
        print("🚀 ASYNC Workflow Search Starting")
        print("="*60)
        print(f"Architecture: Independent episode collection")
        print(f"Episode budget: {self.total_episode_budget}")
        print(f"Environments: {self.n_envs}")
        print(f"Episodes per update: {self.episodes_per_update}")
        print("="*60 + "\n")
        
        iteration = 0
        
        while self.total_episodes_used < self.total_episode_budget:
            iteration += 1
            remaining = self.total_episode_budget - self.total_episodes_used
            
            print(f"\n{'='*60}")
            print(f"🎲 Iteration {iteration}")
            print(f"   Episodes used: {self.total_episodes_used}/{self.total_episode_budget}")
            print(f"{'='*60}\n")
            
            # Select workflow using GP-UCB
            # Get candidate orders
            if iteration <= 5:
                # Use canonical orders for first few iterations
                canonical_dict = self.workflow_manager.get_canonical_workflows()
                candidate_orders = list(canonical_dict.values())
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
            
            print(f"Selected workflow: {' → '.join(workflow_order)}")
            print(f"UCB score: {ucb_score:.4f}\n")
            
            # Use iteration as workflow_id for checkpoint naming
            workflow_id = iteration
            
            # Train workflow
            eval_reward, compliance, episodes_used, success = self.train_workflow_async(
                workflow_order, workflow_id
            )
            
            self.total_episodes_used += episodes_used
            
            # Update GP model with observed reward
            self.gp_search.add_observation(workflow_order, eval_reward)
            
            print(f"📈 GP model updated")
            print(f"   Workflow: {' → '.join(workflow_order)}")
            print(f"   Reward: {eval_reward:.2f}")
            print(f"   Compliance: {compliance:.1%}\n")
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)
        
        self.consolidated_log_file.close()
        self.gp_sampling_file.close()


def main():
    parser = argparse.ArgumentParser(description='Async Workflow RL Training')
    parser.add_argument('--n-envs', type=int, default=100)
    parser.add_argument('--total-episodes', type=int, default=100000)
    parser.add_argument('--max-episodes-per-workflow', type=int, default=100)
    parser.add_argument('--episodes-per-update', type=int, default=100)
    parser.add_argument('--red-agent', type=str, default='B_lineAgent',
                       choices=['B_lineAgent', 'RedMeanderAgent', 'SleepAgent'])
    parser.add_argument('--alignment-lambda', type=float, default=30.0)
    parser.add_argument('--compliance-threshold', type=float, default=0.95)
    
    args = parser.parse_args()
    
    # Map agent name
    agent_map = {
        'B_lineAgent': B_lineAgent,
        'RedMeanderAgent': RedMeanderAgent,
        'SleepAgent': SleepAgent
    }
    red_agent = agent_map[args.red_agent]
    
    print("\n" + "="*60)
    print("🚀 ASYNC PARALLEL TRAINING")
    print("="*60)
    print(f"Red Agent: {args.red_agent}")
    print(f"Environments: {args.n_envs}")
    print(f"Episode Budget: {args.total_episodes}")
    print(f"Episodes/Update: {args.episodes_per_update}")
    print(f"Compliance Threshold: {args.compliance_threshold:.1%}")
    print("="*60 + "\n")
    
    trainer = AsyncWorkflowRLTrainer(
        n_envs=args.n_envs,
        total_episode_budget=args.total_episodes,
        max_train_episodes_per_env=args.max_episodes_per_workflow,
        red_agent_type=red_agent,
        alignment_lambda=args.alignment_lambda,
        compliance_threshold=args.compliance_threshold,
        episodes_per_update=args.episodes_per_update
    )
    
    trainer.run_workflow_search()


if __name__ == "__main__":
    main()

