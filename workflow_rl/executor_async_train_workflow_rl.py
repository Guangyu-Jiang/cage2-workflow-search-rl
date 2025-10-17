"""
ProcessPoolExecutor-based Async Training
TRUE async using Python's built-in concurrent.futures - no external dependencies!
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

from workflow_rl.order_based_workflow import OrderBasedWorkflow
from workflow_rl.gp_ucb_order_search import GPUCBOrderSearch


def collect_single_episode(worker_id: int, scenario_path: str, red_agent_type,
                           policy_weights_cpu: Dict, workflow_encoding: np.ndarray,
                           workflow_order: List[str], alignment_lambda: float = 30.0,
                           max_steps: int = 100):
    """
    Worker function that collects ONE complete episode.
    Runs in a separate process - completely independent!
    
    This function is submitted to ProcessPoolExecutor.
    """
    # Create environment in worker process
    cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    # Action ID to host type mapping (from CAGE2 action space)
    action_to_host_type = {
        # Remove actions
        15: 'defender', 16: 'enterprise', 17: 'enterprise', 18: 'enterprise',
        19: 'op_host', 20: 'op_host', 21: 'op_host', 22: 'op_server',
        23: 'user', 24: 'user', 25: 'user', 26: 'user', 27: 'user',
        # Restore actions  
        132: 'defender', 133: 'enterprise', 134: 'enterprise', 135: 'enterprise',
        136: 'op_host', 137: 'op_host', 138: 'op_host', 139: 'op_server',
        140: 'user', 141: 'user', 142: 'user', 143: 'user', 144: 'user'
    }
    
    # Track which types have been fixed
    fixed_types = set()
    
    # Reconstruct policy from weights (on CPU)
    import torch
    from workflow_rl.order_conditioned_ppo import OrderConditionedActorCritic
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.get_action_space('Blue')
    
    device_local = torch.device('cpu')  # Workers use CPU
    policy = OrderConditionedActorCritic(obs_dim, action_dim, order_dims=25).to(device_local)
    policy.load_state_dict(policy_weights_cpu)
    policy.eval()
    
    workflow_tensor = torch.FloatTensor(workflow_encoding).to(device_local)
    
    # Collect episode
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_dones = []
    episode_log_probs = []
    episode_values = []
    
    # Compliance tracking
    total_fix_actions = 0
    compliant_fix_actions = 0
    prev_true_state = None
    
    state = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(max_steps):
        # Prepare state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device_local)
        order_tensor = workflow_tensor.unsqueeze(0)
        augmented_state = torch.cat([state_tensor, order_tensor], dim=1)
        
        # Get action from policy
        with torch.no_grad():
            logits = policy.actor(augmented_state)
            dist = torch.distributions.Categorical(logits)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
            value = policy.critic(augmented_state)
        
        action = action_tensor.item()
        
        # Execute
        next_state, reward, done, info = env.step(action)
        
        # Track compliance for fix actions using action-to-host-type mapping
        # (Remove: 15-27, Restore: 132-144)
        if action in action_to_host_type:
            total_fix_actions += 1
            target_type = action_to_host_type[action]
            
            # Check if this violates workflow order
            # Violation = fixing this type before higher priority types
            target_priority = workflow_order.index(target_type)
            
            violation = False
            for priority_idx in range(target_priority):
                priority_type = workflow_order[priority_idx]
                if priority_type not in fixed_types:
                    # Higher priority type not fixed yet - violation!
                    violation = True  # ← FIXED: was False!
                    break
            
            if not violation:
                compliant_fix_actions += 1
            
            # Mark this type as fixed
            fixed_types.add(target_type)
        
        # Store
        episode_states.append(augmented_state.squeeze().numpy())
        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_dones.append(done)
        episode_log_probs.append(log_prob.item())
        episode_values.append(value.item())
        
        total_reward += reward
        steps += 1
        state = next_state
        
        if done:
            break
    
    # Compute episode compliance (same as synchronous version)
    if total_fix_actions > 0:
        episode_compliance = compliant_fix_actions / total_fix_actions
    else:
        # No fix actions - return 0.5 (neutral) to match synchronous version
        episode_compliance = 0.5
    
    return {
        'worker_id': worker_id,
        'states': episode_states,
        'actions': episode_actions,
        'rewards': episode_rewards,
        'dones': episode_dones,
        'log_probs': episode_log_probs,
        'values': episode_values,
        'total_reward': total_reward,
        'steps': steps,
        'compliance': episode_compliance,
        'total_fix_actions': total_fix_actions,
        'compliant_fix_actions': compliant_fix_actions
    }


class ExecutorAsyncWorkflowRLTrainer:
    """
    Async trainer using ProcessPoolExecutor.
    Workers collect full episodes independently!
    """
    
    def __init__(self, 
                 n_workers: int = 100,
                 total_episode_budget: int = 100000,
                 max_train_episodes_per_workflow: int = 500,
                 episodes_per_update: int = 100,
                 scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                 red_agent_type=RedMeanderAgent,
                 alignment_lambda: float = 30.0,
                 compliance_threshold: float = 0.95):
        
        self.n_workers = n_workers
        self.total_episode_budget = total_episode_budget
        self.max_train_episodes_per_workflow = max_train_episodes_per_workflow
        self.episodes_per_update = episodes_per_update
        self.scenario_path = scenario_path
        self.red_agent_type = red_agent_type
        self.alignment_lambda = alignment_lambda
        self.compliance_threshold = compliance_threshold
        
        self.total_episodes_used = 0
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pid = os.getpid()
        self.experiment_name = f"exp_executor_async_{timestamp}_{pid}"
        self.checkpoint_dir = os.path.join("logs", self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"\nStarting experiment with PID: {pid}")
        
        # Create ProcessPoolExecutor
        self.executor = ProcessPoolExecutor(max_workers=n_workers)
        
        # Initialize workflow manager and GP-UCB
        self.workflow_manager = OrderBasedWorkflow()
        self.gp_search = GPUCBOrderSearch(beta=2.0)
        
        # Shared agent
        self.shared_agent = None
        
        # Initialize logging
        self._init_logging()
    
    def _init_logging(self):
        """Initialize CSV logging"""
        # Save config
        config_file = os.path.join(self.checkpoint_dir, 'experiment_config.json')
        print(f"Experiment config: {config_file}")
        
        # Training log
        log_filename = os.path.join(self.checkpoint_dir, "training_log.csv")
        self.log_file = open(log_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'Workflow_ID', 'Workflow_Order', 'Update', 'Episodes',
            'Env_Reward', 'Alignment_Bonus', 'Total_Reward', 'Compliance',
            'Avg_Fixes', 'Collection_Time', 'Update_Time'
        ])
        self.log_file.flush()
        print(f"Experiment directory: {self.checkpoint_dir}")
        print(f"Training log: {log_filename}")
        
        # GP log
        gp_log_filename = os.path.join(self.checkpoint_dir, "gp_sampling_log.csv")
        self.gp_log_file = open(gp_log_filename, 'w', newline='')
        self.gp_csv_writer = csv.writer(self.gp_log_file)
        self.gp_csv_writer.writerow(['Iteration', 'Workflow', 'UCB_Score', 'Reward'])
        self.gp_log_file.flush()
        print(f"GP-UCB sampling log: {gp_log_filename}")
    
    def collect_episodes_async(self, agent, workflow_order: List[str], 
                               n_episodes: int) -> Tuple:
        """
        Collect episodes asynchronously using ProcessPoolExecutor.
        Workers run INDEPENDENTLY - TRUE async!
        """
        workflow_encoding = self.workflow_manager.order_to_onehot(workflow_order)
        
        # Move policy weights to CPU for workers
        policy_weights_cpu = {k: v.cpu() for k, v in agent.policy_old.state_dict().items()}
        
        print(f"📦 Submitting {n_episodes} episode collection tasks to {self.n_workers} workers...")
        print(f"   Workers run INDEPENDENTLY (ProcessPoolExecutor async!)")
        
        start_time = time.time()
        
        # Submit episode collection tasks
        futures = []
        for i in range(n_episodes):
            # Each task collects ONE episode
            # Distribute across workers using worker_id % n_workers
            future = self.executor.submit(
                collect_single_episode,
                worker_id=i,
                scenario_path=self.scenario_path,
                red_agent_type=self.red_agent_type,
                policy_weights_cpu=policy_weights_cpu,
                workflow_encoding=workflow_encoding,
                workflow_order=workflow_order,
                alignment_lambda=self.alignment_lambda,
                max_steps=100
            )
            futures.append(future)
        
        print(f"✅ {n_episodes} tasks submitted! Collecting as they complete...")
        
        # Collect results as they complete (async!)
        completed_episodes = []
        collected = 0
        
        for future in as_completed(futures):
            try:
                result = future.result()
                completed_episodes.append(result)
                collected += 1
                
                if collected % 10 == 0 or collected == n_episodes:
                    elapsed = time.time() - start_time
                    rate = collected / elapsed
                    print(f"  {collected}/{n_episodes} episodes ({rate:.1f} eps/sec)")
                    
            except Exception as e:
                print(f"⚠️  Episode collection failed: {e}")
                continue
        
        elapsed = time.time() - start_time
        rate = len(completed_episodes) / elapsed
        print(f"✅ Collected {len(completed_episodes)} episodes in {elapsed:.1f}s ({rate:.1f} eps/sec)")
        print(f"   Workers ran INDEPENDENTLY (true async!)\n")
        
        # Aggregate episode data
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []
        compliances = []
        fix_counts = []
        episode_rewards = []
        
        for episode in completed_episodes:
            all_states.extend(episode['states'])
            all_actions.extend(episode['actions'])
            all_rewards.extend(episode['rewards'])
            all_dones.extend(episode['dones'])
            all_log_probs.extend(episode['log_probs'])
            all_values.extend(episode['values'])
            compliances.append(episode['compliance'])
            fix_counts.append(episode['total_fix_actions'])
            episode_rewards.append(episode['total_reward'])
        
        return (np.array(all_states), np.array(all_actions), np.array(all_rewards),
                np.array(all_dones), np.array(all_log_probs), np.array(all_values),
                compliances, fix_counts, episode_rewards, elapsed)
    
    def train_workflow_async(self, workflow_order: List[str], workflow_id: int):
        """Train workflow using ProcessPoolExecutor async collection"""
        print(f"\n{'='*60}")
        print(f"Training with workflow: {' → '.join(workflow_order)}")
        print(f"Goal: Train until compliance >= {self.compliance_threshold:.1%}")
        print(f"Using {self.n_workers} async workers (ProcessPoolExecutor)")
        print(f"{'='*60}")
        
        # Create or inherit agent
        if self.shared_agent is None:
            print("  Creating new agent (first workflow)")
            cyborg = CybORG(self.scenario_path, 'sim', agents={'Red': self.red_agent_type})
            env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
            obs_dim = env.observation_space.shape[0]
            
            from workflow_rl.parallel_order_conditioned_ppo import ParallelOrderConditionedPPO, device
            
            agent = ParallelOrderConditionedPPO(
                input_dims=obs_dim,
                n_envs=self.n_workers,
                workflow_order=workflow_order,
                workflow_manager=self.workflow_manager,
                alignment_lambda=self.alignment_lambda,
                K_epochs=4,
                eps_clip=0.2,
                gamma=0.99,
                lr=0.002
            )
        else:
            print("  Inheriting policy from previous workflow")
            from workflow_rl.parallel_order_conditioned_ppo import ParallelOrderConditionedPPO, device
            
            obs_dim = self.shared_agent.input_dims
            agent = ParallelOrderConditionedPPO(
                input_dims=obs_dim,
                n_envs=self.n_workers,
                workflow_order=workflow_order,
                workflow_manager=self.workflow_manager,
                alignment_lambda=self.alignment_lambda,
                K_epochs=4,
                eps_clip=0.2,
                gamma=0.99,
                lr=0.002
            )
            agent.policy.load_state_dict(self.shared_agent.policy.state_dict())
            agent.policy_old.load_state_dict(self.shared_agent.policy_old.state_dict())
        
        # Training loop
        total_episodes = 0
        update_num = 0
        all_sampling_times = []
        all_update_times = []
        
        while total_episodes < self.max_train_episodes_per_workflow:
            
            # Collect episodes asynchronously
            (states, actions, rewards, dones, log_probs, values,
             compliances, fix_counts, episode_rewards, collection_time) = self.collect_episodes_async(
                agent, workflow_order, self.episodes_per_update
            )
            
            update_num += 1
            total_episodes += self.episodes_per_update
            
            # Calculate metrics per episode
            avg_env_reward = np.mean(episode_rewards)  # Raw episode rewards
            avg_compliance = np.mean(compliances)
            avg_fixes = np.mean(fix_counts)
            
            # Estimate alignment bonus from compliance
            alignment_bonus = self.alignment_lambda * avg_compliance
            total_reward = avg_env_reward + alignment_bonus
            
            # PPO update
            update_start = time.time()
            
            # Compute returns
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
            all_sampling_times.append(collection_time)
            all_update_times.append(update_time)
            
            # Calculate average timing statistics
            avg_sampling_time = np.mean(all_sampling_times)
            avg_update_time = np.mean(all_update_times)
            update_ratio = avg_update_time / (avg_sampling_time + avg_update_time) * 100
            
            # Print progress in same format as synchronous version
            print(f"\n  Update {update_num}: Episodes: {total_episodes} total")
            print(f"    Env Reward/Episode: {avg_env_reward:.2f}")
            print(f"    Total Reward/Episode: {total_reward:.2f}")
            print(f"    Alignment Bonus (episode-end): {alignment_bonus:+.2f}")
            print(f"    Compliance: {avg_compliance:.2%}")
            print(f"    Avg Fixes/Episode: {avg_fixes:.1f}")
            print(f"    ⏱️ Timing: Sampling={collection_time:.2f}s, Update={update_time:.2f}s")
            print(f"    📊 Average: Sampling={avg_sampling_time:.2f}s, Update={avg_update_time:.2f}s (PPO takes {update_ratio:.1f}% of time)")
            
            # Log - separate environment reward, alignment bonus, and total reward
            self.csv_writer.writerow([
                workflow_id, ' → '.join(workflow_order), update_num, total_episodes,
                f"{avg_env_reward:.2f}",           # Original environment reward
                f"{alignment_bonus:.2f}",          # Compliance/alignment reward
                f"{total_reward:.2f}",             # Customized reward (sum)
                f"{avg_compliance:.4f}",           # Compliance rate
                f"{avg_fixes:.1f}",                # Average fixes per episode
                f"{collection_time:.2f}",          # Collection time
                f"{update_time:.2f}"               # Update time
            ])
            self.log_file.flush()
            
            # Check compliance
            if avg_compliance >= self.compliance_threshold:
                print(f"\n  ✓ Compliance threshold achieved!")
                print(f"    Episodes trained: {total_episodes}")
                print(f"    Latest compliance: {avg_compliance:.2%}")
                break
        
        # Save agent
        self.shared_agent = agent
        checkpoint_path = os.path.join(self.checkpoint_dir, f'workflow_{workflow_id}_agent.pt')
        torch.save(agent.policy.state_dict(), checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}\n")
        
        return np.mean(rewards), avg_compliance, total_episodes
    
    def run_workflow_search(self):
        """Run GP-UCB workflow search with ProcessPoolExecutor async training"""
        print(f"\n{'='*60}")
        print(f"Compliance-Gated Workflow Search (Async)")
        print(f"{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Directory: {self.checkpoint_dir}")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Red Agent: {self.red_agent_type.__name__}")
        print(f"  Async Workers: {self.n_workers}")
        print(f"  Episode Budget: {self.total_episode_budget}")
        print(f"  Compliance Threshold: {self.compliance_threshold:.1%}")
        print(f"  Alignment Lambda: {self.alignment_lambda}")
        print(f"  Architecture: ProcessPoolExecutor (TRUE async!)")
        print(f"\nTraining Strategy:")
        print(f"  1. Train with alignment rewards until compliance >= {self.compliance_threshold:.1%}")
        print(f"  2. Use training environment rewards for GP-UCB")
        print(f"  3. Continue until {self.total_episode_budget} episodes are used")
        print(f"  4. Workers collect episodes INDEPENDENTLY (no sync barriers!)")
        print(f"{'='*60}")
        
        iteration = 0
        
        while self.total_episodes_used < self.total_episode_budget:
            iteration += 1
            
            print(f"\n{'='*50}")
            print(f"Iteration {iteration}")
            print(f"Episode Budget: {self.total_episodes_used}/{self.total_episode_budget} used")
            print(f"{'='*50}")
            
            # Select workflow
            if iteration <= 5:
                canonical_dict = self.workflow_manager.get_canonical_workflows()
                candidate_orders = list(canonical_dict.values())
            else:
                unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
                candidate_orders = []
                for _ in range(10):
                    perm = unit_types.copy()
                    np.random.shuffle(perm)
                    candidate_orders.append(perm)
            
            workflow_order, ucb_score, info = self.gp_search.select_next_order(
                candidate_orders, self.workflow_manager
            )
            
            # Print GP-UCB selection details (match synchronous format)
            print("\n" + "-"*50)
            print("GP-UCB Selection Details:")
            print(f"  Selected: {' → '.join(workflow_order)}")
            print(f"  UCB Score: {ucb_score:.3f}")
            
            if 'selection_method' in info:
                print(f"  Method: {info['selection_method']}")
                print(f"  Reason: {info['reason']}")
            else:
                if 'mean' in info:
                    print(f"  Mean Reward: {info['mean']:.2f}")
                if 'std' in info:
                    print(f"  Uncertainty (std): {info['std']:.3f}")
            print("-"*50)
            
            # Train workflow
            eval_reward, compliance, episodes_used = self.train_workflow_async(
                workflow_order, iteration
            )
            
            self.total_episodes_used += episodes_used
            
            # Update GP model
            self.gp_search.add_observation(workflow_order, eval_reward)
            
            # Log
            self.gp_csv_writer.writerow([iteration, ' → '.join(workflow_order), ucb_score, eval_reward])
            self.gp_log_file.flush()
            
            print(f"📈 GP model updated")
            print(f"   Reward: {eval_reward:.2f}")
            print(f"   Compliance: {compliance:.1%}\n")
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)
        
        self.log_file.close()
        self.gp_log_file.close()
        self.executor.shutdown(wait=True)
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


def main():
    parser = argparse.ArgumentParser(description='ProcessPoolExecutor Async Workflow RL Training')
    parser.add_argument('--n-workers', type=int, default=100)
    parser.add_argument('--total-episodes', type=int, default=100000)
    parser.add_argument('--max-episodes-per-workflow', type=int, default=5000)
    parser.add_argument('--episodes-per-update', type=int, default=100)
    parser.add_argument('--red-agent', type=str, default='B_lineAgent')
    parser.add_argument('--alignment-lambda', type=float, default=30.0)
    parser.add_argument('--compliance-threshold', type=float, default=0.95)
    
    args = parser.parse_args()
    
    agent_map = {
        'B_lineAgent': B_lineAgent,
        'RedMeanderAgent': RedMeanderAgent,
        'SleepAgent': SleepAgent
    }
    red_agent = agent_map[args.red_agent]
    
    print(f"\n{'='*60}")
    print(f"Configuration")
    print(f"{'='*60}")
    print(f"Red Agent: {args.red_agent} ({red_agent.__name__})")
    print(f"Async Workers: {args.n_workers}")
    print(f"Episode Budget: {args.total_episodes}")
    print(f"Max Episodes/Workflow: {args.max_episodes_per_workflow}")
    print(f"Alignment Lambda: {args.alignment_lambda}")
    print(f"Compliance Threshold: {args.compliance_threshold:.1%}")
    print(f"Architecture: ProcessPoolExecutor (Async)")
    print(f"Random Seed: 42")
    print(f"{'='*60}\n")
    
    trainer = ExecutorAsyncWorkflowRLTrainer(
        n_workers=args.n_workers,
        total_episode_budget=args.total_episodes,
        max_train_episodes_per_workflow=args.max_episodes_per_workflow,
        episodes_per_update=args.episodes_per_update,
        red_agent_type=red_agent,
        alignment_lambda=args.alignment_lambda,
        compliance_threshold=args.compliance_threshold
    )
    
    trainer.run_workflow_search()


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on some systems
    mp.set_start_method('spawn', force=True)
    main()

