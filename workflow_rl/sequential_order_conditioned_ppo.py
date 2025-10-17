"""
Sequential Order-Conditioned PPO for single environment training
Collects episodes sequentially before updates
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from collections import deque

from workflow_rl.order_conditioned_ppo import OrderConditionedActorCritic


class RolloutBuffer:
    """Buffer for storing sequential episode data"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def __len__(self):
        return len(self.states)
    
    def get_tensors(self):
        """Convert lists to tensors for training"""
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.dones),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.values)
        )


class SequentialOrderConditionedPPO:
    """PPO agent for sequential single-environment training"""
    
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 workflow_dim: int,
                 alignment_lambda: float = 0.01,
                 episodes_per_update: int = 100,
                 K_epochs: int = 4,
                 eps_clip: float = 0.2,
                 gamma: float = 0.99,
                 lr: float = 0.002):
        """
        Initialize sequential PPO agent
        
        Args:
            episodes_per_update: Number of episodes to collect before update
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.workflow_dim = workflow_dim
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.alignment_lambda = alignment_lambda
        self.episodes_per_update = episodes_per_update
        self.gamma = gamma
        
        # Current workflow
        self.current_workflow = None
        
        # Policy networks
        self.policy = OrderConditionedActorCritic(
            input_dims=obs_dim,
            n_actions=action_dim,
            order_dims=workflow_dim
        )
        
        # Old policy for PPO
        self.policy_old = OrderConditionedActorCritic(
            input_dims=obs_dim,
            n_actions=action_dim,
            order_dims=workflow_dim
        )
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        
        # Experience buffer
        self.buffer = RolloutBuffer()
        
        # Tracking metrics
        self.episode_fixes = 0
        self.episode_compliant_actions = 0
        self.episode_total_actions = 0
        
        # History for compliance calculation
        self.compliance_history = deque(maxlen=100)
        
        # Unit priorities for compliance tracking
        self.unit_priorities = {
            'defender': 5,
            'enterprise': 4,
            'op_server': 3,
            'op_host': 2,
            'user': 1
        }
    
    def set_workflow(self, workflow_id: int):
        """Set the current workflow for conditioning"""
        self.current_workflow = workflow_id
    
    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Get action for a single state"""
        if self.current_workflow is None:
            raise ValueError("Workflow not set")
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        workflow_tensor = torch.tensor([self.current_workflow], dtype=torch.long)
        
        with torch.no_grad():
            action_probs, value = self.policy_old(state_tensor, workflow_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, done, log_prob, value):
        """Store a transition in the buffer"""
        self.buffer.add(state, action, reward, done, log_prob, value)
        
        # Track for compliance (simplified - you'd need actual env info)
        self.episode_total_actions += 1
        if done:
            # Reset episode tracking
            self.episode_fixes = 0
            self.episode_compliant_actions = 0
            self.episode_total_actions = 0
    
    def compute_alignment_reward(self, env, info) -> float:
        """
        Compute alignment reward based on fixes
        
        Note: This is simplified - in practice you'd need proper env interaction
        """
        alignment_bonus = 0.0
        
        # Check if any hosts were fixed (simplified logic)
        if 'Blue' in info and 'action_success' in info['Blue']:
            if info['Blue']['action_success']:
                # Assume successful actions might be fixes
                self.episode_fixes += 1
                alignment_bonus = self.alignment_lambda * 10  # Bonus for fixes
                self.episode_compliant_actions += 1
        
        return alignment_bonus
    
    def get_compliance_rate(self) -> float:
        """Get current compliance rate"""
        if self.episode_total_actions == 0:
            return 0.0
        return self.episode_compliant_actions / self.episode_total_actions
    
    def get_episode_fixes(self) -> int:
        """Get number of fixes in current episode"""
        return self.episode_fixes
    
    def compute_discounted_rewards(self) -> List[float]:
        """Compute discounted rewards for the buffer"""
        _, _, rewards, dones, _, _ = self.buffer.get_tensors()
        
        discounted_rewards = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        return torch.FloatTensor(discounted_rewards)
    
    def update(self):
        """Perform PPO update on collected episodes"""
        if len(self.buffer) == 0:
            print("Warning: Empty buffer, skipping update")
            return
        
        # Get data from buffer
        states, actions, rewards, dones, old_log_probs, old_values = self.buffer.get_tensors()
        
        # Create workflow tensor
        workflows = torch.tensor([self.current_workflow] * len(states), dtype=torch.long)
        
        # Compute discounted rewards
        discounted_rewards = self.compute_discounted_rewards()
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Advantages
        advantages = discounted_rewards - old_values
        
        # PPO epochs
        for _ in range(self.K_epochs):
            # Get new action probabilities and values
            action_probs, values = self.policy(states, workflows)
            dist = torch.distributions.Categorical(action_probs)
            
            # New log probabilities
            new_log_probs = dist.log_prob(actions)
            
            # Entropy for exploration
            entropy = dist.entropy()
            
            # Ratio for PPO
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = self.mse_loss(values.squeeze(), discounted_rewards)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'workflow': self.current_workflow
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_workflow = checkpoint['workflow']
