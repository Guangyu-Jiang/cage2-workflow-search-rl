"""
Simple Sequential Order-Conditioned PPO for single environment training
Simplified version that works with existing components
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque

from workflow_rl.order_conditioned_ppo import OrderConditionedActorCritic, Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleSequentialPPO:
    """Simplified PPO for sequential single-environment training"""
    
    def __init__(self,
                 input_dims: int = 52,
                 lr: float = 0.002,
                 betas: List[float] = [0.9, 0.990],
                 gamma: float = 0.99,
                 K_epochs: int = 4,
                 eps_clip: float = 0.2,
                 workflow_order: List[str] = None,
                 workflow_manager = None,
                 alignment_lambda: float = 30.0,
                 episodes_per_update: int = 100):
        """
        Initialize simple sequential PPO
        """
        self.input_dims = input_dims
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.alignment_lambda = alignment_lambda
        self.episodes_per_update = episodes_per_update
        
        # Workflow setup
        self.workflow_order = workflow_order
        self.workflow_manager = workflow_manager
        
        # Convert order to encoding
        if workflow_order and workflow_manager:
            self.order_encoding = torch.FloatTensor(
                workflow_manager.order_to_onehot(workflow_order)
            ).to(device)
        else:
            # Default encoding
            self.order_encoding = torch.zeros(25).to(device)
        
        # Initialize networks
        self.policy = OrderConditionedActorCritic(
            input_dims=input_dims,
            n_actions=145,  # Full action space
            order_dims=25    # 5x5 one-hot encoding
        ).to(device)
        
        self.policy_old = OrderConditionedActorCritic(
            input_dims=input_dims,
            n_actions=145,
            order_dims=25
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.MseLoss = nn.MSELoss()
        
        # Memory
        self.memory = Memory()
        
        # Tracking
        self.total_fix_actions = 0
        self.total_compliant_actions = 0
        self.compliance_history = deque(maxlen=100)
        
        # Unit priorities
        self.unit_priorities = {
            'defender': 5,
            'enterprise': 4,
            'op_server': 3,
            'op_host': 2,
            'user': 1
        }
        
        # Episode tracking
        self.episode_count = 0
    
    def set_workflow(self, workflow_order: List[str]):
        """Set new workflow order"""
        self.workflow_order = workflow_order
        if workflow_order and self.workflow_manager:
            self.order_encoding = torch.FloatTensor(
                self.workflow_manager.order_to_onehot(workflow_order)
            ).to(device)
    
    def get_action(self, observation):
        """Get action for single observation"""
        state = torch.FloatTensor(observation).to(device)
        
        # Use old policy for sampling during training
        action = self.policy_old.act(
            state, self.order_encoding, self.memory, deterministic=False
        )
        
        return action
    
    def compute_alignment_reward(self, action: int, true_state: Dict,
                                prev_true_state: Optional[Dict] = None) -> float:
        """
        Compute alignment reward based on fixes
        Simplified version
        """
        if not self.workflow_order:
            return 0.0
        
        # Check if action is a fix (action IDs for restore: 132-144)
        is_fix = (action >= 132 and action <= 144)
        
        alignment_bonus = 0.0
        if is_fix:
            self.total_fix_actions += 1
            # Simplified: give bonus for any fix
            alignment_bonus = self.alignment_lambda * 10
            
            # Track compliance (simplified)
            self.total_compliant_actions += 1
        
        return alignment_bonus
    
    def get_compliance_rate(self) -> float:
        """Get current compliance rate"""
        if self.total_fix_actions == 0:
            return 0.0
        return self.total_compliant_actions / max(self.total_fix_actions, 1)
    
    def clear_memory(self):
        """Clear trajectory memory"""
        self.memory.clear_memory()
    
    def update(self):
        """Perform PPO update"""
        if len(self.memory.states) == 0:
            print("Warning: Empty memory, skipping update")
            return
            
        # Get data from memory
        # States are already augmented (state + order encoding) tensors, so stack them
        old_augmented_states = torch.stack([s.squeeze() if s.dim() > 1 else s for s in self.memory.states]).to(device)
        # Actions and logprobs might be tensors too - ensure they are 1D
        old_actions = torch.stack([a if torch.is_tensor(a) else torch.tensor(a) for a in self.memory.actions]).squeeze(-1).to(device)
        old_logprobs = torch.stack([lp if torch.is_tensor(lp) else torch.tensor(lp) for lp in self.memory.logprobs]).squeeze(-1).to(device)
        
        # Calculate rewards to go
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # PPO update - states are already augmented, no need to concatenate again
        for _ in range(self.K_epochs):
            # Forward pass through policy - states already include order encoding
            logits = self.policy.actor(old_augmented_states)
            state_values = self.policy.critic(old_augmented_states).squeeze()
            
            dist = torch.distributions.Categorical(logits)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Compute loss (all components should be 1D tensors of same size)
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.memory.clear_memory()
    
    def save(self, checkpoint_path: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'workflow_order': self.workflow_order
        }, checkpoint_path)
    
    def load(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.workflow_order = checkpoint.get('workflow_order', None)
