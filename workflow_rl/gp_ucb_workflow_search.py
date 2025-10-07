#!/usr/bin/env python3
"""
Gaussian Process Upper Confidence Bound for Workflow Search
Searches over the space of unit type priority orderings
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from scipy.stats import norm
from typing import List, Tuple, Dict, Optional
import random

from workflow_rl.unit_type_workflow import UnitTypeWorkflow


class GPUCBWorkflowSearch:
    """
    GP-UCB search over workflow space
    """
    
    def __init__(self, 
                 beta: float = 2.0,
                 kernel_length_scale: float = 0.5,
                 kernel_nu: float = 2.5,
                 noise_level: float = 0.1):
        
        self.beta = beta  # Exploration parameter
        self.noise_level = noise_level
        
        # Workflow representation
        self.workflow_manager = UnitTypeWorkflow()
        
        # Gaussian Process
        kernel = Matern(length_scale=kernel_length_scale, nu=kernel_nu)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise_level,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        # History
        self.observed_workflows = []  # List of workflow orders
        self.observed_vectors = []    # List of 8D vectors
        self.observed_rewards = []    # List of rewards
        self.observation_counts = {}  # Track revisits
        
        # Initialize with canonical workflows
        self._initialize_canonical()
    
    def _initialize_canonical(self):
        """Start with canonical workflows"""
        canonical = self.workflow_manager.get_canonical_workflows()
        
        for name, order in canonical.items():
            vector = self.workflow_manager.order_to_vector(order)
            self.observed_workflows.append(order)
            self.observed_vectors.append(vector)
            self.observed_rewards.append(0.0)  # Will be updated
            self.observation_counts[tuple(order)] = 0
    
    def select_next_workflow(self) -> Tuple[List[str], np.ndarray]:
        """
        Select next workflow to evaluate using GP-UCB
        
        Returns:
            (workflow_order, workflow_vector)
        """
        
        # If we haven't evaluated all canonical workflows yet
        if len(self.observed_rewards) < len(self.observed_workflows):
            idx = len(self.observed_rewards) - 1
            return self.observed_workflows[idx], self.observed_vectors[idx]
        
        # Fit GP on observed data
        if len(self.observed_rewards) > 0:
            X = np.array(self.observed_vectors)
            y = np.array(self.observed_rewards)
            self.gp.fit(X, y)
        
        # Generate candidate workflows
        candidates = self._generate_candidates(n_candidates=500)
        
        # Compute UCB for each candidate
        best_ucb = -float('inf')
        best_workflow = None
        best_vector = None
        
        for workflow_order in candidates:
            vector = self.workflow_manager.order_to_vector(workflow_order)
            ucb = self._compute_ucb(vector, workflow_order)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_workflow = workflow_order
                best_vector = vector
        
        return best_workflow, best_vector
    
    def _generate_candidates(self, n_candidates: int) -> List[List[str]]:
        """Generate candidate workflows for evaluation"""
        
        candidates = []
        
        # 1. Add some random permutations (exploration)
        for _ in range(n_candidates // 3):
            candidates.append(self.workflow_manager.sample_order('uniform'))
        
        # 2. Add variations of good workflows (exploitation)
        if self.observed_rewards:
            # Get top 20% workflows
            sorted_indices = np.argsort(self.observed_rewards)[::-1]
            n_top = max(1, len(sorted_indices) // 5)
            
            for _ in range(n_candidates // 3):
                # Pick a good workflow
                idx = random.choice(sorted_indices[:n_top])
                base_workflow = self.observed_workflows[idx].copy()
                
                # Make small perturbation (swap two adjacent types)
                if len(base_workflow) > 1:
                    i = random.randint(0, len(base_workflow) - 2)
                    base_workflow[i], base_workflow[i+1] = base_workflow[i+1], base_workflow[i]
                
                candidates.append(base_workflow)
        
        # 3. Add strategy-based samples
        strategies = ['critical_first', 'user_last', 'balanced']
        for _ in range(n_candidates // 3):
            strategy = random.choice(strategies)
            candidates.append(self.workflow_manager.sample_order(strategy))
        
        # Fill remaining with random
        while len(candidates) < n_candidates:
            candidates.append(self.workflow_manager.sample_order('uniform'))
        
        return candidates[:n_candidates]
    
    def _compute_ucb(self, vector: np.ndarray, workflow_order: List[str]) -> float:
        """
        Compute Upper Confidence Bound for a workflow
        
        UCB = μ + β * σ
        """
        
        if len(self.observed_rewards) == 0:
            return np.random.random()
        
        # Get GP prediction
        mean, std = self.gp.predict(vector.reshape(1, -1), return_std=True)
        
        # Adjust beta based on revisit count
        workflow_key = tuple(workflow_order)
        revisit_count = self.observation_counts.get(workflow_key, 0)
        
        # Reduce exploration for heavily visited workflows
        adjusted_beta = self.beta / (1 + 0.5 * revisit_count)
        
        # UCB calculation
        ucb = mean[0] + adjusted_beta * std[0]
        
        return ucb
    
    def update(self, workflow_order: List[str], workflow_vector: np.ndarray, 
               reward: float, compliance_rate: float = 1.0):
        """
        Update GP with new observation
        
        Args:
            workflow_order: The evaluated workflow
            workflow_vector: The 8D vector representation
            reward: Average reward achieved
            compliance_rate: How well the policy followed the workflow
        """
        
        # Adjust reward based on compliance
        # If the policy didn't follow the workflow, the sample is less informative
        adjusted_reward = reward * (0.7 + 0.3 * compliance_rate)
        
        workflow_key = tuple(workflow_order)
        
        # Check if this workflow was already observed
        existing_idx = None
        for i, w in enumerate(self.observed_workflows):
            if w == workflow_order:
                existing_idx = i
                break
        
        if existing_idx is not None:
            # Update existing observation (running average)
            count = self.observation_counts[workflow_key]
            self.observed_rewards[existing_idx] = (
                (self.observed_rewards[existing_idx] * count + adjusted_reward) / 
                (count + 1)
            )
            self.observation_counts[workflow_key] += 1
        else:
            # Add new observation
            self.observed_workflows.append(workflow_order)
            self.observed_vectors.append(workflow_vector)
            self.observed_rewards.append(adjusted_reward)
            self.observation_counts[workflow_key] = 1
    
    def get_best_workflow(self) -> Tuple[List[str], float]:
        """
        Return the best workflow found so far
        
        Returns:
            (best_workflow_order, best_reward)
        """
        
        if not self.observed_rewards:
            # Return a default workflow
            return ['defender', 'op_server', 'enterprise', 'op_host', 'user'], 0.0
        
        best_idx = np.argmax(self.observed_rewards)
        return self.observed_workflows[best_idx], self.observed_rewards[best_idx]
    
    def get_statistics(self) -> Dict:
        """Get search statistics"""
        
        if not self.observed_rewards:
            return {
                'n_evaluated': 0,
                'best_reward': 0.0,
                'mean_reward': 0.0,
                'std_reward': 0.0
            }
        
        return {
            'n_evaluated': len(self.observed_rewards),
            'n_unique': len(set(tuple(w) for w in self.observed_workflows)),
            'best_reward': max(self.observed_rewards),
            'worst_reward': min(self.observed_rewards),
            'mean_reward': np.mean(self.observed_rewards),
            'std_reward': np.std(self.observed_rewards),
            'most_visited': max(self.observation_counts.values()) if self.observation_counts else 0
        }
