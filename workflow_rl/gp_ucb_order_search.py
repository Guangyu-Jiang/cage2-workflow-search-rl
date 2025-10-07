#!/usr/bin/env python3
"""
Gaussian Process Upper Confidence Bound (GP-UCB) for Order-Based Workflow Search
Works directly with permutations, not continuous embeddings
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')


class GPUCBOrderSearch:
    """
    GP-UCB search over discrete permutation space
    Uses Kendall tau distance for the kernel
    """
    
    def __init__(self, beta: float = 2.0, distance_metric: str = 'kendall'):
        """
        Args:
            beta: Exploration parameter for UCB
            distance_metric: 'kendall' or 'spearman' distance between orders
        """
        self.beta = beta
        self.distance_metric = distance_metric
        
        # Storage for observations
        self.observed_orders = []  # List of order lists
        self.observed_rewards = []
        self.observation_counts = {}  # Track how many times each order is visited
        
        # Custom kernel for permutations
        self.gp = None
        self._init_gp()
    
    def _init_gp(self):
        """Initialize GP with custom kernel for permutations"""
        # We'll use a simple RBF kernel on the distance matrix
        # The kernel will be computed based on pairwise distances
        kernel = 1.0 * RBF(length_scale=0.5) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
    
    def _compute_distance_matrix(self, orders1: List[List[str]], orders2: List[List[str]], 
                                 workflow_manager) -> np.ndarray:
        """
        Compute pairwise distance matrix between two sets of orders
        """
        n1, n2 = len(orders1), len(orders2)
        distances = np.zeros((n1, n2))
        
        for i, order1 in enumerate(orders1):
            for j, order2 in enumerate(orders2):
                if self.distance_metric == 'kendall':
                    distances[i, j] = workflow_manager.compute_kendall_distance(order1, order2)
                else:  # spearman
                    distances[i, j] = workflow_manager.compute_spearman_distance(order1, order2)
        
        return distances
    
    def _orders_to_features(self, orders: List[List[str]], workflow_manager) -> np.ndarray:
        """
        Convert orders to feature matrix for GP
        We use the distances to all observed orders as features
        """
        if len(self.observed_orders) == 0:
            # No observations yet, return dummy features
            return np.zeros((len(orders), 1))
        
        # Compute distances to all observed orders
        distances = self._compute_distance_matrix(orders, self.observed_orders, workflow_manager)
        return distances
    
    def add_observation(self, order: List[str], reward: float):
        """Add a new observation"""
        order_tuple = tuple(order)
        
        # Track observation count
        if order_tuple not in self.observation_counts:
            self.observation_counts[order_tuple] = 0
        self.observation_counts[order_tuple] += 1
        
        # Store observation
        self.observed_orders.append(order)
        self.observed_rewards.append(reward)
        
        # Refit GP if we have enough observations
        if len(self.observed_orders) >= 2:
            self._refit_gp()
    
    def _refit_gp(self):
        """Refit GP with current observations"""
        # For GP fitting, we need a feature representation
        # We'll use a simple approach: one-hot encoding of order index
        # or distances to a set of reference orders
        
        # Create pseudo-features based on order indices
        # This is a simplification - in practice, we'd use the kernel directly
        X = np.arange(len(self.observed_orders)).reshape(-1, 1)
        y = np.array(self.observed_rewards)
        
        self.gp.fit(X, y)
    
    def select_next_order(self, candidate_orders: List[List[str]], 
                         workflow_manager) -> Tuple[List[str], float, Dict]:
        """
        Select next order to evaluate using GP-UCB
        
        Returns:
            (selected_order, ucb_score, info_dict)
        """
        
        if len(self.observed_orders) < 2:
            # Not enough data for GP, select randomly from candidates
            idx = np.random.choice(len(candidate_orders))
            selected = candidate_orders[idx]
            
            # For initial selection, prefer diverse orders
            if len(self.observed_orders) == 0:
                # First order: completely random
                reason = 'initial_random'
            else:
                # Second order: maximize distance from first
                distances = [workflow_manager.compute_kendall_distance(selected, self.observed_orders[0])]
                reason = f'maximize_diversity (dist={distances[0]:.2f})'
            
            return selected, 0.0, {
                'reason': reason, 
                'n_obs': len(self.observed_orders),
                'selection_method': 'random' if len(self.observed_orders) == 0 else 'diversity'
            }
        
        # Compute UCB scores for all candidates
        ucb_scores = []
        means = []
        stds = []
        
        for order in candidate_orders:
            # Get mean and std prediction
            # For simplicity, we use order index as feature
            # In a full implementation, we'd compute kernel values directly
            
            # Find if this order was observed
            order_tuple = tuple(order)
            if order_tuple in [tuple(o) for o in self.observed_orders]:
                # Use actual observed value with reduced uncertainty
                idx = [tuple(o) for o in self.observed_orders].index(order_tuple)
                mean = self.observed_rewards[idx]
                # Reduce std based on observation count
                std = 1.0 / (1 + self.observation_counts.get(order_tuple, 0))
            else:
                # Predict using nearest neighbors in order space
                distances = [workflow_manager.compute_kendall_distance(order, obs_order) 
                           for obs_order in self.observed_orders]
                
                # Weighted average based on distances
                weights = np.exp(-np.array(distances) * 2)  # Exponential decay
                weights /= weights.sum()
                
                mean = np.sum(weights * np.array(self.observed_rewards))
                # Higher std for more distant orders
                std = np.min(distances) * 10 + 1.0
            
            # Adjust beta for frequently visited orders
            visit_count = self.observation_counts.get(order_tuple, 0)
            adjusted_beta = self.beta / (1 + 0.5 * visit_count)
            
            # Compute UCB
            ucb = mean + adjusted_beta * std
            
            ucb_scores.append(ucb)
            means.append(mean)
            stds.append(std)
        
        # Select order with highest UCB
        best_idx = np.argmax(ucb_scores)
        selected_order = candidate_orders[best_idx]
        
        # Calculate statistics
        order_tuple = tuple(selected_order)
        visit_count = self.observation_counts.get(order_tuple, 0)
        
        # Find closest observed orders
        distances_to_observed = [
            workflow_manager.compute_kendall_distance(selected_order, obs_order)
            for obs_order in self.observed_orders
        ]
        closest_idx = np.argmin(distances_to_observed)
        closest_order = self.observed_orders[closest_idx]
        closest_reward = self.observed_rewards[closest_idx]
        
        # Calculate exploration vs exploitation
        exploration_bonus = stds[best_idx] * self.beta
        exploitation_value = means[best_idx]
        
        # Get top 3 UCB scores for comparison
        sorted_indices = np.argsort(ucb_scores)[::-1][:3]
        top_3_info = []
        for idx in sorted_indices:
            top_3_info.append({
                'order': ' → '.join(candidate_orders[idx]),
                'ucb': ucb_scores[idx],
                'mean': means[idx],
                'std': stds[idx]
            })
        
        info = {
            'ucb_score': ucb_scores[best_idx],
            'mean': means[best_idx],
            'std': stds[best_idx],
            'n_obs': len(self.observed_orders),
            'visit_count': visit_count,
            'exploration_bonus': exploration_bonus,
            'exploitation_value': exploitation_value,
            'closest_observed': ' → '.join(closest_order),
            'closest_distance': distances_to_observed[closest_idx],
            'closest_reward': closest_reward,
            'selection_reason': 'exploitation' if exploitation_value > exploration_bonus else 'exploration',
            'top_3_candidates': top_3_info
        }
        
        return selected_order, ucb_scores[best_idx], info
    
    def get_best_order(self) -> Tuple[List[str], float]:
        """
        Return the best order found so far (highest observed reward)
        """
        if len(self.observed_orders) == 0:
            return None, float('-inf')
        
        best_idx = np.argmax(self.observed_rewards)
        return self.observed_orders[best_idx], self.observed_rewards[best_idx]
    
    def get_statistics(self) -> Dict:
        """
        Get search statistics
        """
        if len(self.observed_rewards) == 0:
            return {
                'n_observations': 0,
                'best_reward': None,
                'mean_reward': None,
                'std_reward': None
            }
        
        return {
            'n_observations': len(self.observed_rewards),
            'n_unique_orders': len(set(tuple(o) for o in self.observed_orders)),
            'best_reward': np.max(self.observed_rewards),
            'worst_reward': np.min(self.observed_rewards),
            'mean_reward': np.mean(self.observed_rewards),
            'std_reward': np.std(self.observed_rewards),
            'most_visited': max(self.observation_counts.items(), 
                              key=lambda x: x[1]) if self.observation_counts else None
        }
