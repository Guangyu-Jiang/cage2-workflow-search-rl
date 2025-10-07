#!/usr/bin/env python3
"""
GP-UCB Workflow Search with Meaningful Workflow Representation
Addresses: 1) Workflow distance metric, 2) Workflow compliance verification,
3) Structured rewards for workflow following
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from typing import Dict, List, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# PART 1: MEANINGFUL WORKFLOW REPRESENTATION
# ============================================================================

class WorkflowEmbedding:
    """
    Ensures workflow embeddings are meaningful and distinguishable
    """
    
    def __init__(self, dim=8):
        self.dim = dim
        # Define semantic anchors for each dimension
        self.dimension_semantics = {
            0: ("late_fortify", "early_fortify"),
            1: ("no_decoys", "heavy_decoys"),
            2: ("no_analysis", "constant_analysis"),
            3: ("prefer_remove", "prefer_restore"),
            4: ("delayed_response", "immediate_response"),
            5: ("ignore_user", "focus_user"),
            6: ("ignore_enterprise", "focus_enterprise"),
            7: ("ignore_operational", "focus_operational")
        }
        
    def create_canonical_workflows(self):
        """Create interpretable reference workflows"""
        canonicals = {
            "aggressive_early": np.array([1.0, 0.8, 0.3, 0.8, 1.0, 0.2, 0.5, 0.3]),
            "defensive_late": np.array([0.0, 0.2, 0.7, -0.8, 0.5, 0.4, 0.4, 0.2]),
            "information_first": np.array([0.5, 0.0, 1.0, 0.0, 0.3, 0.3, 0.3, 0.4]),
            "fortify_heavy": np.array([0.8, 1.0, 0.2, 0.5, 0.8, 0.3, 0.4, 0.3]),
            "reactive_minimal": np.array([0.2, 0.1, 0.2, -0.5, 0.8, 0.6, 0.3, 0.1])
        }
        return canonicals
    
    def distance(self, w1: np.ndarray, w2: np.ndarray, 
                 semantic_weights: Optional[np.ndarray] = None) -> float:
        """
        Compute semantically meaningful distance between workflows
        
        Key idea: Some dimensions are more important than others
        E.g., fortify timing (dim 0) has bigger impact than subnet focus
        """
        if semantic_weights is None:
            # Default weights based on empirical importance
            semantic_weights = np.array([
                1.5,  # Fortify timing - very important
                1.2,  # Fortify intensity - important
                0.8,  # Analysis frequency - moderate
                1.3,  # Remove vs restore - important
                1.0,  # Response speed - standard
                0.6,  # User focus - less critical
                0.7,  # Enterprise focus
                0.6   # Operational focus
            ])
        
        # Weighted Euclidean distance
        diff = (w1 - w2) * semantic_weights
        return np.linalg.norm(diff)
    
    def interpolate(self, w1: np.ndarray, w2: np.ndarray, alpha: float) -> np.ndarray:
        """Interpolate between workflows (for exploration)"""
        return w1 * (1 - alpha) + w2 * alpha


# ============================================================================
# PART 2: WORKFLOW COMPLIANCE VERIFICATION
# ============================================================================

class WorkflowComplianceMonitor:
    """
    Monitors whether the policy is actually following the intended workflow
    """
    
    def __init__(self, workflow: np.ndarray, window_size: int = 20):
        self.workflow = workflow
        self.window_size = window_size
        self.action_history = deque(maxlen=window_size)
        self.compliance_metrics = {}
        
    def update(self, action: int, timestep: int, observation: np.ndarray):
        """Track action and update compliance metrics"""
        self.action_history.append({
            'action': action,
            'timestep': timestep,
            'observation': observation
        })
        
        if len(self.action_history) >= self.window_size:
            self._compute_compliance()
    
    def _compute_compliance(self):
        """Compute how well actions match workflow intent"""
        
        # Count action types in window
        action_counts = {
            'fortify': 0,
            'analyse': 0,
            'remove': 0,
            'restore': 0,
            'other': 0
        }
        
        for item in self.action_history:
            action = item['action']
            if action in range(1000, 1009) or action in [28, 41, 54, 67, 80, 93, 106, 119]:
                action_counts['fortify'] += 1
            elif action in [2, 3, 4, 5, 9, 11, 12, 13, 14]:
                action_counts['analyse'] += 1
            elif action in [15, 16, 17, 18, 22, 24, 25, 26, 27]:
                action_counts['remove'] += 1
            elif action in [132, 133, 134, 135, 139, 141, 142, 143, 144]:
                action_counts['restore'] += 1
            else:
                action_counts['other'] += 1
        
        total = sum(action_counts.values())
        
        # Compute compliance scores for each dimension
        self.compliance_metrics = {
            # Dimension 0: Fortify timing
            'fortify_timing': self._check_fortify_timing(),
            
            # Dimension 1: Fortify intensity
            'fortify_intensity': action_counts['fortify'] / total,
            
            # Dimension 2: Analysis frequency
            'analysis_freq': action_counts['analyse'] / total,
            
            # Dimension 3: Remove vs Restore preference
            'remediation_pref': self._compute_remediation_preference(action_counts),
            
            # Overall compliance
            'overall': self._compute_overall_compliance()
        }
    
    def _check_fortify_timing(self) -> float:
        """Check if fortification happens early (workflow[0] > 0.5) or late"""
        early_fortify_count = sum(
            1 for item in self.action_history[:10]
            if item['action'] in range(1000, 1009)
        )
        expected_early = self.workflow[0] > 0.5
        actual_early = early_fortify_count > 2
        return 1.0 if expected_early == actual_early else 0.0
    
    def _compute_remediation_preference(self, counts: Dict) -> float:
        """Check if remove/restore preference matches workflow[3]"""
        if counts['remove'] + counts['restore'] == 0:
            return 0.5  # No remediation actions
        
        restore_ratio = counts['restore'] / (counts['remove'] + counts['restore'])
        expected_restore_pref = (self.workflow[3] + 1) / 2  # Convert [-1,1] to [0,1]
        
        # Compute similarity
        return 1.0 - abs(restore_ratio - expected_restore_pref)
    
    def _compute_overall_compliance(self) -> float:
        """Aggregate compliance score"""
        scores = [v for k, v in self.compliance_metrics.items() 
                 if k != 'overall' and isinstance(v, (int, float))]
        return np.mean(scores) if scores else 0.0
    
    def is_compliant(self, threshold: float = 0.6) -> bool:
        """Check if policy is sufficiently following the workflow"""
        return self.compliance_metrics.get('overall', 0) >= threshold


# ============================================================================
# PART 3: STRUCTURED REWARD SHAPING FOR WORKFLOW FOLLOWING
# ============================================================================

class WorkflowAwareRewardShaper:
    """
    Shapes rewards to encourage workflow following without changing optimal policy
    Uses potential-based shaping to maintain optimality
    """
    
    def __init__(self, workflow: np.ndarray, base_gamma: float = 0.99):
        self.workflow = workflow
        self.gamma = base_gamma
        self.compliance_monitor = WorkflowComplianceMonitor(workflow)
        
    def shape_reward(self, 
                     base_reward: float,
                     action: int,
                     observation: np.ndarray,
                     next_observation: np.ndarray,
                     timestep: int) -> Tuple[float, Dict]:
        """
        Add shaping reward that encourages workflow following
        
        Key principle: Use potential-based shaping F(s,a,s') = γΦ(s') - Φ(s)
        This preserves optimal policy while accelerating learning
        """
        
        # Update compliance tracking
        self.compliance_monitor.update(action, timestep, observation)
        
        # Compute potential function based on workflow alignment
        potential_current = self._compute_potential(observation, timestep)
        potential_next = self._compute_potential(next_observation, timestep + 1)
        
        # Potential-based shaping reward
        shaping_reward = self.gamma * potential_next - potential_current
        
        # Add small compliance bonus (careful not to change optimal policy)
        compliance_bonus = 0.0
        if self.compliance_monitor.compliance_metrics:
            compliance = self.compliance_monitor.compliance_metrics.get('overall', 0)
            compliance_bonus = 0.01 * compliance  # Small bonus
        
        # Total shaped reward
        shaped_reward = base_reward + shaping_reward + compliance_bonus
        
        # Debugging info
        info = {
            'base_reward': base_reward,
            'shaping_reward': shaping_reward,
            'compliance_bonus': compliance_bonus,
            'compliance_metrics': self.compliance_monitor.compliance_metrics.copy()
        }
        
        return shaped_reward, info
    
    def _compute_potential(self, observation: np.ndarray, timestep: int) -> float:
        """
        Compute potential function that encodes workflow progress
        Higher potential = closer to workflow goals
        """
        potential = 0.0
        
        # Early fortification potential
        if self.workflow[0] > 0.5 and timestep < 10:
            # Reward being in early phase when early fortify is desired
            potential += 0.1 * (10 - timestep) / 10
        
        # Information gathering potential
        if self.workflow[2] > 0.5:
            # Estimate information level from observation
            info_level = self._estimate_information_level(observation)
            potential += 0.05 * info_level
        
        # Remediation readiness potential
        threat_level = self._estimate_threat_level(observation)
        if threat_level > 0:
            if self.workflow[3] > 0:  # Prefer restore
                potential += 0.1 * threat_level  # Ready to restore is good
            else:  # Prefer remove
                potential += 0.05 * threat_level  # Ready to remove is good
        
        return potential
    
    def _estimate_information_level(self, obs: np.ndarray) -> float:
        """Estimate how much information we have from observation"""
        # Count non-zero elements (activity indicators)
        return np.sum(obs != 0) / len(obs)
    
    def _estimate_threat_level(self, obs: np.ndarray) -> float:
        """Estimate threat level from observation"""
        # Check compromise indicators (every 4th element starting at index 2)
        compromise_indicators = obs[2::4] if len(obs) >= 52 else []
        return np.mean(compromise_indicators) if len(compromise_indicators) > 0 else 0.0


# ============================================================================
# PART 4: GP-UCB WORKFLOW SEARCH
# ============================================================================

class GPUCBWorkflowSearch:
    """
    Gaussian Process Upper Confidence Bound for workflow search
    """
    
    def __init__(self, 
                 workflow_dim: int = 8,
                 kernel_length_scale: float = 0.5,
                 beta: float = 2.0,
                 allow_revisit: bool = True):
        
        self.workflow_dim = workflow_dim
        self.beta = beta
        self.allow_revisit = allow_revisit
        
        # Workflow embedding handler
        self.embedding = WorkflowEmbedding(workflow_dim)
        
        # Gaussian Process
        kernel = Matern(length_scale=kernel_length_scale, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        # History
        self.X_observed = []  # Workflows
        self.y_observed = []  # Rewards
        self.observation_counts = {}  # Track revisits
        
        # Initialize with canonical workflows
        self._initialize_with_canonical()
    
    def _initialize_with_canonical(self):
        """Start with known good workflows"""
        canonicals = self.embedding.create_canonical_workflows()
        for name, workflow in canonicals.items():
            self.X_observed.append(workflow)
            self.y_observed.append(0.0)  # Placeholder, will be updated
            self.observation_counts[tuple(workflow)] = 0
    
    def select_next_workflow(self, n_candidates: int = 1000) -> np.ndarray:
        """
        Select next workflow using GP-UCB acquisition
        """
        if len(self.y_observed) < len(self.X_observed):
            # Return unobserved canonical workflow
            return self.X_observed[len(self.y_observed) - 1]
        
        # Fit GP
        if len(self.X_observed) > 0:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)
        
        # Generate candidates
        candidates = self._generate_candidates(n_candidates)
        
        # Compute UCB for each candidate
        ucb_values = []
        for candidate in candidates:
            ucb = self._compute_ucb(candidate)
            ucb_values.append(ucb)
        
        # Select best candidate
        best_idx = np.argmax(ucb_values)
        selected_workflow = candidates[best_idx]
        
        # Track selection
        workflow_key = tuple(selected_workflow)
        if workflow_key not in self.observation_counts:
            self.observation_counts[workflow_key] = 0
        
        return selected_workflow
    
    def _generate_candidates(self, n: int) -> List[np.ndarray]:
        """Generate candidate workflows for evaluation"""
        candidates = []
        
        # 1. Random exploration (40%)
        for _ in range(int(n * 0.4)):
            candidates.append(np.random.uniform(-1, 1, self.workflow_dim))
        
        # 2. Local exploration around best (30%)
        if self.y_observed:
            best_idx = np.argmax(self.y_observed)
            best_workflow = self.X_observed[best_idx]
            for _ in range(int(n * 0.3)):
                noise = np.random.randn(self.workflow_dim) * 0.2
                candidate = np.clip(best_workflow + noise, -1, 1)
                candidates.append(candidate)
        
        # 3. Interpolation between good workflows (20%)
        if len(self.X_observed) >= 2:
            for _ in range(int(n * 0.2)):
                idx1, idx2 = np.random.choice(len(self.X_observed), 2, replace=False)
                alpha = np.random.random()
                interpolated = self.embedding.interpolate(
                    self.X_observed[idx1], 
                    self.X_observed[idx2], 
                    alpha
                )
                candidates.append(interpolated)
        
        # 4. Include observed workflows for potential revisit (10%)
        if self.allow_revisit and self.X_observed:
            for _ in range(int(n * 0.1)):
                idx = np.random.choice(len(self.X_observed))
                candidates.append(self.X_observed[idx].copy())
        
        # Fill remaining with random
        while len(candidates) < n:
            candidates.append(np.random.uniform(-1, 1, self.workflow_dim))
        
        return candidates[:n]
    
    def _compute_ucb(self, workflow: np.ndarray) -> float:
        """Compute Upper Confidence Bound for workflow"""
        if len(self.X_observed) == 0:
            return np.random.random()  # Random if no data
        
        # Get GP prediction
        mean, std = self.gp.predict([workflow], return_std=True)
        
        # Adjust beta based on revisit count
        workflow_key = tuple(workflow)
        revisit_count = self.observation_counts.get(workflow_key, 0)
        adjusted_beta = self.beta / (1 + revisit_count * 0.5)  # Reduce exploration for revisited
        
        # UCB = mean + beta * std
        ucb = mean[0] + adjusted_beta * std[0]
        
        return ucb
    
    def update(self, workflow: np.ndarray, reward: float, compliance: float):
        """
        Update GP with new observation
        
        Args:
            workflow: The evaluated workflow
            reward: Environment reward
            compliance: How well the policy followed the workflow
        """
        # Adjust reward based on compliance
        # If policy didn't follow workflow, the sample is less informative
        adjusted_reward = reward * (0.5 + 0.5 * compliance)
        
        # Update history
        workflow_key = tuple(workflow)
        
        if self.allow_revisit and workflow_key in self.observation_counts:
            # Update existing observation (running average)
            idx = None
            for i, w in enumerate(self.X_observed):
                if np.allclose(w, workflow, atol=1e-6):
                    idx = i
                    break
            
            if idx is not None:
                count = self.observation_counts[workflow_key]
                self.y_observed[idx] = (self.y_observed[idx] * count + adjusted_reward) / (count + 1)
                self.observation_counts[workflow_key] += 1
            else:
                self.X_observed.append(workflow.copy())
                self.y_observed.append(adjusted_reward)
                self.observation_counts[workflow_key] = 1
        else:
            self.X_observed.append(workflow.copy())
            self.y_observed.append(adjusted_reward)
            self.observation_counts[workflow_key] = 1
    
    def get_best_workflow(self) -> Tuple[np.ndarray, float]:
        """Return best workflow found so far"""
        if not self.y_observed:
            return None, float('-inf')
        best_idx = np.argmax(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]


# ============================================================================
# PART 5: INTEGRATED TRAINING LOOP
# ============================================================================

def train_with_gpucb_workflow_search(
    env_fn,
    policy_class,
    num_workflows: int = 50,
    episodes_per_workflow: int = 10,
    compliance_threshold: float = 0.6,
    max_steps: int = 50
):
    """
    Main training loop with GP-UCB workflow search
    """
    
    # Initialize search
    searcher = GPUCBWorkflowSearch(
        workflow_dim=8,
        kernel_length_scale=0.5,
        beta=2.0,
        allow_revisit=True
    )
    
    for workflow_idx in range(num_workflows):
        print(f"\n{'='*60}")
        print(f"Workflow {workflow_idx + 1}/{num_workflows}")
        
        # Select workflow
        workflow = searcher.select_next_workflow()
        print(f"Selected workflow: {workflow[:4]}...")
        
        # Initialize policy for this workflow
        policy = policy_class(workflow)
        
        # Initialize reward shaper and compliance monitor
        reward_shaper = WorkflowAwareRewardShaper(workflow)
        compliance_monitor = WorkflowComplianceMonitor(workflow)
        
        # Train policy with this workflow
        workflow_rewards = []
        workflow_compliances = []
        
        for episode in range(episodes_per_workflow):
            env = env_fn()
            obs = env.reset()
            episode_reward = 0
            episode_shaped_reward = 0
            
            for step in range(max_steps):
                # Get action from policy
                action = policy.get_action(obs, workflow)
                
                # Step environment
                next_obs, base_reward, done, _ = env.step(action)
                
                # Shape reward
                shaped_reward, shaping_info = reward_shaper.shape_reward(
                    base_reward, action, obs, next_obs, step
                )
                
                # Update policy with shaped reward
                policy.update(obs, action, shaped_reward, next_obs, done)
                
                # Track metrics
                episode_reward += base_reward
                episode_shaped_reward += shaped_reward
                compliance_monitor.update(action, step, obs)
                
                obs = next_obs
                if done:
                    break
            
            # Check compliance
            compliance = compliance_monitor.compliance_metrics.get('overall', 0)
            workflow_rewards.append(episode_reward)
            workflow_compliances.append(compliance)
            
            print(f"  Episode {episode + 1}: Reward={episode_reward:.1f}, "
                  f"Compliance={compliance:.2f}")
            
            # Early stopping if not compliant
            if episode >= 3 and np.mean(workflow_compliances[-3:]) < compliance_threshold:
                print(f"  Early stopping: Low compliance ({np.mean(workflow_compliances[-3:]):.2f})")
                break
        
        # Update GP-UCB with results
        avg_reward = np.mean(workflow_rewards)
        avg_compliance = np.mean(workflow_compliances)
        searcher.update(workflow, avg_reward, avg_compliance)
        
        print(f"  Workflow performance: Reward={avg_reward:.1f}, Compliance={avg_compliance:.2f}")
        
        # Report best so far
        best_workflow, best_reward = searcher.get_best_workflow()
        print(f"  Best so far: {best_reward:.1f}")
    
    return searcher


if __name__ == "__main__":
    print("GP-UCB Workflow Search Framework")
    print("="*60)
    
    # Demonstrate workflow distance metric
    embedding = WorkflowEmbedding()
    w1 = np.array([0.9, 0.4, 0.5, 0.5, 0.9, 0.3, 0.4, 0.3])  # Proactive
    w2 = np.array([0.2, 0.1, 0.2, -0.5, 0.8, 0.6, 0.3, 0.1])  # Reactive
    
    distance = embedding.distance(w1, w2)
    print(f"\nDistance between Proactive and Reactive workflows: {distance:.3f}")
    
    # Demonstrate compliance monitoring
    monitor = WorkflowComplianceMonitor(w1)
    print(f"\nCompliance monitoring initialized for Proactive workflow")
    
    # Demonstrate reward shaping
    shaper = WorkflowAwareRewardShaper(w1)
    print(f"Reward shaping initialized")
    
    print("\nFramework ready for training with meaningful workflow representations!")
