"""
Custom parallel training script with adjustable parameters
"""

import numpy as np
import torch
from workflow_rl.parallel_train_workflow_rl import ParallelWorkflowRLTrainer

def main():
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configure training parameters
    trainer = ParallelWorkflowRLTrainer(
        # Parallel environment settings
        n_envs=25,                      # Number of parallel environments
        
        # Workflow search settings
        n_workflows=10,                 # Number of workflows to explore
        
        # Training settings
        train_episodes_per_env=30,      # Episodes per environment (30*25 = 750 total)
        max_steps=100,                  # Steps per episode
        
        # Alignment reward weights
        alignment_alpha=0.1,            # Bonus for following workflow
        alignment_beta=0.2,             # Penalty for violating workflow
        
        # GP-UCB settings
        gp_beta=2.0,                    # Exploration parameter
        
        # Early stopping
        compliance_threshold=0.95,      # Stop at 95% compliance
        min_episodes=10,                # Minimum episodes before checking
        
        # PPO update frequency
        update_every_steps=100,         # Update after full episode (2500 transitions)
        
        # Output directory
        checkpoint_dir='my_parallel_checkpoints'
    )
    
    # Run the training
    print("\n" + "="*60)
    print("CUSTOM PARALLEL TRAINING CONFIGURATION")
    print("="*60)
    print(f"Parallel Environments: {trainer.n_envs}")
    print(f"Workflows to Explore: {trainer.n_workflows}")
    print(f"Episodes per Environment: {trainer.train_episodes_per_env}")
    print(f"Total Episodes per Workflow: {trainer.n_envs * trainer.train_episodes_per_env}")
    print(f"Transitions per Update: {trainer.n_envs * trainer.update_every_steps}")
    print("="*60 + "\n")
    
    trainer.run_workflow_search()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Results saved in: {trainer.checkpoint_dir}/")
    print("Check parallel_training_history.json for detailed results")

if __name__ == "__main__":
    main()
