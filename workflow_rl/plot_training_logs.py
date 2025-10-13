"""
Visualize training logs from CSV files

Usage:
    python plot_training_logs.py <workflow_id>
    
Example:
    python plot_training_logs.py 0
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_workflow_training(workflow_id, checkpoint_dir='compliance_checkpoints'):
    """Plot training curves for a specific workflow"""
    
    # Load CSV files
    detailed_file = os.path.join(checkpoint_dir, f'workflow_{workflow_id}_detailed_log.csv')
    summary_file = os.path.join(checkpoint_dir, f'workflow_{workflow_id}_summary_log.csv')
    
    if not os.path.exists(detailed_file):
        print(f"Error: {detailed_file} not found!")
        return
    
    # Load data
    detailed_df = pd.read_csv(detailed_file)
    summary_df = pd.read_csv(summary_file) if os.path.exists(summary_file) else None
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Workflow {workflow_id} Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Environment Reward over time (all envs)
    ax = axes[0, 0]
    for env_id in detailed_df['Env_ID'].unique():
        env_data = detailed_df[detailed_df['Env_ID'] == env_id]
        ax.plot(env_data['Episode'], env_data['Env_Reward'].astype(float), 
                alpha=0.3, linewidth=0.5)
    if summary_df is not None:
        ax.plot(summary_df['Total_Episodes'], summary_df['Avg_Env_Reward'].astype(float),
                color='red', linewidth=2, label='Average')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Environment Reward')
    ax.set_title('Environment Reward per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Total Reward over time (all envs)
    ax = axes[0, 1]
    for env_id in detailed_df['Env_ID'].unique():
        env_data = detailed_df[detailed_df['Env_ID'] == env_id]
        ax.plot(env_data['Episode'], env_data['Total_Reward'].astype(float),
                alpha=0.3, linewidth=0.5)
    if summary_df is not None:
        ax.plot(summary_df['Total_Episodes'], summary_df['Avg_Total_Reward'].astype(float),
                color='red', linewidth=2, label='Average')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (Env + Alignment)')
    ax.set_title('Total Reward per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Alignment Bonus over time
    ax = axes[0, 2]
    for env_id in detailed_df['Env_ID'].unique():
        env_data = detailed_df[detailed_df['Env_ID'] == env_id]
        ax.plot(env_data['Episode'], env_data['Alignment_Bonus'].astype(float),
                alpha=0.3, linewidth=0.5)
    if summary_df is not None:
        ax.plot(summary_df['Total_Episodes'], summary_df['Avg_Alignment_Bonus'].astype(float),
                color='red', linewidth=2, label='Average')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Alignment Bonus')
    ax.set_title('Alignment Bonus per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Compliance over time
    ax = axes[1, 0]
    for env_id in detailed_df['Env_ID'].unique():
        env_data = detailed_df[detailed_df['Env_ID'] == env_id]
        ax.plot(env_data['Episode'], detailed_df[detailed_df['Env_ID'] == env_id]['Compliance'].astype(float),
                alpha=0.3, linewidth=0.5)
    if summary_df is not None:
        ax.plot(summary_df['Total_Episodes'], summary_df['Avg_Compliance'].astype(float),
                color='red', linewidth=2, label='Average')
    ax.axhline(y=0.95, color='green', linestyle='--', label='Target (95%)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Compliance Rate')
    ax.set_title('Compliance Rate per Episode')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Fixes per episode
    ax = axes[1, 1]
    for env_id in detailed_df['Env_ID'].unique():
        env_data = detailed_df[detailed_df['Env_ID'] == env_id]
        ax.plot(env_data['Episode'], env_data['Fixes'],
                alpha=0.3, linewidth=0.5)
    if summary_df is not None:
        ax.plot(summary_df['Total_Episodes'], summary_df['Avg_Fixes'].astype(float),
                color='red', linewidth=2, label='Average')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Number of Fixes')
    ax.set_title('Fixes per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate summary stats
    final_env_reward = summary_df['Avg_Env_Reward'].astype(float).iloc[-1] if summary_df is not None else detailed_df['Env_Reward'].astype(float).mean()
    final_compliance = summary_df['Avg_Compliance'].astype(float).iloc[-1] if summary_df is not None else detailed_df['Compliance'].astype(float).mean()
    avg_fixes = detailed_df['Fixes'].mean()
    total_episodes = detailed_df['Episode'].max()
    n_envs = detailed_df['Env_ID'].nunique()
    
    stats_text = f"""
    SUMMARY STATISTICS
    {'='*30}
    
    Total Episodes: {total_episodes}
    Parallel Envs: {n_envs}
    
    Final Env Reward: {final_env_reward:.2f}
    Final Compliance: {final_compliance:.2%}
    
    Avg Fixes/Episode: {avg_fixes:.1f}
    
    Min Env Reward: {detailed_df['Env_Reward'].astype(float).min():.2f}
    Max Env Reward: {detailed_df['Env_Reward'].astype(float).max():.2f}
    
    Min Compliance: {detailed_df['Compliance'].astype(float).min():.2%}
    Max Compliance: {detailed_df['Compliance'].astype(float).max():.2%}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(checkpoint_dir, f'workflow_{workflow_id}_training_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_training_logs.py <workflow_id>")
        print("Example: python plot_training_logs.py 0")
        sys.exit(1)
    
    workflow_id = sys.argv[1]
    checkpoint_dir = sys.argv[2] if len(sys.argv) > 2 else 'compliance_checkpoints'
    
    plot_workflow_training(workflow_id, checkpoint_dir)

