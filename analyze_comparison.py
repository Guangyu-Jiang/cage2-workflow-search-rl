#!/usr/bin/env python3
"""
Analyze and compare results from baseline PPO vs workflow search experiments
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
import argparse

def load_baseline_reduced_results(folder_pattern):
    """Load results from original baseline (22 actions)"""
    # This would need custom parsing since original doesn't save CSV
    # For now, return placeholder
    return None

def load_baseline_full_results(log_path):
    """Load results from baseline with full action space"""
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        return df
    return None

def load_workflow_search_results(exp_dir):
    """Load results from workflow search experiment"""
    training_log = os.path.join(exp_dir, "training_log.csv")
    gp_log = os.path.join(exp_dir, "gp_sampling_log.csv")
    
    results = {}
    if os.path.exists(training_log):
        results['training'] = pd.read_csv(training_log)
    if os.path.exists(gp_log):
        results['gp_sampling'] = pd.read_csv(gp_log)
    
    return results if results else None

def calculate_metrics(data, method_name):
    """Calculate comparison metrics"""
    
    metrics = {
        'method': method_name,
        'total_episodes': 0,
        'total_steps': 0,
        'best_episode_reward': None,
        'final_avg_reward': None,
        'convergence_episode': None,
        'sample_efficiency': None
    }
    
    if data is None:
        return metrics
    
    if isinstance(data, pd.DataFrame):
        # Baseline data
        if 'Episode' in data.columns and 'Reward' in data.columns:
            metrics['total_episodes'] = len(data)
            metrics['total_steps'] = len(data) * 100  # Assuming 100 steps per episode
            metrics['best_episode_reward'] = data['Reward'].max()
            
            # Final average (last 100 episodes)
            if len(data) >= 100:
                metrics['final_avg_reward'] = data['Reward'].tail(100).mean()
            else:
                metrics['final_avg_reward'] = data['Reward'].mean()
            
            # Convergence (first time reaching 80% of final performance)
            target = metrics['final_avg_reward'] * 0.8
            converged = data[data['Reward'].rolling(10).mean() >= target]
            if len(converged) > 0:
                metrics['convergence_episode'] = converged.iloc[0]['Episode']
    
    elif isinstance(data, dict) and 'training' in data:
        # Workflow search data
        df = data['training']
        episode_data = df[df['Type'] == 'episode']
        
        if len(episode_data) > 0:
            metrics['total_episodes'] = len(episode_data)
            metrics['total_steps'] = len(episode_data) * 100
            
            # Use Env_Reward for fair comparison (not Total_Reward which includes alignment)
            rewards = episode_data['Env_Reward'].astype(float)
            metrics['best_episode_reward'] = rewards.max()
            
            if len(rewards) >= 100:
                metrics['final_avg_reward'] = rewards.tail(100).mean()
            else:
                metrics['final_avg_reward'] = rewards.mean()
            
            # Convergence
            target = metrics['final_avg_reward'] * 0.8
            rolling_mean = rewards.rolling(10).mean()
            converged = episode_data[rolling_mean >= target]
            if len(converged) > 0:
                metrics['convergence_episode'] = converged.index[0]
    
    # Calculate sample efficiency (reward per 1000 steps)
    if metrics['total_steps'] > 0 and metrics['final_avg_reward'] is not None:
        metrics['sample_efficiency'] = (metrics['final_avg_reward'] / metrics['total_steps']) * 1000
    
    return metrics

def plot_comparison(results_dict, output_file='comparison.png'):
    """Create comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Learning curves
    ax1 = axes[0, 0]
    for name, data in results_dict.items():
        if data is None:
            continue
            
        if isinstance(data, pd.DataFrame) and 'Reward' in data.columns:
            # Baseline data
            episodes = data['Episode']
            rewards = data['Reward'].rolling(100).mean()
            ax1.plot(episodes, rewards, label=name, linewidth=2)
            
        elif isinstance(data, dict) and 'training' in data:
            # Workflow search data
            df = data['training']
            episode_data = df[df['Type'] == 'episode']
            if len(episode_data) > 0:
                rewards = episode_data['Env_Reward'].astype(float).rolling(100).mean()
                ax1.plot(range(len(rewards)), rewards, label=name, linewidth=2)
    
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Reward (100-ep rolling)')
    ax1.set_title('Learning Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Action space impact
    ax2 = axes[0, 1]
    categories = []
    values = []
    colors = []
    
    for name, metrics in results_dict.items():
        if metrics and isinstance(metrics, dict) and 'final_avg_reward' in metrics:
            categories.append(name.replace('_', '\n'))
            values.append(metrics.get('final_avg_reward', 0))
            if 'reduced' in name:
                colors.append('orange')  # 22 actions
            else:
                colors.append('blue')    # 145 actions
    
    bars = ax2.bar(categories, values, color=colors)
    ax2.set_ylabel('Final Average Reward')
    ax2.set_title('Performance by Action Space Size')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add legend
    import matplotlib.patches as mpatches
    orange_patch = mpatches.Patch(color='orange', label='22 actions (reduced)')
    blue_patch = mpatches.Patch(color='blue', label='145 actions (full)')
    ax2.legend(handles=[orange_patch, blue_patch])
    
    # Plot 3: Sample efficiency
    ax3 = axes[1, 0]
    efficiency_data = []
    labels = []
    
    for name, metrics in results_dict.items():
        if metrics and isinstance(metrics, dict):
            eff = metrics.get('sample_efficiency', 0)
            if eff:
                efficiency_data.append(eff)
                labels.append(name.replace('_', '\n'))
    
    if efficiency_data:
        bars = ax3.bar(labels, efficiency_data, color=['orange' if 'reduced' in l else 'blue' 
                                                       for l in labels])
        ax3.set_ylabel('Reward per 1000 Steps')
        ax3.set_title('Sample Efficiency')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    for name, metrics in results_dict.items():
        if metrics and isinstance(metrics, dict):
            row = [
                name.replace('_', ' ').title(),
                f"{metrics.get('best_episode_reward', 'N/A'):.1f}" if metrics.get('best_episode_reward') else 'N/A',
                f"{metrics.get('final_avg_reward', 'N/A'):.1f}" if metrics.get('final_avg_reward') else 'N/A',
                f"{metrics.get('convergence_episode', 'N/A')}" if metrics.get('convergence_episode') else 'N/A'
            ]
            table_data.append(row)
    
    if table_data:
        table = ax4.table(cellText=table_data,
                         colLabels=['Method', 'Best Episode', 'Final Avg', 'Convergence'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    
    plt.suptitle('Baseline PPO vs Workflow Search Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    
    return fig

def print_analysis_summary(metrics_dict):
    """Print detailed analysis summary"""
    
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS SUMMARY")
    print("="*60)
    
    # Check if we have both baseline and workflow results
    has_baseline_reduced = any('reduced' in k for k in metrics_dict.keys())
    has_baseline_full = any('baseline_full' in k for k in metrics_dict.keys())
    has_workflow = any('workflow' in k for k in metrics_dict.keys())
    
    if has_baseline_reduced and has_baseline_full:
        # Calculate action space impact
        reduced_perf = next((v['final_avg_reward'] for k, v in metrics_dict.items() 
                           if 'reduced' in k and v.get('final_avg_reward')), None)
        full_perf = next((v['final_avg_reward'] for k, v in metrics_dict.items() 
                        if 'baseline_full' in k and v.get('final_avg_reward')), None)
        
        if reduced_perf and full_perf:
            advantage = (reduced_perf - full_perf) / abs(full_perf) * 100
            print(f"\n📊 Action Space Impact:")
            print(f"  Baseline (22 actions):  {reduced_perf:.2f}")
            print(f"  Baseline (145 actions): {full_perf:.2f}")
            print(f"  Advantage of reduction: {advantage:.1f}%")
    
    if has_baseline_full and has_workflow:
        # Fair comparison
        baseline_perf = next((v['final_avg_reward'] for k, v in metrics_dict.items() 
                            if 'baseline_full' in k and v.get('final_avg_reward')), None)
        workflow_perf = next((v['final_avg_reward'] for k, v in metrics_dict.items() 
                            if 'workflow' in k and v.get('final_avg_reward')), None)
        
        if baseline_perf and workflow_perf:
            improvement = (workflow_perf - baseline_perf) / abs(baseline_perf) * 100
            print(f"\n🎯 Fair Comparison (Full Action Space):")
            print(f"  Baseline PPO:     {baseline_perf:.2f}")
            print(f"  Workflow Search:  {workflow_perf:.2f}")
            print(f"  Improvement:      {improvement:.1f}%")
    
    print(f"\n📈 Detailed Metrics:")
    print(f"{'Method':<20} {'Best':<10} {'Final Avg':<12} {'Convergence':<12} {'Efficiency':<12}")
    print("-" * 66)
    
    for name, metrics in metrics_dict.items():
        if isinstance(metrics, dict):
            best = f"{metrics.get('best_episode_reward', 0):.1f}" if metrics.get('best_episode_reward') else 'N/A'
            final = f"{metrics.get('final_avg_reward', 0):.1f}" if metrics.get('final_avg_reward') else 'N/A'
            conv = f"{metrics.get('convergence_episode', 0)}" if metrics.get('convergence_episode') else 'N/A'
            eff = f"{metrics.get('sample_efficiency', 0):.3f}" if metrics.get('sample_efficiency') else 'N/A'
            
            print(f"{name[:19]:<20} {best:<10} {final:<12} {conv:<12} {eff:<12}")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Analyze comparison results')
    parser.add_argument('--baseline-reduced', type=str,
                       help='Path to baseline reduced action results')
    parser.add_argument('--baseline-full', type=str,
                       help='Path to baseline full action CSV')
    parser.add_argument('--workflow-search', type=str,
                       help='Path to workflow search experiment directory')
    parser.add_argument('--output', type=str, default='comparison_analysis.png',
                       help='Output plot filename')
    parser.add_argument('--compare-all', action='store_true',
                       help='Find and compare all recent experiments')
    
    args = parser.parse_args()
    
    results = {}
    metrics = {}
    
    if args.compare_all:
        # Find recent experiments
        print("Searching for recent experiments...")
        
        # Find baseline full action results
        baseline_dirs = glob("Models/baseline_ppo_full_action_*/training_log.csv")
        if baseline_dirs:
            latest_baseline = sorted(baseline_dirs)[-1]
            print(f"Found baseline (full): {latest_baseline}")
            data = load_baseline_full_results(latest_baseline)
            results['baseline_full'] = data
            metrics['baseline_full'] = calculate_metrics(data, 'baseline_full')
        
        # Find workflow search results
        workflow_dirs = glob("logs/exp_*/")
        if workflow_dirs:
            latest_workflow = sorted(workflow_dirs)[-1]
            print(f"Found workflow search: {latest_workflow}")
            data = load_workflow_search_results(latest_workflow)
            results['workflow_search'] = data
            metrics['workflow_search'] = calculate_metrics(data, 'workflow_search')
    
    else:
        # Load specific experiments
        if args.baseline_reduced:
            data = load_baseline_reduced_results(args.baseline_reduced)
            results['baseline_reduced'] = data
            metrics['baseline_reduced'] = calculate_metrics(data, 'baseline_reduced')
        
        if args.baseline_full:
            data = load_baseline_full_results(args.baseline_full)
            results['baseline_full'] = data
            metrics['baseline_full'] = calculate_metrics(data, 'baseline_full')
        
        if args.workflow_search:
            data = load_workflow_search_results(args.workflow_search)
            results['workflow_search'] = data
            metrics['workflow_search'] = calculate_metrics(data, 'workflow_search')
    
    if not results:
        print("No results found to analyze!")
        return
    
    # Create plots
    plot_comparison(results, args.output)
    
    # Print summary
    print_analysis_summary(metrics)
    
    print(f"\nAnalysis complete! Plot saved to: {args.output}")

if __name__ == '__main__':
    main()
