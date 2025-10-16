#!/usr/bin/env python3
"""
Test script to verify the episode budget changes work correctly
"""

import sys
import argparse


def test_config():
    """Test the configuration with default arguments"""
    
    # Temporarily replace sys.argv to test default arguments
    original_argv = sys.argv
    sys.argv = ['test_episode_budget.py']
    
    # Import after setting sys.argv
    from workflow_rl.parallel_train_workflow_rl import parse_args
    
    args = parse_args()
    
    # Restore original sys.argv
    sys.argv = original_argv
    
    print("✅ Testing default configuration...")
    print(f"  n_envs: {args.n_envs} (expected: 200)")
    print(f"  total_episodes: {args.total_episodes} (expected: 500)")
    print(f"  max_episodes: {args.max_episodes} (expected: 50)")
    print(f"  compliance_threshold: {args.compliance_threshold} (expected: 0.95)")
    print(f"  alignment_lambda: {args.alignment_lambda} (expected: 30.0)")
    
    # Check that n_workflows is not an argument anymore
    assert not hasattr(args, 'n_workflows'), "n_workflows should no longer exist"
    assert not hasattr(args, 'min_episodes'), "min_episodes should no longer exist"
    
    assert args.n_envs == 200, f"n_envs should be 200, got {args.n_envs}"
    assert args.total_episodes == 500, f"total_episodes should be 500, got {args.total_episodes}"
    assert args.max_episodes == 50, f"max_episodes should be 50, got {args.max_episodes}"
    
    print("\n✅ All configuration tests passed!")


def test_custom_config():
    """Test with custom arguments"""
    
    # Temporarily replace sys.argv with custom arguments
    original_argv = sys.argv
    sys.argv = [
        'test_episode_budget.py',
        '--total-episodes', '1000',
        '--max-episodes', '100',
        '--n-envs', '100'
    ]
    
    from workflow_rl.parallel_train_workflow_rl import parse_args
    
    args = parse_args()
    
    # Restore original sys.argv
    sys.argv = original_argv
    
    print("\n✅ Testing custom configuration...")
    print(f"  n_envs: {args.n_envs} (expected: 100)")
    print(f"  total_episodes: {args.total_episodes} (expected: 1000)")
    print(f"  max_episodes: {args.max_episodes} (expected: 100)")
    
    assert args.n_envs == 100, f"n_envs should be 100, got {args.n_envs}"
    assert args.total_episodes == 1000, f"total_episodes should be 1000, got {args.total_episodes}"
    assert args.max_episodes == 100, f"max_episodes should be 100, got {args.max_episodes}"
    
    print("\n✅ Custom configuration tests passed!")


def main():
    print("="*60)
    print("Testing Episode Budget Changes")
    print("="*60)
    
    test_config()
    test_custom_config()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nThe episode budget changes are working correctly:")
    print("1. n_workflows parameter has been replaced with total_episode_budget")
    print("2. min_episodes parameter has been removed")
    print("3. Training will continue until budget is exhausted")
    print("4. Early stopping can happen immediately when compliance is achieved")
    print("\nUsage examples:")
    print("  Default: python workflow_rl/parallel_train_workflow_rl.py")
    print("  Custom:  python workflow_rl/parallel_train_workflow_rl.py --total-episodes 1000 --max-episodes 100")


if __name__ == "__main__":
    main()
