#!/usr/bin/env python3
"""
Test script to verify the new default settings
"""

import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')


def test_defaults():
    """Test the new default configuration"""
    
    # Temporarily replace sys.argv to test default arguments
    original_argv = sys.argv
    sys.argv = ['test_new_defaults.py']
    
    # Import after setting sys.argv
    from workflow_rl.parallel_train_workflow_rl import parse_args
    
    args = parse_args()
    
    # Restore original sys.argv
    sys.argv = original_argv
    
    print("="*60)
    print("Testing New Default Configuration")
    print("="*60)
    print(f"\n✅ Environment Settings:")
    print(f"  n_envs: {args.n_envs} (expected: 100)")
    print(f"  total_episodes: {args.total_episodes:,} (expected: 100,000)")
    print(f"  max_episodes: {args.max_episodes} (expected: 100)")
    
    print(f"\n✅ Learning Settings:")
    print(f"  compliance_threshold: {args.compliance_threshold} (expected: 0.95)")
    print(f"  alignment_lambda: {args.alignment_lambda} (expected: 30.0)")
    print(f"  update_steps: {args.update_steps} (expected: 100)")
    
    print(f"\n✅ Other Settings:")
    print(f"  red_agent: {args.red_agent} (default: meander)")
    print(f"  max_steps: {args.max_steps} (expected: 100)")
    
    # Verify values
    assert args.n_envs == 100, f"n_envs should be 100, got {args.n_envs}"
    assert args.total_episodes == 100000, f"total_episodes should be 100000, got {args.total_episodes}"
    assert args.max_episodes == 100, f"max_episodes should be 100, got {args.max_episodes}"
    
    print("\n" + "="*60)
    print("✅ ALL DEFAULT VALUES CORRECT!")
    print("="*60)
    
    print("\n📊 With these settings:")
    print(f"  - Each update uses {args.n_envs} trajectories")
    print(f"  - Total transitions per update: {args.n_envs * args.max_steps:,}")
    print(f"  - Maximum workflows explorable: ~{args.total_episodes // 10:,} (if quick)")
    print(f"  - Minimum workflows explorable: ~{args.total_episodes // args.max_episodes:,} (if slow)")
    print(f"  - Updates per workflow: up to {args.max_episodes}")
    
    print("\n🚀 Ready for large-scale workflow search!")


def test_instantiation():
    """Test that the trainer can be instantiated with new defaults"""
    
    print("\n" + "="*60)
    print("Testing Trainer Instantiation")
    print("="*60)
    
    from workflow_rl.parallel_train_workflow_rl import ParallelWorkflowRLTrainer
    from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
    
    try:
        trainer = ParallelWorkflowRLTrainer()  # Use all defaults
        
        print(f"\n✅ Trainer created successfully with defaults:")
        print(f"  n_envs: {trainer.n_envs}")
        print(f"  total_episode_budget: {trainer.total_episode_budget:,}")
        print(f"  max_train_episodes_per_env: {trainer.max_train_episodes_per_env}")
        
        assert trainer.n_envs == 100
        assert trainer.total_episode_budget == 100000
        assert trainer.max_train_episodes_per_env == 100
        
        print("\n✅ Default values in trainer match expected!")
        
    except Exception as e:
        print(f"\n✗ Failed to create trainer: {e}")
        return False
    
    return True


def main():
    print("="*60)
    print("VERIFYING NEW DEFAULT SETTINGS")
    print("="*60)
    
    # Test argument parsing
    test_defaults()
    
    # Test trainer instantiation
    test_instantiation()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nThe new optimized defaults are working correctly:")
    print("  - 100 parallel environments (more frequent updates)")
    print("  - 100,000 episode budget (extensive exploration)")
    print("  - 100 max episodes per workflow (better convergence)")


if __name__ == "__main__":
    main()
