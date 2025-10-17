#!/usr/bin/env python3
"""
Quick test to compare training speeds
"""

import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import time
import subprocess
import os

def test_speed(script_name, label, episodes=1000):
    """Test training speed for a given script"""
    print(f"\n{'='*60}")
    print(f"Testing {label}")
    print(f"{'='*60}")
    
    cmd = [
        "python", script_name,
        "--total-episodes", str(episodes),
        "--n-envs", "100",
        "--max-episodes", "10"
    ]
    
    start_time = time.time()
    
    try:
        # Run with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout for test
        )
        
        elapsed = time.time() - start_time
        
        # Parse output for episodes completed
        lines = result.stdout.split('\n')
        episodes_completed = 0
        for line in lines:
            if "Total episodes used:" in line:
                parts = line.split(":")[-1].strip().split("/")
                if parts:
                    episodes_completed = int(parts[0])
                    break
        
        if episodes_completed > 0:
            eps_per_sec = episodes_completed / elapsed
            print(f"✅ Completed {episodes_completed} episodes in {elapsed:.1f}s")
            print(f"   Speed: {eps_per_sec:.1f} episodes/second")
            return eps_per_sec
        else:
            print(f"⚠️ Could not determine episodes completed")
            return 0
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout after 120s (test limit)")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0


def main():
    print("\n" + "="*60)
    print("🏁 TRAINING SPEED COMPARISON TEST")
    print("="*60)
    
    # Test original
    print("\n1️⃣ Testing ORIGINAL implementation...")
    original_speed = test_speed(
        "workflow_rl/parallel_train_workflow_rl.py",
        "Original (SharedMemory + K_epochs=4)",
        episodes=500
    )
    
    # Test optimized
    print("\n2️⃣ Testing OPTIMIZED implementation...")
    optimized_speed = test_speed(
        "workflow_rl/parallel_train_workflow_rl_fast.py",
        "Optimized (Vectorized + K_epochs=2)",
        episodes=500
    )
    
    # Summary
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    print(f"Original:  {original_speed:.1f} episodes/sec")
    print(f"Optimized: {optimized_speed:.1f} episodes/sec")
    
    if original_speed > 0 and optimized_speed > 0:
        speedup = optimized_speed / original_speed
        print(f"\n🚀 Speedup: {speedup:.2f}x faster")
        
        if speedup > 2:
            print("✅ Excellent! Achieved >2x speedup")
        elif speedup > 1.5:
            print("👍 Good! Achieved >1.5x speedup")
        elif speedup > 1:
            print("📈 Improved! Some speedup achieved")
        else:
            print("⚠️ No speedup - check implementation")
    
    print("\n💡 For production training, use:")
    print("   python workflow_rl/parallel_train_workflow_rl_fast.py \\")
    print("       --n-envs 200 --total-episodes 100000")


if __name__ == "__main__":
    main()
