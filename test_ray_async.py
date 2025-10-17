#!/usr/bin/env python3
"""
Test Ray async training with different configurations
"""

import sys
import time
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import subprocess
import numpy as np

def run_test(n_workers, n_episodes, episodes_per_update):
    """Run a single test configuration"""
    print(f"\n{'='*70}")
    print(f"🧪 Testing Ray Async: {n_workers} workers, {n_episodes} episodes")
    print(f"{'='*70}")
    
    cmd = [
        'python', 'workflow_rl/ray_async_train_workflow_rl.py',
        '--n-workers', str(n_workers),
        '--total-episodes', str(n_episodes),
        '--episodes-per-update', str(episodes_per_update),
        '--max-episodes-per-workflow', str(n_episodes),
        '--red-agent', 'B_lineAgent'
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout
            cwd='/home/ubuntu/CAGE2/-cyborg-cage-2'
        )
        
        elapsed = time.time() - start_time
        
        # Parse output
        output = result.stdout + result.stderr
        
        # Look for collection rates
        collection_rates = []
        for line in output.split('\n'):
            if 'eps/sec)' in line and 'Collected' in line:
                try:
                    # Extract rate from "Collected X episodes in Y.Zs (R.R eps/sec)"
                    rate_str = line.split('(')[1].split('eps/sec')[0].strip()
                    rate = float(rate_str)
                    collection_rates.append(rate)
                except:
                    pass
        
        # Count episodes collected
        episodes_collected = output.count('Collected')
        
        success = result.returncode == 0 or 'Compliance threshold' in output
        
        print(f"\n📊 Results:")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Episodes collected: {episodes_collected} batches")
        if collection_rates:
            print(f"  Collection rates: {collection_rates}")
            print(f"  Avg rate: {np.mean(collection_rates):.2f} eps/sec")
            print(f"  Min rate: {np.min(collection_rates):.2f} eps/sec")
            print(f"  Max rate: {np.max(collection_rates):.2f} eps/sec")
        print(f"  Success: {'✅' if success else '❌'}")
        
        return {
            'n_workers': n_workers,
            'n_episodes': n_episodes,
            'elapsed': elapsed,
            'rates': collection_rates,
            'success': success
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏱️  Test timed out after 180s")
        return {
            'n_workers': n_workers,
            'n_episodes': n_episodes,
            'elapsed': 180,
            'rates': [],
            'success': False
        }
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return {
            'n_workers': n_workers,
            'n_episodes': n_episodes,
            'elapsed': 0,
            'rates': [],
            'success': False
        }


def main():
    print("\n" + "="*70)
    print("🚀 Ray Async Training Test Suite")
    print("="*70)
    
    # Test configurations
    tests = [
        # (n_workers, total_episodes, episodes_per_update)
        (5, 20, 10),      # Small test
        (10, 40, 20),     # Medium test
        (25, 50, 25),     # Larger test
    ]
    
    results = []
    
    for n_workers, n_episodes, eps_per_update in tests:
        result = run_test(n_workers, n_episodes, eps_per_update)
        results.append(result)
        
        # Brief pause between tests
        time.sleep(2)
    
    # Summary
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    
    print(f"\n{'Workers':<10} {'Episodes':<12} {'Avg Rate':<15} {'Success':<10}")
    print("-" * 50)
    
    for r in results:
        avg_rate = np.mean(r['rates']) if r['rates'] else 0
        success = '✅' if r['success'] else '❌'
        print(f"{r['n_workers']:<10} {r['n_episodes']:<12} {avg_rate:<15.2f} {success:<10}")
    
    # Scaling analysis
    print("\n" + "="*70)
    print("📈 Scaling Analysis")
    print("="*70)
    
    successful_results = [r for r in results if r['rates']]
    
    if len(successful_results) >= 2:
        for i in range(len(successful_results) - 1):
            r1, r2 = successful_results[i], successful_results[i+1]
            
            rate1 = np.mean(r1['rates'])
            rate2 = np.mean(r2['rates'])
            
            worker_ratio = r2['n_workers'] / r1['n_workers']
            speed_ratio = rate2 / rate1
            efficiency = (speed_ratio / worker_ratio) * 100
            
            print(f"\n{r1['n_workers']} → {r2['n_workers']} workers:")
            print(f"  Worker increase: {worker_ratio:.1f}x")
            print(f"  Speed increase: {speed_ratio:.2f}x")
            print(f"  Scaling efficiency: {efficiency:.1f}%")
    
    # Recommendations
    print("\n" + "="*70)
    print("💡 Recommendations")
    print("="*70)
    
    if successful_results:
        best = max(successful_results, key=lambda r: np.mean(r['rates']) if r['rates'] else 0)
        best_rate = np.mean(best['rates'])
        
        print(f"\n✅ Best configuration: {best['n_workers']} workers")
        print(f"   Achieved: {best_rate:.2f} eps/sec")
        
        # Extrapolate to 100 workers
        if len(successful_results) >= 2:
            avg_efficiency = np.mean([
                (np.mean(r2['rates']) / np.mean(r1['rates'])) / (r2['n_workers'] / r1['n_workers'])
                for r1, r2 in zip(successful_results[:-1], successful_results[1:])
            ])
            
            estimated_100 = best_rate * (100 / best['n_workers']) * avg_efficiency
            print(f"\n📊 Estimated performance with 100 workers:")
            print(f"   ~{estimated_100:.1f} eps/sec (based on {avg_efficiency:.1%} scaling efficiency)")
    
    print("\n" + "="*70)
    print("✅ Testing Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

