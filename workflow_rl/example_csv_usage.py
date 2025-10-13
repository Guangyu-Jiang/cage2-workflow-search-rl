"""
Example: How to use the consolidated training log CSV

The new single CSV file contains both episode-level and summary-level data.
Use the 'Type' column to filter between them.
"""

import pandas as pd

# Load the training log
df = pd.read_csv('compliance_checkpoints/workflow_0_training_log.csv')

print("="*60)
print("EXAMPLE: Working with Consolidated Training Log")
print("="*60)

# 1. Get all episode data
episodes = df[df['Type'] == 'episode']
print(f"\n1. Episode rows: {len(episodes)} entries")
print(episodes.head())

# 2. Get all summary data
summaries = df[df['Type'] == 'summary']
print(f"\n2. Summary rows: {len(summaries)} entries")
print(summaries.head())

# 3. Analyze specific environment
print("\n3. Environment 5 performance:")
env_5 = episodes[episodes['Env_ID'] == 5]
print(f"   Episodes: {len(env_5)}")
print(f"   Avg Env Reward: {env_5['Env_Reward'].astype(float).mean():.2f}")
print(f"   Avg Compliance: {env_5['Compliance'].astype(float).mean():.2%}")

# 4. Track progress over time (from summaries)
print("\n4. Training progress (from summaries):")
print(summaries[['Total_Episodes', 'Env_Reward', 'Compliance']].to_string(index=False))

# 5. Find best episode
best_idx = episodes['Env_Reward'].astype(float).idxmax()
best = episodes.loc[best_idx]
print(f"\n5. Best episode:")
print(f"   Episode: {best['Episode']} (Env {best['Env_ID']})")
print(f"   Env Reward: {best['Env_Reward']}")
print(f"   Compliance: {float(best['Compliance']):.2%}")

# 6. Compare early vs late training
print("\n6. Early vs Late training:")
early = episodes[episodes['Episode'].astype(float) <= 10]
late = episodes[episodes['Episode'].astype(float) >= 90]
print(f"   Early (ep 1-10): Avg Reward={early['Env_Reward'].astype(float).mean():.2f}, "
      f"Compliance={early['Compliance'].astype(float).mean():.2%}")
print(f"   Late  (ep 90+) : Avg Reward={late['Env_Reward'].astype(float).mean():.2f}, "
      f"Compliance={late['Compliance'].astype(float).mean():.2%}")

print("\n" + "="*60)
print("TIP: Use Type column to filter episode vs summary rows")
print("="*60)

