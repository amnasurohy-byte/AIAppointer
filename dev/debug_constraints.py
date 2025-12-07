"""
Debug script to trace constraint enforcement
"""
import pandas as pd
import json
from src.inference import Predictor
from config import DATASET_PATH

print("Loading constraints...")
with open('models/all_constraints.json') as f:
    constraints = json.load(f)

# Check a specific role
role = "Director R&D - Plasma Systems"
if role in constraints:
    print(f"\nConstraints for '{role}':")
    print(f"  Ranks: {constraints[role]['ranks']}")
    print(f"  Branches: {constraints[role]['branches']}")
else:
    print(f"\n'{role}' not found in constraints")

print("\n" + "="*60)
print("Testing Rank Matching Logic")
print("="*60)

rank_order = ['Ensign', 'Lieutenant (jg)', 'Lieutenant', 'Lieutenant Commander', 'Commander', 'Captain', 'Commodore', 'Rear Admiral', 'Unknown']
rank_map = {r: i for i, r in enumerate(rank_order)}

user_rank = 'Commander'
user_rank_idx = rank_map[user_rank]

print(f"\nUser Rank: {user_rank} (index={user_rank_idx})")

# Test against Captain-only role
allowed_ranks = ['Captain']
print(f"\nTesting against role with allowed_ranks: {allowed_ranks}")

for r in allowed_ranks:
    r_idx = rank_map.get(str(r).strip(), -1)
    print(f"  Role rank '{r}' has index: {r_idx}")
    print(f"  Exact match (r_idx == user_rank_idx): {r_idx} == {user_rank_idx} = {r_idx == user_rank_idx}")
    print(f"  Promotion (+1): {r_idx} == {user_rank_idx + 1} = {r_idx == user_rank_idx + 1}")
    print(f"  SHOULD ALLOW: {r_idx == user_rank_idx or r_idx == user_rank_idx + 1}")

print("\n" + "="*60)
print("Now testing actual predictor...")
print("="*60)

predictor = Predictor()
df = pd.read_csv(DATASET_PATH)

commander_eng = df[(df['Rank'] == 'Commander') & (df['Branch'] == 'Engineering')].head(1)
results = predictor.predict(commander_eng)

print("\nPredictions:")
print(results[['Prediction', 'Confidence']].to_string())

# Check if Captain-only roles appear
for idx, row in results.iterrows():
    role_name = row['Prediction']
    if role_name in constraints:
        allowed = constraints[role_name]['ranks']
        if 'Captain' in allowed and 'Commander' not in allowed:
            print(f"\n⚠️  VIOLATION: '{role_name}' is Captain-only but was recommended!")
            print(f"   Allowed ranks: {allowed}")
