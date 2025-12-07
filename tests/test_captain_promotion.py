"""
Comprehensive test of promotion flexibility for ALL ranks
"""
from src.inference import Predictor
from config import DATASET_PATH
import pandas as pd
import json

print("Initializing Predictor...")
predictor = Predictor()

print("Loading Data...")
df = pd.read_csv(DATASET_PATH)

# Load constraints
with open('models/all_constraints.json') as f:
    constraints = json.load(f)

# Test Captain → Commodore promotion
print("\n" + "="*60)
print("TEST: Captain (Engineering) → Commodore roles")
print("="*60)

captain = df[(df['Rank'] == 'Captain') & (df['Branch'] == 'Engineering')].head(1)

if not captain.empty:
    print(f"\nOfficer: {captain.iloc[0]['Name']}")
    print(f"Rank: {captain.iloc[0]['Rank']}")
    print(f"Branch: {captain.iloc[0]['Branch']}")
    
    # Test with promotion flexibility
    print("\n" + "-"*60)
    print("With rank_flex_up=1 (should allow Commodore roles)")
    print("-"*60)
    results = predictor.predict(captain, rank_flex_up=1, rank_flex_down=0)
    print(results[['Prediction', 'Confidence']].to_string())
    
    # Check ranks
    print("\nRank distribution:")
    commodore_roles = 0
    captain_roles = 0
    for idx, row in results.iterrows():
        role = row['Prediction']
        if role in constraints:
            allowed_ranks = constraints[role]['ranks']
            print(f"  {role}: {allowed_ranks}")
            if 'Commodore' in allowed_ranks:
                commodore_roles += 1
            elif 'Captain' in allowed_ranks:
                captain_roles += 1
    
    print(f"\nSummary:")
    print(f"  Commodore roles: {commodore_roles}")
    print(f"  Captain roles: {captain_roles}")
    
    if commodore_roles > 0:
        print("\n✅ Captain → Commodore promotion working!")
    else:
        print("\n❌ No Commodore roles found")
        
        # Check if Commodore Engineering roles exist
        print("\n   Checking for Commodore Engineering roles in constraints...")
        commodore_eng_roles = []
        for role_name, role_const in constraints.items():
            if 'Commodore' in role_const.get('ranks', []) and 'Engineering' in role_const.get('branches', []):
                commodore_eng_roles.append(role_name)
        
        if commodore_eng_roles:
            print(f"   Found {len(commodore_eng_roles)} Commodore Engineering roles:")
            for role in commodore_eng_roles[:10]:
                print(f"     - {role}")
        else:
            print("   ⚠️  NO Commodore Engineering roles in constraints!")

# Also test the rank order
print("\n" + "="*60)
print("RANK ORDER VERIFICATION")
print("="*60)

rank_order = ['Ensign', 'Lieutenant (jg)', 'Lieutenant', 'Lieutenant Commander', 'Commander', 'Captain', 'Commodore', 'Rear Admiral', 'Unknown']
rank_map = {r: i for i, r in enumerate(rank_order)}

print("Rank indices:")
for rank, idx in rank_map.items():
    print(f"  {rank:25s} = {idx}")

print("\nTesting promotion logic:")
user_rank = 'Captain'
user_rank_idx = rank_map[user_rank]
print(f"\nUser rank: {user_rank} (idx={user_rank_idx})")

test_roles = ['Captain', 'Commodore', 'Rear Admiral', 'Commander']
for test_rank in test_roles:
    test_idx = rank_map[test_rank]
    rank_diff = test_idx - user_rank_idx
    
    print(f"\n  Testing role rank: {test_rank} (idx={test_idx})")
    print(f"    rank_diff = {test_idx} - {user_rank_idx} = {rank_diff}")
    
    if rank_diff >= 0:
        print(f"    This is a PROMOTION (diff={rank_diff})")
        print(f"    With rank_flex_up=1: {rank_diff} <= 1? {rank_diff <= 1}")
        print(f"    With rank_flex_up=2: {rank_diff} <= 2? {rank_diff <= 2}")
    else:
        print(f"    This is a DEMOTION (diff={rank_diff})")
