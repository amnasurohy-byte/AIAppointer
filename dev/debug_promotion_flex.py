"""
Debug script to verify directional rank flexibility logic
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

# Test Commander with promotion flexibility
print("\n" + "="*60)
print("TEST: Commander (Engineering) with Promotion Flexibility")
print("="*60)

commander = df[(df['Rank'] == 'Commander') & (df['Branch'] == 'Engineering')].head(1)

if not commander.empty:
    print(f"\nOfficer: {commander.iloc[0]['Name']}")
    print(f"Rank: {commander.iloc[0]['Rank']}")
    print(f"Branch: {commander.iloc[0]['Branch']}")
    
    # Test 1: Strict (no flexibility)
    print("\n" + "-"*60)
    print("Test 1: STRICT (rank_flex_up=0, rank_flex_down=0)")
    print("-"*60)
    results = predictor.predict(commander, rank_flex_up=0, rank_flex_down=0)
    print(results[['Prediction', 'Confidence']].to_string())
    
    # Check ranks
    print("\nRank distribution:")
    for idx, row in results.iterrows():
        role = row['Prediction']
        if role in constraints:
            allowed_ranks = constraints[role]['ranks']
            print(f"  {role}: {allowed_ranks}")
    
    # Test 2: Allow promotions
    print("\n" + "-"*60)
    print("Test 2: PROMOTION ALLOWED (rank_flex_up=1, rank_flex_down=0)")
    print("-"*60)
    results = predictor.predict(commander, rank_flex_up=1, rank_flex_down=0)
    print(results[['Prediction', 'Confidence']].to_string())
    
    # Check ranks
    print("\nRank distribution:")
    captain_roles = 0
    commander_roles = 0
    for idx, row in results.iterrows():
        role = row['Prediction']
        if role in constraints:
            allowed_ranks = constraints[role]['ranks']
            print(f"  {role}: {allowed_ranks}")
            if 'Captain' in allowed_ranks and 'Commander' not in allowed_ranks:
                captain_roles += 1
            elif 'Commander' in allowed_ranks:
                commander_roles += 1
    
    print(f"\nSummary:")
    print(f"  Captain-only roles: {captain_roles}")
    print(f"  Commander roles: {commander_roles}")
    
    if captain_roles > 0:
        print("\n✅ SUCCESS: Promotion flexibility is working!")
    else:
        print("\n❌ ISSUE: No Captain roles found even with rank_flex_up=1")
        print("   This could mean:")
        print("   1. No Captain roles exist for Engineering branch in constraints")
        print("   2. The logic is not working correctly")
        
        # Check if Captain Engineering roles exist
        print("\n   Checking for Captain Engineering roles in constraints...")
        captain_eng_roles = []
        for role_name, role_const in constraints.items():
            if 'Captain' in role_const.get('ranks', []) and 'Engineering' in role_const.get('branches', []):
                captain_eng_roles.append(role_name)
        
        if captain_eng_roles:
            print(f"   Found {len(captain_eng_roles)} Captain Engineering roles:")
            for role in captain_eng_roles[:5]:
                print(f"     - {role}")
        else:
            print("   ⚠️  NO Captain Engineering roles found in constraints!")
            print("   The model cannot recommend what doesn't exist in the data.")
