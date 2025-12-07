"""
Test strict rank constraint enforcement (rank_flexibility=0)
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

# Test Commander with STRICT mode
print("\n" + "="*60)
print("TEST: Commander (Engineering) with STRICT mode (flex=0)")
print("="*60)

commander = df[(df['Rank'] == 'Commander') & (df['Branch'] == 'Engineering')].head(1)

if not commander.empty:
    results = predictor.predict(commander, rank_flexibility=0)
    
    print(f"\nOfficer: {commander.iloc[0]['Name']}")
    print(f"Rank: {commander.iloc[0]['Rank']}")
    print(f"Branch: {commander.iloc[0]['Branch']}")
    
    print("\nTop 5 Predictions:")
    print(results[['Prediction', 'Confidence']].to_string())
    
    print("\n" + "-"*60)
    print("Constraint Verification:")
    violations = 0
    
    for idx, row in results.iterrows():
        role = row['Prediction']
        if role in constraints:
            allowed = constraints[role]['ranks']
            if 'Commander' not in allowed:
                print(f"\n⚠️  VIOLATION: '{role}'")
                print(f"   Allowed ranks: {allowed}")
                print(f"   Commander is NOT in allowed ranks!")
                violations += 1
            else:
                print(f"\n✓ '{role}' - Commander IS allowed")
        else:
            print(f"\n? '{role}' - No constraints found")
    
    print("\n" + "="*60)
    if violations == 0:
        print("✅ SUCCESS: No violations found! Strict mode working correctly.")
    else:
        print(f"❌ FAILURE: {violations} violations found!")
    print("="*60)
else:
    print("No Commander (Engineering) found in dataset")
