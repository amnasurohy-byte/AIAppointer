"""
Test strict rank constraint enforcement (rank_flex_up=0, rank_flex_down=0)
"""
from src.inference import Predictor
from config import DATASET_PATH
import pandas as pd
import json

print("Initializing Predictor...")
predictor = Predictor()

print("Loading Data...")
df = pd.read_csv(DATASET_PATH)

# Test Commander with STRICT mode
print("\n" + "="*60)
print("TEST: Commander (Engineering) with STRICT mode (flex=0)")
print("="*60)

commander = df[(df['Rank'] == 'Commander') & (df['Branch'] == 'Engineering')].head(1)

if not commander.empty:
    # UPDATED API CALL: Using new parameters
    results = predictor.predict(commander, rank_flex_up=0, rank_flex_down=0)
    
    print(f"\nOfficer: {commander.iloc[0]['Name']}")
    print(f"Rank: {commander.iloc[0]['Rank']}")
    print(f"Branch: {commander.iloc[0]['Branch']}")
    
    print("\nTop 5 Predictions:")
    print(results[['Prediction', 'Confidence']].to_string())
    
    # We cannot strictly check constraints against 'all_constraints.json' anymore
    # because 'all_constraints.json' now contains keys for GENERALIZED roles (e.g. "Div Officer"),
    # but 'results' contains SPECIFIC roles (e.g. "Div Officer USS Vanguard").
    # We must normalize the prediction before checking.

    print("\n" + "-"*60)
    print("Constraint Verification:")
    violations = 0
    
    for idx, row in results.iterrows():
        specific_role = row['Prediction']
        # Normalize to check constraints
        gen_role = predictor.dp.normalize_role_title(specific_role)

        if gen_role in predictor.constraints:
            allowed = predictor.constraints[gen_role]['ranks']
            if 'Commander' not in allowed:
                print(f"\n⚠️  VIOLATION: '{specific_role}' (Type: {gen_role})")
                print(f"   Allowed ranks: {allowed}")
                print(f"   Commander is NOT in allowed ranks!")
                violations += 1
            else:
                print(f"\n✓ '{specific_role}' (Type: {gen_role}) - Commander IS allowed")
        else:
            print(f"\n? '{specific_role}' - No constraints found (Type: {gen_role})")
    
    print("\n" + "="*60)
    if violations == 0:
        print("✅ SUCCESS: No violations found! Strict mode working correctly.")
    else:
        print(f"❌ FAILURE: {violations} violations found!")
    print("="*60)
else:
    print("No Commander (Engineering) found in dataset")
