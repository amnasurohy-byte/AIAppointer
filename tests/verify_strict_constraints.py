"""
Verify that strict rank constraints are being enforced
"""
import pandas as pd
from src.inference import Predictor
from config import DATASET_PATH

print("Initializing Predictor...")
predictor = Predictor()

print("Loading Data...")
df = pd.read_csv(DATASET_PATH)

# Test Case 1: Captain (Science) should NOT get Commander-only roles
print("\n" + "="*60)
print("TEST 1: Captain (Science) Predictions")
print("="*60)
captain_science = df[(df['Rank'] == 'Captain') & (df['Branch'] == 'Science')].head(1)

if not captain_science.empty:
    results = predictor.predict(captain_science)
    print(f"\nOfficer: {captain_science.iloc[0]['Name']}")
    print(f"Rank: {captain_science.iloc[0]['Rank']}")
    print(f"Branch: {captain_science.iloc[0]['Branch']}")
    print("\nTop 5 Predictions:")
    print(results[['Prediction', 'Confidence', 'Explanation']].to_string())
    
    # Check constraints for each prediction
    print("\n" + "-"*60)
    print("Constraint Verification:")
    for idx, row in results.iterrows():
        role = row['Prediction']
        if role in predictor.constraints:
            allowed_ranks = predictor.constraints[role].get('ranks', [])
            print(f"\n{role}:")
            print(f"  Allowed Ranks: {allowed_ranks}")
            if 'Commander' in allowed_ranks and 'Captain' not in allowed_ranks:
                print(f"  ⚠️  WARNING: This is a Commander-only role!")
        else:
            print(f"\n{role}: No constraints found")
else:
    print("No Captain (Science) found in dataset")

# Test Case 2: Commander (Engineering) should NOT get Captain-only roles
print("\n" + "="*60)
print("TEST 2: Commander (Engineering) Predictions")
print("="*60)
commander_eng = df[(df['Rank'] == 'Commander') & (df['Branch'] == 'Engineering')].head(1)

if not commander_eng.empty:
    results = predictor.predict(commander_eng)
    print(f"\nOfficer: {commander_eng.iloc[0]['Name']}")
    print(f"Rank: {commander_eng.iloc[0]['Rank']}")
    print(f"Branch: {commander_eng.iloc[0]['Branch']}")
    print("\nTop 5 Predictions:")
    print(results[['Prediction', 'Confidence', 'Explanation']].to_string())
    
    print("\n" + "-"*60)
    print("Constraint Verification:")
    for idx, row in results.iterrows():
        role = row['Prediction']
        if role in predictor.constraints:
            allowed_ranks = predictor.constraints[role].get('ranks', [])
            print(f"\n{role}:")
            print(f"  Allowed Ranks: {allowed_ranks}")
            if 'Captain' in allowed_ranks and 'Commander' not in allowed_ranks:
                print(f"  ⚠️  WARNING: This is a Captain-only role!")
        else:
            print(f"\n{role}: No constraints found")
else:
    print("No Commander (Engineering) found in dataset")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
