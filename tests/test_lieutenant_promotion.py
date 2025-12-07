"""
Test promotion flexibility with Lieutenant (junior rank with common promotions)
"""
from src.inference import Predictor
import pandas as pd
import json

print("="*60)
print("TEST: Lieutenant → Lieutenant Commander Promotion")
print("="*60)

predictor = Predictor()

# Load constraints
with open('models/all_constraints.json') as f:
    constraints = json.load(f)

# Create dummy Lieutenant officer
dummy_data = {
    'Employee_ID': [999999],
    'Rank': ['Lieutenant'],
    'Branch': ['Tactical Systems'],
    'Pool': ['Planetary Surface'],
    'Entry_type': ['Cadet'],
    'Appointment_history': ['Div Officer USS Vanguard (01 JAN 2300 - )'],
    'Training_history': ['Basic Training (01 JAN 2290 - 01 FEB 2290), Advanced Tactical Systems Course (01 JAN 2295 - 01 JUN 2295)'],
    'Promotion_history': ['Lieutenant (01 JAN 2300 - )'],
    'current_appointment': ['Div Officer USS Vanguard'],
    'appointed_since': ['01/01/2300']
}

dummy_df = pd.DataFrame(dummy_data)

print("\nOfficer Profile:")
print(f"  Rank: Lieutenant")
print(f"  Branch: Tactical Systems")
print(f"  Current Role: Div Officer USS Vanguard")

# Test 1: Strict mode
print("\n" + "-"*60)
print("TEST 1: STRICT (Promotion=0, Demotion=0)")
print("-"*60)
results = predictor.predict(dummy_df, rank_flex_up=0, rank_flex_down=0)
print("\nTop 5 Predictions:")
for idx, row in results.iterrows():
    role = row['Prediction']
    conf = row['Confidence']
    ranks = constraints.get(role, {}).get('ranks', [])
    print(f"  {idx+1}. {role:50s} {conf:.1%} {ranks}")

# Test 2: Allow promotion to Lt. Commander
print("\n" + "-"*60)
print("TEST 2: PROMOTION ALLOWED (Promotion=1, Demotion=0)")
print("Expected: Should see Lieutenant Commander roles")
print("-"*60)
results = predictor.predict(dummy_df, rank_flex_up=1, rank_flex_down=0)

lt_cdr_count = 0
lt_count = 0

print("\nTop 5 Predictions:")
for idx, row in results.iterrows():
    role = row['Prediction']
    conf = row['Confidence']
    ranks = constraints.get(role, {}).get('ranks', [])
    
    rank_label = ""
    if 'Lieutenant Commander' in ranks:
        rank_label = "[LT CDR]"
        lt_cdr_count += 1
    elif 'Lieutenant' in ranks:
        rank_label = "[LT]"
        lt_count += 1
    
    print(f"  {idx+1}. {role:45s} {conf:.1%} {rank_label} {ranks}")

print(f"\nSummary:")
print(f"  Lieutenant Commander roles: {lt_cdr_count}")
print(f"  Lieutenant roles: {lt_count}")

if lt_cdr_count > 0:
    print("\n✅ SUCCESS: Promotion working for Lieutenant!")
else:
    print("\n❌ ISSUE: No Lt Commander roles even with promotion flexibility")

# Check if Lt Commander Tactical roles exist
print("\n" + "-"*60)
print("Checking available roles...")
print("-"*60)

lt_tac_roles = [r for r, v in constraints.items() if 'Lieutenant' in v.get('ranks', []) and 'Tactical Systems' in v.get('branches', [])]
ltcdr_tac_roles = [r for r, v in constraints.items() if 'Lieutenant Commander' in v.get('ranks', []) and 'Tactical Systems' in v.get('branches', [])]

print(f"\nLieutenant Tactical roles: {len(lt_tac_roles)}")
print(f"Lt Commander Tactical roles: {len(ltcdr_tac_roles)}")

if ltcdr_tac_roles:
    print(f"\nSample Lt Commander Tactical roles:")
    for role in ltcdr_tac_roles[:5]:
        print(f"  - {role}")
