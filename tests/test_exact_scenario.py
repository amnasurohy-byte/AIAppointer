"""
Exact simulation of what user sees in UI
"""
from src.inference import Predictor
import pandas as pd

print("="*60)
print("SIMULATING EXACT USER SCENARIO")
print("="*60)

predictor = Predictor()

# Create the EXACT dummy data that Simulation mode creates
# User settings: Captain, Engineering, (any pool), (any entry), (any current role)
dummy_data = {
    'Employee_ID': [999999],
    'Rank': ['Captain'],
    'Branch': ['Engineering'],
    'Pool': ['Planetary Surface'],  # Default from UI
    'Entry_type': ['Cadet'],  # User mentioned "cadet"
    'Appointment_history': ['Director R&D - Plasma Systems (01 JAN 2300 - )'],  # Example Captain Engineering role
    'Training_history': ['Basic Training (01 JAN 2290 - 01 FEB 2290), Advanced Engineering Course (01 JAN 2295 - 01 JUN 2295)'],
    'Promotion_history': ['Captain (01 JAN 2300 - )'],
    'current_appointment': ['Director R&D - Plasma Systems'],
    'appointed_since': ['01/01/2300']
}

dummy_df = pd.DataFrame(dummy_data)

print("\nSimulation Settings:")
print(f"  Rank: Captain")
print(f"  Branch: Engineering")
print(f"  Entry: Cadet")
print(f"  Current Role: Director R&D - Plasma Systems")

# Test 1: Strict mode (both sliders at 0)
print("\n" + "-"*60)
print("TEST 1: STRICT MODE (Promotion=0, Demotion=0)")
print("-"*60)
results = predictor.predict(dummy_df, rank_flex_up=0, rank_flex_down=0)
print("\nTop 5 Predictions:")
for idx, row in results.iterrows():
    print(f"  {idx+1}. {row['Prediction']:50s} {row['Confidence']:.1%}")

# Test 2: Promotion allowed (up=1, down=0)
print("\n" + "-"*60)
print("TEST 2: PROMOTION ALLOWED (Promotion=1, Demotion=0)")
print("-"*60)
print("Expected: Should see Commodore roles")
results = predictor.predict(dummy_df, rank_flex_up=1, rank_flex_down=0)
print("\nTop 5 Predictions:")
commodore_count = 0
captain_count = 0

import json
with open('models/all_constraints.json') as f:
    constraints = json.load(f)

for idx, row in results.iterrows():
    role = row['Prediction']
    conf = row['Confidence']
    
    # Check rank
    rank_label = ""
    if role in constraints:
        ranks = constraints[role].get('ranks', [])
        if 'Commodore' in ranks:
            rank_label = "[COMMODORE]"
            commodore_count += 1
        elif 'Captain' in ranks:
            rank_label = "[CAPTAIN]"
            captain_count += 1
    
    print(f"  {idx+1}. {role:45s} {conf:.1%} {rank_label}")

print(f"\nSummary:")
print(f"  Commodore roles: {commodore_count}")
print(f"  Captain roles: {captain_count}")

if commodore_count > 0:
    print("\n✅ SUCCESS: Promotion flexibility is working!")
else:
    print("\n❌ ISSUE: No Commodore roles even with rank_flex_up=1")
    print("   This suggests the model is not learning to predict higher ranks")
    print("   even when constraints allow it.")

# Test 3: Higher promotion (up=2, down=0)
print("\n" + "-"*60)
print("TEST 3: HIGHER PROMOTION (Promotion=2, Demotion=0)")
print("-"*60)
print("Expected: Should see Commodore and possibly Rear Admiral roles")
results = predictor.predict(dummy_df, rank_flex_up=2, rank_flex_down=0)
print("\nTop 5 Predictions:")
for idx, row in results.iterrows():
    role = row['Prediction']
    conf = row['Confidence']
    
    rank_label = ""
    if role in constraints:
        ranks = constraints[role].get('ranks', [])
        if 'Rear Admiral' in ranks:
            rank_label = "[REAR ADMIRAL]"
        elif 'Commodore' in ranks:
            rank_label = "[COMMODORE]"
        elif 'Captain' in ranks:
            rank_label = "[CAPTAIN]"
    
    print(f"  {idx+1}. {role:45s} {conf:.1%} {rank_label}")
