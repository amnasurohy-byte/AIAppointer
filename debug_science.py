
import pandas as pd
import json
import re
from src.predictor import Predictor

def main():
    print("Loading Data...")
    try:
        # Use checked path from list_dir
        df = pd.read_csv('data/hr_star_trek_v4c_modernized_clean_modified_v4.csv')
        print(f"Loaded {len(df)} rows from raw CSV.")
    except Exception as e:
        print(f"Could not load data: {e}")
        return

    # 1. Find a Junior Science Officer (Lieutenant or Lt jg)
    science_juniors = df[
        (df['Branch'] == 'Science') & 
        (df['Rank'].isin(['Lieutenant', 'Lieutenant (jg)']))
    ]
    
    if science_juniors.empty:
        print("No Junior Science Officers found in data!")
    else:
        target_officer = science_juniors.iloc[0]
        print(f"\n--- Analysis Target ---")
        print(f"ID: {target_officer['Employee_ID']}")
        print(f"Rank: {target_officer['Rank']}")
        print(f"Branch: {target_officer['Branch']}")
        
        # 2. Reproduce the bug (Run Prediction)
        print("\n--- Running Prediction (Reproduction) ---")
        predictor = Predictor()
        # Mock context like app.py
        # predictor.predict expects a single row dict/series?
        # predictor.predict uses self.all_candidates/valid_roles for search space.
        
        # We need to construct the input dict
        officer_input = target_officer.to_dict()
        officer_input['last_role_title'] = officer_input.get('current_appointment', 'Unknown') # Hack for test
        
        try:
            results = predictor.predict(officer_input)
            print(f"Prediction Candidates Found: {len(results)}")
            if len(results) == 0:
                print("BUG REPRODUCED: No candidates returned.")
            else:
                print("Result Examples:")
                print(results[['Role', 'Confidence']].head())
        except Exception as e:
            print(f"Prediction failed: {e}")

    # 3. Analyze Senior Science Officers
    print("\n--- Analyzing Senior Science History ---")
    seniors = df[
        (df['Branch'] == 'Science') & 
        (df['Rank'].isin(['Captain', 'Commander']))
    ]
    
    all_past_roles = []
    
    for idx, row in seniors.iterrows():
        hist = str(row.get('Appointment_history', ''))
        # Parse history
        items = hist.split(',')
        for item in items:
            clean = re.sub(r'\s*\(.*?\)', '', item).strip()
            if clean:
                all_past_roles.append(clean)
                
    unique_past = set(all_past_roles)
    print(f"Found {len(unique_past)} unique past roles held by Senior Science Officers.")
    
    # 4. Check these roles in Constraints
    print("\n--- Checking Constraints for Top Past Roles ---")
    
    # Load constraints
    try:
        with open('models/all_constraints.json', 'r') as f:
            cons = json.load(f)
    except:
        print("Could not load constraints json.")
        cons = {}
        
    # Check top 10 most common past roles (if we counted, but here just set)
    # Let's count them
    from collections import Counter
    counts = Counter(all_past_roles)
    
    matched_count = 0
    mismatched_roles = []
    
    for role, freq in counts.most_common(20):
        if role in cons:
            c = cons[role]
            branches = c.get('branches', [])
            ranks = c.get('ranks', [])
            
            print(f"Role: {role} (Freq: {freq})")
            print(f"  > Constraint Branches: {branches}")
            print(f"  > Constraint Ranks: {ranks}")
            
            if 'Science' not in branches:
                print("  > MISMATCH: Held by Science officers but constraint excludes Science!")
                mismatched_roles.append(role)
            else:
                print("  > OK: Science allowed.")
                matched_count += 1
        else:
            print(f"Role: {role} (Freq: {freq}) - NOT IN CONSTRAINTS JSON")

    print(f"\nSummary: Found {len(mismatched_roles)} roles that Science officers held but are blocked by constraints.")

if __name__ == "__main__":
    main()
