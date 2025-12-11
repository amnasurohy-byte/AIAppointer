
import pandas as pd
import json
import difflib
import re
from src.predictor import Predictor
from src.explainer import Explainer

def main():
    print("--- Debugging Mismatch ---")
    
    # 1. Load Predictor (Source of '20%')
    pred = Predictor()
    print("Predictor loaded.")
    
    # Check stats for the specific mismatch reported
    # From: 'Training Officer - Fleet Technical Training Center'
    # To: 'Dy Director R&D - Quantum Computing / Post 3'
    
    t_from = 'Training Officer - Fleet Technical Training Center'
    t_to = 'Dy Director R&D - Quantum Computing / Post 3'
    
    # Normalize using Predictor's logic? Predictor uses raw strings usually or normalized?
    # Predictor stores keys in transition_stats['title_trans']
    
    stats = pred.transition_stats.get('title_trans', {})
    
    # Check if exact key exists
    if t_from in stats:
        print(f"Predictor Stats for '{t_from}':")
        targets = stats[t_from]
        if t_to in targets:
            print(f"  -> '{t_to}': {targets[t_to]} (Count in Offline Stats)")
        else:
            print(f"  -> '{t_to}': NOT FOUND in targets.")
            # Fuzzy check targets
            print("  Closest targets:")
            print(difflib.get_close_matches(t_to, targets.keys()))
    else:
        print(f"'{t_from}' NOT FOUND in Predictor Stats keys.")
        
    # 2. Check Explainer (Source of 'N=0')
    print("\nLoading Data for Explainer...")
    df = pd.read_csv('data/hr_star_trek_v4c_modernized_clean_modified_v4.csv')
    
    # Initialize Explainer
    # We pass known titles from stats to ensure normalization matches
    known_titles = list(stats.keys())
    exp = Explainer(df, known_titles=known_titles)
    
    # Check Index
    # Explainer uses normalized keys
    k_from = exp._normalize_title(t_from)
    k_to = exp._normalize_title(t_to)
    
    print(f"\nExplainer Normalized Keys:")
    print(f"  From: '{k_from}'")
    print(f"  To:   '{k_to}'")
    
    matches = exp.get_precedents(t_from, t_to)
    print(f"Explainer.get_precedents returned: {len(matches)} matches.")
    
    if len(matches) == 0:
        print("Mismatch Confirmed.")
        
        # Deep Dive: Why did Explainer miss it?
        # Scan the DF for an officer who supposedly made this move.
        # We look for someone with both strings in history.
        print("\nScanning raw dataframe for potential matches...")
        
        for idx, row in df.iterrows():
            hist = str(row['Appointment_history'])
            if t_from in hist and t_to in hist:
                print(f"  Found Candidate: Officer {row['Employee_ID']}")
                print(f"  History snippet: {hist[:100]}...")
                
                # Trace Explainer logic on this row
                # Normalize titles found in history
                items = hist.split(',')
                titles = [re.sub(r'\s*\(.*?\)', '', item).strip() for item in items]
                norm_titles = [exp._normalize_title(t) for t in titles]
                
                print(f"  Parsed Titles: {titles}")
                print(f"  Norm Titles:   {norm_titles}")
                
                # Check for pair
                found_pair = False
                for i in range(len(norm_titles)-1):
                    if norm_titles[i] == k_from and norm_titles[i+1] == k_to:
                        found_pair = True
                        print("  -> Explainer SHOULD have indexed this pair!")
                        break
                
                if not found_pair:
                    print("  -> Explainer pair check FAILED. Normalization mismatch?")
                    if k_from not in norm_titles:
                        print(f"     '{k_from}' not in normalized list.")
                    if k_to not in norm_titles:
                        print(f"     '{k_to}' not in normalized list.")

if __name__ == "__main__":
    main()
