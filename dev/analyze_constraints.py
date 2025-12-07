import pandas as pd
import json
import os
from src.data_processor import DataProcessor
from config import DATASET_PATH

print("Loading Data...")
df = pd.read_csv(DATASET_PATH)

dp = DataProcessor()
print("Exploding Transitions to map Role -> Constraints...")
df_transitions = dp.create_transition_dataset(df)

# We want maps:
# Role -> Allowed Ranks (Already done, but let's refresh)
# Role -> Allowed Branches
# Role -> Allowed Pools

role_constraints = {}

all_roles = df_transitions['Target_Next_Role'].unique()

print(f"Analyzing {len(all_roles)} roles...")

for role in all_roles:
    # Filter rows where this role was the target
    subset = df_transitions[df_transitions['Target_Next_Role'] == role]
    
    allowed_ranks = list(subset['Rank'].unique())
    allowed_branches = list(subset['Branch'].unique())
    allowed_pools = list(subset['Pool'].unique())
    
    # Filter out NaNs if any
    allowed_ranks = [x for x in allowed_ranks if pd.notna(x) and x != 'Unknown']
    allowed_branches = [x for x in allowed_branches if pd.notna(x)]
    allowed_pools = [x for x in allowed_pools if pd.notna(x)]
    
    role_constraints[role] = {
        'ranks': allowed_ranks,
        'branches': allowed_branches,
        'pools': allowed_pools
    }

# Specific check for user query: "Staff Officer" and Science Branch
full_map_check = False
for role, const in role_constraints.items():
    if "Staff Officer" in role:
        print(f"\nConstraint for '{role}':")
        print(f"  Branches: {const['branches']}")
        if "Science" in const['branches']:
            print("  WARNING: Science IS allowed based on data.")
        else:
            print("  Science is correctly EXCLUDED.")

# Save
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/all_constraints.json', 'w') as f:
    json.dump(role_constraints, f, indent=2)

print("\nSaved constraints to models/all_constraints.json")
