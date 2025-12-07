import pandas as pd
from src.data_processor import DataProcessor

print("Loading Data...")
df = pd.read_csv(DATASET_PATH)

# We want to know: For each "Appointment" string, what "Ranks" held it?
# This will build our allowable transition map.

# Explode history similarly to training
dp = DataProcessor()
print("Exploding Transitions to map Role -> Rank...")
df_transitions = dp.create_transition_dataset(df)

# Group by Target Role and see the Ranks needed
role_rank_map = df_transitions.groupby('Target_Next_Role')['Rank'].unique()

print(f"\nAnalyzed {len(role_rank_map)} unique roles.")

# Check for consistency
inconsistent_roles = 0
for role, ranks in role_rank_map.items():
    if len(ranks) > 1:
        # print(f"Role '{role}' held by multiple ranks: {ranks}")
        inconsistent_roles += 1

print(f"Roles held by multiple ranks: {inconsistent_roles} / {len(role_rank_map)}")

# Sample Mapping
print("\nSample Role -> Rank requirements:")
for role in list(role_rank_map.index)[:10]:
     print(f"{role} : {role_rank_map[role]}")

# Save this mapping logic?
# We can create a 'RankConstraint' dictionary.
import json
import os
from config import DATASET_PATH

constraints = {}
for role, ranks in role_rank_map.items():
    # We take the lowest rank seen? Or list?
    # To be safe: List of allowed ranks.
    constraints[role] = list(ranks)

if not os.path.exists('models'):
    os.makedirs('models')

with open('models/rank_constraints.json', 'w') as f:
    json.dump(constraints, f, indent=2)
    
print("Saved constraints to models/rank_constraints.json")
