"""
Utility functions for constraint generation
"""
import pandas as pd
import json
import os
from src.data_processor import DataProcessor

def generate_constraints(dataset_path, output_dir='models', verbose=True):
    """
    Generate role constraints (Rank, Branch, Pool) from dataset.
    
    Args:
        dataset_path: Path to the CSV dataset
        output_dir: Directory to save constraints JSON
        verbose: Print progress messages
    
    Returns:
        dict: Role constraints mapping
    """
    if verbose:
        print(f"Generating constraints from {dataset_path}...")
    
    df = pd.read_csv(dataset_path)
    dp = DataProcessor()
    
    if verbose:
        print("Creating transition dataset...")
    df_transitions = dp.create_transition_dataset(df)
    
    role_constraints = {}
    role_map = {} # Generalized -> [Specific Roles]

    all_roles = df_transitions['Target_Next_Role'].unique()
    
    if verbose:
        print(f"Analyzing {len(all_roles)} unique roles...")
    
    for role in all_roles:
        # Get Generalized Name
        gen_role = dp.normalize_role(role)
        if gen_role not in role_map:
            role_map[gen_role] = []
        if role not in role_map[gen_role]:
            role_map[gen_role].append(role)

        subset = df_transitions[df_transitions['Target_Next_Role'] == role]
        
        allowed_ranks = list(subset['Rank'].unique())
        allowed_branches = list(subset['Branch'].unique())
        allowed_pools = list(subset['Pool'].unique())
        
        # Filter out NaNs
        allowed_ranks = [x for x in allowed_ranks if pd.notna(x) and x != 'Unknown']
        allowed_branches = [x for x in allowed_branches if pd.notna(x)]
        allowed_pools = [x for x in allowed_pools if pd.notna(x)]
        
        role_constraints[role] = {
            'ranks': allowed_ranks,
            'branches': allowed_branches,
            'pools': allowed_pools,
            'generalized_role': gen_role
        }
    
    # Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'all_constraints.json')
    with open(output_path, 'w') as f:
        json.dump(role_constraints, f, indent=2)

    map_path = os.path.join(output_dir, 'role_map.json')
    with open(map_path, 'w') as f:
        json.dump(role_map, f, indent=2)
    
    if verbose:
        print(f"✓ Saved {len(role_constraints)} role constraints to {output_path}")
        print(f"✓ Saved {len(role_map)} generalized mappings to {map_path}")
    
    return role_constraints

if __name__ == "__main__":
    from config import DATASET_PATH
    generate_constraints(DATASET_PATH)
