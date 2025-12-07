import json

with open('models/all_constraints.json') as f:
    constraints = json.load(f)

# Count roles by rank and branch
rank_branch_counts = {}

for role_name, role_const in constraints.items():
    ranks = role_const.get('ranks', [])
    branches = role_const.get('branches', [])
    
    for rank in ranks:
        for branch in branches:
            key = f"{rank} - {branch}"
            if key not in rank_branch_counts:
                rank_branch_counts[key] = []
            rank_branch_counts[key].append(role_name)

# Print Commodore Engineering roles
print("Commodore Engineering roles:")
key = "Commodore - Engineering"
if key in rank_branch_counts:
    print(f"  Found {len(rank_branch_counts[key])} roles")
    for role in rank_branch_counts[key][:10]:
        print(f"    - {role}")
else:
    print("  NO Commodore Engineering roles found!")

# Print Captain Engineering roles for comparison
print("\nCaptain Engineering roles:")
key = "Captain - Engineering"
if key in rank_branch_counts:
    print(f"  Found {len(rank_branch_counts[key])} roles")
    for role in rank_branch_counts[key][:10]:
        print(f"    - {role}")
