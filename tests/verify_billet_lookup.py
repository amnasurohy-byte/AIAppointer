import pandas as pd
from src.inference import Predictor
from config import DATASET_PATH

print("Initializing Predictor...")
predictor = Predictor()

print("Loading Data...")
df = pd.read_csv(DATASET_PATH)

target_role = "Chief Engineer USS Enterprise"
print(f"Looking for candidates for: {target_role}")

# Constraints
if target_role in predictor.constraints:
    const = predictor.constraints[target_role]
    allowed_branches = const.get('branches', [])
    print(f"Allowed Branches: {allowed_branches}")
    
    # Pre-filter
    candidates = df[df['Branch'].isin(allowed_branches)].copy()
    print(f"filtered candidates from {len(df)} down to {len(candidates)}")
else:
    candidates = df.head(100) # Fallback

# Predict
print("Running batch prediction...")
# Limit to 50 for speed in test
candidates = candidates.head(50)
results_list = predictor.predict(candidates)

matches = []
for i, res_df in enumerate(results_list):
    if target_role in res_df['Prediction'].values:
        row = res_df[res_df['Prediction'] == target_role].iloc[0]
        matches.append({
            'Employee_ID': candidates.iloc[i]['Employee_ID'],
            'Rank': candidates.iloc[i]['Rank'],
            'Confidence': row['Confidence'],
            'Explanation': row['Explanation']
        })

if matches:
    match_df = pd.DataFrame(matches).sort_values(by='Confidence', ascending=False)
    print("\nTop 5 Candidates:")
    print(match_df.head(5).to_string())
else:
    print("No matches found in this sample.")
