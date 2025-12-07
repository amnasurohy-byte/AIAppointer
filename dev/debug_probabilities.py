"""
Debug: Show ALL probabilities before/after constraint filtering
"""
from src.inference import Predictor
import pandas as pd
import numpy as np
import json

predictor = Predictor()

# Load constraints
with open('models/all_constraints.json') as f:
    constraints = json.load(f)

# Captain Engineering scenario
dummy_data = {
    'Employee_ID': [999999],
    'Rank': ['Captain'],
    'Branch': ['Engineering'],
    'Pool': ['Planetary Surface'],
    'Entry_type': ['Cadet'],
    'Appointment_history': ['Director R&D - Plasma Systems (01 JAN 2300 - )'],
    'Training_history': ['Basic Training (01 JAN 2290 - 01 FEB 2290), Advanced Engineering Course (01 JAN 2295 - 01 JUN 2295)'],
    'Promotion_history': ['Captain (01 JAN 2300 - )'],
    'current_appointment': ['Director R&D - Plasma Systems'],
    'appointed_since': ['01/01/2300']
}

dummy_df = pd.DataFrame(dummy_data)

print("="*60)
print("DEEP DIVE: Captain Engineering Probabilities")
print("="*60)

# Manually run prediction logic to see probabilities
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer

df = dummy_df.copy()
dp = DataProcessor()
fe = FeatureEngineer()

df = dp.get_current_features(df)
df = fe.extract_features(df)

# Encoding
cat_features = ['Rank', 'Branch', 'Pool', 'Entry_type', 'last_role_title']
for col in cat_features:
    if col in predictor.encoders:
        le = predictor.encoders[col]
        known_classes = set(le.classes_)
        def encode_safe(val):
            val_str = str(val)
            if val_str in known_classes:
                return le.transform([val_str])[0]
            if 'Unknown' in known_classes:
                 return le.transform(['Unknown'])[0]
            return 0 
        df[col] = df[col].apply(encode_safe)

for col in predictor.feature_cols:
    if col not in df.columns:
        df[col] = 0

X = df[predictor.feature_cols]
probas = predictor.model.predict_proba(X)[0]

all_classes = predictor.target_encoder.classes_

# Find top Commodore Engineering roles by RAW probability
commodore_eng_probs = []
captain_eng_probs = []

for idx, role_name in enumerate(all_classes):
    if role_name in constraints:
        ranks = constraints[role_name].get('ranks', [])
        branches = constraints[role_name].get('branches', [])
        
        if 'Engineering' in branches:
            if 'Commodore' in ranks:
                commodore_eng_probs.append((role_name, probas[idx]))
            elif 'Captain' in ranks:
                captain_eng_probs.append((role_name, probas[idx]))

# Sort by probability
commodore_eng_probs.sort(key=lambda x: x[1], reverse=True)
captain_eng_probs.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 COMMODORE Engineering roles by RAW model probability:")
for role, prob in commodore_eng_probs[:10]:
    print(f"  {role:60s} {prob:.6f} ({prob*100:.4f}%)")

print("\nTop 10 CAPTAIN Engineering roles by RAW model probability:")
for role, prob in captain_eng_probs[:10]:
    print(f"  {role:60s} {prob:.6f} ({prob*100:.4f}%)")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if commodore_eng_probs:
    max_commodore_prob = commodore_eng_probs[0][1]
    max_captain_prob = captain_eng_probs[0][1]
    
    print(f"\nHighest Commodore probability: {max_commodore_prob:.6f} ({max_commodore_prob*100:.4f}%)")
    print(f"Highest Captain probability: {max_captain_prob:.6f} ({max_captain_prob*100:.4f}%)")
    print(f"Ratio: {max_commodore_prob/max_captain_prob:.4f}")
    
    if max_commodore_prob < max_captain_prob * 0.01:
        print("\n❌ ISSUE: Commodore probabilities are 100x lower than Captain")
        print("   The model strongly learned Captain→Captain patterns")
        print("   Even with constraints allowing Commodore, they won't appear in Top-5")
    else:
        print("\n✓ Commodore probabilities are reasonable")
