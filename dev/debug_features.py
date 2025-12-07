import pandas as pd
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer
import warnings
from config import DATASET_PATH

# Suppress warnings
warnings.filterwarnings('ignore')

print("Loading Data...")
df = pd.read_csv(DATASET_PATH).head(50)

print("Running DataProcessor...")
dp = DataProcessor()
df = dp.get_current_features(df)

print("Running FeatureEngineer...")
fe = FeatureEngineer()
df = fe.extract_features(df)

print("\n--- Verification ---")
columns_to_show = ['Employee_ID', 'Name', 'current_appointment', 'last_role_title', 'days_in_last_role', 'years_service', 'num_prior_roles']
print(df[columns_to_show].head(10))

# Check for target leakage
# The 'last_role_title' should NOT be equal to 'current_appointment'
leakages = df[df['last_role_title'] == df['current_appointment']]
if not leakages.empty:
    print(f"\n[WARNING] POTENTIAL LEAKAGE detected in {len(leakages)} rows!")
    print(leakages[['current_appointment', 'last_role_title']])
else:
    print("\n[SUCCESS] No obvious Target Leakage (last_role != current_appointment).")
    
# Check for negative days?
neg_days = df[df['days_in_last_role'] < 0]
if not neg_days.empty:
    print(f"\n[WARNING] Negative days in last role detected!")
    print(neg_days[['days_in_last_role', 'appointed_since']])
