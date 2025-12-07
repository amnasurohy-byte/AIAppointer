import pandas as pd
import re
from config import DATASET_PATH

# Load the dataset
file_path = DATASET_PATH
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

print(f"Shape: {df.shape}")
print("-" * 30)

# Check columns
print("Columns:", df.columns.tolist())
print("-" * 30)

# Target Variable Analysis
target_col = 'current_appointment'
if target_col in df.columns:
    unique_appointments = df[target_col].nunique()
    print(f"Unique Predicted Classes (current_appointment): {unique_appointments}")
    print(f"Top 10 most common appointments:\n{df[target_col].value_counts().head(10)}")
else:
    print(f"Warning: {target_col} not found.")

print("-" * 30)

# Feature Cardinality
for col in ['Rank', 'Branch', 'Pool', 'Entry_type']:
    if col in df.columns:
        print(f"Unique {col}: {df[col].nunique()}")
        print(df[col].unique()[:10])
print("-" * 30)

# History Field Parser Check
def parse_history_field(text):
    if pd.isna(text):
        return []
    # Values might be separated by comma, but entries themselves contain text.
    # Looking at the raw data, it seems to be quoted.
    # We will try to just print one example to see the structure for now.
    return text

history_cols = ['Appointment_history', 'Training_history', 'Promotion_history']
for col in history_cols:
    if col in df.columns:
        print(f"\nSample {col} (First Non-Null):")
        sample = df[col].dropna().iloc[0]
        print(sample)
