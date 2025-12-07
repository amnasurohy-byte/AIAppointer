import pandas as pd
from src.data_processor import DataProcessor
from config import DATASET_PATH

df = pd.read_csv(DATASET_PATH)
dp = DataProcessor()

# Test on Appointment_history
print("Sample Parsed Appointments:")
for text in df['Appointment_history'].head(5):
    print(f"RAW: {text}")
    parsed = dp.parse_history_column(text)
    print(f"PARSED: {parsed}")
    print("-" * 20)
    
# Check for empty parses where text is NOT empty
print("\nChecking for failures...")
errors = 0
for i, row in df.iterrows():
    raw = row['Appointment_history']
    if pd.isna(raw) or str(raw) == '0':
        continue
    parsed = dp.parse_history_column(raw)
    if not parsed:
        print(f"FAILED TO PARSE: {raw}")
        errors += 1
        if errors > 10: break

if errors == 0:
    print("No obvious parsing failures found in first scan.")
