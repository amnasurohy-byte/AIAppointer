import pandas as pd
from config import DATASET_PATH
df = pd.read_csv(DATASET_PATH)
print(f"Max Commas: {df['Appointment_history'].str.count(',').max()}")
print(f"Avg Commas: {df['Appointment_history'].str.count(',').mean()}")
# Also check if commas are used as separators or inside texts
print("Sample with max commas:")
max_commas = df['Appointment_history'].str.count(',').max()
print(df[df['Appointment_history'].str.count(',') == max_commas]['Appointment_history'].iloc[0])
