import pandas as pd
from src.inference import Predictor

print("Initializing Predictor...")
predictor = Predictor()

# 1. Create a Base Case
# A Lieutenant who just finished being a "Div Officer".
# Model likely predicts "Department Head" or "Instructor".
base_data = {
    'Employee_ID': [999990],
    'Rank': ['Lieutenant'],
    'Branch': ['Tactical Systems'],
    'Pool': ['Deep Space'],
    'Entry_type': ['Cadet'],
    'Appointment_history': ["Ensign USS Test (01 JAN 2295 - 01 JAN 2298)"], 
    'Training_history': ["Advanced Tactics (01 JAN 2298)"], 
    'Promotion_history': ["Lieutenant (01 JAN 2298)"], 
    'current_appointment': ["Div Officer USS Vanguard"], 
    'appointed_since': ["01/01/2298"]
}
df_base = pd.DataFrame(base_data)

print("\n--- BASELINE PREDICTION ---")
res_base = predictor.predict(df_base)
print(res_base.head(3).to_string())

top_prediction = res_base.iloc[0]['Prediction']
print(f"\nTop Prediction is: '{top_prediction}'")

# 2. Inject this prediction into History to test Penalty
print(f"\n--- TESTING PENALTY: Injecting '{top_prediction}' into History ---")

# We append the top prediction to the history string
base_data['Appointment_history'] = [f"Ensign USS Test (01 JAN 2295 - 01 JAN 2298), {top_prediction} (01 JAN 2298 - 01 JAN 2299)"]
# Update current to be something else to avoid current-role penalty confusion (or keep it same)
# Let's say they are CURRENTLY doing 'Div Officer' but they ALSO did 'Top Prediction' in the past.
# Or better: make 'Top Prediction' the Current Role.
base_data['current_appointment'] = [top_prediction]
df_penalty = pd.DataFrame(base_data)

res_penalty = predictor.predict(df_penalty)
print(res_penalty.head(3).to_string())

# Check
new_top = res_penalty.iloc[0]['Prediction']
if new_top != top_prediction:
    print(f"\nSUCCESS: '{top_prediction}' was penalized and is no longer Top-1. New Top: '{new_top}'")
else:
    print(f"\nFAILURE: '{top_prediction}' is still Top-1 despite penalty.")
