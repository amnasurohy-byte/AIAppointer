import pandas as pd
from src.inference import Predictor

print("Initializing Predictor...")
predictor = Predictor()

# Simulate the dummy data payload from app.py
print("Constructing Dummy Data...")
last_role = "Div Officer USS Vanguard"
dummy_data = {
    'Employee_ID': [999999],
    'Rank': ['Lieutenant'],
    'Branch': ['Tactical Systems'],
    'Pool': ['Deep Space'],
    'Entry_type': ['Cadet'],
    'Appointment_history': [f"{last_role} (01 JAN 2300 - )"], 
    'Training_history': ["Basic Training (01 JAN 2290 - 01 FEB 2290)"], 
    'Promotion_history': [""], # This was the fix (list of length 1)
    'current_appointment': [last_role], 
    'appointed_since': ["01/01/2300"]
}

try:
    dummy_df = pd.DataFrame(dummy_data)
    print("DataFrame Created Successfully.")
    
    print("Predicting on Dummy Data...")
    results = predictor.predict(dummy_df)
    print("\nSimulation Results:")
    print(results.to_string())
    print("\nSUCCESS: Simulation Logic Verified.")
except Exception as e:
    print(f"\nFAILURE: {e}")
    exit(1)
