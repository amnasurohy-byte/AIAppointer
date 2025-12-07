import pandas as pd
from src.data_processor import DataProcessor
from config import DATASET_PATH

df = pd.read_csv(DATASET_PATH)
dp = DataProcessor()

# Focus on an employee who might have "Unknown" ranks in history
# We iterate until we find a case where rank determination fails.

print("Searching for Rank Failures...")

for idx, row in df.iterrows():
    # Parse
    row['parsed_promotions'] = dp.parse_history_column(row['Promotion_history'])
    
    # Simulate the logic in create_transition_dataset
    history = dp.parse_history_column(row['Appointment_history'])
    current_role = row['current_appointment']
    current_start = dp.parse_date(row['appointed_since'])
    
    timeline = history + [{'title': current_role, 'start_date': current_start, 'end_date': pd.NaT}]
    
    for i in range(len(timeline)-1):
        role_next = timeline[i+1]
        decision_date = role_next['start_date']
        
        rank = dp.get_rank_at_date(row['parsed_promotions'], decision_date)
        
        if rank == "Unknown":
            print(f"\n--- Failure at Index {idx} ---")
            print(f"Employee ID: {row['Employee_ID']}")
            print(f"Promotion History: {row['Promotion_history']}")
            print(f"Parsed Promos: {row['parsed_promotions']}")
            print(f"Target Role: {role_next['title']}")
            print(f"Decision Date: {decision_date}")
            print("Why Unknown? Date comparison failed?")
            # Break after first few
            if idx > 5: exit()
            break
