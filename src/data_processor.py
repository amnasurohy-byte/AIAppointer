import pandas as pd
import numpy as np
import re
from datetime import datetime

class DataProcessor:
    def __init__(self):
        # Regex to capture: Title (Start - End)
        # Using non-greedy .*? to capture title up to the date parenthesis
        # CHANGED: \s+ to \s* to handle "Title(Date)" without space
        self.entry_pattern = re.compile(r'(.*?)\s*\((\d{2}\s+[A-Z]{3}\s+\d{4})\s*-\s*(.*?)\)')

    def normalize_title(self, title):
        """
        Normalizes a role title by stripping specific unit identifiers and post numbers.
        Example: "Head of Deptt USS Orion / Post 1" -> "Head of Deptt"
        """
        if pd.isna(title): return "Unknown"
        title = str(title).strip()

        # 1. Strip Post suffix (e.g. " / Post 1")
        title = re.sub(r'\s*/\s*Post\s*\d+', '', title, flags=re.IGNORECASE)

        # 2. Strip Ship Names (USS, ISS, etc.)
        # Look for the pattern: "Role Title UNIT_PREFIX Name"
        # We assume the Role Title comes before the Unit Prefix.
        # Prefixes: USS, ISS, SS, CS, RS
        title = re.sub(r'\s+(USS|ISS|SS|CS|RS)\s+.*', '', title)

        # 3. Strip " - " separated suffixes (e.g. "Staff Officer SFHQ - Resource Management")
        # But allow "Staff Officer - SFHQ"? No, usually "Role - Unit".
        # We strip the suffix.
        if ' - ' in title:
            title = title.split(' - ')[0]

        # 4. specific known unit prefixes that might not be caught
        # "Dy Manager Fleet Maintenance Wing Alpha" -> "Dy Manager"
        units = [
            'Fleet Maintenance', 'Systems Maintenance', 'Starbase Support',
            'Dockyard', 'Planetary Repair', 'Hull Repair'
        ]
        for unit in units:
            if unit in title:
                # Split and take first part
                parts = title.split(unit)
                if len(parts) > 1 and len(parts[0].strip()) > 0:
                    title = parts[0].strip()

        return title.strip()
        
    def parse_date(self, date_str):
        """Parses dates like '25 NOV 1975' or '25/11/1975'. Returns datetime object or NaT."""
        if not date_str or str(date_str).strip() == '':
            return pd.NaT
        # Try standard formats
        for fmt in ['%d %b %Y', '%d/%m/%Y', '%Y-%m-%d']:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        return pd.NaT

    def parse_history_column(self, text):
        """
        Parses a history text field into a list of dictionaries.
        Format: [{'title': '...', 'start': datetime, 'end': datetime}, ...]
        """
        if pd.isna(text) or text == '0': # '0' seen in sample for nulls
            return []

        entries = []
        # Find all matches
        # We need to be careful about splitting. 
        # Strategy: Iterate through matches in the string.
        
        for match in self.entry_pattern.finditer(str(text)):
            title = match.group(1).strip().strip(',').strip()
            start_str = match.group(2).strip()
            end_str = match.group(3).strip() # Could be empty or a date
            
            start_date = self.parse_date(start_str)
            end_date = self.parse_date(end_str) if end_str else pd.NaT # Empty means current/ongoing
            
            entries.append({
                'title': title,
                'start_date': start_date,
                'end_date': end_date
            })
            
        # Sort by start date
        entries.sort(key=lambda x: x['start_date'] if pd.notna(x['start_date']) else pd.Timestamp.min)
        return entries

    def get_current_features(self, df):
        """Extracts features from the history and current state."""
        
        # 1. Parse Histories
        print("Parsing Appointment History...")
        df['parsed_appointments'] = df['Appointment_history'].apply(self.parse_history_column)
        
        print("Parsing Training History...")
        df['parsed_training'] = df['Training_history'].apply(self.parse_history_column)
        
        print("Parsing Promotion History...")
        df['parsed_promotions'] = df['Promotion_history'].apply(self.parse_history_column)
        
        # 2. Extract Features
        # Recency: Years since last promotion
        # We need a reference date. For training, we can use the conceptual "today" 
        # or the max date in the dataset. 
        # For this specific task, "current_appointment" is the target? 
        # No, "current_appointment" is a COLUMN in the input csv. 
        # Wait, the PROMPT says: "Predict an officer's next appointment".
        # BUT the CSV has "current_appointment". 
        # Usually in these datasets:
        # - "current_appointment" is the LABEL (Next Role)? 
        #   OR 
        # - "current_appointment" is the CURRENT Role, and we predict the NEXT one?
        # Re-reading prompt: "Experiment Goal: Predict an officer’s next appointment... Example first row... current_appointment: 'Div Officer USS Vanguard'".
        # If the row represents the CURRENT state, then 'current_appointment' is likely the TARGET (the role they just got or are in).
        # OR, does the row represent their state BEFORE the appointment?
        # The prompt says: "Predict an officer’s next appointment (next posting/role) given their career record...".
        # And "Example first row... current_appointment: 'Div Officer USS Vanguard', appointed_since: 25/11/1975".
        # And "Appointment_history: Div Officer USS Vanguard (25 NOV 1975 - )".
        # So 'current_appointment' IS the most recent item in the history.
        # So we are likely training to predict 'current_appointment' given the History MINUS the current appointment?
        # OR is the dataset a snapshot of past appointments? 
        # Let's assume for a standard supervised problem:
        # X = History BEFORE the target event.
        # Y = The Target Event (current_appointment).
        # So we must verify if 'current_appointment' appears in 'Appointment_history'.
        # In the sample: 'Div Officer USS Vanguard' is in 'Appointment_history' with start date '25 NOV 1975'.
        # So: To avoid leakage, we must REMOVE the latest entry from history if it matches the target.
        
        return df

    def get_rank_at_date(self, promotion_history, target_date):
        """
        Determines the rank held at a specific date.
        promotion_history: List of parsed dicts [{'title': 'Lt', 'start_date': ...}]
        target_date: datetime
        """
        if pd.isna(target_date) or not promotion_history:
            return "Unknown"
            
        # Sort by date
        sorted_promos = sorted([p for p in promotion_history if pd.notna(p['start_date'])], 
                               key=lambda x: x['start_date'])
                               
        current_rank = "Unknown"
        for p in sorted_promos:
            if p['start_date'] <= target_date:
                current_rank = p['title']
            else:
                break
        return current_rank

    def create_transition_dataset(self, df):
        """
        Explodes the dataframe into valid transitions:
        (State at T) -> (Role at T+1)
        """
        print("Exploding dataset into transitions...")
        transitions = []
        
        # Ensure parsing is done
        if 'parsed_appointments' not in df.columns:
            df = self.get_current_features(df)
            
        for idx, row in df.iterrows():
            # Combine history + current into a full timeline
            history = row['parsed_appointments']
            
            # Current appointment (Target in original CSV)
            current_role_title = row['current_appointment']
            current_start = pd.NaT
            if 'appointed_since' in row:
                current_start = self.parse_date(row['appointed_since'])
            
            # Create a full timeline
            # We must filter out the current role from history if it's there (to avoid duplicates)
            # but for the timeline construction, we want everything chronological.
            
            timeline = [h for h in history]
            
            # Append current if not already the last item (avoid duplicates)
            if not timeline or (str(timeline[-1]['title']).strip() != str(current_role_title).strip()):
                timeline.append({
                    'title': current_role_title,
                    'start_date': current_start,
                    'end_date': pd.NaT
                })
                
            # Now generate pairs
            # We need at least 2 items to have a transition
            if len(timeline) < 2:
                continue
                
            # Static attributes
            static_branch = row['Branch']
            static_pool = row['Pool'] 
            emp_id = row['Employee_ID']
            
            # Iterate through valid transitions
            # A -> B
            for i in range(len(timeline) - 1):
                role_now = timeline[i]
                role_next = timeline[i+1]
                
                # We need the state at the moment BEFORE role_next started.
                # So we take history up to i (inclusive)
                current_history = timeline[:i+1]
                
                # Determine Rank at this time
                # We use role_next['start_date'] as the decision point? 
                # No, strict causality: At time T (during role_now), what is my rank?
                # We can use role_next['start_date'] - 1 second.
                decision_date = role_next['start_date']
                
                rank_at_time = "Unknown"
                if pd.notna(decision_date):
                     rank_at_time = self.get_rank_at_date(row['parsed_promotions'], decision_date)
                else:
                    # If we don't know when the next role started, we can't reliably place the rank.
                    # Fallback to row['Rank'] if it's the final transition? 
                    # But row['Rank'] is the FINAL rank.
                    rank_at_time = row['Rank'] # Hazardous assumption but better than null
                
                transition_row = {
                    'Employee_ID': emp_id,
                    'Branch': static_branch, # Assumption: Branch doesn't change often or we lack history
                    'Pool': static_pool,     # Assumption: Pool is current? Risk.
                    'Entry_type': row['Entry_type'],
                    'Rank': rank_at_time,
                    'Target_Next_Role': role_next['title'],
                    'Target_General_Role': self.normalize_title(role_next['title']),
                    'snapshot_history': current_history, # Special field for FeatureEngineer
                    'snapshot_date': decision_date
                }
                transitions.append(transition_row)
                
        return pd.DataFrame(transitions)

if __name__ == "__main__":
    # Test script
    dp = DataProcessor()
    
    sample_text = "Div Officer USS Vanguard (25 NOV 1975 - ), Ensign USS Enterprise (01 JAN 1970 - 24 NOV 1975)"
    print(f"Testing Parser on: {sample_text}")
    parsed = dp.parse_history_column(sample_text)
    for p in parsed:
        print(p)
