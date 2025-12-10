import pandas as pd
import numpy as np
import re
from datetime import datetime

class DataProcessor:
    def __init__(self):
        # Regex to capture: Title (Start - End)
        self.entry_pattern = re.compile(r'(.*?)\s*\((\d{2}\s+[A-Z]{3}\s+\d{4})\s*-\s*(.*?)\)')
        
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

    def normalize_role_title(self, role):
        """
        Generalizes a role title by removing specific identifiers.
        E.g., "Div Officer USS Vanguard / Post 1" -> "Div Officer"
        """
        if pd.isna(role):
            return "Unknown"

        role = str(role)

        # Strategy 1: Remove " / Post X"
        role = re.sub(r'\s*/\s*Post\s*\d+', '', role, flags=re.IGNORECASE)

        # Strategy 2: Remove Unit identifiers
        splitters = [
            'Starbase', 'Dockyard', 'Field HQ', 'Fleet', 'Systems Maintenance',
            'USS', 'SS', 'RS', 'CS', 'ISS',
            'Starfleet', 'Advanced', 'Enlisted', 'Systems Design', 'Warp'
        ]

        for splitter in splitters:
            if f" {splitter}" in role:
                role = role.split(f" {splitter}")[0]
                break

        return role.strip()

    def extract_unit(self, role):
        """
        Extracts the 'Unit' from a specific role title.
        E.g., "Div Officer USS Vanguard" -> "USS Vanguard"
        """
        if pd.isna(role): return "Unknown"
        role = str(role)

        # Prefixes that denote a unit
        prefixes = ['USS', 'SS', 'RS', 'CS', 'ISS', 'Starbase', 'Dockyard', 'Field HQ']

        for p in prefixes:
            if p in role:
                # Regex to grab "Prefix Name" (e.g., "USS Vanguard")
                # We assume the unit name ends at the end of string or before " /"
                # Remove " / Post X" first
                clean_role = re.sub(r'\s*/\s*Post\s*\d+.*', '', role, flags=re.IGNORECASE).strip()

                # Split by prefix
                parts = clean_role.split(p)
                if len(parts) > 1:
                    suffix = parts[1].strip()
                    # Take first word of suffix? Or entire suffix?
                    # "USS Vanguard" -> suffix="Vanguard"
                    # "Starbase 12" -> suffix="12"
                    # "Dockyard Complex One" -> suffix="Complex One"
                    return f"{p} {suffix}".strip()

        # Special cases
        if "Wing" in role:
             clean_role = re.sub(r'\s*/\s*Post\s*\d+.*', '', role, flags=re.IGNORECASE).strip()
             parts = clean_role.split("Wing")
             if len(parts) > 1:
                  return f"Wing {parts[1].strip()}"

        return "Generic"

    def parse_history_column(self, text):
        """
        Parses a history text field into a list of dictionaries.
        """
        if pd.isna(text) or text == '0':
            return []

        entries = []
        for match in self.entry_pattern.finditer(str(text)):
            title = match.group(1).strip().strip(',').strip()
            start_str = match.group(2).strip()
            end_str = match.group(3).strip()
            
            start_date = self.parse_date(start_str)
            end_date = self.parse_date(end_str) if end_str else pd.NaT

            # Normalize title for consistent history tracking
            norm_title = self.normalize_role_title(title)
            
            entries.append({
                'title': title, # Keep raw title
                'normalized_title': norm_title,
                'start_date': start_date,
                'end_date': end_date
            })
            
        entries.sort(key=lambda x: x['start_date'] if pd.notna(x['start_date']) else pd.Timestamp.min)
        return entries

    def get_current_features(self, df):
        """Extracts features from the history and current state."""
        print("Parsing Appointment History...")
        df['parsed_appointments'] = df['Appointment_history'].apply(self.parse_history_column)
        
        print("Parsing Training History...")
        df['parsed_training'] = df['Training_history'].apply(self.parse_history_column)
        
        print("Parsing Promotion History...")
        df['parsed_promotions'] = df['Promotion_history'].apply(self.parse_history_column)
        
        return df

    def get_rank_at_date(self, promotion_history, target_date):
        if pd.isna(target_date) or not promotion_history:
            return "Unknown"
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
        Explodes the dataframe into valid transitions.
        """
        print("Exploding dataset into transitions...")
        transitions = []
        
        if 'parsed_appointments' not in df.columns:
            df = self.get_current_features(df)
            
        for idx, row in df.iterrows():
            history = row['parsed_appointments']
            current_role_title = row['current_appointment']
            current_start = pd.NaT
            if 'appointed_since' in row:
                current_start = self.parse_date(row['appointed_since'])
            
            timeline = [h for h in history]
            
            if not timeline or (str(timeline[-1]['title']).strip() != str(current_role_title).strip()):
                timeline.append({
                    'title': current_role_title,
                    'normalized_title': self.normalize_role_title(current_role_title),
                    'start_date': current_start,
                    'end_date': pd.NaT
                })
                
            if len(timeline) < 2:
                continue
                
            static_branch = row['Branch']
            static_pool = row['Pool'] 
            emp_id = row['Employee_ID']
            
            for i in range(len(timeline) - 1):
                role_now = timeline[i]
                role_next = timeline[i+1]
                
                current_history = timeline[:i+1]
                decision_date = role_next['start_date']
                
                rank_at_time = "Unknown"
                if pd.notna(decision_date):
                     rank_at_time = self.get_rank_at_date(row['parsed_promotions'], decision_date)
                else:
                    rank_at_time = row['Rank']

                # Extract Unit for Target
                target_unit = self.extract_unit(role_next['title'])
                
                transition_row = {
                    'Employee_ID': emp_id,
                    'Branch': static_branch,
                    'Pool': static_pool,
                    'Entry_type': row['Entry_type'],
                    'Rank': rank_at_time,
                    'Target_Next_Role_Raw': role_next['title'],
                    'Target_Next_Role': role_next['normalized_title'],
                    'Target_Next_Unit': target_unit, # NEW TARGET
                    'snapshot_history': current_history,
                    'snapshot_date': decision_date
                }
                transitions.append(transition_row)
                
        return pd.DataFrame(transitions)

if __name__ == "__main__":
    dp = DataProcessor()
    print(dp.normalize_role_title("Div Officer USS Vanguard / Post 1"))
