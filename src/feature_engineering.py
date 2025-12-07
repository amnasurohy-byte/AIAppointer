import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    def __init__(self):
        pass
        
    def extract_features(self, df):
        """
        Extracts features for the model.
        Supports both Transition Dataset (with 'snapshot_history') and Raw Dataset (Inference).
        """
        print("Extracting features...")
        
        # 1. Standardize History Input
        if 'snapshot_history' in df.columns:
            # Transition Training Mode
            df['prior_appointments'] = df['snapshot_history']
        else:
            # Inference Mode or Legacy
            # We assume the input IS the current state.
            # So 'prior_appointments' is just the full history parsed.
            # We assume df has 'parsed_appointments' (if not, we'd need to parse).
            # For inference, the "Target" is unknown, so we don't exclude the "Current" row 
            # (unless the user provided it as history).
            if 'parsed_appointments' not in df.columns:
                 # Should have been processed by DP
                 pass
            df['prior_appointments'] = df['parsed_appointments']
        
        # 2. Temporal Features
        df['num_prior_roles'] = df['prior_appointments'].apply(len)
        df['years_service'] = df.apply(self._calculate_service_time, axis=1)
        
        # New Feature: Years in Current Rank
        # We need access to promotion history.
        # Transition dataset (snapshot_history) and Inference (parsed_promotions) need unification.
        # But 'parsed_promotions' is available in both cases (propagated in DP).
        df['years_in_current_rank'] = df.apply(self._calculate_years_in_rank, axis=1)
        
        # 3. Specific History Features
        df['last_role_title'] = df['prior_appointments'].apply(lambda x: x[-1]['title'] if x and isinstance(x, list) else 'Recruit')
        df['days_in_last_role'] = df.apply(self._calculate_time_in_last_role, axis=1)
        
        # 4. Training Features (Granular)
        # We count occurrences of specific keywords in the training history
        if 'parsed_training' in df.columns:
             df['num_training_courses'] = df['parsed_training'].apply(len)
             df['count_command_training'] = df['parsed_training'].apply(lambda x: sum(1 for i in x if 'Command' in i['title']))
             df['count_tactical_training'] = df['parsed_training'].apply(lambda x: sum(1 for i in x if 'Tactical' in i['title'] or 'Security' in i['title'] or 'Weapon' in i['title']))
             df['count_science_training'] = df['parsed_training'].apply(lambda x: sum(1 for i in x if 'Science' in i['title'] or 'Physics' in i['title'] or 'Biology' in i['title']))
             df['count_engineering_training'] = df['parsed_training'].apply(lambda x: sum(1 for i in x if 'Engineering' in i['title'] or 'Propulsion' in i['title'] or 'Warp' in i['title']))
             df['count_medical_training'] = df['parsed_training'].apply(lambda x: sum(1 for i in x if 'Medical' in i['title'] or 'Surgery' in i['title']))
             # Keep binary flags for legacy or explicit checks if needed, but counts are better for trees.
             df['has_command_training'] = df['count_command_training'].apply(lambda x: 1 if x > 0 else 0)
        else:
             # Fallback
             df['num_training_courses'] = 0
             df['count_command_training'] = 0
             df['count_tactical_training'] = 0
             df['count_science_training'] = 0
             df['count_engineering_training'] = 0
             df['count_medical_training'] = 0
             df['has_command_training'] = 0
        
        return df

    def _calculate_service_time(self, row):
        history = row['prior_appointments']
        if not history or not isinstance(history, list):
            return 0
            
        start = history[0]['start_date']
        
        # End date reference: 'snapshot_date' (if training) or Now?
        end_ref = pd.Timestamp.now() # Fallback
        if 'snapshot_date' in row and pd.notna(row['snapshot_date']):
             end_ref = row['snapshot_date']
        elif 'appointed_since' in row:
             # Inference case
             dt = pd.to_datetime(row['appointed_since'], dayfirst=True, errors='coerce')
             if pd.notna(dt): end_ref = dt
             
        if pd.isna(start): return 0
        return max(0, (end_ref - start).days / 365.25)

    def _calculate_time_in_last_role(self, row):
        prior = row['prior_appointments']
        if not prior or not isinstance(prior, list):
            return 0 # New recruit
            
        last_role = prior[-1]
        start = last_role['start_date']
        
        # End date reference
        end_ref = pd.Timestamp.now()
        if 'snapshot_date' in row and pd.notna(row['snapshot_date']):
             end_ref = row['snapshot_date']
        elif 'appointed_since' in row:
             dt = pd.to_datetime(row['appointed_since'], dayfirst=True, errors='coerce')
             if pd.notna(dt): end_ref = dt
            
        if pd.isna(start) or pd.isna(end_ref):
            return 0
            
        return (end_ref - start).days

    def _calculate_years_in_rank(self, row):
        # Determine the date of the query (snapshot or now)
        ref_date = pd.Timestamp.now()
        if 'snapshot_date' in row and pd.notna(row['snapshot_date']):
             ref_date = row['snapshot_date']
        elif 'appointed_since' in row:
             dt = pd.to_datetime(row['appointed_since'], dayfirst=True, errors='coerce')
             if pd.notna(dt): ref_date = dt
             
        # Get promotions
        promos = row.get('parsed_promotions', [])
        if not promos: return 0
        
        # Determine Current Rank (Title)
        current_rank_title = row.get('Rank', 'Unknown')
        
        # Find when this rank started
        # We look for the LATEST start date of a promotion matching this rank title
        # that is BEFORE ref_date
        
        # Filter valid promos
        valid_promos = [p for p in promos if pd.notna(p['start_date']) and p['start_date'] <= ref_date]
        if not valid_promos: return 0
        
        # Sort desc
        valid_promos.sort(key=lambda x: x['start_date'], reverse=True)
        
        # Match title? Or just take the last one?
        # If the rank title matches, use it. If not, maybe use the last promotion date anyway?
        # Using last promotion date is safer because "Rank" column might be "Unknown" 
        # in some rows (like in transition dataset if we failed to map).
        
        last_promo_date = valid_promos[0]['start_date']
        
        delta = (ref_date - last_promo_date).days
        return max(0, delta / 365.25)

