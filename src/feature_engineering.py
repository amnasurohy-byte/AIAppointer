import pandas as pd
import numpy as np
from datetime import datetime
# Import DataProcessor only for normalization helper if needed?
# Better to inject it or duplicate simple logic.
# We'll rely on the fact that 'snapshot_history' already has normalized titles
# IF the DataProcessor put them there.
# Let's check DP: yes, it puts 'normalized_title' in the dicts.

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
            if 'parsed_appointments' not in df.columns:
                 # Should have been processed by DP
                 pass
            df['prior_appointments'] = df['parsed_appointments']
        
        # 2. Temporal Features
        df['num_prior_roles'] = df['prior_appointments'].apply(len)
        df['years_service'] = df.apply(self._calculate_service_time, axis=1)
        df['years_in_current_rank'] = df.apply(self._calculate_years_in_rank, axis=1)
        
        # 3. Specific History Features (SEQUENTIAL ENRICHMENT)
        # We need to extract the LAST 3 Roles as distinct categorical features
        # to allow the model to learn sequences (A -> B -> C).
        # Important: We must use NORMALIZED titles for this to work generalizeable.

        def get_prior_role(history, lag=1):
            """
            Get the (normalized) title of the role 'lag' steps back.
            lag=1 is the most recent role.
            """
            if not history or not isinstance(history, list):
                return 'Recruit'

            if len(history) < lag:
                return 'Recruit' # Or 'None'

            entry = history[-lag]
            # Use normalized title if available, else raw
            return entry.get('normalized_title', entry.get('title', 'Unknown'))

        df['last_role_title'] = df['prior_appointments'].apply(lambda x: get_prior_role(x, 1))
        df['prev_role_2'] = df['prior_appointments'].apply(lambda x: get_prior_role(x, 2))
        df['prev_role_3'] = df['prior_appointments'].apply(lambda x: get_prior_role(x, 3))

        df['days_in_last_role'] = df.apply(self._calculate_time_in_last_role, axis=1)
        
        # 4. Training Features (SEQUENTIAL + AGGREGATE)
        if 'parsed_training' in df.columns:
             df['num_training_courses'] = df['parsed_training'].apply(len)
             df['count_command_training'] = df['parsed_training'].apply(lambda x: sum(1 for i in x if 'Command' in i['title']))
             df['count_tactical_training'] = df['parsed_training'].apply(lambda x: sum(1 for i in x if 'Tactical' in i['title'] or 'Security' in i['title'] or 'Weapon' in i['title']))
             df['count_science_training'] = df['parsed_training'].apply(lambda x: sum(1 for i in x if 'Science' in i['title'] or 'Physics' in i['title'] or 'Biology' in i['title']))
             df['count_engineering_training'] = df['parsed_training'].apply(lambda x: sum(1 for i in x if 'Engineering' in i['title'] or 'Propulsion' in i['title'] or 'Warp' in i['title']))
             df['count_medical_training'] = df['parsed_training'].apply(lambda x: sum(1 for i in x if 'Medical' in i['title'] or 'Surgery' in i['title']))

             # Sequential Training: Last Course
             def get_last_training(history):
                 if not history or not isinstance(history, list): return 'None'
                 # Assuming sorted
                 return history[-1].get('normalized_title', history[-1].get('title', 'None'))

             df['last_training_title'] = df['parsed_training'].apply(get_last_training)

             # Days since last training
             def get_days_since_training(row):
                 history = row.get('parsed_training', [])
                 if not history: return 9999
                 last = history[-1].get('end_date', pd.NaT)
                 if pd.isna(last): last = history[-1].get('start_date', pd.NaT)

                 ref = pd.Timestamp.now()
                 if 'snapshot_date' in row and pd.notna(row['snapshot_date']): ref = row['snapshot_date']
                 elif 'appointed_since' in row:
                      dt = pd.to_datetime(row['appointed_since'], dayfirst=True, errors='coerce')
                      if pd.notna(dt): ref = dt

                 if pd.isna(last) or pd.isna(ref): return 9999
                 return max(0, (ref - last).days)

             df['days_since_last_training'] = df.apply(get_days_since_training, axis=1)

        else:
             # Fallback
             df['num_training_courses'] = 0
             for c in ['count_command_training', 'count_tactical_training', 'count_science_training', 'count_engineering_training', 'count_medical_training']:
                 df[c] = 0
             df['last_training_title'] = 'None'
             df['days_since_last_training'] = 9999
        
        return df

    def _calculate_service_time(self, row):
        history = row['prior_appointments']
        if not history or not isinstance(history, list):
            return 0
            
        start = history[0]['start_date']
        
        # End date reference
        end_ref = pd.Timestamp.now()
        if 'snapshot_date' in row and pd.notna(row['snapshot_date']):
             end_ref = row['snapshot_date']
        elif 'appointed_since' in row:
             dt = pd.to_datetime(row['appointed_since'], dayfirst=True, errors='coerce')
             if pd.notna(dt): end_ref = dt
             
        if pd.isna(start): return 0
        return max(0, (end_ref - start).days / 365.25)

    def _calculate_time_in_last_role(self, row):
        prior = row['prior_appointments']
        if not prior or not isinstance(prior, list):
            return 0
            
        last_role = prior[-1]
        start = last_role['start_date']
        
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
        ref_date = pd.Timestamp.now()
        if 'snapshot_date' in row and pd.notna(row['snapshot_date']):
             ref_date = row['snapshot_date']
        elif 'appointed_since' in row:
             dt = pd.to_datetime(row['appointed_since'], dayfirst=True, errors='coerce')
             if pd.notna(dt): ref_date = dt
             
        promos = row.get('parsed_promotions', [])
        if not promos: return 0
        
        valid_promos = [p for p in promos if pd.notna(p['start_date']) and p['start_date'] <= ref_date]
        if not valid_promos: return 0
        
        valid_promos.sort(key=lambda x: x['start_date'], reverse=True)
        last_promo_date = valid_promos[0]['start_date']
        
        delta = (ref_date - last_promo_date).days
        return max(0, delta / 365.25)
