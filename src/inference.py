import pandas as pd
import numpy as np
import joblib
import os
import lightgbm as lgb
import json
import re
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer

class Predictor:
    def __init__(self, models_dir='models'):
        print(f"Loading models from {models_dir}...")
        self.role_model = joblib.load(os.path.join(models_dir, 'role_model.pkl'))
        self.unit_model = joblib.load(os.path.join(models_dir, 'unit_model.pkl'))
        self.encoders = joblib.load(os.path.join(models_dir, 'feature_encoders.pkl'))
        self.role_encoder = joblib.load(os.path.join(models_dir, 'role_encoder.pkl'))
        self.unit_encoder = joblib.load(os.path.join(models_dir, 'unit_encoder.pkl'))
        self.feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))
        
        # Load Constraints
        constraints_path = os.path.join(models_dir, 'all_constraints.json')
        if os.path.exists(constraints_path):
             print(f"Loading Extended Constraints from {constraints_path}")
             with open(constraints_path, 'r') as f:
                 self.constraints = json.load(f)
        else:
             self.constraints = {}

        # Load Knowledge Base for CBR
        kb_path = os.path.join(models_dir, 'knowledge_base.csv')
        if os.path.exists(kb_path):
            print(f"Loading Knowledge Base from {kb_path}")
            self.kb_df = pd.read_csv(kb_path)
            # Pre-filter: Group by Normalized Role
            self.kb_by_role = self.kb_df.groupby('Target_Next_Role')
        else:
             print("Warning: No Knowledge Base found. Specificity will be limited.")
             self.kb_df = pd.DataFrame()
        
        # Initialize processors
        self.dp = DataProcessor()
        self.fe = FeatureEngineer()
        
    def predict(self, input_df, rank_flex_up=0, rank_flex_down=0):
        """
        Predicts next specific appointment via Dual-Model + CBR process.
        """
        # 1. Processing
        df = input_df.copy()
        df = self.dp.get_current_features(df)
        df = self.fe.extract_features(df)
        
        # 2. Encoding - DYNAMIC
        current_ranks = input_df['Rank'].tolist()
        current_branches = input_df['Branch'].tolist()
        
        for col, le in self.encoders.items():
            if col in df.columns:
                known_classes = set(le.classes_)
                def encode_safe(val):
                    val_str = str(val)
                    if val_str in known_classes:
                        return le.transform([val_str])[0]
                    if 'Unknown' in known_classes:
                         return le.transform(['Unknown'])[0]
                    return 0 
                df[col] = df[col].apply(encode_safe)
            else:
                df[col] = 0

        # Ensure all numeric features exist too
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_cols]
        
        # 3. Model Predictions (Dual)
        # Role Prediction
        role_probas = self.role_model.predict_proba(X)
        # Unit Prediction
        unit_probas = self.unit_model.predict_proba(X)

        results = []
        all_roles = self.role_encoder.classes_
        all_units = self.unit_encoder.classes_
        
        rank_order = ['Ensign', 'Lieutenant (jg)', 'Lieutenant', 'Lieutenant Commander', 'Commander', 'Captain', 'Commodore', 'Rear Admiral', 'Unknown']
        rank_map = {r: i for i, r in enumerate(rank_order)}
        
        for i in range(len(X)):
            r_probs = role_probas[i].copy()
            u_probs = unit_probas[i].copy()

            user_rank = str(current_ranks[i]).strip()
            user_rank_idx = rank_map.get(user_rank, -1)
            user_branch = str(current_branches[i]).strip()
            
            # Repetition Handling
            history = df.iloc[i]['prior_appointments']
            past_titles = set()
            if history and isinstance(history, list):
                past_titles = {self.dp.normalize_role_title(h['title']) for h in history}
            
            # --- STAGE 1: Filter Roles by Constraints ---
            for c_idx, role_name in enumerate(all_roles):
                # Apply constraints (Logic unchanged)
                if role_name in self.constraints:
                    const = self.constraints[role_name]
                    # Rank
                    allowed_ranks = const.get('ranks', [])
                    rank_match = False
                    if not allowed_ranks or 'Unknown' in allowed_ranks:
                        rank_match = True
                    else:
                        for r in allowed_ranks:
                             r_idx = rank_map.get(str(r).strip(), -1)
                             if r_idx == -1: continue
                             rank_diff = r_idx - user_rank_idx
                             if rank_diff >= 0:
                                 if rank_diff <= rank_flex_up:
                                     rank_match = True
                                     if rank_diff > 0 and rank_flex_up > 0: r_probs[c_idx] *= (2.0 ** rank_diff)
                                     break
                             else:
                                 if abs(rank_diff) <= rank_flex_down:
                                     rank_match = True
                                     break
                    if not rank_match:
                        r_probs[c_idx] = 0.0
                        continue
                    # Branch
                    allowed_branches = const.get('branches', [])
                    if allowed_branches and user_branch not in allowed_branches:
                             r_probs[c_idx] = 0.0
                             continue

                # Repetition Penalty
                if role_name in past_titles:
                     r_probs[c_idx] *= 0.1

            # Normalize Role Probabilities
            total = r_probs.sum()
            if total > 0: r_probs = r_probs / total
            
            # Get Top 5 Roles and Top 5 Units
            top_k_r = np.argsort(r_probs)[-5:][::-1]
            top_k_u = np.argsort(u_probs)[-5:][::-1]
            
            # --- STAGE 2: Combine Role + Unit to find Specific Billet ---
            # Instead of iterating specific billets, we iterate (Role, Unit) pairs
            # and calculate Score = P(Role) * P(Unit)
            
            candidates = []
            
            for r_idx in top_k_r:
                role_name = all_roles[r_idx]
                p_role = r_probs[r_idx]
                if p_role < 0.01: continue

                # Get KB candidates for this role
                if role_name in self.kb_by_role.groups:
                    kb_subset = self.kb_by_role.get_group(role_name)
                    # Get unique (Raw Title, Unit) pairs from this subset
                    # To efficiently match units
                    # We iterate raw titles in KB
                    unique_raws = kb_subset['Target_Next_Role_Raw'].unique()

                    for raw in unique_raws:
                        # Extract Unit of this raw title
                        unit_of_raw = self.dp.extract_unit(raw)

                        # Check if this Unit is in our Top K Units
                        p_unit = 0.001 # Base prob

                        # Find index of this unit in model
                        try:
                            # Is 'unit_of_raw' in 'all_units'?
                            # self.unit_encoder.transform([unit_of_raw]) logic
                            # We can just look it up if we have a map, or scan
                            # Optimization: pre-map all_units to indices
                            # For now, just iterate top_k_u

                            for u_idx in top_k_u:
                                if all_units[u_idx] == unit_of_raw:
                                    p_unit = u_probs[u_idx]
                                    break
                        except:
                            pass

                        score = p_role * p_unit
                        candidates.append({
                            'Prediction': raw,
                            'Confidence': score,
                            'Explanation': f"Role: {role_name}, Unit: {unit_of_raw}"
                        })
                else:
                    # Fallback if no KB
                    candidates.append({
                        'Prediction': role_name,
                        'Confidence': p_role * 0.01,
                        'Explanation': "Generic Role (No History)"
                    })

            # Create Result DataFrame
            res = pd.DataFrame(candidates)
            if not res.empty:
                res = res.sort_values('Confidence', ascending=False).head(5)
                # Normalize confidence to sum to 1 (optional, for UI)
                # res['Confidence'] = res['Confidence'] / res['Confidence'].sum()
                res['Rank Info'] = range(1, len(res) + 1)
            else:
                # Absolute fallback
                res = pd.DataFrame({'Rank Info': [1], 'Prediction': ['Unknown'], 'Confidence': [0.0], 'Explanation': ['No Match']})

            results.append(res)
            
        return results[0] if len(results) == 1 else results
    
    def predict_for_role(self, input_df, target_role, rank_flex_up=0, rank_flex_down=0):
        # Simplified for brevity - Billet Lookup Logic remains similar but could use unit model
        # For now, keeping legacy logic or just returning empty if not prioritized
        return pd.DataFrame() # Placeholder as focus is on predict()
