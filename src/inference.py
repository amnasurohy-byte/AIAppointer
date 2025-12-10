import pandas as pd
import numpy as np
import joblib
import os
import lightgbm as lgb
import json
import re
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.sequential_recommender import SequentialRecommender

class Predictor:
    def __init__(self, models_dir='models'):
        print(f"Loading models from {models_dir}...")
        self.role_model = joblib.load(os.path.join(models_dir, 'role_model.pkl'))
        self.unit_model = joblib.load(os.path.join(models_dir, 'unit_model.pkl'))
        self.seq_model = joblib.load(os.path.join(models_dir, 'seq_model.pkl')) # NEW

        self.encoders = joblib.load(os.path.join(models_dir, 'feature_encoders.pkl'))
        self.role_encoder = joblib.load(os.path.join(models_dir, 'role_encoder.pkl'))
        self.unit_encoder = joblib.load(os.path.join(models_dir, 'unit_encoder.pkl'))
        self.feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))
        
        # Load Constraints
        constraints_path = os.path.join(models_dir, 'all_constraints.json')
        if os.path.exists(constraints_path):
             with open(constraints_path, 'r') as f:
                 self.constraints = json.load(f)
        else:
             self.constraints = {}

        # Load Knowledge Base for CBR
        kb_path = os.path.join(models_dir, 'knowledge_base.csv')
        if os.path.exists(kb_path):
            self.kb_df = pd.read_csv(kb_path)
            self.kb_by_role = self.kb_df.groupby('Target_Next_Role')
        else:
             self.kb_df = pd.DataFrame()
        
        self.dp = DataProcessor()
        self.fe = FeatureEngineer()
        
    def predict(self, input_df, rank_flex_up=0, rank_flex_down=0):
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
                    if val_str in known_classes: return le.transform([val_str])[0]
                    if 'Unknown' in known_classes: return le.transform(['Unknown'])[0]
                    return 0 
                df[col] = df[col].apply(encode_safe)
            else: df[col] = 0

        for col in self.feature_cols:
            if col not in df.columns: df[col] = 0
        
        X = df[self.feature_cols]
        
        # 3. Model Predictions (Dual)
        lgbm_role_probas = self.role_model.predict_proba(X)
        lgbm_unit_probas = self.unit_model.predict_proba(X)

        results = []
        all_roles = self.role_encoder.classes_
        all_units = self.unit_encoder.classes_
        
        rank_order = ['Ensign', 'Lieutenant (jg)', 'Lieutenant', 'Lieutenant Commander', 'Commander', 'Captain', 'Commodore', 'Rear Admiral', 'Unknown']
        rank_map = {r: i for i, r in enumerate(rank_order)}
        
        for i in range(len(X)):
            # --- ENSEMBLE: LightGBM + Markov Chain ---
            # Get Current State
            last_role_title = 'Unknown'
            # We can get last role title from FeatureEngineer output before encoding
            # Or re-extract from history
            hist = df.iloc[i]['prior_appointments']
            if hist and isinstance(hist, list):
                last_role_title = hist[-1].get('normalized_title', 'Unknown')
                last_role_raw = hist[-1].get('title', '')
                last_unit = self.dp.extract_unit(last_role_raw)
            else:
                last_role_title = 'Recruit'
                last_unit = 'Unknown'

            # Get Sequential Probabilities
            seq_role_probs = self.seq_model.predict_role_probs(last_role_title, all_roles)
            seq_unit_probs = self.seq_model.predict_unit_probs(last_unit, all_units)

            # Combine Role Probs
            # Weighted Ensemble: 0.7 LGBM + 0.3 Seq
            combined_role_probs = lgbm_role_probas[i].copy()
            for r_idx, role_label in enumerate(all_roles):
                p_seq = seq_role_probs.get(role_label, 0.0)
                combined_role_probs[r_idx] = (combined_role_probs[r_idx] * 0.7) + (p_seq * 0.3)

            # Combine Unit Probs
            combined_unit_probs = lgbm_unit_probas[i].copy()
            for u_idx, unit_label in enumerate(all_units):
                p_seq = seq_unit_probs.get(unit_label, 0.0)
                combined_unit_probs[u_idx] = (combined_unit_probs[u_idx] * 0.7) + (p_seq * 0.3)

            # Normalize
            if combined_role_probs.sum() > 0: combined_role_probs /= combined_role_probs.sum()
            if combined_unit_probs.sum() > 0: combined_unit_probs /= combined_unit_probs.sum()

            # --- STAGE 1: Filter Roles by Constraints ---
            user_rank = str(current_ranks[i]).strip()
            user_rank_idx = rank_map.get(user_rank, -1)
            user_branch = str(current_branches[i]).strip()
            
            past_titles = set()
            if hist and isinstance(hist, list):
                past_titles = {self.dp.normalize_role_title(h['title']) for h in hist}
            
            for c_idx, role_name in enumerate(all_roles):
                if role_name in self.constraints:
                    const = self.constraints[role_name]
                    # Rank
                    allowed_ranks = const.get('ranks', [])
                    rank_match = False
                    if not allowed_ranks or 'Unknown' in allowed_ranks: rank_match = True
                    else:
                        for r in allowed_ranks:
                             r_idx = rank_map.get(str(r).strip(), -1)
                             if r_idx == -1: continue
                             rank_diff = r_idx - user_rank_idx
                             if rank_diff >= 0:
                                 if rank_diff <= rank_flex_up:
                                     rank_match = True
                                     if rank_diff > 0 and rank_flex_up > 0: combined_role_probs[c_idx] *= (2.0 ** rank_diff)
                                     break
                             else:
                                 if abs(rank_diff) <= rank_flex_down: rank_match = True; break
                    if not rank_match: combined_role_probs[c_idx] = 0.0; continue

                    # Branch
                    allowed_branches = const.get('branches', [])
                    if allowed_branches and user_branch not in allowed_branches:
                             combined_role_probs[c_idx] = 0.0; continue

                if role_name in past_titles: combined_role_probs[c_idx] *= 0.1

            # Normalize again
            if combined_role_probs.sum() > 0: combined_role_probs /= combined_role_probs.sum()
            
            # --- STAGE 2: Combine Role + Unit ---
            top_k_r = np.argsort(combined_role_probs)[-5:][::-1]
            top_k_u = np.argsort(combined_unit_probs)[-5:][::-1]
            
            candidates = []
            
            for r_idx in top_k_r:
                role_name = all_roles[r_idx]
                p_role = combined_role_probs[r_idx]
                if p_role < 0.01: continue

                # Get KB candidates for this role
                if role_name in self.kb_by_role.groups:
                    kb_subset = self.kb_by_role.get_group(role_name)
                    unique_raws = kb_subset['Target_Next_Role_Raw'].unique()

                    for raw in unique_raws:
                        unit_of_raw = self.dp.extract_unit(raw)
                        p_unit = 0.001

                        # Find p_unit in combined_unit_probs
                        for u_idx in top_k_u:
                            if all_units[u_idx] == unit_of_raw:
                                p_unit = combined_unit_probs[u_idx]
                                break

                        score = p_role * p_unit
                        candidates.append({
                            'Prediction': raw,
                            'Confidence': score,
                            'Explanation': f"Role: {role_name}, Unit: {unit_of_raw}"
                        })
                else:
                    candidates.append({
                        'Prediction': role_name,
                        'Confidence': p_role * 0.01,
                        'Explanation': "Generic Role (No History)"
                    })

            res = pd.DataFrame(candidates)
            if not res.empty:
                res = res.sort_values('Confidence', ascending=False).head(5)
                res['Rank Info'] = range(1, len(res) + 1)
            else:
                res = pd.DataFrame({'Rank Info': [1], 'Prediction': ['Unknown'], 'Confidence': [0.0], 'Explanation': ['No Match']})

            results.append(res)
            
        return results[0] if len(results) == 1 else results

    def predict_for_role(self, input_df, target_role, rank_flex_up=0, rank_flex_down=0):
        return pd.DataFrame()
