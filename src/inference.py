import pandas as pd
import numpy as np
import joblib
import os
import lightgbm as lgb
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer

class Predictor:
    def __init__(self, models_dir='models'):
        print(f"Loading models from {models_dir}...")
        self.model = joblib.load(os.path.join(models_dir, 'lgbm_model.pkl'))
        self.encoders = joblib.load(os.path.join(models_dir, 'feature_encoders.pkl'))
        self.target_encoder = joblib.load(os.path.join(models_dir, 'target_encoder.pkl'))
        self.feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))
        
        # Load Constraints (New Unified File)
        constraints_path = os.path.join(models_dir, 'all_constraints.json')
        if os.path.exists(constraints_path):
             print(f"Loading Extended Constraints from {constraints_path}")
             import json
             with open(constraints_path, 'r') as f:
                 self.constraints = json.load(f)
        else:
             # Fallback to old rank constraints if new one missing (backward comp)
             old_path = os.path.join(models_dir, 'rank_constraints.json')
             if os.path.exists(old_path):
                 print("Loading Basic Rank Constraints")
                 import json
                 with open(old_path, 'r') as f:
                     # Adapt to new structure: 'Role': {'ranks': [...]}
                     raw = json.load(f)
                     self.constraints = {k: {'ranks': v, 'branches': [], 'pools': []} for k, v in raw.items()}
             else:
                 self.constraints = {}
        
        # Initialize processors
        self.dp = DataProcessor()
        self.fe = FeatureEngineer()
        
    def predict(self, input_df, rank_flex_up=0, rank_flex_down=0):
        """
        Predicts next appointment.
        
        Args:
            input_df: DataFrame with officer data
            rank_flex_up: How many ranks UP to allow (0=strict, 1=+1 rank for promotions)
            rank_flex_down: How many ranks DOWN to allow (0=strict, 1=-1 rank for demotions)
        """
        # 1. Processing
        df = input_df.copy()
        df = self.dp.get_current_features(df)
        df = self.fe.extract_features(df)
        
        # 2. Encoding (Same as before)
        cat_features = ['Rank', 'Branch', 'Pool', 'Entry_type', 'last_role_title']
        current_ranks = input_df['Rank'].tolist() # Use raw input for logic
        current_branches = input_df['Branch'].tolist()
        current_pools = input_df['Pool'].tolist()
        
        for col in cat_features:
            if col in self.encoders:
                le = self.encoders[col]
                known_classes = set(le.classes_)
                def encode_safe(val):
                    val_str = str(val)
                    if val_str in known_classes:
                        return le.transform([val_str])[0]
                    if 'Unknown' in known_classes:
                         return le.transform(['Unknown'])[0]
                    return 0 
                df[col] = df[col].apply(encode_safe)

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_cols]
        probas = self.model.predict_proba(X)
        
        # 4. Apply Constraints & Format Output
        results = []
        all_classes = self.target_encoder.classes_
        
        rank_order = ['Ensign', 'Lieutenant (jg)', 'Lieutenant', 'Lieutenant Commander', 'Commander', 'Captain', 'Commodore', 'Rear Admiral', 'Unknown']
        rank_map = {r: i for i, r in enumerate(rank_order)}
        
        for i in range(len(X)):
            probs = probas[i].copy()
            user_rank = str(current_ranks[i]).strip()
            user_rank_idx = rank_map.get(user_rank, -1)
            user_branch = str(current_branches[i]).strip()
            user_pool = str(current_pools[i]).strip()
            
            # Repetition Handling (History)
            history = df.iloc[i]['prior_appointments']
            past_titles = set()
            if history and isinstance(history, list):
                past_titles = {h['title'].strip() for h in history}
            current_role_str = str(input_df.iloc[i]['current_appointment']).strip()
            past_titles.add(current_role_str)
            
            valid_indices = []
            
            for c_idx, role_name in enumerate(all_classes):
                # Default allow
                probs[c_idx] *= 1.0 
                
                if role_name not in self.constraints:
                    # No info? Risk. Allow but maybe penalty?
                    pass
                else:
                    const = self.constraints[role_name]
                    
                    # 1. Rank Constraint (Directional Flexibility)
                    allowed_ranks = const.get('ranks', [])
                    
                    rank_match = False
                    if not allowed_ranks or 'Unknown' in allowed_ranks:
                        # No constraint data - allow but risky
                        rank_match = True
                    else:
                        # Check if role's allowed ranks are within flexibility range
                        for r in allowed_ranks:
                             r_idx = rank_map.get(str(r).strip(), -1)
                             if r_idx == -1: 
                                 continue
                             
                             # Directional matching:
                             # rank_flex_up: allow roles UP TO N ranks above (promotions)
                             # rank_flex_down: allow roles UP TO N ranks below (demotions)
                             rank_diff = r_idx - user_rank_idx
                             
                             if rank_diff >= 0:  # Role is same rank or higher (promotion)
                                 if rank_diff <= rank_flex_up:
                                     rank_match = True
                                     # BOOST: If this is a promotion role and user enabled promotion flexibility
                                     # Apply a boost to make promotions more visible
                                     if rank_diff > 0 and rank_flex_up > 0:
                                         # Boost factor: 2x for each rank level up
                                         boost_factor = 2.0 ** rank_diff
                                         probs[c_idx] *= boost_factor
                                     break
                             else:  # Role is lower rank (demotion)
                                 if abs(rank_diff) <= rank_flex_down:
                                     rank_match = True
                                     break
                    
                    if not rank_match:
                        probs[c_idx] = 0.0
                        continue
                        
                    # 2. Branch Constraint
                    allowed_branches = const.get('branches', [])
                    if allowed_branches:
                        if user_branch not in allowed_branches:
                             probs[c_idx] = 0.0
                             continue
                             
                    # 3. Pool Constraint
                    allowed_pools = const.get('pools', [])
                    if allowed_pools:
                         if user_pool not in allowed_pools:
                              # Pools can be fluid? Maybe soft penalty instead of hard kill?
                              # "we dont want to pursue novelty at the cost of accuracy" -> Hard constraint is safer.
                              probs[c_idx] = 0.0
                              continue

                # 4. Repetition Penalty
                if role_name.strip() in past_titles:
                     probs[c_idx] *= 0.1 # 10% probability remaining
                
                if probs[c_idx] > 0:
                    valid_indices.append(c_idx)

            # Normalize
            total = probs.sum()
            if total > 0:
                probs = probs / total
            
            # Top 5
            top_k_idx = np.argsort(probs)[-5:][::-1]
            top_k_labels = self.target_encoder.inverse_transform(top_k_idx)
            top_k_probs = probs[top_k_idx]
            
            # Explanations
            explanations = []
            
            # Extract row features for explanation
            # We need to access the derived features (years_service, etc) from 'df' (the processed one)
            # But 'df' has N rows matching len(X).
            row_features = df.iloc[i]
            
            for idx, label in zip(top_k_idx, top_k_labels):
                 p = probs[idx]
                 reasons = []
                 
                 # 1. Rank Progression
                 role_ranks = self.constraints.get(label, {}).get('ranks', [])
                 avg_role_rank_idx = -1
                 if role_ranks:
                      for rr in role_ranks:
                           rid = rank_map.get(rr, -1)
                           if rid != -1: 
                               avg_role_rank_idx = rid
                               break
                 
                 if avg_role_rank_idx > user_rank_idx:
                      reasons.append("Promotion Step")
                 elif avg_role_rank_idx == user_rank_idx:
                      reasons.append("Lateral Move")
                 elif avg_role_rank_idx < user_rank_idx and avg_role_rank_idx != -1:
                      reasons.append("Seniority Consolidation") # Euphemism for lower rank role
                 
                 # 2. Branch Fit
                 role_branches = self.constraints.get(label, {}).get('branches', [])
                 if user_branch in role_branches:
                      reasons.append(f"{user_branch} Branch Fit")
                 
                 # 3. Experience / Training Fit (The "Why")
                 # We simply check if the Role Title maps to a training category
                 label_lower = label.lower()
                 if 'command' in label_lower or 'captain' in label_lower or 'xo' in label_lower:
                      if row_features.get('count_command_training', 0) > 0:
                           reasons.append("Command Trained")
                 if 'tactical' in label_lower or 'security' in label_lower:
                      if row_features.get('count_tactical_training', 0) > 0:
                           reasons.append("Tactical Trained")
                 if 'engineer' in label_lower or 'warp' in label_lower:
                      if row_features.get('count_engineering_training', 0) > 0:
                           reasons.append("Engineering Trained")
                 if 'science' in label_lower or 'sensor' in label_lower:
                      if row_features.get('count_science_training', 0) > 0:
                           reasons.append("Science Trained")
                 
                 # 4. Seniority Fit
                 # If years_service > 10 and role is senior
                 if row_features.get('years_service', 0) > 15:
                      reasons.append("High Seniority")
                 elif row_features.get('years_service', 0) < 3:
                      reasons.append("Early Career")
                 
                 # 5. History Penalty Mention
                 if label in past_titles:
                      reasons.append("Repeat Role (Caution)")
                 
                 # Join
                 # Limit to top 3 reasons to keep UI clean
                 explanations.append(", ".join(reasons[:3]))

            res = pd.DataFrame({
                'Rank Info': range(1, 6),
                'Prediction': top_k_labels,
                'Confidence': top_k_probs,
                'Explanation': explanations
            })
            results.append(res)
            
        return results[0] if len(results) == 1 else results
    
    def predict_for_role(self, input_df, target_role, rank_flex_up=0, rank_flex_down=0):
        """
        Predicts confidence for a SPECIFIC target role across all candidates.
        Used for Billet Lookup (reverse search).
        
        Args:
            input_df: DataFrame with candidate data
            target_role: Specific role to evaluate
            rank_flex_up: How many ranks UP to allow
            rank_flex_down: How many ranks DOWN to allow
            
        Returns:
            DataFrame with Employee_ID, Rank, Current_Role, Confidence, Explanation.
        """
        # 1. Processing (same as predict)
        df = input_df.copy()
        df = self.dp.get_current_features(df)
        df = self.fe.extract_features(df)
        
        # 2. Encoding
        cat_features = ['Rank', 'Branch', 'Pool', 'Entry_type', 'last_role_title']
        current_ranks = input_df['Rank'].tolist()
        current_branches = input_df['Branch'].tolist()
        current_pools = input_df['Pool'].tolist()
        
        for col in cat_features:
            if col in self.encoders:
                le = self.encoders[col]
                known_classes = set(le.classes_)
                def encode_safe(val):
                    val_str = str(val)
                    if val_str in known_classes:
                        return le.transform([val_str])[0]
                    if 'Unknown' in known_classes:
                         return le.transform(['Unknown'])[0]
                    return 0 
                df[col] = df[col].apply(encode_safe)

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_cols]
        probas = self.model.predict_proba(X)
        
        # Find target role index
        all_classes = self.target_encoder.classes_
        try:
            target_idx = list(all_classes).index(target_role)
        except ValueError:
            # Role not in training data
            return pd.DataFrame()
        
        rank_order = ['Ensign', 'Lieutenant (jg)', 'Lieutenant', 'Lieutenant Commander', 'Commander', 'Captain', 'Commodore', 'Rear Admiral', 'Unknown']
        rank_map = {r: i for i, r in enumerate(rank_order)}
        
        results = []
        
        for i in range(len(X)):
            probs = probas[i].copy()
            user_rank = str(current_ranks[i]).strip()
            user_rank_idx = rank_map.get(user_rank, -1)
            user_branch = str(current_branches[i]).strip()
            user_pool = str(current_pools[i]).strip()
            
            # Apply same constraints as predict
            history = df.iloc[i]['prior_appointments']
            past_titles = set()
            if history and isinstance(history, list):
                past_titles = {h['title'].strip() for h in history}
            current_role_str = str(input_df.iloc[i]['current_appointment']).strip()
            past_titles.add(current_role_str)
            
            # Apply constraints to target role
            if target_role in self.constraints:
                const = self.constraints[target_role]
                
                # Rank constraint (Directional Flexibility)
                allowed_ranks = const.get('ranks', [])
                rank_match = False
                if not allowed_ranks or 'Unknown' in allowed_ranks:
                    rank_match = True
                else:
                    for r in allowed_ranks:
                         r_idx = rank_map.get(str(r).strip(), -1)
                         if r_idx == -1: 
                             continue
                         # Directional matching (same as predict)
                         rank_diff = r_idx - user_rank_idx
                         if rank_diff >= 0:  # Promotion
                             if rank_diff <= rank_flex_up:
                                 rank_match = True
                                 # Apply promotion boost
                                 if rank_diff > 0 and rank_flex_up > 0:
                                     boost_factor = 2.0 ** rank_diff
                                     probs[target_idx] *= boost_factor
                                 break
                         else:  # Demotion
                             if abs(rank_diff) <= rank_flex_down:
                                 rank_match = True
                                 break
                
                if not rank_match:
                    probs[target_idx] = 0.0
                    
                # Branch constraint
                allowed_branches = const.get('branches', [])
                if allowed_branches:
                    if user_branch not in allowed_branches:
                         probs[target_idx] = 0.0
                         
                # Pool constraint
                allowed_pools = const.get('pools', [])
                if allowed_pools:
                     if user_pool not in allowed_pools:
                          probs[target_idx] = 0.0
            
            # Repetition penalty
            if target_role.strip() in past_titles:
                 probs[target_idx] *= 0.1
            
            # Normalize
            total = probs.sum()
            if total > 0:
                probs = probs / total
            
            confidence = probs[target_idx]
            
            # Generate explanation for this specific role
            row_features = df.iloc[i]
            reasons = []
            
            # Rank progression
            role_ranks = self.constraints.get(target_role, {}).get('ranks', [])
            avg_role_rank_idx = -1
            if role_ranks:
                 for rr in role_ranks:
                      rid = rank_map.get(rr, -1)
                      if rid != -1: 
                          avg_role_rank_idx = rid
                          break
            
            if avg_role_rank_idx > user_rank_idx:
                 reasons.append("Promotion Step")
            elif avg_role_rank_idx == user_rank_idx:
                 reasons.append("Lateral Move")
            elif avg_role_rank_idx < user_rank_idx and avg_role_rank_idx != -1:
                 reasons.append("Seniority Consolidation")
            
            # Branch fit
            role_branches = self.constraints.get(target_role, {}).get('branches', [])
            if user_branch in role_branches:
                 reasons.append(f"{user_branch} Branch Fit")
            
            # Training fit
            label_lower = target_role.lower()
            if 'command' in label_lower or 'captain' in label_lower or 'xo' in label_lower:
                 if row_features.get('count_command_training', 0) > 0:
                      reasons.append("Command Trained")
            if 'tactical' in label_lower or 'security' in label_lower:
                 if row_features.get('count_tactical_training', 0) > 0:
                      reasons.append("Tactical Trained")
            if 'engineer' in label_lower or 'warp' in label_lower:
                 if row_features.get('count_engineering_training', 0) > 0:
                      reasons.append("Engineering Trained")
            if 'science' in label_lower or 'sensor' in label_lower:
                 if row_features.get('count_science_training', 0) > 0:
                      reasons.append("Science Trained")
            
            # Seniority
            if row_features.get('years_service', 0) > 15:
                 reasons.append("High Seniority")
            elif row_features.get('years_service', 0) < 3:
                 reasons.append("Early Career")
            
            if target_role in past_titles:
                 reasons.append("Repeat Role (Caution)")
            
            explanation = ", ".join(reasons[:3])
            
            results.append({
                'Employee_ID': input_df.iloc[i]['Employee_ID'],
                'Name': input_df.iloc[i]['Name'],
                'Rank': user_rank,
                'Branch': user_branch,
                'Current_Role': current_role_str,
                'Confidence': confidence,
                'Explanation': explanation
            })
        
        result_df = pd.DataFrame(results)
        # Filter out zero confidence and sort
        result_df = result_df[result_df['Confidence'] > 0].sort_values(by='Confidence', ascending=False)
        return result_df
