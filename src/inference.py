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
        
        # Load Role Map (Generalized -> Specific)
        map_path = os.path.join(models_dir, 'role_map.json')
        if os.path.exists(map_path):
             print(f"Loading Role Map from {map_path}")
             import json
             with open(map_path, 'r') as f:
                 self.role_map = json.load(f)
        else:
             print("Warning: role_map.json not found. Two-stage prediction will be limited.")
             self.role_map = {}

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
            row_features = df.iloc[i]

            # Repetition Handling (History)
            history = df.iloc[i]['prior_appointments']
            past_titles = set()
            if history and isinstance(history, list):
                past_titles = {h['title'].strip() for h in history}
            current_role_str = str(input_df.iloc[i]['current_appointment']).strip()
            past_titles.add(current_role_str)
            
            # TWO-STAGE PREDICTION LOGIC
            # 1. We have probabilities for GENERALIZED ROLES (all_classes)
            # 2. We need to expand these to SPECIFIC ROLES (using self.role_map)
            # 3. We apply constraints to SPECIFIC ROLES and re-score

            candidate_roles = []

            # Threshold to consider generalized role (optimization)
            # Only consider top N generalized roles to expand? Or all?
            # All is safer but slower. Let's do all with > 0 probability.
            
            for gen_idx, gen_role_name in enumerate(all_classes):
                gen_prob = probs[gen_idx]
                # if gen_prob < 0.001: continue # Skip low prob
                
                # Get specific roles mapping to this generalized role
                specific_roles = self.role_map.get(gen_role_name, [gen_role_name])
                if not specific_roles: specific_roles = [gen_role_name]

                # Distribute probability among specific roles?
                # Or give each specific role the generalized probability?
                # Heuristic: Each specific role inherits the generalized probability,
                # then we filter by constraints.

                for specific_role in specific_roles:
                    candidate_roles.append({
                        'role': specific_role,
                        'base_prob': gen_prob,
                        'gen_role': gen_role_name
                    })

            # Now filter and rank specific candidates
            final_candidates = []

            for item in candidate_roles:
                role_name = item['role']
                base_prob = item['base_prob']

                # Get constraints for SPECIFIC role
                # Note: role_constraints in all_constraints.json are keyed by SPECIFIC role
                const = self.constraints.get(role_name, {})

                # 1. Rank Constraint
                allowed_ranks = const.get('ranks', [])
                rank_match = False
                boost_mult = 1.0

                if not allowed_ranks or 'Unknown' in allowed_ranks:
                    rank_match = True
                else:
                    for r in allowed_ranks:
                         r_idx = rank_map.get(str(r).strip(), -1)
                         if r_idx == -1: continue

                         rank_diff = r_idx - user_rank_idx
                         if rank_diff >= 0: # Promotion or Lateral
                             if rank_diff <= rank_flex_up:
                                 rank_match = True
                                 if rank_diff > 0 and rank_flex_up > 0:
                                     # Stronger boost for promotions to ensure they appear
                                     # Base prob for next rank is often 10-100x lower than current rank
                                     boost_mult = 10.0 ** rank_diff
                                 break
                         else: # Demotion
                             if abs(rank_diff) <= rank_flex_down:
                                 rank_match = True
                                 break

                if not rank_match: continue

                # 2. Branch Constraint
                allowed_branches = const.get('branches', [])
                if allowed_branches:
                    if user_branch not in allowed_branches:
                         continue

                # 3. Pool Constraint
                allowed_pools = const.get('pools', [])
                if allowed_pools:
                     if user_pool not in allowed_pools:
                          continue

                # 4. Repetition Penalty
                penalty_mult = 1.0
                if role_name.strip() in past_titles:
                     penalty_mult = 0.1
                
                final_score = base_prob * boost_mult * penalty_mult

                final_candidates.append({
                    'role': role_name,
                    'score': final_score,
                    'const': const
                })
            
            # Sort by score
            final_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # Take Top 5
            top_5 = final_candidates[:5]
            
            # Normalize scores for display (Confidence)
            total_score = sum(x['score'] for x in top_5) if top_5 else 1.0
            if total_score == 0: total_score = 1.0
            
            prediction_labels = []
            confidence_values = []
            explanations = []

            for cand in top_5:
                label = cand['role']
                score = cand['score']
                normalized_score = score / total_score # This is relative confidence among top 5
                # Or should we use original probability?
                # Original prob is better but might be low.
                # Let's show relative confidence of the top recommendations.

                prediction_labels.append(label)
                confidence_values.append(normalized_score)

                # Generate Explanation
                reasons = []
                const = cand['const']

                # Rank
                role_ranks = const.get('ranks', [])
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

                # Branch
                role_branches = const.get('branches', [])
                if user_branch in role_branches:
                      reasons.append(f"{user_branch} Branch Fit")

                # Training
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

                if row_features.get('years_service', 0) > 15:
                      reasons.append("High Seniority")
                elif row_features.get('years_service', 0) < 3:
                      reasons.append("Early Career")

                if label in past_titles:
                      reasons.append("Repeat Role (Caution)")

                explanations.append(", ".join(reasons[:3]))

            # Fill if empty
            if not prediction_labels:
                prediction_labels = ["No suitable role found"]
                confidence_values = [0.0]
                explanations = ["Constraints too strict"]

            res = pd.DataFrame({
                'Rank Info': range(1, len(prediction_labels) + 1),
                'Prediction': prediction_labels,
                'Confidence': confidence_values,
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
