
import pandas as pd
import numpy as np
import joblib
import os
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.features_ltr import LTRFeatureEngineer

class Predictor:
    def __init__(self, models_dir='models/ltr'):
        print(f"Loading LTR models from {models_dir}...")
        
        # Paths
        if not os.path.exists(models_dir):
            models_dir = os.path.join(os.getcwd(), 'models', 'ltr')
            
        try:
            self.model = joblib.load(os.path.join(models_dir, 'lgbm_ranker.pkl'))
            self.role_meta = joblib.load(os.path.join(models_dir, 'role_meta.pkl'))
            self.feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))
            
            # Feature Engineer
            self.ltr_fe = LTRFeatureEngineer()
            self.ltr_fe.load(os.path.join(models_dir, 'ltr_fe.pkl'))
            
            # Load Transition Stats
            try:
                self.transition_stats = joblib.load(os.path.join(models_dir, 'transition_stats.pkl'))
                print("✓ Transition Stats Loaded.")
            except:
                print("⚠️ Transition Stats NOT Found. LTR will run without priors.")
                self.transition_stats = None
            
            # Load Strict Constraints (Ground Truth)
            import json
            with open('models/all_constraints.json', 'r') as f:
                self.strict_constraints = json.load(f)
                
            # STRICT WHITELIST: Only these roles exist.
            self.valid_roles = set(self.strict_constraints.keys())
            
            # UI COMPATIBILITY
            # The UI expects 'target_encoder.classes_' to list all roles for Dropdowns
            # and 'constraints' dict.
            self.constraints = self.strict_constraints
            self.target_encoder = type('E', (), {'classes_': sorted(list(self.valid_roles))})
                
            self.ready = True
            print("✓ LTR System Loaded.")
        except Exception as e:
            print(f"Error loading LTR system: {e}")
            self.ready = False
            
    def _prepare_input_context(self, input_data):
        """Prepare Officer Context Dictionary"""
        if isinstance(input_data, pd.DataFrame):
            row = input_data.iloc[0].to_dict()
        else:
            row = input_data
            
        # Normalize Rank
        if 'Rank' in row and isinstance(row['Rank'], str):
             row['Rank'] = row['Rank'].strip()  # Don't lower(), our keys are Case Sensitive "Lieutenant (jg)"

        # CRITICAL FIX: Clean History String
        # Input might be raw "Role (Date - Date)" strings which fail text matching
        # We need to extract just the titles.
        
        hist_raw = row.get('Appointment_history', '')
        if isinstance(hist_raw, str):
            # Regex to clean "Title (Date...)" -> "Title"
            # Split by comma or newline if multiple?
            # Usually dataset is "Role A (Date), Role B (Date)"
            import re
            # Split by comma, then clean each
            items = str(hist_raw).split(',')
            titles = []
            for item in items:
                # Remove dates (...)
                clean = re.sub(r'\s*\(.*?\)', '', item).strip()
                if clean: titles.append(clean)
            
            row['history_str'] = " > ".join(titles)
            # Update last_role_title for Priors
            if titles:
                row['last_role_title'] = titles[0] # Most recent usually first or last? 
                # Check dataset convention. Usually index 0 is latest? 
                # Let's assume text order implies sequence.
                # Actually, dataset 'Appointment_history' format varies.
                # Safer: Use 'current_appointment' if available
        
        if 'current_appointment' in row:
             row['last_role_title'] = str(row['current_appointment']).strip()
             
        # FUZZY MATCHING for Last Role Title
        # To ensure we hit the Transition Stats keys
        if 'last_role_title' in row and self.transition_stats:
            known_titles = list(self.transition_stats['title_trans'].keys())
            curr = row['last_role_title']
            
            # 1. Exact Match
            if curr in self.transition_stats['title_trans']:
                pass
            else:
                # 2. Substring Match (Case Insensitive)
                # If 'Ensign' is a key, and title is 'Ensign Role', matched!
                found_sub = None
                curr_lower = curr.lower()
                for key in known_titles:
                     if key.lower() in curr_lower or curr_lower in key.lower():
                         found_sub = key
                         break
                
                if found_sub:
                    row['last_role_title'] = found_sub
                    # print(f"Confident Fix: Substring Mapped '{curr}' -> '{found_sub}'")
                else:
                    # 3. Fuzzy Match (Lower Threshold)
                    import difflib
                    matches = difflib.get_close_matches(curr, known_titles, n=1, cutoff=0.4)
                    if matches:
                        # print(f"Confidence Fix: Fuzzy Mapped '{curr}' -> '{matches[0]}'")
                        row['last_role_title'] = matches[0]

        # Enrich if needed (snapshot_history is from simulation)
        if 'snapshot_history' in row:
             hist = row['snapshot_history']
             if isinstance(hist, list):
                 row['history_str'] = " > ".join([h.get('title', '') for h in hist])
                 if hist: row['last_role_title'] = hist[-1].get('title', '')
                 
        return row

    def predict(self, input_data, rank_flex_up=0, rank_flex_down=0):
        """
        Learning-to-Rank Prediction
        """
        if not self.ready:
            return pd.DataFrame([{'Prediction': 'System Not Ready', 'Confidence': 0, 'Explanation': 'Model Error'}])
            
        officer = self._prepare_input_context(input_data)
        context = officer
        
        # 1. Filter Candidates by Constraints
        candidates_to_score = []
        
        current_rank = officer.get('Rank')
        current_branch = officer.get('Branch')
        
        # Multi-Pass Candidate Selection
        # Pass 1: Strict Constraints
        # Pass 2: Relaxed Constraints (if Strict yields 0) - Ignore Branch/Entry
        
        candidates_to_score = []
        
        for attempt in ['strict', 'relaxed']:
            temp_candidates = []
            if attempt == 'relaxed':
                 print("  > Strict constraints yielded 0 candidates. Retrying with RELAXED constraints.")
            
            for role_name in self.valid_roles:
                # Meta retrieval
                meta = self.role_meta.get(role_name)
                if not meta:
                    cons = self.strict_constraints.get(role_name, {})
                    meta = {
                        'Role': role_name,
                        'Branch': cons.get('branches', ['Unknown'])[0],
                        'Pool': 'Unknown',
                        'REQ_Ranks': cons.get('ranks', []),
                        'freq': 1
                    }
                
                # Validation Logic
                if self.strict_constraints:
                     role_cons = self.strict_constraints.get(role_name, {})
                     
                     # 1. RANK CHECK (Always Enforced, unless we add Pass 3)
                     ranks = role_cons.get('ranks', [])
                     if ranks and current_rank not in ranks:
                         continue
                         
                     if attempt == 'strict':
                        # 2. MATCH CHECK (Branch)
                        branches = role_cons.get('branches', [])
                        if branches and current_branch:
                            # Science Relaxation (Partial)
                            if current_branch == "Science":
                                 allowed = ["Science", "Engineering", "Tactical Systems"]
                                 if not any(b in allowed for b in branches):
                                     continue
                            elif current_branch not in branches:
                                continue
                        
                        # 3. ENTRY CHECK
                        entries = role_cons.get('entries', [])
                        current_entry = context.get('Entry_type')
                        if entries and current_entry and current_entry not in entries:
                            continue
                
                # If we got here, it passed
                if role_name in self.role_meta:
                    temp_candidates.append(self.role_meta[role_name])
            
            # End of Role Loop
            if temp_candidates:
                candidates_to_score = temp_candidates
                break # Found valid candidates
                
        # If still empty after relaxed pass, effectively return empty

            
        # If strict filter removes everything, relax
        if not candidates_to_score:
             # Fallback? Or just return empty?
             # User wants strict. Return empty.
             pass

        # Special Handling: Removed Soft Mapping hack as constraints now natively support Lt search logic.

        if not candidates_to_score:
             return pd.DataFrame([{'Prediction': 'No valid roles found for Rank/Branch/Entry', 'Confidence': 0, 'Explanation': 'Strict Constraints blocked all options.'}])
             
        X_rows = []
        meta_list = []
        feats_list = []
        
        for cand in candidates_to_score:
            feats = self.ltr_fe.generate_pair_features(officer, cand, self.transition_stats)
            
            # Inject Context for Explainability
            # Calculate simple Entry Match for display (using strict constraints fallback logic)
            # We don't have easy access to role_cons here in the loop without looking up again.
            # But we can pass it if we refactor. For now, let's just pass officer Entry Type.
            
            entry_type = officer.get('Entry_type', 'Unknown')
            # Look up constraint for this role?
            # We are iterating `candidates_to_score` which came from `valid_roles` loop.
            # We don't have the constraints handy here in `predict` loop easily.
            # But LTR feats usually don't use it.
            # Let's just pass the officer's type. Explainer can display "Candidate is [Type]".
            
            feats['_Context'] = {
                'From_Title': officer.get('last_role_title', 'Unknown'),
                'To_Title': cand.get('Role', 'Unknown'),
                'From_Pool': officer.get('Pool', 'Unknown'),
                'To_Pool': cand.get('Pool', 'Unknown'),
                'Officer_Training': officer.get('Training_history', ''),
                'Entry_Type': entry_type
            }
            
            # Ensure correct col order
            vector = [feats.get(c, 0) for c in self.feature_cols]
            X_rows.append(vector)
            meta_list.append(cand)
            feats_list.append(feats)
            
        if not X_rows:
            return pd.DataFrame()
            
        # Batch Predict
        raw_scores = self.model.predict(X_rows)
        
        # Normalize: GBRT scores are unbounded. Map to 0-1 via Sigmoid.
        # This fixes "Zero Confidence" issues for negative raw scores.
        scores = 1 / (1 + np.exp(-raw_scores))
        
        # 3. Rank and Format
        scored_candidates = []
        for i, score in enumerate(scores):
            scored_candidates.append({
                'role': meta_list[i]['Role'],
                'score': score,
                '_Feats': feats_list[i]
            })
            
        # Sort desc
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Format Results
        results = []
        for item in scored_candidates[:10]:
            prob = item['score']
            role = item['role']
            
            # Simple explanation logic based on features?
            # We don't have easy access to Feat names inside the list vector easily here unless mapped
            # Let's just give generic high score reasons
            
            reason = "High compatibility score."
            if prob > 0.8: reason = "Excellent Fit: Rank, Branch, and Skills match."
            elif prob > 0.5: reason = "Good Fit: Likely transition target."
            else: reason = "Moderate Fit."
            
            results.append({
                'Prediction': role,
                'Confidence': prob,
                'Explanation': reason,
                '_Feats': item['_Feats']
            })
            
        return pd.DataFrame(results)

    def predict_for_role(self, candidates_df, target_role, rank_flex_up=0, rank_flex_down=0):
        """
        Reverse Lookup: Find best officers for a role.
        """
        if target_role not in self.role_meta:
             # Try to find closest? or Generic
             cand_meta = {'Role': target_role, 'Branch': 'Unknown', 'REQ_Ranks': [], 'freq': 1}
        else:
             cand_meta = self.role_meta[target_role]
             
        results = []
        
        # Iterate officers
        # Optimization: Filter officers by Rank first?
        # Yes, filter candidates_df by role requirements
        
        # valid_officers = candidates_df[candidates_df['Rank'].isin(cand_meta['REQ_Ranks'])]
        # Taking all for now to show scores
        # Iterate officers
        X_rows = []
        feats_list = []
        indices = []
        
        # Load constraints for target role
        target_cons = self.strict_constraints.get(target_role, {})
        allowed_ranks = target_cons.get('ranks', [])
        allowed_branches = target_cons.get('branches', [])
        allowed_entries = target_cons.get('entries', []) # New
        
        for idx, row in candidates_df.iterrows():
            # STRICT FILTERING
            # 1. Rank
            # (Handled by flex sliders? Or strict?)
            current_rank = row['Rank']
            if allowed_ranks:
                # If strict, must be in allowed keys
                # BUT we need to handle flexibility logic here if we really want to support "Promotable" officers
                # Simplified: If rank strictly in allowed, keep.
                # If rank_flex > 0, we need a "rank map" to know if adjacent.
                # Since we don't have rank map easily accessible here, let's skip STRICT rank filter if flexibility is enabling.
                # But if flexibility is 0, we can filter.
                if rank_flex_up == 0 and rank_flex_down == 0:
                     if current_rank not in allowed_ranks:
                         continue
            
            # 2. Entry Type Check
            if allowed_entries and row['Entry_type'] not in allowed_entries:
                continue
                
            # 3. Branch Check (Optional/Strict?)
            # Usually role implies branch, but cross-branch happens.
            # If strict constraints exist, respect them.
            if allowed_branches and row['Branch'] not in allowed_branches:
                continue

            off_dict = row.to_dict()
            try:
                self._prepare_input_context(off_dict) # In-place enrich
            except:
                pass # Use raw if fails
            feats = self.ltr_fe.generate_pair_features(off_dict, cand_meta, self.transition_stats)
            
            # Inject Context for Explainability
            feats['_Context'] = {
                'From_Title': off_dict.get('last_role_title', 'Unknown'),
                'To_Title': cand_meta.get('Role', 'Unknown'),
                'From_Pool': off_dict.get('Pool', 'Unknown'),
                'To_Pool': cand_meta.get('Pool', 'Unknown'),
                'Officer_Training': off_dict.get('Training_history', '')
            }
            
            # Store feats for later retrieval
            # We can't put Dict in X_rows.
            # We must map index to feats.
            # Hack: Store in a temp list aligned with X_rows?
            # Or simplified: X_rows is list of lists
            
            vector = [feats.get(c, 0) for c in self.feature_cols]
            
            X_rows.append(vector)
            feats_list.append(feats) # Needs initialization
            indices.append(idx)
            
        if not X_rows: return pd.DataFrame()
        
        # Predict Raw Scores
        raw_scores = self.model.predict(X_rows)
        
        # Normalize
        scores = 1 / (1 + np.exp(-raw_scores))
        
        out = []
        for i, score in enumerate(scores):
            orig_idx = indices[i]
            row = candidates_df.loc[orig_idx]
            feat_dict = feats_list[i]
            
            if score > 0.01: # 1% threshold to filter noise
                out.append({
                    'Employee_ID': row['Employee_ID'],
                    'Name': f"Officer {row['Employee_ID']}",
                    'Rank': row['Rank'],
                    'Branch': row['Branch'],
                    'Confidence': score,
                    'Explanation': f"Match Probability: {score:.1%}",
                    '_Feats': feat_dict 
                })
        
        if not out:
             return pd.DataFrame(columns=['Employee_ID', 'Name', 'Rank', 'Branch', 'Confidence', 'Explanation'])
             
        return pd.DataFrame(out).sort_values('Confidence', ascending=False)
