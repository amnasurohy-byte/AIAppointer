import pandas as pd
import re
import difflib

class Explainer:
    def __init__(self, df=None, known_titles=None):
        self.transition_index = {} # (from, to) -> [(id, name, rank, branch)]
        self.known_titles = known_titles if known_titles else []
        if df is not None:
            self.index_transitions(df)

    def _normalize_title(self, raw_title):
        """
        Maps raw history title to canonical known title using Fuzzy logic.
        Matches Predictor logic to ensure Index keys align with Probability keys.
        """
        if not self.known_titles:
            return raw_title.strip()
            
        t = raw_title.strip()
        # 1. Exact
        if t in self.known_titles:
            return t
        
        # 2. Substring (Case Insensitive)
        t_lower = t.lower()
        for k in self.known_titles:
            if k.lower() in t_lower: # Role name inside history string? "Ensign (jg)" contains "Ensign"?
                return k
            if t_lower in k.lower(): # Reverse? "Ensign" inside "Ensign Role"?
                return k
                
        # 3. Fuzzy
        matches = difflib.get_close_matches(t, self.known_titles, n=1, cutoff=0.4)
        if matches:
            return matches[0]
            
        return t

    def index_transitions(self, df):
        """
        Builds a map of (FromTitle, ToTitle) -> List of Officers who made this move.
        Also builds Role -> Training Profile map (Consensus Training).
        """
        self.role_training_counts = {} # role -> Counter(course -> count)
        self.role_incumbent_counts = {} # role -> total_unique_officers
        
        # Iterate over all officers and parse their history
        for idx, row in df.iterrows():
            emp_id = row.get('Employee_ID', 'Unknown')
            name = row.get('Name', f"Officer {emp_id}")
            rank = row.get('Rank', 'Unknown')
            branch = row.get('Branch', 'Unknown')
            
            # Parse Training
            train_raw = row.get('Training_history', '')
            courses = set()
            if isinstance(train_raw, str) and train_raw:
                # Split by comma
                for t in train_raw.split(','):
                    # Clean dates: "Course (Date)" -> "Course"
                    # Regex: Remove parenthesis and everything inside it
                    clean_course = re.sub(r'\s*\(.*?\)', '', t).strip()
                    if clean_course: courses.add(clean_course)
            
            hist_raw = row.get('Appointment_history', '')
            if not isinstance(hist_raw, str):
                continue
                
            # Parse history (Same logic as Predictor)
            items = str(hist_raw).split(',')
            titles = []
            for item in items:
                clean = re.sub(r'\s*\(.*?\)', '', item).strip()
                if clean: 
                    norm = self._normalize_title(clean)
                    titles.append(norm)
            
            unique_roles_held = set(titles)
            for role in unique_roles_held:
                if role not in self.role_training_counts:
                    self.role_training_counts[role] = {}
                    self.role_incumbent_counts[role] = 0
                
                self.role_incumbent_counts[role] += 1
                
                for course in courses:
                    self.role_training_counts[role][course] = self.role_training_counts[role].get(course, 0) + 1
            
            if len(titles) < 2:
                continue
            
            for i in range(len(titles) - 1):
                t_from = titles[i]
                t_to = titles[i+1]
                
                key = (t_from, t_to)
                if key not in self.transition_index:
                    self.transition_index[key] = []
                
                existing = self.transition_index[key]
                if existing and existing[-1]['Employee_ID'] == emp_id:
                    continue
                    
                self.transition_index[key].append({
                    'Employee_ID': emp_id,
                    'Name': name,
                    'Rank': rank,
                    'Branch': branch,
                    'Pool': row.get('Pool', 'Unknown'),
                    'Entry_type': row.get('Entry_type', 'Unknown'),
                    'Appointment_history': hist_raw,
                    'Training_history': train_raw,
                    'From': t_from,
                    'To': t_to
                })
                
    def get_precedents(self, from_title, to_title, limit=5, fuzzy=False):
        """
        Finds officers who moved From -> To.
        Auto-normalizes inputs to match Index keys.
        """
        # Normalize inputs to match how we indexed them
        k_from = self._normalize_title(from_title)
        k_to = self._normalize_title(to_title)
        
        key = (k_from, k_to)
        
        matches = self.transition_index.get(key, [])
        
        if not matches and fuzzy:
            # Try to partial match? Expensive.
            pass
            
        return matches[:limit]

    def format_feature_explanation(self, feats, score=0.0, constraints=None):
        """
        Converts raw feature dict into rich human-readable explanation objects.
        Merges related metrics to save space and provide clearer context.
        """
        ctx = feats.get('_Context', {})
        t_from = ctx.get('From_Title', 'Unknown')
        t_to = ctx.get('To_Title', 'Unknown')
        p_from = ctx.get('From_Pool', 'Unknown')
        p_to = ctx.get('To_Pool', 'Unknown')
        
        # Get constraints for expectations
        role_const = constraints.get(t_to, {}) if constraints else {}
        expected_ranks = role_const.get('ranks', [])
        expected_branches = role_const.get('branches', [])
        
        ex = {}
        
        # 0. AI Probability Score (NEW)
        # Dynamic Driver Analysis
        drivers = []
        if feats.get('prior_title_prob', 0) > 0.05:
            drivers.append(f"**Strong History Signal ({feats['prior_title_prob']:.1%})**")
        if feats.get('rank_match_exact', 0):
            drivers.append("**Rank Eligibility**")
        if feats.get('branch_match', 0):
            drivers.append("**Branch Fit**")
        
        driver_text = ", ".join(drivers) if drivers else "**Baseline Structural Fit**"
        
        # Determine Status for text
        rank_status = 'Met' if feats.get('rank_match_exact') else 'Miss'
        branch_status = 'Met' if feats.get('branch_match') else 'Miss'
        
        ex['AI Score'] = {
            'value': f"{score:.1%}",
            # FIX: Use f-string to evaluate variables
            'desc': (
                f"**Score Drivers**: {driver_text}\n\n"
                f"The AI calculates this probability by comparing this candidate's profile against:\n"
                f"1.  **Past Patterns**: How often officers moved from '{t_from}' to '{t_to}'.\n"
                f"2.  **Hard Constraints**: Rank ({rank_status}) and Branch ({branch_status}).\n"
                f"3.  **Training Overlap**: Shared skills with previous incumbents."
            )
        }
        
        # 1. Historical Precedent
        # Ensure we use normalized lookup here too for consistency
        norm_from = self._normalize_title(t_from)
        norm_to = self._normalize_title(t_to)
        
        prob = feats.get('prior_title_prob', 0)
        # Look up exact count if possible using normalized keys
        count = len(self.transition_index.get((norm_from, norm_to), []))
        
        suffix = "officer" if count == 1 else "officers"
        
        ex['History Strength'] = {
            'value': f"{prob:.1%}",
            'desc': f"**{prob:.1%} ({count} {suffix})** of those who held the role **'{t_from}'** immediately moved to **'{t_to}'**. This strong historical pattern suggests this is a standard career progression." if prob > 0 else f"No direct historical precedent (N=0) found for moving strictly from **'{t_from}'** to **'{t_to}'**."
        }
        
        # 2. Branch & Pool Fit (MERGED)
        pool_prob = feats.get('prior_pool_prob', 0)
        branch_match = feats.get('branch_match', 0)
        
        # Format expectations
        exp_branch_str = f"Expected Branches: {', '.join(expected_branches)}" if expected_branches else "Expected Branches: Any"
        # Pool constraints not strictly defined in JSON usually, imply 'Any' or context based
        exp_pool_str = "Expected Pool: Open (No strict constraint)"
        
        if branch_match and pool_prob > 0:
            bp_val = "Strong Fit"
            bp_desc = f"**Branch Match**: Candidate is in the correct branch.\n\n{exp_branch_str}\n{exp_pool_str}\n\n**Pool Flow**: {pool_prob:.1%} historical flow from {p_from} to {p_to}."
        elif branch_match:
            bp_val = "Branch Match"
            bp_desc = f"Candidate matches Branch requirement, but Pool transition ({p_from} -> {p_to}) is uncommon ({pool_prob:.1%}).\n\n{exp_branch_str}\n{exp_pool_str}"
        elif pool_prob > 0.1:
            bp_val = "Flow Match"
            bp_desc = f"Cross-Branch candidate, but valid Pool transition flow ({pool_prob:.1%}).\n\n{exp_branch_str}\n{exp_pool_str}"
        else:
            bp_val = "Weak Fit"
            bp_desc = f"Cross-Branch candidate with low historical pool flow ({pool_prob:.1%}).\n\n{exp_branch_str}\n{exp_pool_str}"
            
        ex['Branch & Pool Fit'] = {
            'value': bp_val,
            'desc': bp_desc
        }
        
        # 3. Rank & Entry Fit (MERGED)
        rank_match = feats.get('rank_match_exact', 0)
        entry = ctx.get('Entry_Type', 'Unknown')
        
        # Format expectations
        exp_rank_str = f"Expected Ranks: {', '.join(expected_ranks)}" if expected_ranks else "Expected Ranks: Any"
        exp_entry_str = "Expected Entry: Any (No strict constraint)"
        
        re_val = "Rank Met" if rank_match else "Rank Mismatch"
        re_desc = "Rank requirement is met." if rank_match else "Rank does not match standard constraints."
        re_desc += f"\n\n{exp_rank_str}"
        re_desc += f"\n{exp_entry_str}"
        re_desc += f"\n\nCandidate Entry Protocol: **{entry}**"
        
        ex['Rank & Entry Fit'] = {
            'value': re_val,
            'desc': re_desc
        }
        
        # 4. Training Match (Consensus-Based)
        cand_training_str = ctx.get('Officer_Training', '')
        cand_courses = set()
        if isinstance(cand_training_str, str) and cand_training_str:
            for t in cand_training_str.split(','):
                # Clean dates here too
                clean_t = re.sub(r'\s*\(.*?\)', '', t).strip()
                if clean_t: cand_courses.add(clean_t)
        
        # Get profile for Target Title
        norm_target = self._normalize_title(t_to)
        
        relevant_courses = getattr(self, 'role_training_counts', {}).get(norm_target, {})
        total_incumbents = getattr(self, 'role_incumbent_counts', {}).get(norm_target, 1)
        
        # Find overlaps
        hits = []
        for course, count in relevant_courses.items():
            if course in cand_courses:
                 # Calculate popularity among incumbents
                 # "What was the most popular training?"
                prevalence = count
                total = total_incumbents
                hits.append({'course': course, 'n': count, 'pct': count/max(1,total)})
        
        # Sort by popularity (count desc)
        hits.sort(key=lambda x: x['n'], reverse=True)
        
        if hits:
            # Format top matches
            # "Most popular: X (used by 5), Y (used by 3)"
            details = ", ".join([f"{h['course']} (used by {h['n']})" for h in hits[:3]])
            
            ex['Training Match'] = {
                'value': f"Matched ({len(hits)})",
                'desc': f"The officer possesses **{len(hits)}** qualifications commonly held by previous incumbents. Most popular overlaps: **{details}**."
            }
        else:
            # If no match, show what ARE the popular ones they miss?
            # Find top popular courses for role
            popular_all = sorted(relevant_courses.items(), key=lambda x: x[1], reverse=True)[:3]
            if popular_all:
                missing = ", ".join([f"{c} ({n})" for c, n in popular_all])
                ex['Training Match'] = {
                    'value': "None",
                    'desc': f"Officer lacks common qualifications for this role. Previous incumbents typically held: **{missing}**."
                }
            else:
                ex['Training Match'] = {
                    'value': "N/A",
                    'desc': f"No historical training data for role **'{t_to}'**."
                }
            
        return ex
