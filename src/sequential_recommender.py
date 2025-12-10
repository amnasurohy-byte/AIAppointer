import pandas as pd
import numpy as np
from collections import defaultdict

class SequentialRecommender:
    def __init__(self):
        # Nested dict: matrix[current_state][next_state] = probability
        self.role_transitions = {}
        self.unit_transitions = {}

        # Global priors (for cold start)
        self.role_priors = defaultdict(float)
        self.unit_priors = defaultdict(float)

    def fit(self, transitions_df):
        """
        Learns transition probabilities from the transitions dataframe.
        """
        print("Training Sequential Markov Models...")

        # 1. Role Transitions
        # Count occurrences
        role_counts = defaultdict(lambda: defaultdict(int))
        role_totals = defaultdict(int)

        for _, row in transitions_df.iterrows():
            curr = str(row['last_role_title'])
            next_role = str(row['Target_Next_Role'])

            if curr != 'Unknown' and next_role != 'Unknown':
                role_counts[curr][next_role] += 1
                role_totals[curr] += 1

            self.role_priors[next_role] += 1

        # Normalize to probabilities
        for curr, targets in role_counts.items():
            total = role_totals[curr]
            if curr not in self.role_transitions:
                self.role_transitions[curr] = {}
            for tgt, count in targets.items():
                self.role_transitions[curr][tgt] = count / total

        # Normalize priors
        total_roles = sum(self.role_priors.values())
        for r in self.role_priors: self.role_priors[r] /= total_roles

        # 2. Unit Transitions
        # We need to extract Last Unit -> Next Unit
        # The transition_df has 'Target_Next_Unit' but not 'Last_Unit' explicitly?
        # We can extract it from 'last_role_title' using DataProcessor logic?
        # But `last_role_title` is normalized (stripped of unit).
        # We need the RAW previous role.
        # `df_transitions` usually doesn't store raw previous role title in a column?
        # Let's check `create_transition_dataset`. It stores `snapshot_history`.

        unit_counts = defaultdict(lambda: defaultdict(int))
        unit_totals = defaultdict(int)

        from src.data_processor import DataProcessor
        dp = DataProcessor()

        for _, row in transitions_df.iterrows():
            hist = row.get('snapshot_history', [])
            if not hist: continue

            # Get last raw title
            last_raw = hist[-1].get('title', '')
            next_unit = str(row['Target_Next_Unit'])

            curr_unit = dp.extract_unit(last_raw)

            if curr_unit != 'Unknown' and next_unit != 'Unknown':
                unit_counts[curr_unit][next_unit] += 1
                unit_totals[curr_unit] += 1

            self.unit_priors[next_unit] += 1

        # Normalize
        for curr, targets in unit_counts.items():
            total = unit_totals[curr]
            if curr not in self.unit_transitions:
                self.unit_transitions[curr] = {}
            for tgt, count in targets.items():
                self.unit_transitions[curr][tgt] = count / total

        total_units = sum(self.unit_priors.values())
        for u in self.unit_priors: self.unit_priors[u] /= total_units

        print(f"Learned {len(self.role_transitions)} Role patterns and {len(self.unit_transitions)} Unit patterns.")

    def predict_role_probs(self, current_role, candidates):
        """
        Returns probability dict for candidates given current_role.
        """
        probs = {}
        transitions = self.role_transitions.get(current_role, {})

        for cand in candidates:
            # P(cand | current)
            p_trans = transitions.get(cand, 0.0)
            # Smoothing with prior?
            # P_smooth = 0.9 * P_trans + 0.1 * P_prior
            p_prior = self.role_priors.get(cand, 0.0)

            if p_trans > 0:
                probs[cand] = 0.9 * p_trans + 0.1 * p_prior
            else:
                probs[cand] = 0.01 * p_prior # Penalty if no transition seen

        return probs

    def predict_unit_probs(self, current_unit, candidates):
        """
        Returns probability dict for candidates given current_unit.
        """
        probs = {}
        transitions = self.unit_transitions.get(current_unit, {})

        for cand in candidates:
            p_trans = transitions.get(cand, 0.0)
            p_prior = self.unit_priors.get(cand, 0.0)

            if p_trans > 0:
                probs[cand] = 0.9 * p_trans + 0.1 * p_prior
            else:
                probs[cand] = 0.01 * p_prior

        return probs
