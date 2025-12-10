import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import json

from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer

class ModelTrainer:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        self.encoders = {}
        self.role_encoder = None
        self.unit_encoder = None
        self.role_model = None
        self.unit_model = None
        self.feature_cols = []

    def train(self, csv_path):
        print("="*60)
        print("STEP 1: Generating Role Constraints")
        print("="*60)
        from src.constraint_generator import generate_constraints
        generate_constraints(csv_path, output_dir=self.models_dir, verbose=True)
        
        print("\n" + "="*60)
        print("STEP 2: Processing Data")
        print("="*60)
        df = pd.read_csv(csv_path)
        dp = DataProcessor()
        df_transitions = dp.create_transition_dataset(df)
        
        fe = FeatureEngineer()
        df_transitions = fe.extract_features(df_transitions)
        
        # Save KB
        kb_cols = ['Rank', 'Branch', 'last_role_title', 'Target_Next_Role', 'Target_Next_Role_Raw']
        kb_df = df_transitions[kb_cols].copy()
        kb_df.to_csv(os.path.join(self.models_dir, 'knowledge_base.csv'), index=False)

        # Features
        cat_features = ['Rank', 'Branch', 'Pool', 'Entry_type',
                        'last_role_title', 'prev_role_2', 'prev_role_3', 'last_training_title']
        num_features = ['years_service', 'days_in_last_role', 'years_in_current_rank', 'num_prior_roles', 
                        'num_training_courses', 
                        'count_command_training', 'count_tactical_training', 'count_science_training',
                        'count_engineering_training', 'count_medical_training',
                        'days_since_last_training']
        
        X = df_transitions[cat_features + num_features].copy()
        self.feature_cols = cat_features + num_features
        
        # Encoding Features
        print("Encoding Features...")
        for col in cat_features:
            le = LabelEncoder()
            X[col] = X[col].fillna('Unknown').astype(str)
            X[col] = le.fit_transform(X[col])
            self.encoders[col] = le
            
        # --- MODEL A: ROLE PREDICTION ---
        print("\n" + "="*40)
        print("TRAINING MODEL A: Generalized Role")
        print("="*40)
        y_role = df_transitions['Target_Next_Role'].copy()

        # Filter Rare Roles
        role_counts = y_role.value_counts()
        valid_roles = role_counts[role_counts >= 2].index
        mask_role = y_role.isin(valid_roles)
        X_role = X[mask_role]
        y_role_filtered = y_role[mask_role]

        self.role_encoder = LabelEncoder()
        y_role_enc = self.role_encoder.fit_transform(y_role_filtered.astype(str))

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_role, y_role_enc, test_size=0.2, random_state=42, stratify=y_role_enc)

        self.role_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(self.role_encoder.classes_),
            n_estimators=50,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        cat_indices = [X.columns.get_loc(c) for c in cat_features]
        self.role_model.fit(X_train, y_train, categorical_feature=cat_indices)
        
        print(f"Role Top-1 Acc: {accuracy_score(y_test, self.role_model.predict(X_test)):.3f}")
        
        # --- MODEL B: UNIT PREDICTION ---
        print("\n" + "="*40)
        print("TRAINING MODEL B: Target Unit")
        print("="*40)
        y_unit = df_transitions['Target_Next_Unit'].copy()
        self.unit_encoder = LabelEncoder()
        y_unit_enc = self.unit_encoder.fit_transform(y_unit.astype(str))
        
        # Split (Stratified on Units)
        # Note: Some units are rare, StratifiedKFold might warn, but train_test_split handles singletons by putting in train usually?
        # Or we filter.
        valid_units = y_unit.value_counts()[y_unit.value_counts() >= 2].index
        mask_unit = y_unit.isin(valid_units)
        X_unit = X[mask_unit]
        y_unit_enc_filtered = self.unit_encoder.transform(y_unit[mask_unit].astype(str))
        
        X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_unit, y_unit_enc_filtered, test_size=0.2, random_state=42, stratify=y_unit_enc_filtered)
        
        self.unit_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(self.unit_encoder.classes_),
            n_estimators=50,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        self.unit_model.fit(X_train_u, y_train_u, categorical_feature=cat_indices)

        print(f"Unit Top-1 Acc: {accuracy_score(y_test_u, self.unit_model.predict(X_test_u)):.3f}")

        self.save_artifacts()

    def save_artifacts(self):
        print(f"Saving models to {self.models_dir}...")
        # Compress
        joblib.dump(self.role_model, os.path.join(self.models_dir, 'role_model.pkl'), compress=3)
        joblib.dump(self.unit_model, os.path.join(self.models_dir, 'unit_model.pkl'), compress=3)
        joblib.dump(self.encoders, os.path.join(self.models_dir, 'feature_encoders.pkl'), compress=3)
        joblib.dump(self.role_encoder, os.path.join(self.models_dir, 'role_encoder.pkl'), compress=3)
        joblib.dump(self.unit_encoder, os.path.join(self.models_dir, 'unit_encoder.pkl'), compress=3)
        joblib.dump(self.feature_cols, os.path.join(self.models_dir, 'feature_cols.pkl'), compress=3)
        print("Save Complete.")

if __name__ == "__main__":
    from config import DATASET_PATH
    trainer = ModelTrainer()
    trainer.train(DATASET_PATH)
