import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer

class ModelTrainer:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        self.encoders = {}
        self.target_encoder = None
        self.model = None
        self.feature_cols = []

    def train(self, csv_path):
        print("="*60)
        print("STEP 1: Generating Role Constraints")
        print("="*60)
        from src.constraint_generator import generate_constraints
        generate_constraints(csv_path, output_dir=self.models_dir, verbose=True)
        
        print("\n" + "="*60)
        print("STEP 2: Loading and Processing Data")
        print("="*60)
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # 1. Processing Pipeline
        print("Running Data Pipeline (Transition Mode)...")
        dp = DataProcessor()
        # Create Transition Dataset (Exploded)
        # Note: create_transition_dataset internall calls get_current_features if needed.
        df_transitions = dp.create_transition_dataset(df)
        print(f"Dataset exploded from {len(df)} officers to {len(df_transitions)} transitions.")
        
        fe = FeatureEngineer()
        df_transitions = fe.extract_features(df_transitions)
        
        # 2. Prepare Features (X) and Target (y)
        # Define categorical features
        cat_features = ['Rank', 'Branch', 'Pool', 'Entry_type', 'last_role_title']
        # Updated feature list (Phase 3)
        num_features = ['years_service', 'days_in_last_role', 'years_in_current_rank', 'num_prior_roles', 
                        'num_training_courses', 
                        'count_command_training', 'count_tactical_training', 'count_science_training',
                        'count_engineering_training', 'count_medical_training']
        
        target_col = 'Target_Next_Role' # <-- CHANGED TARGET
        
        # Filter columns
        X = df_transitions[cat_features + num_features].copy()
        y = df_transitions[target_col].copy()
        
        # Filter out classes with < 2 samples (cannot stratify/train/test split properly)
        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        mask = y.isin(valid_classes)
        print(f"Dropping {len(y) - mask.sum()} rows due to rare target classes (freq < 2).")
        X = X[mask]
        y = y[mask]

        # 3. Encoding
        print("Encoding Features...")
        for col in cat_features:
            le = LabelEncoder()
            X[col] = X[col].fillna('Unknown').astype(str)
            X[col] = le.fit_transform(X[col])
            self.encoders[col] = le
            
        # Encode Target
        print("Encoding Target...")
        self.target_encoder = LabelEncoder()
        y = self.target_encoder.fit_transform(y.astype(str))
        
        # Save feature names for inference
        self.feature_cols = cat_features + num_features
        
        # 4. Split (Stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 5. Train LightGBM
        print(f"Training LightGBM on {len(X_train)} samples...")
        # Create dataset
        # dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features) # using indices
        # We passed encoded integers, so we should tell LGBM which are categorical
        # Get indices of cat features
        cat_indices = [X.columns.get_loc(c) for c in cat_features]
        
        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(self.target_encoder.classes_),
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='multi_logloss',
            categorical_feature=cat_indices
        )
        
        # 6. Evaluate
        print("Evaluating...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Top-1 Accuracy: {acc:.4f}")
        
        # Top-5 Accuracy?
        # verify dimensionality
        probas = self.model.predict_proba(X_test)
        top5_acc = self._top_k_accuracy(y_test, probas, k=5)
        print(f"Top-5 Accuracy: {top5_acc:.4f}")
        
        # 7. Save Artifacts
        self.save_artifacts()
        
    def _top_k_accuracy(self, y_true, y_proba, k=5):
        # Setup
        best_n = np.argsort(y_proba, axis=1)[:,-k:]
        ts = 0
        for i in range(len(y_true)):
            if y_true[i] in best_n[i,:]:
                ts += 1
        return ts / len(y_true)

    def save_artifacts(self):
        print(f"Saving model and encoders to {self.models_dir}...")
        joblib.dump(self.model, os.path.join(self.models_dir, 'lgbm_model.pkl'))
        joblib.dump(self.encoders, os.path.join(self.models_dir, 'feature_encoders.pkl'))
        joblib.dump(self.target_encoder, os.path.join(self.models_dir, 'target_encoder.pkl'))
        joblib.dump(self.feature_cols, os.path.join(self.models_dir, 'feature_cols.pkl'))
        print("Save Complete.")

if __name__ == "__main__":
    from config import DATASET_PATH
    trainer = ModelTrainer()
    trainer.train(DATASET_PATH)
