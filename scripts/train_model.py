"""
Train AI Appointer Assist Model

This script trains the LightGBM model for career path prediction.
Run this script whenever you update the dataset or want to retrain the model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_trainer import ModelTrainer
from src.constraint_generator import generate_constraints
from config import DATASET_PATH, MODELS_DIR

def main():
    print("="*60)
    print("AI Appointer Assist - Model Training")
    print("="*60)
    print()
    
    print(f"Dataset: {DATASET_PATH}")
    print(f"Models Dir: {MODELS_DIR}")
    print()
    
    # 1. Generate Constraints
    print("Step 1: Generating Constraints...")
    generate_constraints(DATASET_PATH, output_dir=MODELS_DIR)
    print()
    
    # 2. Train Model
    print("Step 2: Training Model...")
    trainer = ModelTrainer(models_dir=MODELS_DIR)
    trainer.train(DATASET_PATH)
    
    print()
    print("="*60)
    print("Training Complete!")
    print("="*60)
    print()
    print("Models saved to: models/")
    print("- lgbm_model.pkl")
    print("- all_constraints.json")
    print("- target_encoder.pkl")
    print("- encoders.pkl")
    print("- feature_cols.pkl")
    print()

if __name__ == "__main__":
    main()
