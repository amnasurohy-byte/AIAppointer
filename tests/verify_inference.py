import pandas as pd
from src.inference import Predictor
from config import DATASET_PATH

print("Initializing Predictor...")
predictor = Predictor()

print("Loading Data...")
df = pd.read_csv(DATASET_PATH)
sample_row = df[df['Employee_ID'] == 200009]

print("Predicting...")
try:
    result = predictor.predict(sample_row)
    print("\nPrediction Result:")
    print(result.to_string())
    print("\nSUCCESS: Inference Logic Verified.")
except Exception as e:
    print(f"\nFAILURE: {e}")
    exit(1)
