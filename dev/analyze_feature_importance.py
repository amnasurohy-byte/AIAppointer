"""
Analyze feature importance to understand why current role has minimal impact
"""
import joblib
import pandas as pd

print("Loading model and feature columns...")
model = joblib.load('models/lgbm_model.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

# Get feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

print("\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

print("\n" + "-"*60)
print("Key Feature Rankings:")
print("-"*60)

# Find specific features
for feature in ['last_role_title', 'Rank', 'Branch', 'Pool', 'Entry_type', 'years_service']:
    if feature in importance_df['feature'].values:
        rank = importance_df[importance_df['feature'] == feature].index[0] + 1
        importance = importance_df[importance_df['feature'] == feature]['importance'].values[0]
        print(f"{feature:20s}: Rank #{rank:2d}, Importance: {importance:.4f}")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

last_role_importance = importance_df[importance_df['feature'] == 'last_role_title']['importance'].values[0]
rank_importance = importance_df[importance_df['feature'] == 'Rank']['importance'].values[0]

if last_role_importance < rank_importance * 0.1:
    print("⚠️  ISSUE: last_role_title has very low importance compared to Rank")
    print(f"   Ratio: {last_role_importance / rank_importance:.2%}")
    print("\nRECOMMENDATION:")
    print("- The model is over-relying on Rank/Branch and ignoring current role")
    print("- This explains why changing current role has minimal effect")
    print("- Consider retraining with balanced feature weights or adding role-based features")
else:
    print("✓ last_role_title has reasonable importance")
