# AI Officer Appointer - Quick Start Guide

## Changing the Dataset

To use a new dataset:

1. **Update config.py**:
   ```python
   DATASET_PATH = 'your_new_dataset.csv'
   ```

2. **Retrain the model** (this automatically updates constraints):
   ```bash
   python -m src.model_trainer
   ```
   
   This single command will:
   - ✓ Generate/update role constraints (`models/all_constraints.json`)
   - ✓ Process the data and create transition dataset
   - ✓ Train the LightGBM model
   - ✓ Save all artifacts (model, encoders, feature columns)

3. **Restart the Streamlit app**:
   ```bash
   streamlit run src/app.py
   ```

## What Happens During Training?

When you run `python -m src.model_trainer`, the system automatically:

### Step 1: Constraint Generation
- Analyzes the dataset to map each role to valid Ranks, Branches, and Pools
- Creates `models/all_constraints.json` (this ensures predictions respect organizational rules)
- Example: "Chief Engineer" → Only Engineering Branch, Commander/Captain ranks

### Step 2: Data Processing
- Parses appointment, training, and promotion histories
- Creates transition dataset (learns "State A → Role B" patterns)
- Extracts features (years of service, training counts, etc.)

### Step 3: Model Training
- Trains LightGBM classifier on career transitions
- Validates with Top-1 and Top-5 accuracy metrics
- Saves model and encoders to `models/` directory

## Dataset Requirements

Your CSV must have these columns:
- `Employee_ID`
- `Rank`
- `Branch`
- `Pool`
- `Entry_type`
- `Appointment_history`
- `Training_history`
- `Promotion_history`
- `current_appointment`
- `appointed_since`

## UI Improvements

The UI now features:
- **Responsive design** that adapts to window size
- **Text wrapping** in tables for long explanations
- **Column width optimization** for better readability
- **Styled metrics** with background colors
- **Rank Flexibility Slider**: Control how strict rank constraints are

### Rank Flexibility Feature

The system includes a **Rank Flexibility** slider in the sidebar with 4 levels:

- **0 - Strict** (🔒): Only exact rank matches (Commander → Commander roles only)
- **1 - Flexible** (⚡): ±1 rank (Commander → Lt.Cdr, Commander, or Captain roles)
- **2 - Creative** (🎨): ±2 ranks (broader options for exploration)
- **3 - Very Creative** (🌟): ±3 ranks (maximum flexibility)

**Default**: 0 (Strict) - ensures predictions respect organizational hierarchy.

**When to adjust**: 
- Use Strict (0) for official recommendations
- Use Flexible (1-2) to explore career progression paths
- Use Creative (3) for "what-if" scenarios and brainstorming

## Troubleshooting

### Predictions seem inaccurate
1. Check that training completed successfully (no errors)
2. Verify `models/all_constraints.json` exists and is recent
3. Check dataset quality (no missing values in key columns)
4. Ensure dataset has enough samples per role (minimum 2)

### UI not updating after retraining
1. Stop the Streamlit app (Ctrl+C)
2. Clear Streamlit cache: `streamlit cache clear`
3. Restart: `streamlit run src/app.py`

### Constraints not being enforced
- The constraints are automatically regenerated during training
- If you manually edited constraints, they will be overwritten
- To customize constraints, edit them AFTER training completes
