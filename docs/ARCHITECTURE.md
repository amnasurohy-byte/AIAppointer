# AI Appointer Assist - Architecture

## System Overview

AI Appointer Assist is a machine learning-powered career path prediction system that recommends next best assignments for officers based on their career history, qualifications, and organizational constraints.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Employee   │  │  Simulation  │  │    Billet    │     │
│  │    Lookup    │  │     Mode     │  │    Lookup    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Inference Engine                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Predictor Class                                   │    │
│  │  - predict(officer_data, rank_flex_up/down)       │    │
│  │  - predict_for_role(candidates, target_role)      │    │
│  │  - apply_constraints(predictions)                 │    │
│  │  - generate_explanations(predictions)             │    │
│  └────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌────────────────┐ ┌────────────┐ ┌──────────────┐
│ Data Processor │ │  Feature   │ │  Constraint  │
│                │ │ Engineering│ │  Generator   │
│ - Parse history│ │ - Extract  │ │ - Rank rules │
│ - Clean data   │ │   features │ │ - Branch fit │
│ - Validate     │ │ - Encode   │ │ - Pool match │
└────────────────┘ └────────────┘ └──────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   LightGBM Model                            │
│  - Multi-class classification                              │
│  - Gradient boosting                                       │
│  - ~1000 role classes                                      │
│  - Features: rank, branch, pool, history, training         │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data & Models Storage                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Dataset    │  │    Trained   │  │  Constraints │     │
│  │   (CSV)      │  │    Models    │  │    (JSON)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Web UI Layer (`src/app.py`)
- **Technology**: Streamlit
- **Modes**:
  - Employee Lookup: Search by ID, predict next role
  - Simulation: Hypothetical career scenarios
  - Billet Lookup: Find candidates for specific role
- **Features**:
  - Rank flexibility sliders (promotion/demotion)
  - Interactive visualizations
  - Detailed explanations

### 2. Inference Engine (`src/inference.py`)
- **Predictor Class**: Main prediction logic
- **Methods**:
  - `predict()`: Top-K role predictions for officer
  - `predict_for_role()`: Confidence scores for specific role
- **Constraint Enforcement**:
  - Rank matching (with flexibility)
  - Branch compatibility
  - Pool requirements
  - Repetition penalty
- **Explanation Generation**:
  - Rank progression analysis
  - Branch fit reasoning
  - Training/experience match
  - Seniority considerations

### 3. Data Processing (`src/data_processor.py`)
- **Responsibilities**:
  - Parse appointment history
  - Parse training history
  - Parse promotion history
  - Extract current features
  - Data validation

### 4. Feature Engineering (`src/feature_engineering.py`)
- **Features Extracted**:
  - Numerical: years_service, years_in_current_rank, num_prior_roles
  - Categorical: Rank, Branch, Pool, Entry_type, last_role_title
  - Training counts: command, tactical, science, engineering, medical
  - Derived: seniority scores, progression patterns

### 5. Model Training (`src/model_trainer.py`)
- **Algorithm**: LightGBM Gradient Boosting
- **Training Process**:
  1. Load and validate data
  2. Generate constraints
  3. Extract features
  4. Encode categorical variables
  5. Train model
  6. Save artifacts
- **Outputs**:
  - Trained model (`.pkl`)
  - Encoders (`.pkl`)
  - Feature columns list
  - Constraints (`.json`)

### 6. Constraint Generator (`src/constraint_generator.py`)
- **Purpose**: Extract organizational rules from data
- **Constraints Generated**:
  - Valid ranks per role
  - Required branches per role
  - Pool requirements per role
- **Format**: JSON dictionary

## Data Flow

### Prediction Flow
```
User Input
    ↓
Data Processor (parse history)
    ↓
Feature Engineering (extract features)
    ↓
Encoding (categorical → numerical)
    ↓
LightGBM Model (predict probabilities)
    ↓
Constraint Filtering (apply rules)
    ↓
Rank Flexibility Boost (if enabled)
    ↓
Top-K Selection
    ↓
Explanation Generation
    ↓
Display Results
```

### Training Flow
```
CSV Dataset
    ↓
Constraint Generation (analyze data)
    ↓
Data Processing (clean, parse)
    ↓
Feature Engineering (extract features)
    ↓
Encoding (fit encoders)
    ↓
Train/Test Split
    ↓
LightGBM Training
    ↓
Model Evaluation
    ↓
Save Artifacts (model, encoders, constraints)
```

## Key Design Decisions

### 1. Multi-Class Classification
- **Choice**: Treat each role as a separate class
- **Rationale**: Simple, interpretable, fast inference
- **Trade-off**: Doesn't model sequential patterns (future: RNN/Transformer)

### 2. Constraint-Based Filtering
- **Choice**: Apply hard constraints after model prediction
- **Rationale**: Ensures organizational rules are never violated
- **Implementation**: Zero out probabilities for invalid roles

### 3. Rank Flexibility with Boost
- **Choice**: Separate up/down sliders + probability boost
- **Rationale**: Model learned from data where promotions are rare
- **Boost Factor**: 2^rank_difference for promotion roles

### 4. Explanation Generation
- **Choice**: Rule-based explanations from features
- **Rationale**: Interpretability, trust, debugging
- **Components**: Rank progression, branch fit, training match

## Performance Characteristics

### Metrics
- **Top-5 Accuracy**: ~19.4%
- **Inference Speed**: <100ms per prediction
- **Model Size**: ~50MB
- **Memory Usage**: ~500MB (loaded model + data)

### Scalability
- **Officers**: Tested up to 10,000
- **Roles**: Tested up to 1,000 unique roles
- **Concurrent Users**: Streamlit handles ~10-50 (single instance)

## Security Considerations

### Data Privacy
- All processing done locally
- No external API calls
- No data leaves the server

### Input Validation
- Employee ID validation
- Role name sanitization
- Feature value bounds checking

### Error Handling
- Graceful degradation
- User-friendly error messages
- Detailed logging for debugging

## Deployment Modes

### 1. On-Premise Air-Gapped
- Self-contained package
- No internet required
- All dependencies pre-bundled

### 2. Cloud/Server
- Git-based deployment
- Systemd service
- Reverse proxy (Nginx) for HTTPS

### 3. Docker
- Containerized deployment
- Volume mounts for data/models
- Easy scaling

## Future Enhancements

### Short-Term
- Markov Chain enhancement (+3-8% accuracy)
- More granular constraints
- User authentication

### Medium-Term
- Hybrid LightGBM + RNN (+12-18% accuracy)
- Career path visualization
- Batch prediction API

### Long-Term
- Full sequential model (CAPER-style)
- Multi-objective optimization
- Explainable AI (SHAP values)

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| UI Framework | Streamlit | 1.29+ |
| ML Library | LightGBM | 4.1+ |
| Data Processing | Pandas | 2.1+ |
| Numerical | NumPy | 1.26+ |
| ML Utilities | scikit-learn | 1.3+ |
| Serialization | joblib | 1.3+ |
| Language | Python | 3.8+ |

## File Structure

```
AIAppointer/
├── src/
│   ├── app.py                 # Streamlit UI
│   ├── inference.py           # Prediction engine
│   ├── data_processor.py      # Data processing
│   ├── feature_engineering.py # Feature extraction
│   ├── model_trainer.py       # Training pipeline
│   └── constraint_generator.py# Constraint extraction
├── models/                    # Trained artifacts
├── data/                      # Dataset files
├── scripts/                   # Deployment scripts
├── tests/                     # Test suite
├── dev/                       # Development tools
├── docs/                      # Documentation
└── config.py                  # Configuration
```

## Configuration

### Environment Variables
- `APPOINTER_DATA_DIR`: Data directory path
- `APPOINTER_MODELS_DIR`: Models directory path
- `DEFAULT_RANK_FLEX`: Default rank flexibility
- `UI_TITLE`: Application title
- `LOG_LEVEL`: Logging verbosity

### config.py
- Dataset path
- Model directory
- UI settings
- Feature engineering parameters

## Monitoring & Logging

### Logs
- Application logs: `logs/appointer.log`
- Error tracking
- Performance metrics

### Health Checks
- Streamlit health endpoint: `/healthz`
- Model validation on startup
- Data integrity checks

## References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Deployment Guide](DEPLOYMENT.md)
