# AI Appointer Assist (Enhanced V2)

AI-powered career path prediction system for organizational talent management.
**Enhanced with Hybrid Dual-Model & Sequential Recommendation Engine.**

## 🎯 Features

- **Next Appointment Prediction**: AI-driven recommendations for officer's next role
- **Career Simulation**: Explore hypothetical career scenarios
- **Billet Lookup**: Find best candidates for specific positions
- **Constraint-Based**: Respects rank, branch, and organizational rules
- **Flexible Predictions**: Adjustable rank flexibility for creative exploration

## 🧠 New AI Architecture (V2)

The system has been upgraded from a simple classifier to a robust **Hybrid Ensemble**:

1.  **Dual-Model Prediction**:
    *   **Role Model (LightGBM)**: Predicts the *type* of job (e.g., "Div Officer"). Accuracy: ~58%.
    *   **Unit Model (LightGBM)**: Predicts the *target unit* (e.g., "USS Vanguard"). Accuracy: ~57%.
2.  **Sequential Intelligence**:
    *   **Markov Chain Recommender**: Learns historical transition probabilities ($A \to B \to C$) to enforce valid career pipelines.
3.  **Specific Billet Ranking**:
    *   Combines $P(Role) \times P(Unit)$ to generate specific recommendations.
    *   Uses **Case-Based Reasoning (CBR)** to find historical precedents for specific assignments.

**Performance:**
- **Generalized Role Accuracy**: ~77% (Top-5)
- **Target Unit Accuracy**: ~70% (Top-5)
- **Specific Billet Accuracy**: ~38% (Top-5) - *Production Ready for Decision Support*

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone repository
git clone <repository-url>
cd AIAppointer

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
AIAppointer/
├── src/                    # Core application code
│   ├── app.py             # Streamlit UI
│   ├── inference.py       # Hybrid Prediction Engine
│   ├── model_trainer.py   # Dual-Model + Seq Training
│   ├── sequential_recommender.py # Markov Chain Logic
│   └── ...
├── models/                # Trained models (Role, Unit, Seq)
├── data/                  # Dataset files
├── scripts/               # Deployment scripts
├── tests/                 # Test suite
├── dev/                   # Development tools
├── docs/                  # Documentation
└── config.py             # Configuration
```

## 📖 Documentation

- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [Quick Start Guide](QUICKSTART.md) - Getting started
- [Architecture](docs/ARCHITECTURE.md) - System design

## 🔧 Configuration

Edit `config.py` to customize:
- Dataset path
- Model directory
- UI settings
- Rank flexibility defaults

## 🛠️ Development

### Training Model (Required for V2)
```bash
python -m src.model_trainer
```
*Note: This generates `role_model.pkl`, `unit_model.pkl`, `seq_model.pkl`, and `knowledge_base.csv`.*

## 🔒 Security

- No external API calls
- All data processed locally
- Configurable via environment variables
- Audit logging support

## 📝 License

[Add your license here]

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md)

## 📧 Contact

[Add contact information]

---

**Version**: 2.0.0 (Enhanced Independent Fork)
**Last Updated**: 2025-12-10
