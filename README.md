# AI Appointer Assist

AI-powered career path prediction system for organizational talent management.

## 🎯 Features

- **Next Appointment Prediction**: AI-driven recommendations for officer's next role
- **Career Simulation**: Explore hypothetical career scenarios
- **Billet Lookup**: Find best candidates for specific positions
- **Constraint-Based**: Respects rank, branch, and organizational rules
- **Flexible Predictions**: Adjustable rank flexibility for creative exploration

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
│   ├── inference.py       # Prediction engine
│   ├── model_trainer.py   # Model training
│   └── ...
├── models/                # Trained models
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

### Running Tests
```bash
python -m pytest tests/
```

### Training Model
```bash
python scripts/train_model.py
```

## 📊 Model Performance

- **Algorithm**: LightGBM Gradient Boosting
- **Top-5 Accuracy**: ~19.4%
- **Inference Speed**: <100ms
- **Features**: Rank, Branch, Pool, Career History, Training

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

**Version**: 1.0.0  
**Last Updated**: 2025-12-08
