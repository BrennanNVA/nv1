# Nova Aetus - Automated Trading System

An AI-powered trading system that combines technical analysis, machine learning, and sentiment analysis to make swing trading decisions.

## What It Does

1. **Analyzes stocks** using 88+ technical indicators, ML models, and sentiment
2. **Generates signals** when multiple factors agree (Technical + Sentiment + Fundamental)
3. **Manages risk** automatically using mathematical formulas (Kelly Criterion)
4. **Executes trades** via Alpaca API when signals pass all checks
5. **Monitors performance** via dashboard and Discord notifications

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Alpaca and Discord credentials
```

### 2. Start Database
```bash
docker-compose up -d
```

### 3. Train Models
```bash
# Train individual stock models (one per symbol)
python scripts/train_models.py --all --years 3

# Train master ensemble model (combines all individual models)
python scripts/train_master_model.py --all --years 2
```

### 4. Run System
```bash
# Start trading loop
python -m src.nova.main

# Start dashboard (separate terminal)
streamlit run src/nova/dashboard/app.py
```

## Documentation

**Complete Guide**: See [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) for everything you need to know.

**Recent Changes**: See [`docs/CHANGELOG.md`](docs/CHANGELOG.md) for what's new.

## Tech Stack

- **Python 3.12+**
- **XGBoost** (GPU-accelerated ML)
- **Polars** (fast data processing)
- **TimescaleDB** (time-series database)
- **Ollama** (local LLM for sentiment)
- **Streamlit** (dashboard)
- **Alpaca API** (trading)

## How It Works

**Two-Layer Model System:**
1. **Individual Models** (24 models, one per stock) - Learn stock-specific patterns
2. **Master Model** (1 unified model) - Combines all predictions, learns cross-symbol patterns

**Trading Flow:**
```
Market Data → Features → Individual Models → Master Model → Confluence → Risk Management → Execution
```

See [`docs/GUIDE.md`](docs/GUIDE.md) for detailed explanation.

## Project Structure

```
nova_aetus/
├── .env              # API keys (create from .env.example)
├── config.toml       # Strategy parameters
├── docker-compose.yml # Database setup
├── requirements.txt  # Python dependencies
├── models/          # Trained models (individual + master)
├── logs/            # Application logs
└── src/nova/
    ├── main.py      # Main trading loop
    ├── core/        # Config, logging, notifications
    ├── data/        # Data loading/storage
    ├── features/    # Technical indicators, sentiment
    ├── models/      # Training/prediction
    ├── strategy/    # Risk management, execution
    └── dashboard/   # Streamlit dashboard
```

## Requirements

- Python 3.12+
- Docker & Docker Compose (for TimescaleDB)
- NVIDIA GPU (optional, for faster training)
- Ollama (optional, for sentiment analysis)

## Configuration

**Environment Variables** (`.env`):
- `ALPACA_API_KEY` - Trading API key
- `ALPACA_SECRET_KEY` - Trading API secret
- `DISCORD_WEBHOOK_URL` - Notifications webhook

**Strategy Settings** (`config.toml`):
- Technical indicator parameters
- ML model hyperparameters
- Risk management thresholds

## Troubleshooting

- **"No models found"** → Train individual models first
- **"Database connection failed"** → Start TimescaleDB: `docker-compose up -d`
- **"CUDA out of memory"** → Reduce batch size or use CPU mode

For more help, see [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).
