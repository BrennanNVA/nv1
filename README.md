# Nova Aetus - Institutional-Grade Swing Trading System

A quantitative trading system that combines Technical Analysis, Fundamental Analysis, and Sentiment Analysis to make swing trading decisions.

## Core Philosophy

1. **Confluence of Factors**: Trades require agreement from Technical, Fundamental, and Sentiment models
2. **Hardware Optimization**: Leverages local Nvidia 5070 Ti for XGBoost training and Ollama LLM inference
3. **Observability**: Chat-queryable via MCP and health reports via Discord

## Technical Stack

- **Language**: Python 3.12+
- **Data Engine**: Polars (Rust-based DataFrame), Pydantic (Validation)
- **ML Engine**: XGBoost (GPU Enabled), Optuna (Tuning)
- **Sentiment**: Ollama (Local Llama-3/Mistral via ollama-python)
- **Dashboard**: Streamlit + Plotly
- **Database**: TimescaleDB (PostgreSQL) via Docker
- **Notifications**: Discord Webhooks
- **IDE Integration**: Model Context Protocol (MCP) for DB access

## Quick Start - Training

**Want to start training immediately?** See the [Training Quick Start Guide](docs/guides/TRAINING_QUICK_START.md) or [How to Train](docs/guides/training/HOW_TO_TRAIN.md) for a step-by-step walkthrough from terminal to training.

### Fastest Way to Start Training

```bash
# Option 1: Use the launcher script (easiest)
./START_TRAINING.sh AAPL 3

# Option 2: Manual commands
cd /home/brennan/nac/nova_aetus
export PATH="$HOME/miniconda3/envs/nova_aetus/bin:$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nova_aetus
python scripts/train_models.py --symbols AAPL --years 3
```

**Full guides:**
- [Training Quick Start Guide](docs/guides/TRAINING_QUICK_START.md) - Complete step-by-step guide
- [How to Train](docs/guides/training/HOW_TO_TRAIN.md) - Quick reference
- [Training Commands](docs/guides/training/TRAINING_COMMANDS.md) - All commands reference

## Project Structure

```
nova_aetus/
├── .cursorrules           # Cursor IDE rules
├── .env                   # API Keys & Webhook URLs (create from .env.example)
├── config.toml            # Strategy Parameters
├── docker-compose.yml     # DB + Observability stack
├── prometheus.yml         # Prometheus configuration
├── requirements.txt       # Python dependencies
├── pytest.ini            # Test configuration
├── START_TRAINING.sh      # Training launcher script (NEW!)
├── launch_dashboard.sh    # Dashboard launcher script
├── run.sh                 # Main system launcher
├── setup_docker.sh        # Docker setup script
├── docs/                  # Documentation (see docs/README.md)
│   ├── guides/            # User guides and manuals
│   ├── architecture/      # System architecture docs
│   └── development/       # Development docs
├── knowledge/            # Research documentation
├── logs/                  # Application logs
├── models/                # Trained model registry
├── mcp_server/            # MCP Integration for Cursor
│   └── db_connector.py
└── src/
    └── nova/
        ├── main.py        # Main trading loop
        ├── core/          # Configuration, logging, notifications
        ├── data/          # Data loading and storage
        ├── features/      # Technical indicators and sentiment
        ├── models/        # XGBoost training and prediction
        ├── strategy/      # Risk management and execution
        ├── dashboard/     # Streamlit dashboard
        └── api/           # Health/metrics API server
```

## Setup

### 1. Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Nvidia GPU with CUDA (for XGBoost GPU acceleration)
- Ollama installed and running (for sentiment analysis)

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Discord webhook URL and other settings
```

### 4. Start TimescaleDB

```bash
docker-compose up -d
```

### 5. Initialize Database Schema

```bash
# Run database migrations (to be implemented)
python -m src.nova.data.storage --init
```

### 6. Configure Ollama

```bash
# Pull required model
ollama pull llama3
# or
ollama pull mistral
```

### 7. Run the System

```bash
# Main trading system
python -m src.nova.main

# Dashboard (separate terminal)
streamlit run src/nova/dashboard/app.py
```

## Configuration

Edit `config.toml` to adjust:
- Technical indicator parameters
- ML model hyperparameters
- Risk management thresholds
- Circuit breaker limits
- Sentiment analysis settings

## Documentation

📚 **All documentation is in the `docs/` directory:**

- **User Guides**: `docs/guides/` - Operations manual, dashboard guides, training guides
- **Architecture**: `docs/architecture/` - System design and schematics
- **Reference**: `docs/reference/` - Technical references (API capabilities, database, secrets)
- **Research**: `docs/research/` - Research notes and knowledge base
- **Development**: `docs/development/` - Development notes and changelog

**Quick Start**: See `docs/guides/OPERATION_MANUAL.md` for complete operations guide.

For detailed documentation structure, see `docs/README.md`.

## MCP Integration

The MCP server allows querying the TimescaleDB database from Cursor chat. Configure in your Cursor/VSCode MCP settings.

## Circuit Breaker

The system includes a circuit breaker that halts trading if more than 5 errors occur within 60 seconds. This prevents cascading failures and protects capital.

## Development

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality. **Note:** Pre-commit requires Git. If the project is not yet a Git repository, initialize it first with `git init`.

Set up hooks before committing:

```bash
# Activate virtual environment
source venv/bin/activate  # or conda activate nova_aetus

# Install pre-commit (if not already installed)
pip install pre-commit

# Initialize git repository (if not already done)
git init

# Install the git hooks
pre-commit install

# Run hooks manually on all files (optional)
pre-commit run --all-files
```

The hooks will automatically:
- Format code with Black
- Lint with Ruff
- Type check with mypy
- Check for security issues with Bandit
- Validate YAML, TOML, and JSON files
- Check for trailing whitespace and other common issues

Hooks run automatically on `git commit`. To skip hooks (not recommended), use `git commit --no-verify`.
