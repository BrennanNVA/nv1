# Nova Aetus Trading System - Operation Manual

## Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Dashboard Usage](#dashboard-usage)
6. [Model Training](#model-training)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## System Overview

### What is Nova Aetus?

**Nova Aetus** is an automated trading system that analyzes stocks and executes trades based on multiple factors agreeing. Think of it as a sophisticated stock-picking robot that:

1. **Analyzes stocks** using 88+ technical indicators, AI sentiment analysis, and fundamental metrics
2. **Generates signals** when multiple factors agree (technical + sentiment + fundamentals)
3. **Manages risk** automatically using mathematical formulas and safety limits
4. **Executes trades** via Alpaca API when signals pass all checks
5. **Tracks performance** and adjusts based on what works

### Why Use This System?

**The Problem:** Manually analyzing hundreds of stocks, combining multiple signals, managing risk, and executing trades is:
- Time-consuming (hours per day)
- Emotionally difficult (fear/greed interfere)
- Mathematically complex (Kelly Criterion, risk calculations)
- Error-prone (miss signals, miscalculate positions)

**The Solution:** Nova Aetus automates all of this:
- **24/7 Monitoring**: Never misses opportunities
- **Emotion-Free**: Follows rules, not feelings
- **Mathematically Optimal**: Uses proven formulas (Kelly Criterion, etc.)
- **Consistent**: Same process every time

### How It Works (High Level)

**Step 1: Data Collection**
- **What:** Fetches stock prices, volume, news, social media
- **Why:** Need data to analyze
- **How:** Alpaca API for prices, Yahoo for fundamentals, Ollama for sentiment

**Step 2: Feature Engineering**
- **What:** Calculates 88+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Why:** Raw prices aren't enough - need derived metrics to spot patterns
- **How:** Polars (fast Rust-based DataFrame library) processes data

**Step 3: Machine Learning Prediction**
- **What:** AI model predicts if stock will go up or down
- **Why:** ML finds patterns humans miss
- **How:** XGBoost (gradient boosting) trained on historical data

**Step 4: Sentiment Analysis**
- **What:** Analyzes news and social media for bullish/bearish sentiment
- **Why:** Market moves on news and emotions, not just charts
- **How:** Local LLM (Ollama) processes text and scores sentiment

**Step 5: Signal Confluence**
- **What:** Combines technical + sentiment + fundamental signals
- **Why:** Multiple signals agreeing = higher confidence trade
- **How:** Regime-aware weighting (adjusts based on market conditions)

**Step 6: Risk Management**
- **What:** Calculates position size, sets stop-loss, checks limits
- **Why:** Protects capital - one bad trade shouldn't wipe you out
- **How:** Kelly Criterion (optimal sizing) + circuit breakers (safety stops)

**Step 7: Order Execution**
- **What:** Places buy/sell orders via Alpaca API
- **Why:** Actually makes the trades
- **How:** Alpaca Trading API integration

**Step 8: Monitoring**
- **What:** Tracks positions, P&L, performance metrics
- **Why:** Need to know if strategy is working
- **How:** Dashboard, Prometheus metrics, Discord alerts

### Key Components Explained

- **Technical Analysis**: 88+ indicators with fractional differencing
  - **What:** Mathematical formulas that analyze price patterns (RSI, MACD, etc.)
  - **Why:** Charts show patterns that predict future moves
  - **Example:** RSI > 70 = overbought (might fall), RSI < 30 = oversold (might rise)

- **Sentiment Analysis**: Local LLM (Ollama) for news/social media
  - **What:** AI reads news headlines and social posts, determines if bullish/bearish
  - **Why:** News drives prices - "Apple beats earnings" = stock goes up
  - **Example:** "Tesla announces breakthrough" → Bullish sentiment → Higher score

- **Fundamental Analysis**: Value and quality scoring
  - **What:** Analyzes company financials (P/E ratio, ROE, debt levels)
  - **Why:** Good companies at good prices tend to outperform
  - **Example:** Low P/E + High ROE = Good value stock

- **Machine Learning**: GPU-accelerated XGBoost with NPMM labeling
  - **What:** AI model trained on historical data to predict price movements
  - **Why:** Finds complex patterns humans can't see
  - **Example:** Model sees pattern: "When RSI < 30 AND volume spikes AND sentiment turns positive → Price rises 5% in 3 days"

- **Risk Management**: Kelly Criterion, circuit breakers, drawdown control
  - **What:** Mathematical formulas to size positions and limit losses
  - **Why:** Prevents one bad trade from destroying your account
  - **Example:** Kelly says "bet 2% of capital" → System bets exactly 2%, not more/less

- **Signal Confluence**: Regime-aware multi-signal combination
  - **What:** Combines all signals into one final decision
  - **Why:** Multiple signals agreeing = higher probability trade
  - **Example:** Technical says BUY + Sentiment says BULLISH + Fundamentals say VALUE → Strong BUY signal

### Architecture Components

```
┌─────────────────┐
│  Data Pipeline  │ → TimescaleDB (Hypertables, Continuous Aggregates)
└─────────────────┘
         ↓
┌─────────────────┐
│ Feature Engine  │ → 88+ Technical Indicators + Fractional Diff
└─────────────────┘
         ↓
┌─────────────────┐
│  ML Pipeline    │ → XGBoost (GPU) + NPMM Labeling
└─────────────────┘
         ↓
┌─────────────────┐
│ Confluence Layer│ → Regime-Aware Signal Combination
└─────────────────┘
         ↓
┌─────────────────┐
│ Risk Manager    │ → Kelly Criterion + Circuit Breakers
└─────────────────┘
         ↓
┌─────────────────┐
│ Execution Engine│ → Alpaca API Order Execution
└─────────────────┘
```

---

## Prerequisites & Setup

### Required Software

1. **Python 3.12+**
   - **What:** Programming language the system is written in
   - **Why:** Python has excellent libraries for data science and trading
   - **Check:** `python3 --version` should show 3.12 or higher

2. **PostgreSQL 14+** (for TimescaleDB)
   - **What:** Database that stores all market data, signals, positions
   - **Why:** Need persistent storage - can't lose trading history
   - **Note:** TimescaleDB is PostgreSQL with time-series extensions (faster queries)

3. **Docker & Docker Compose**
   - **What:** Containerization tool - runs database in isolated environment
   - **Why:** Easy setup, consistent environment, no manual PostgreSQL install needed
   - **Check:** `docker --version` and `docker-compose --version`

4. **NVIDIA GPU** (optional, for GPU-accelerated XGBoost)
   - **What:** Graphics card for faster ML model training
   - **Why:** Training models on GPU is 10-100x faster than CPU
   - **Note:** System works without GPU, just slower training

5. **Ollama** (for local LLM sentiment analysis)
   - **What:** Tool to run AI language models locally
   - **Why:** Analyzes news/social media sentiment without sending data to external APIs
   - **Note:** Optional - system can run without sentiment analysis

### Installation Steps

1. **Clone and navigate to project:**
   ```bash
   cd /home/brennan/nac/nova_aetus
   ```
   **What:** Go to project directory
   **Why:** Need to be in right location for commands

2. **Create virtual environment:**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```
   **What:** Creates isolated Python environment
   **Why:** Keeps project dependencies separate from system Python (prevents conflicts)
   **Note:** You'll see `(venv)` in terminal prompt when active

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   **What:** Installs all Python packages needed (Polars, XGBoost, Alpaca API, etc.)
   **Why:** System needs these libraries to function
   **Takes:** 2-5 minutes depending on internet speed

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and webhook URLs
   ```
   **What:** Creates configuration file with API keys
   **Why:** System needs credentials to access trading APIs and send notifications
   **Required:** Alpaca API keys (get from https://alpaca.markets), Discord webhook URL

5. **Start TimescaleDB and observability stack:**
   ```bash
   docker-compose up -d
   ```
   **What:** Starts database and monitoring tools in Docker containers
   **Why:** Database stores all data, Prometheus/Grafana monitor system health
   **Verify:** `docker-compose ps` should show all services as "Up"

6. **Initialize database schema:**
   ```bash
   python -m nova.main --init-db
   ```
   **What:** Creates tables, indexes, and database structure
   **Why:** Database needs structure before it can store data
   **Note:** Only need to do this once (first time setup)

### Environment Variables (.env)

```bash
# Alpaca API (Required for trading)
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://api.alpaca.markets

# Discord Notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Database (if not using docker-compose defaults)
TIMESCALEDB_HOST=localhost
TIMESCALEDB_PORT=5432
TIMESCALEDB_DB=nova_aetus
TIMESCALEDB_USER=postgres
TIMESCALEDB_PASSWORD=your_password

# Ollama (for sentiment analysis)
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Configuration

### config.toml Structure

The system uses `config.toml` for all configuration. Key sections:

```toml
[data]
symbols = ["AAPL", "MSFT", "GOOGL"]  # Trading universe
default_timeframe = "1Day"
lookback_periods = 252  # Days of historical data

[technical]
rsi_period = 14
macd_fast = 12
macd_slow = 26
# ... more indicator settings

[ml]
n_estimators = 100
max_depth = 6
learning_rate = 0.1
use_gpu = true  # Enable GPU acceleration

[risk]
risk_per_trade_pct = 0.02  # 2% risk per trade
max_drawdown_pct = 0.15  # 15% max drawdown
max_position_size_pct = 0.10  # 10% max position size

[circuit_breaker]
max_errors_per_minute = 5
error_window_seconds = 60
```

### Key Configuration Files

- **`config.toml`**: Main system configuration
- **`.env`**: API keys and secrets (not in git)
- **`prometheus.yml`**: Prometheus metrics scraping config
- **`docker-compose.yml`**: Database and observability services

---

## Running the System

### What Happens When You Run the System?

**The Trading Loop:**
1. Every 5 minutes, system checks all configured symbols
2. Fetches latest market data (prices, volume)
3. Calculates technical indicators (RSI, MACD, etc.)
4. Gets sentiment scores (if news/social data available)
5. Fetches fundamental data (P/E, ROE, etc.)
6. Generates confluence signal (combines all factors)
7. Evaluates trade with risk manager (position size, stop-loss)
8. Executes approved trades via Alpaca API
9. Stores signals and updates positions in database
10. Sends notifications to Discord (if configured)

**The Dashboard:**
- Shows real-time positions, P&L, signals
- Displays performance metrics and charts
- Updates automatically as system runs

### 1. Start Infrastructure

```bash
# Start TimescaleDB, Prometheus, Grafana
docker-compose up -d

# Verify services are running
docker-compose ps
```

**What:** Starts database and monitoring tools
**Why:** System needs database to store data, monitoring to track health
**Expected:** All services show status "Up"

### 2. Initialize Database (First Time Only)

```bash
python -m nova.main --init-db
```

**What:** Creates database structure
**Why:** Database needs tables before it can store data

This creates:
- **Hypertables** for market data (optimized for time-series)
- **Continuous aggregates** (1min, 1hour, daily views - pre-calculated for speed)
- **Feature store** (stores ML features for training)
- **Signal tables** (stores all generated trading signals)
- **Portfolio positions** (tracks open/closed trades)

### 3. Train Models (Before Trading)

**What:** Trains AI models on historical data
**Why:** Models need to learn patterns from past data to predict future moves
**When:** Before first trade, or periodically (monthly/quarterly) to adapt to market changes
**Takes:** 10-60 minutes depending on data amount and symbols

```bash
# Train single symbol
python -c "
from nova.core.config import load_config
from nova.data.loader import DataLoader
from nova.models.training_pipeline import TrainingPipeline
import asyncio

async def train():
    config = load_config()
    data_loader = DataLoader(config.data)
    pipeline = TrainingPipeline(config, data_loader)

    result = await pipeline.train_universe(
        symbols=['AAPL'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        use_walk_forward=True
    )
    print(f'Training complete: {result}')

asyncio.run(train())
"
```

**What this does:**
1. Fetches 4 years of historical data for AAPL
2. Calculates all technical indicators
3. Generates labels (when to buy/sell) using NPMM method
4. Trains XGBoost model to predict labels
5. Validates model with walk-forward optimization
6. Saves model to `models/` directory

**Note:** System can run without trained models, but predictions will be limited.

### 4. Start Trading System

```bash
# Run main trading loop
python -m nova.main

# Or with specific config
python -m nova.main --config /path/to/config.toml
```

**What:** Starts the automated trading system
**Why:** This is the main process that analyzes stocks and executes trades
**Expected Output:**
- "Database connected"
- "Components initialized"
- "Trading loop started"
- Then continuous log messages as it processes symbols

The system will:
1. **Connect to TimescaleDB** - Verifies database is accessible
2. **Initialize components** - Loads data loader, feature calculator, models, risk manager
3. **Start health server** - Exposes `/health` and `/metrics` endpoints on port 8000
4. **Begin trading loop** - Checks all symbols every 5 minutes, generates signals, executes trades

**Note:** System runs continuously until you stop it (Ctrl+C). Keep terminal open.

### 5. Start Dashboard

```bash
# In a separate terminal
streamlit run src/nova/dashboard/app.py

# Or use the launcher script
./launch_dashboard.sh

# Dashboard will open at http://localhost:8501
```

**What:** Starts web-based dashboard for monitoring
**Why:** Visual interface to see positions, signals, performance - easier than reading logs
**What You'll See:**
- Overview page: Equity, positions count, daily P&L, recent signals
- Positions page: Open trades with entry/current prices, unrealized P&L
- Performance page: Historical metrics, win rate, drawdown charts
- Models page: Model status and recent predictions
- Settings page: Current configuration values

**Note:** Dashboard reads from database, so it shows data even if trading system is stopped (historical data).

---

## Dashboard Usage

### Overview Page

**Key Metrics:**
- **Total Equity**: Current account equity from Alpaca
- **Open Positions**: Number of active positions
- **Daily P&L**: Unrealized profit/loss for the day
- **Sharpe Ratio**: Risk-adjusted returns

**Charts:**
- **Equity Curve**: Portfolio value over time
- **Recent Signals Table**: Latest generated trading signals with confluence scores

### Positions Page

**Position Table:**
- Symbol, quantity, entry price, current price
- Unrealized P&L (dollar and percentage)
- Market value

**Visualizations:**
- Position distribution pie chart (by market value)

### Performance Page

**Metrics:**
- Max Drawdown
- Win Rate
- Profit Factor
- Average Trade P&L

**Charts:**
- Returns distribution histogram
- Equity curve from closed trades

### Models Page

**Model Status:**
- Technical model: Shows latest trained model and metadata
- Sentiment analyzer: Status and model name
- Recent predictions: Latest model predictions with scores

### Settings Page

Displays current system configuration:
- Technical indicator parameters
- Risk management settings
- Circuit breaker thresholds

---

## Model Training

### Training Workflow

1. **Data Collection**: Fetches historical bars for training universe
2. **Feature Engineering**: Calculates 88+ technical indicators
3. **NPMM Labeling**: Generates swing trading labels at local extrema
4. **Walk-Forward Validation**: Trains on IS, validates on OOS
5. **Statistical Validation**: Calculates DSR, PBO, Sharpe ratios
6. **Model Registry**: Saves models with versioning and metadata

### Training Commands

**Single Symbol:**
```python
from nova.models.training_pipeline import TrainingPipeline
# ... setup config and data_loader ...
result = await pipeline.train_single_symbol(
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2024-12-31'
)
```

**Universe Training:**
```python
result = await pipeline.train_universe(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    use_walk_forward=True
)
```

### Model Validation Criteria

Before deploying models, ensure:
- **DSR > 0.95**: Deflated Sharpe Ratio indicates robustness
- **PBO < 0.20**: Low probability of backtest overfitting
- **OOS Sharpe > 1.0**: Out-of-sample performance acceptable
- **Win Rate > 45%**: Reasonable win rate for swing trading

---

## Monitoring & Maintenance

### System Health

**Health Endpoint:**
```bash
curl http://localhost:8000/health
```

Returns:
- Database connectivity
- Component status
- Circuit breaker state

**Metrics Endpoint:**
```bash
curl http://localhost:8000/metrics
```

Prometheus-formatted metrics:
- `signals_generated_total`
- `orders_executed_total`
- `errors_total`
- `latency_seconds`

### Prometheus & Grafana

**Access Grafana:**
- URL: http://localhost:3000
- Default credentials: `admin` / `admin`

**Key Dashboards:**
1. **Trading Metrics**: Signals, orders, P&L
2. **System Health**: Errors, latency, circuit breakers
3. **Model Performance**: Prediction accuracy, confidence

### Discord Alerts

The system sends alerts to Discord for:
- Trade signals generated
- Orders executed
- Circuit breaker triggers
- System errors

### Database Maintenance

**Continuous Aggregates:**
- Auto-refresh policies update 1min/1hour/daily views
- No manual intervention needed

**Data Retention:**
```sql
-- Set retention policy (e.g., keep 2 years of 1min data)
SELECT add_retention_policy('ohlcv_1min', INTERVAL '2 years');
```

**Vacuum & Analyze:**
```sql
-- Run periodically for performance
VACUUM ANALYZE market_bars;
```

---

## Troubleshooting

### Common Issues

**1. Database Connection Failed**
```
Error: Could not connect to TimescaleDB
```
**Solution:**
- Check `docker-compose ps` - is TimescaleDB running?
- Verify `.env` has correct database credentials
- Check firewall/network settings

**2. Model Not Found**
```
Warning: No trained model found
```
**Solution:**
- Train models first: `python scripts/train_models.py`
- Check `models/` directory for `.json` files
- Verify model path in logs

**3. Alpaca API Errors**
```
Error: Alpaca API error executing order
```
**Solution:**
- Verify API keys in `.env`
- Check account status (not blocked, has buying power)
- Ensure market is open (for market orders)
- Check rate limits (Alpaca has request limits)

**4. GPU Not Available**
```
Warning: GPU not available, using CPU
```
**Solution:**
- Install CUDA toolkit and cuDNN
- Verify GPU: `nvidia-smi`
- Set `use_gpu = false` in config if GPU unavailable

**5. Ollama Connection Failed**
```
Error: Could not connect to Ollama
```
**Solution:**
- Start Ollama: `ollama serve`
- Verify `OLLAMA_BASE_URL` in config
- Pull required model: `ollama pull llama3`

**6. Circuit Breaker Triggered**
```
Trading halted: Too many errors
```
**Solution:**
- Check error logs: `tail -f logs/nova_aetus.log`
- Fix underlying issue (API, database, etc.)
- Reset circuit breaker: `python -c "from nova.strategy.risk import RiskManager; rm = RiskManager(...); rm.reset_circuit_breaker()"`

### Log Files

**Location:** `logs/nova_aetus.log`

**Log Levels:**
- `DEBUG`: Detailed execution flow
- `INFO`: Normal operations
- `WARNING`: Non-critical issues
- `ERROR`: Errors requiring attention

**View Logs:**
```bash
# Follow logs in real-time
tail -f logs/nova_aetus.log

# Search for errors
grep ERROR logs/nova_aetus.log

# Last 100 lines
tail -n 100 logs/nova_aetus.log
```

---

## API Reference

### MCP Server (Cursor IDE Integration)

The MCP server exposes database queries via Cursor IDE:

**Available Tools:**
- `get_portfolio_summary`: Get open positions and P&L
- `get_recent_signals`: Get latest trading signals
- `get_system_health`: Get system metrics and health

**Usage in Cursor:**
Ask Cursor: "What are my open positions?" or "Show me recent signals"

### Health Server Endpoints

**GET /health**
```json
{
  "status": "healthy",
  "database": "connected",
  "components": {
    "data_loader": "ready",
    "risk_manager": "ready"
  }
}
```

**GET /metrics**
Prometheus-formatted metrics:
```
# HELP signals_generated_total Total signals generated
# TYPE signals_generated_total counter
signals_generated_total{symbol="AAPL"} 15.0
```

---

## Best Practices

### Risk Management

1. **Start Small**: Begin with paper trading or small position sizes
2. **Monitor Drawdown**: Set `max_drawdown_pct` conservatively (10-15%)
3. **Diversify**: Trade multiple symbols, don't concentrate risk
4. **Review Regularly**: Check performance weekly, adjust as needed

### Model Training

1. **Use Walk-Forward**: Always validate with walk-forward optimization
2. **Check DSR/PBO**: Ensure models pass statistical validation
3. **Retrain Periodically**: Retrain models quarterly or when performance degrades
4. **Version Control**: Keep model versions and metadata for rollback

### System Operations

1. **Monitor Health**: Check `/health` endpoint regularly
2. **Review Logs**: Monitor logs for warnings/errors
3. **Backup Database**: Regular backups of TimescaleDB
4. **Test Changes**: Test configuration changes in paper trading first

### Performance Optimization

1. **GPU Acceleration**: Use GPU for XGBoost training (10-100x faster)
2. **Continuous Aggregates**: Leverage TimescaleDB aggregates for fast queries
3. **Batch Operations**: Fetch fundamentals in batches
4. **Connection Pooling**: Reuse database connections

---

## Support & Resources

### Documentation
- `docs/architecture/SYSTEM_ARCHITECTURE.md`: Complete system schematic
- `versionlog.md`: Version history and changes
- `nova_aetus_phase_3_operationalization_20bc1fd5.plan.md`: Phase 3 implementation plan

### Key Files
- `src/nova/main.py`: Main trading loop
- `src/nova/dashboard/app.py`: Streamlit dashboard
- `config.toml`: System configuration
- `.env`: API keys and secrets

### External Resources
- [TimescaleDB Docs](https://docs.timescale.com/)
- [Alpaca API Docs](https://alpaca.markets/docs/)
- [XGBoost GPU Guide](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [Ollama Docs](https://ollama.ai/docs)

---

## Quick Start Checklist

- [ ] Install Python 3.12+, Docker, PostgreSQL
- [ ] Clone repository and install dependencies
- [ ] Configure `.env` with API keys
- [ ] Start `docker-compose up -d`
- [ ] Initialize database: `python -m nova.main --init-db`
- [ ] Train models: `python scripts/train_models.py`
- [ ] Start trading: `python -m nova.main`
- [ ] Open dashboard: `streamlit run src/nova/dashboard/app.py`
- [ ] Monitor health: `curl http://localhost:8000/health`
- [ ] Check Grafana: http://localhost:3000

---

**Last Updated:** January 2026
**Version:** Nova Aetus Phase 3 (Operational)
