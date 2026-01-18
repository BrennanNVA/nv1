# Quick Start Guide - Running Nova Aetus

## 🚀 Quick Summary

**What is Nova Aetus?** An automated trading system that uses AI/ML to analyze stocks and execute trades.

**What does it do?**
1. Analyzes stocks using technical indicators, sentiment, and fundamentals
2. Generates trading signals when multiple factors agree
3. Executes trades via Alpaca API
4. Tracks performance and manages risk automatically

**Why use it?** Automates quantitative trading strategies that would be difficult to execute manually.

---

## 📋 Step-by-Step: First Time Setup

### Step 1: Install Prerequisites

**What:** Install required software
**Why:** System needs these to run

```bash
# Check Python version (need 3.12+)
python3 --version

# Install Docker (if not installed)
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

### Step 2: Clone and Navigate to Project

**What:** Go to project directory
**Why:** Need to be in the right location

```bash
cd /home/brennan/nac/nova_aetus
```

### Step 3: Create Virtual Environment

**What:** Isolated Python environment
**Why:** Keeps dependencies separate from system Python

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies

**What:** Install Python packages
**Why:** System needs these libraries to function

```bash
pip install -r requirements.txt
```

This installs: Polars, XGBoost, Alpaca API, Streamlit, etc.

### Step 5: Configure Environment Variables

**What:** Set up API keys and secrets
**Why:** System needs credentials to access trading APIs and send notifications

```bash
# Copy example file
cp .env.example .env

# Edit with your credentials
nano .env  # or use your preferred editor
```

**Required in `.env`:**
```bash
# Alpaca API (get from https://alpaca.markets)
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here

# Discord Webhook (for notifications)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### Step 6: Start Database

**What:** Start TimescaleDB (PostgreSQL with time-series extensions)
**Why:** System stores all market data, signals, and positions here

```bash
docker-compose up -d
```

**Verify it's running:**
```bash
docker-compose ps
```

You should see `timescaledb` status as "Up".

### Step 7: Initialize Database Schema

**What:** Create tables and indexes
**Why:** Database needs structure to store data

```bash
python -m nova.main --init-db
```

Or if that doesn't work:
```python
python -c "
from nova.core.config import load_config
from nova.data.storage import StorageService
import asyncio

async def init():
    config = load_config()
    storage = StorageService(config.data)
    await storage.init_schema()
    print('Database initialized!')

asyncio.run(init())
"
```

### Step 8: Configure Trading Symbols

**What:** Set which stocks to trade
**Why:** System needs to know what to analyze

Edit `config.toml`:
```toml
[data]
symbols = ["AAPL", "MSFT", "GOOGL"]  # Add your symbols here
```

---

## 🎯 Step-by-Step: Running the System

### Option A: Run Everything (Recommended)

**Terminal 1 - Trading System:**
```bash
cd /home/brennan/nac/nova_aetus
source venv/bin/activate
python -m nova.main
```

**What happens:** System starts, connects to database, begins analyzing stocks every 5 minutes, generates signals, executes trades.

**Terminal 2 - Dashboard:**
```bash
cd /home/brennan/nac/nova_aetus
source venv/bin/activate
./launch_dashboard.sh
```

**What happens:** Dashboard opens at http://localhost:8501 showing positions, signals, performance.

### Option B: Just View Dashboard (No Trading)

**What:** See dashboard without running trading system
**Why:** Useful for viewing existing data or testing

```bash
cd /home/brennan/nac/nova_aetus
source venv/bin/activate
./launch_dashboard.sh
```

Dashboard will show empty/placeholder data if trading system hasn't run.

---

## 🔍 What to Expect

### When Trading System Starts:

1. **Initialization** (10-30 seconds):
   - Connects to database ✅
   - Loads configuration ✅
   - Initializes components ✅
   - Checks for trained models ⚠️ (may show warning if none)

2. **Trading Loop** (runs continuously):
   - Every 5 minutes: Fetches data → Calculates features → Generates signals → Evaluates risk → Executes trades
   - Logs activity to console and `logs/nova_aetus.log`

3. **Signals Generated**:
   - When technical + sentiment + fundamental factors agree
   - Stored in database
   - Sent to Discord (if configured)

### When Dashboard Opens:

1. **Overview Page**: Shows equity, positions, recent signals
2. **Positions Page**: Lists open trades with P&L
3. **Performance Page**: Historical metrics and charts
4. **Models Page**: Model status and predictions
5. **Settings Page**: Current configuration

---

## ⚠️ Common First-Time Issues

### "Module not found" errors
**Why:** Dependencies not installed
**Fix:** `pip install -r requirements.txt`

### "Cannot connect to database"
**Why:** TimescaleDB not running
**Fix:** `docker-compose up -d` then check `docker-compose ps`

### "No trained model found"
**Why:** Models need to be trained first
**Fix:** See "Training Models" section below (optional for testing)

### "Alpaca API error"
**Why:** Invalid API keys or account issues
**Fix:** Verify keys in `.env`, check Alpaca account status

---

## 🎓 Training Models (Optional)

**What:** Train ML models on historical data
**Why:** Better predictions = better trading signals

```python
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
        symbols=['AAPL', 'MSFT'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        use_walk_forward=True
    )
    print(f'Training complete: {result}')

asyncio.run(train())
"
```

**Takes:** 10-60 minutes depending on data and symbols

---

## 📊 Monitoring

### Check System Health:
```bash
curl http://localhost:8000/health
```

### View Logs:
```bash
tail -f logs/nova_aetus.log
```

### View Metrics:
```bash
curl http://localhost:8000/metrics
```

### Grafana Dashboard:
Open http://localhost:3000 (default: admin/admin)

---

## 🛑 Stopping the System

**Trading System:** Press `Ctrl+C` in terminal

**Dashboard:** Press `Ctrl+C` in terminal

**Database:** `docker-compose down` (stops but keeps data)

**Full Shutdown:** `docker-compose down -v` (removes data - be careful!)

---

## ✅ Success Checklist

- [ ] Prerequisites installed (Python 3.12+, Docker)
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` configured with API keys
- [ ] Database running (`docker-compose up -d`)
- [ ] Database schema initialized
- [ ] `config.toml` has trading symbols
- [ ] Trading system starts without errors
- [ ] Dashboard opens at http://localhost:8501
- [ ] Can see system status in dashboard

---

## 🆘 Need Help?

- **Detailed Operations**: See `docs/guides/OPERATION_MANUAL.md`
- **Dashboard Guide**: See `docs/guides/dashboard/DASHBOARD_GUIDE.md`
- **Troubleshooting**: See `docs/guides/OPERATION_MANUAL.md` - Troubleshooting section

---

**Ready to start?** Begin with Step 1 above! 🚀
