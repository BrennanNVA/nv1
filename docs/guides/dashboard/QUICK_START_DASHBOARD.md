# Quick Start: Viewing the Dashboard

## ⚡ Quick Summary - What to Do

**3 Simple Steps:**
1. **Open Terminal** → Navigate to project directory
2. **Enter Command** → `./launch_dashboard.sh` (or `streamlit run src/nova/dashboard/app.py`)
3. **Click Link** → Open http://localhost:8501 in your browser

**That's it!** Dashboard will show positions, signals, and performance.

---

## Prerequisites Check

Before starting, ensure you have:
- [ ] Python 3.12+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] TimescaleDB running (`docker-compose up -d`)
- [ ] Database initialized (optional, but recommended)

## Step 1: Install Streamlit (if not already installed)

```bash
cd /home/brennan/nac/nova_aetus
pip install streamlit plotly
```

## Step 2: Start the Dashboard

### Easy Way (using launch script):
```bash
./launch_dashboard.sh
```

### Manual Way:
```bash
streamlit run src/nova/dashboard/app.py
```

## Step 3: Open in Browser

The dashboard will automatically open at:
**http://localhost:8501**

If it doesn't open automatically, copy the URL from the terminal output.

## What You'll See

### If System Has Been Running:
- Real positions, signals, and performance data
- Live equity curve and P&L
- Model predictions and status

### If System Hasn't Run Yet:
- Placeholder data showing the dashboard structure
- Configuration settings
- Empty tables (waiting for data)

## To Generate Real Data

1. **Start the trading system** (in a separate terminal):
   ```bash
   python -m nova.main
   ```

2. **Wait for signals** (system checks every 5 minutes)

3. **Refresh the dashboard** (press `R` or click "Rerun")

## Dashboard Pages Overview

1. **Overview**: System health, equity, recent signals
2. **Positions**: Open positions with P&L
3. **Performance**: Historical metrics and charts
4. **Models**: Model status and predictions
5. **Settings**: Current configuration

## Troubleshooting

**"Module not found" errors:**
```bash
pip install streamlit plotly polars
```

**"Cannot connect to database":**
```bash
# Check if TimescaleDB is running
docker-compose ps

# Start if not running
docker-compose up -d
```

**Dashboard shows "No data":**
- This is normal if the trading system hasn't run yet
- Start the trading system to generate data
- Or check if you have historical data in the database

## Next Steps

1. ✅ View the dashboard structure
2. Start the trading system to generate real data
3. Monitor positions and performance
4. Adjust configuration as needed

For detailed dashboard usage, see `docs/guides/dashboard/DASHBOARD_GUIDE.md`
For complete operations manual, see `docs/guides/OPERATION_MANUAL.md`
