# Nova Aetus Dashboard - Quick Guide

## Starting the Dashboard

### Option 1: Using the Launch Script
```bash
./launch_dashboard.sh
```

### Option 2: Manual Start
```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Install dependencies (if needed)
pip install streamlit plotly

# Start dashboard
streamlit run src/nova/dashboard/app.py
```

### Option 3: With Custom Port
```bash
streamlit run src/nova/dashboard/app.py --server.port 8502
```

## Accessing the Dashboard

Once started, the dashboard will be available at:
- **Local**: http://localhost:8501
- **Network**: http://your-ip:8501 (if server.address is set to 0.0.0.0)

## Dashboard Pages

### 1. Overview Page
**Purpose**: System-wide metrics and recent activity

**What You'll See:**
- **Total Equity**: Current account value from Alpaca
- **Open Positions**: Number of active trades
- **Daily P&L**: Unrealized profit/loss for today
- **Sharpe Ratio**: Risk-adjusted performance metric
- **Equity Curve**: Portfolio value over time (last 7 days)
- **Recent Signals Table**: Latest trading signals with:
  - Symbol
  - Direction (LONG/SHORT/HOLD)
  - Strength (0-1)
  - Confidence (0-1)
  - Confluence Score
  - Timestamp

**Use Cases:**
- Quick health check of the system
- See if signals are being generated
- Monitor account equity changes

### 2. Positions Page
**Purpose**: Detailed view of all open positions

**What You'll See:**
- **Position Table**:
  - Symbol
  - Quantity (shares)
  - Entry Price
  - Current Price
  - Unrealized P&L (dollar amount)
  - P&L Percentage
  - Market Value
- **Position Distribution Chart**: Pie chart showing allocation by market value

**Use Cases:**
- Monitor individual position performance
- Check if positions are profitable
- See position sizing and diversification

### 3. Performance Page
**Purpose**: Historical performance analysis

**What You'll See:**
- **Returns Distribution**: Histogram of trade P&L
- **Risk Metrics**:
  - Max Drawdown (%)
  - Win Rate (%)
  - Profit Factor
  - Average Trade P&L
- **Equity Curve**: Portfolio value from closed trades

**Use Cases:**
- Evaluate strategy performance
- Identify winning/losing patterns
- Assess risk metrics

### 4. Models Page
**Purpose**: Model status and predictions

**What You'll See:**
- **Technical Model Status**:
  - Model file name
  - Last modified timestamp
  - Training metadata (if available)
- **Sentiment Analyzer Status**: Active model name
- **Recent Predictions Table**:
  - Symbol
  - Technical Score
  - Sentiment Score
  - Confluence Score
  - Direction
  - Confidence

**Use Cases:**
- Verify models are loaded
- Check model predictions
- Monitor prediction confidence

### 5. Settings Page
**Purpose**: View current system configuration

**What You'll See:**
- Technical indicator parameters (RSI period, MACD settings, etc.)
- Risk management settings (risk per trade, max drawdown, etc.)
- Circuit breaker thresholds

**Use Cases:**
- Verify configuration
- Check parameter values
- Understand system limits

## Dashboard Features

### Real-Time Updates
- Dashboard refreshes every 60 seconds (cached data)
- Click "Rerun" button or press `R` to manually refresh
- Data is fetched from TimescaleDB

### Interactive Charts
- **Plotly Integration**: All charts are interactive
- Hover over data points for details
- Zoom and pan on charts
- Download charts as images

### Data Sources
- **Positions**: From `portfolio_positions` table in TimescaleDB
- **Signals**: From `trading_signals` table
- **Performance**: From closed positions
- **Account Info**: From Alpaca API (via ExecutionEngine)

## Troubleshooting Dashboard

### Issue: "No module named 'streamlit'"
**Solution:**
```bash
pip install streamlit plotly
```

### Issue: "Could not connect to database"
**Solution:**
1. Check if TimescaleDB is running: `docker-compose ps`
2. Verify database credentials in `.env`
3. Check if database schema is initialized

### Issue: "No data available"
**Solution:**
1. Ensure trading system has been running
2. Check if positions/signals exist in database
3. Verify date ranges in queries

### Issue: Dashboard shows placeholder data
**Solution:**
- This means the system hasn't generated any real data yet
- Start the trading system: `python -m nova.main`
- Wait for signals/positions to be generated

### Issue: Charts not displaying
**Solution:**
1. Check browser console for errors
2. Verify plotly is installed: `pip install plotly`
3. Try clearing browser cache

## Tips for Best Experience

1. **Keep Trading System Running**: Dashboard reads from database, so the trading system should be running to populate data

2. **Monitor in Real-Time**: Open dashboard in a separate window and refresh periodically

3. **Use Multiple Tabs**: Open different pages in separate tabs for comprehensive monitoring

4. **Check Logs**: If dashboard shows errors, check `logs/nova_aetus.log` for details

5. **Grafana Integration**: For advanced monitoring, use Grafana (http://localhost:3000) which has more detailed metrics

## Keyboard Shortcuts

- `R`: Rerun/refresh dashboard
- `C`: Clear cache
- `?`: Show keyboard shortcuts
- `Ctrl+C`: Stop dashboard server

## Next Steps

After viewing the dashboard:
1. **Start Trading**: Run `python -m nova.main` to begin signal generation
2. **Train Models**: Use training pipeline to create models
3. **Monitor Performance**: Check dashboard regularly for system health
4. **Adjust Configuration**: Modify `config.toml` based on performance

---

**Note**: The dashboard requires the trading system to be running (or have run previously) to show real data. If you're seeing placeholder data, start the trading system first.
