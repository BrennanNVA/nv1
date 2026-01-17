# Portfolio Optimization: Automatic Operation

## ✅ Yes, It's Fully Automatic!

Portfolio optimization is **enabled by default** and runs automatically. No manual configuration needed.

## What Happens Automatically

### 1. Initialization (Automatic)
```python
# In main.py, automatically created:
portfolio_optimizer = PortfolioOptimizer(
    method=OptimizationMethod.MEAN_VARIANCE,
    risk_aversion=1.0,
    max_position_weight=0.10,  # From config.risk.max_position_size_pct
    long_only=True,
)
```

**No action needed** - happens when system starts.

### 2. Daily Signal Collection (Automatic)
- Trading loop checks all symbols
- Collects signals for each symbol
- Stores signal data for portfolio optimization

**No action needed** - happens every iteration (every 5 minutes).

### 3. Portfolio Optimization (Automatic, Once Per Day)
- Checks if it's a new day (first iteration of the day)
- Collects all signals across all symbols
- Runs portfolio optimizer
- Calculates optimal weights

**No action needed** - happens once per day automatically.

### 4. Rebalancing (Automatic)
- Compares current positions to target weights
- Calculates trades needed
- Executes rebalancing trades
- Updates positions in database

**No action needed** - trades execute automatically.

### 5. Holding (Swing Trading)
- Positions held for 2-7 days
- No new trades until next rebalancing
- Stop-losses still active

**No action needed** - system holds positions.

## What You'll See

### In Logs:
```
Portfolio optimization enabled - collecting signals and optimizing portfolio
Optimizing portfolio for 5 symbols...
Portfolio optimization complete: expected_return=0.0012, volatility=0.0150, sharpe=0.0800
Rebalancing AAPL: BUY 10 shares (target: 10, current: 0, weight: 15.00%)
Portfolio rebalanced: 5 positions, Sharpe=0.08
```

### In Dashboard:
- **Portfolio tab**: Shows current portfolio composition
- **Positions tab**: Shows all positions with weights
- **Performance tab**: Portfolio-level metrics

### In Notifications (Discord):
```
Portfolio rebalanced: 5 positions, Sharpe=0.08
```

## Timeline (Automatic)

**Monday 9:30 AM:**
- ✅ System collects all signals
- ✅ Optimizes portfolio
- ✅ Executes rebalancing trades
- ✅ Portfolio set, now HOLDING

**Monday 9:35 AM - Friday:**
- ✅ Monitoring positions
- ✅ No new trades (swing trading)
- ✅ Holding for 2-7 days

**Next Monday:**
- ✅ Repeat: rebalance portfolio
- ✅ Continue holding

## Configuration (Optional)

You can customize optimization method via code (default is MEAN_VARIANCE):

```python
# In main.py, you can change:
portfolio_optimizer = PortfolioOptimizer(
    method=OptimizationMethod.RISK_PARITY,  # or KELLY, MIN_VARIANCE, EQUAL_WEIGHT
    risk_aversion=1.5,  # Higher = more risk-averse
    max_position_weight=0.15,  # 15% max per position
)
```

**But this is optional** - default settings work well for swing trading.

## To Disable (If Needed)

If you want to go back to individual trades (original behavior):

```python
# In main.py, change:
portfolio_optimizer = None  # Disable portfolio optimization
```

But **you don't need to** - portfolio optimization is better for swing trading.

## Summary

✅ **Fully Automatic** - No manual steps needed
✅ **Once Per Day** - Rebalances daily (swing trading)
✅ **PDT-Safe** - Holds positions 2-7 days
✅ **Better Returns** - Diversification + optimal sizing

**Just start the system and it works!** 🚀
