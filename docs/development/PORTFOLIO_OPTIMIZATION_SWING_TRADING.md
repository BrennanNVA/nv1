# Portfolio Optimization for Swing Trading (PDT-Safe)

## Why Portfolio Optimization is Perfect for Swing Trading

### Your Current Setup
- **Timeframe**: Daily bars (`1Day`)
- **Holding Period**: 2-7 days (swing trading, NOT day trading)
- **Target Frequency**: 10-50 trades per year
- **Trading Loop**: Checks every 5 minutes but **holds positions for days**

### Portfolio Optimization ≠ Day Trading

**Portfolio optimization is PERFECT for swing trading because:**

1. **No Day Trading Required**
   - Portfolio optimization happens **once per day** (or even weekly)
   - You're holding positions for **multiple days** (swing trading)
   - **PDT rules don't apply** - you're not opening/closing same day

2. **Swing Trading = Multi-Day Holds**
   - Buy on Day 1, hold for 2-7 days, sell on Day 3-8
   - Portfolio optimization determines **what to hold**, not how often to trade
   - Rebalancing happens weekly/monthly, not intraday

3. **Better Than Individual Trades**
   - Current: One signal → one trade → hold individually
   - Portfolio: Multiple signals → optimize positions → hold portfolio
   - Same holding period, better diversification

---

## How Portfolio Optimization Works in Swing Trading

### Example: Weekly Portfolio Rebalancing

```python
# Monday: Check all signals, optimize portfolio
signals = {
    "AAPL": 0.8,   # Strong buy signal
    "MSFT": 0.6,   # Moderate buy signal
    "GOOGL": -0.3, # Weak sell signal
    "NVDA": 0.9,   # Very strong buy signal
}

# Optimize portfolio (takes 5 minutes)
portfolio_weights = portfolio_optimizer.optimize(
    signals=signals,
    current_positions=current_positions,
    target_holdings=[2, 5, 7]  # days
)

# Execute trades to reach target portfolio (all same day = OK)
# Then HOLD for 2-7 days (swing trading)
# No day trading = No PDT restrictions
```

**Timeline:**
- **Monday 9:30 AM**: Check signals, optimize portfolio, execute trades
- **Monday 9:35 AM**: Portfolio positions set, HOLD
- **Wednesday-Friday**: Still holding (swing trading)
- **Next Monday**: Rebalance if needed

**Result:**
- ✅ No day trading (holding > 1 day)
- ✅ No PDT restrictions
- ✅ Better diversification
- ✅ Optimal position sizing

---

## PDT Rule Explanation

### Pattern Day Trader (PDT) Rule
- **Applies if**: Opening and closing position **same day** (4+ times in 5 business days)
- **Requires**: $25k minimum account balance

### Swing Trading = NOT Day Trading
- **Opening**: Monday
- **Closing**: Wednesday-Friday (2-7 days later)
- **Result**: ✅ No day trading, no PDT restrictions

### Portfolio Optimization = Still Swing Trading
- **Portfolio check**: Once per day (or weekly)
- **Execute trades**: Same day as check (OK - setting up portfolio)
- **Hold positions**: 2-7 days (swing trading)
- **Result**: ✅ No day trading, no PDT restrictions

---

## Portfolio Optimization Schedule (Swing Trading)

### Option 1: Daily Check (Recommended)
```
Monday 9:30 AM:
  1. Check all signals
  2. Optimize portfolio weights
  3. Execute rebalancing trades
  4. HOLD portfolio for 2-7 days

Tuesday-Sunday:
  - Monitor positions
  - No new trades (unless stop-loss hit)
```

### Option 2: Weekly Rebalancing (More Conservative)
```
Monday 9:30 AM:
  1. Check all signals
  2. Optimize portfolio
  3. Rebalance if needed (>5% drift)
  4. HOLD for the week

Next Monday:
  - Repeat
```

### Option 3: Bi-Weekly (Very Conservative)
```
First Monday of Month:
  - Full portfolio optimization

Mid-Month Monday:
  - Check signals, rebalance if major changes
```

**All options = Swing Trading, No PDT Issues**

---

## What Changes: Current vs. Portfolio Approach

### Current Approach (Individual Trades)
```
Signal for AAPL → Buy AAPL → Hold 5 days → Sell
Signal for MSFT → Buy MSFT → Hold 3 days → Sell
Signal for GOOGL → Skip (risk limit reached)
```

**Problems:**
- One signal at a time
- No diversification across positions
- Can't account for correlations
- Risk limits per trade, not portfolio

### Portfolio Approach (Swing Trading)
```
Monday Morning:
  1. Collect all signals: AAPL=0.8, MSFT=0.6, GOOGL=-0.3, NVDA=0.9
  2. Optimize portfolio weights:
     - AAPL: 15% (strong signal, low correlation)
     - MSFT: 10% (moderate signal)
     - NVDA: 20% (very strong signal)
     - GOOGL: 0% (weak sell, avoid)
  3. Execute trades to reach target weights
  4. HOLD portfolio for 2-7 days (swing trading)

Wednesday-Friday:
  - Still holding (no new trades)
  - Only rebalance if major signal changes
```

**Benefits:**
- ✅ Diversification (not all-in on one stock)
- ✅ Correlation-aware (avoid correlated positions)
- ✅ Optimal position sizing
- ✅ Portfolio-level risk management
- ✅ **Still swing trading** (holding 2-7 days)

---

## Implementation: Swing Trading Portfolio

### In `main.py` Trading Loop

**Current (Individual Trades):**
```python
for symbol in config.data.symbols:
    # Generate signal
    signal = generate_signal(symbol)

    # Execute if approved
    if signal.approved:
        execute_order(symbol, signal)
```

**Portfolio Approach (Swing Trading):**
```python
# Collect all signals first
all_signals = {}
for symbol in config.data.symbols:
    signal = generate_signal(symbol)
    all_signals[symbol] = signal

# Optimize portfolio ONCE per day (or weekly)
if should_rebalance():  # Once per day/week
    portfolio_weights = portfolio_optimizer.optimize(all_signals)

    # Rebalance positions (execute same day = OK, then HOLD)
    rebalance_portfolio(portfolio_weights)
else:
    # Just monitor positions (swing trading = holding)
    monitor_positions()
```

**Holding Period:**
- Positions held for **2-7 days** (swing trading)
- Rebalance **once per day/week**
- ✅ No day trading, no PDT restrictions

---

## Real-World Example

### Week of Trading (Portfolio Approach)

**Monday 9:30 AM:**
- Check signals: AAPL=0.8, MSFT=0.6, NVDA=0.9
- Optimize: AAPL=15%, MSFT=10%, NVDA=20%
- Execute: Buy AAPL, MSFT, NVDA
- **Status**: Portfolio set, HOLD

**Tuesday-Friday:**
- **No new trades** (swing trading = holding)
- Monitor positions
- Stop-losses active
- **Status**: Still holding (2-4 days so far)

**Next Monday:**
- Check signals again
- Optimize portfolio (some positions may change)
- Rebalance if needed (>5% drift)
- **Status**: Rebalance, then HOLD again

**Result:**
- ✅ 2 trades per week (Monday only)
- ✅ Holding periods: 2-7 days
- ✅ No day trading
- ✅ No PDT restrictions
- ✅ Better diversification

---

## Configuration for Swing Trading Portfolio

### In `config.toml`:

```toml
[portfolio]
# Rebalancing frequency (for swing trading)
rebalance_frequency = "daily"  # or "weekly", "biweekly"

# Portfolio optimization method
optimization_method = "mean_variance"  # or "risk_parity", "kelly"

# Rebalancing threshold (only rebalance if drift > threshold)
rebalance_threshold_pct = 0.05  # 5% drift before rebalancing

# Maximum positions in portfolio
max_positions = 10  # Diversify across 10 stocks

# Minimum holding period (enforce swing trading)
min_holding_days = 2  # Must hold at least 2 days
```

---

## Key Points

1. **Portfolio Optimization ≠ High Frequency**
   - Happens once per day/week (not multiple times per day)
   - Positions held for 2-7 days (swing trading)

2. **No PDT Issues**
   - Not day trading (holding > 1 day)
   - PDT rule only applies to same-day open/close
   - Portfolio optimization sets positions, then you HOLD

3. **Better Than Current Approach**
   - Diversification across multiple positions
   - Correlation-aware position sizing
   - Portfolio-level risk management
   - Same holding period, better returns

4. **RenTec-Style Thinking**
   - Think in portfolios, not individual trades
   - Optimize across all positions simultaneously
   - Account for correlations and diversification

---

## Conclusion

**Portfolio optimization is PERFECT for swing trading** because:
- ✅ Same holding periods (2-7 days)
- ✅ No day trading (PDT-safe)
- ✅ Better diversification
- ✅ Optimal position sizing
- ✅ Portfolio-level risk management

**What changes:**
- **Before**: One signal → one trade → hold individually
- **After**: All signals → optimize portfolio → hold portfolio

**What stays the same:**
- Holding periods: 2-7 days (swing trading)
- Trading frequency: 10-50 trades per year
- No day trading: PDT-safe

Portfolio optimization makes your swing trading **better**, not faster.
