# Implementation Patterns and Best Practices

## Strategy Patterns

### Multi-Indicator Signal Generation
```python
# Example: Combine multiple indicators for signal generation
signals = []
for i in range(lookback, len(data)):
    trend_bullish = sma_fast[i] > sma_slow[i] and sma_fast[i-1] <= sma_slow[i-1]
    momentum_ok = 40 < rsi[i] < 60
    volume_confirm = obv[i] > obv[i-1]

    if trend_bullish and momentum_ok and volume_confirm:
        signals.append(1)  # Buy
    elif sma_fast[i] < sma_slow[i]:
        signals.append(-1)  # Sell
    else:
        signals.append(0)  # Hold
```

### ATR-Based Stop Loss
```python
# Calculate stop loss using ATR
atr_value = calculate_atr(high, low, close, period=14)
stop_loss_long = entry_price - (atr_value * 2.0)
stop_loss_short = entry_price + (atr_value * 2.0)

# Position sizing based on risk
risk_per_share = abs(entry_price - stop_loss)
max_risk_amount = capital * 0.02  # 2% risk per trade
shares = int(max_risk_amount / risk_per_share)
```

### Walk-Forward Analysis Pattern
```python
# Train on period 1, test on period 2, slide forward
results = []
start_idx = 0
train_period = 252  # 1 year
test_period = 63    # 1 quarter

while start_idx + train_period + test_period <= len(data):
    train_data = data.iloc[start_idx:start_idx + train_period]
    test_data = data.iloc[start_idx + train_period:start_idx + train_period + test_period]

    # Optimize on training data
    best_params = optimize_strategy(train_data)

    # Test on out-of-sample data
    test_result = backtest_strategy(test_data, best_params)
    results.append(test_result)

    start_idx += test_period
```

## Risk Management Patterns

### Position Sizing
```python
def calculate_position_size(
    entry_price: float,
    stop_loss: float,
    capital: float,
    risk_per_trade: float = 0.02
) -> int:
    """Calculate position size based on risk per trade."""
    risk_per_share = abs(entry_price - stop_loss)
    max_risk_amount = capital * risk_per_trade
    shares = int(max_risk_amount / risk_per_share)
    return max(1, shares)  # Minimum 1 share
```

### Drawdown Protection
```python
def check_drawdown(
    current_equity: float,
    peak_equity: float,
    max_drawdown: float = 0.10
) -> bool:
    """Check if drawdown limit exceeded."""
    if peak_equity == 0:
        return True
    drawdown = (peak_equity - current_equity) / peak_equity
    return drawdown <= max_drawdown
```

## Performance Optimization

### Async Data Loading
```python
async def load_multiple_symbols(symbols: list[str]) -> dict:
    """Load data for multiple symbols concurrently."""
    tasks = [load_symbol_data(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    return dict(zip(symbols, results))
```

### Batch Processing
```python
# Process indicators in batches for efficiency
def calculate_all_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate multiple indicators in single pass."""
    return df.with_columns([
        pl.col("close").rolling_mean(20).alias("sma_20"),
        pl.col("close").rolling_mean(50).alias("sma_50"),
        # ... more indicators
    ])
```

## Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, max_errors: int = 5, time_window: int = 60):
        self.max_errors = max_errors
        self.time_window = time_window
        self.errors = []

    def record_error(self):
        """Record an error with timestamp."""
        now = time.time()
        self.errors.append(now)
        # Remove errors outside time window
        self.errors = [e for e in self.errors if now - e < self.time_window]

    def is_open(self) -> bool:
        """Check if circuit breaker should halt system."""
        return len(self.errors) >= self.max_errors
```
