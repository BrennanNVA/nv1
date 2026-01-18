# Swing Trading Strategy Research

## Core Swing Trading Characteristics
- **Time Horizon**: 2-10 days (holding positions for multiple days)
- **Goal**: Capture price swings within a trend
- **Entry/Exit**: Based on technical analysis, not fundamental analysis
- **Risk Management**: Stop-losses and position sizing critical

## Essential Technical Indicators for Swing Trading

### Trend Indicators
- **Simple Moving Average (SMA)**: Common periods: 20, 50, 100, 200 days
- **Exponential Moving Average (EMA)**: Common periods: 12, 26, 50, 200
  - Formula: `EMA = (Price - Previous EMA) * (2 / (Period + 1)) + Previous EMA`
- **Moving Average Convergence Divergence (MACD)**:
  - Components: MACD line (12 EMA - 26 EMA), Signal line (9 EMA of MACD), Histogram
  - Critical for: Identifying entry/exit points via crossovers

### Momentum Indicators
- **Relative Strength Index (RSI)**: Period: Typically 14
  - Range: 0-100, Overbought: >70, Oversold: <30
  - Formula: `RSI = 100 - (100 / (1 + RS))` where `RS = Avg Gain / Avg Loss`
- **Stochastic Oscillator**: Components %K and %D
  - Formula: `%K = 100 * (Current Close - Lowest Low) / (Highest High - Lowest Low)`
- **Commodity Channel Index (CCI)**: Period: Typically 20
  - Formula: `CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)`

### Volatility Indicators
- **Bollinger Bands**:
  - Components: Middle band (SMA), Upper/Lower bands (SMA ± 2*StdDev)
  - Period: Typically 20 with 2 standard deviations
  - Critical for: Identifying squeeze patterns (low volatility before big moves)
- **Average True Range (ATR)**: Period: Typically 14
  - Formula: `ATR = SMA of True Range`, where `True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)`
  - Critical for stop-loss placement

### Volume Indicators
- **Volume Weighted Average Price (VWAP)**:
  - Formula: `VWAP = Σ(Price * Volume) / Σ(Volume)` over period
- **On-Balance Volume (OBV)**:
  - Formula: `OBV = Previous OBV + Volume` (if close up) or `-Volume` (if close down)

## Proven Swing Trading Strategy Patterns

### Pattern 1: Moving Average Crossover
- **Entry**: Fast MA crosses above slow MA (golden cross)
- **Exit**: Fast MA crosses below slow MA (death cross)
- **Confirmation**: Volume increase, RSI > 50

### Pattern 2: RSI Divergence
- **Entry**: Price makes lower low, RSI makes higher low (bullish divergence)
- **Exit**: Price makes higher high, RSI makes lower high (bearish divergence)
- **Confirmation**: Volume, trend alignment

### Pattern 3: Bollinger Band Squeeze
- **Entry**: Bands contract (low volatility), then expand with price breakout
- **Exit**: Price reaches opposite band or RSI overbought/oversold
- **Confirmation**: Volume spike on breakout

### Pattern 4: MACD Crossover with Trend
- **Entry**: MACD line crosses above signal line in uptrend
- **Exit**: MACD line crosses below signal line
- **Confirmation**: Price above 200 EMA (uptrend)

## Multi-Indicator Confirmation System
**Key Finding**: Successful swing traders combine multiple indicators to reduce false signals by 60-80%.

**Components**:
- **Primary Trend**: EMA(8) vs EMA(50) crossover
- **Momentum Confirmation**: RSI(14) between 40-60 for entry (not extreme)
- **Volume Confirmation**: OBV trending in same direction as price
- **Volatility Filter**: ATR-based position sizing (2x ATR for stop-loss)
- **Entry Signal**: All conditions align (trend + momentum + volume)
- **Exit Signal**: RSI > 70 (overbought) OR MACD bearish crossover

**Performance Metrics** (from research):
- Sharpe Ratio: 1.5-2.0 (good strategies)
- Win Rate: 45-55% (acceptable with proper risk/reward)
- Profit Factor: > 1.5 (gross profit / gross loss)

## Critical Strategy Design Principles

1. **Avoid Look-Ahead Bias**: Indicators must use only past data up to decision point
2. **Survivorship Bias Prevention**: Include delisted stocks in backtests
3. **Realistic Transaction Costs**: ~0.2-0.7% per round trip
4. **Walk-Forward Analysis**: Train on period 1, test on period 2
5. **Regime Filters**: Identify market conditions: trending vs ranging

## Risk Management
- **Risk per Trade**: 1-2% of capital
- **Stop-Loss**: ATR-based levels (2x ATR typical)
- **Position Sizing**: Based on risk per share and maximum risk amount
