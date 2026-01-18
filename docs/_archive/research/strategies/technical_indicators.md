# Technical Indicators Reference

## Indicator Formulas and Implementation Notes

### Moving Averages

#### Simple Moving Average (SMA)
- Formula: `SMA = Σ(Prices) / Period`
- Implementation: Sliding window with running sum for efficiency

#### Exponential Moving Average (EMA)
- Formula: `EMA = (Price - Previous EMA) * (2 / (Period + 1)) + Previous EMA`
- More responsive than SMA, gives more weight to recent prices
- Recursive calculation for efficiency

### Momentum Indicators

#### Relative Strength Index (RSI)
- Formula: `RSI = 100 - (100 / (1 + RS))`
- Where `RS = Average Gain / Average Loss` over period
- Period: Typically 14
- Range: 0-100
- Overbought: >70, Oversold: <30
- Edge cases: Handle all gains/all losses scenarios

#### MACD (Moving Average Convergence Divergence)
- Components:
  - MACD Line = 12 EMA - 26 EMA
  - Signal Line = 9 EMA of MACD Line
  - Histogram = MACD Line - Signal Line
- Use case: Momentum and trend changes

### Volatility Indicators

#### Bollinger Bands
- Middle Band = SMA(period)
- Upper Band = SMA + (StdDev * multiplier)
- Lower Band = SMA - (StdDev * multiplier)
- Default: Period 20, Multiplier 2.0
- Use case: Volatility expansion/contraction, mean reversion

#### Average True Range (ATR)
- True Range = max(
    High - Low,
    |High - Previous Close|,
    |Low - Previous Close|
  )
- ATR = SMA(True Range, period)
- Default period: 14
- Critical for: Stop-loss placement (`StopLoss = EntryPrice ± (ATR * multiplier)`)

### Volume Indicators

#### Volume Weighted Average Price (VWAP)
- Formula: `VWAP = Σ(Price * Volume) / Σ(Volume)` over period
- Use case: Intraday price levels, institutional activity

#### On-Balance Volume (OBV)
- Cumulative: `OBV = Previous OBV + Volume` (if close up) or `-Volume` (if close down)
- Use case: Volume confirmation, divergence detection

### Support/Resistance Tools

#### Fibonacci Retracements
- Levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
- Calculate retracement levels from swing high/low
- Use case: Identify pullback entry points within trends

#### Pivot Points
- Pivot = (High + Low + Close) / 3
- Resistance: R1 = 2*Pivot - Low, R2 = Pivot + (High - Low), R3 = High + 2*(Pivot - Low)
- Support: S1 = 2*Pivot - High, S2 = Pivot - (High - Low), S3 = Low - 2*(High - Pivot)
- Use case: Intraday support/resistance levels

## Implementation Notes for Polars

When implementing these indicators in Polars:
- Use `rolling_*` functions for window calculations
- Leverage Polars' lazy evaluation for performance
- Use `over()` for grouped operations by symbol
- Consider using `cumsum()` for cumulative indicators like OBV
- Pre-allocate result columns when possible

## Performance Targets

- Single indicator (10K points): < 1ms
- Multi-indicator strategy: < 10ms
- Use vectorized operations where possible
