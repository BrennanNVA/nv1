# Indicator Validation & Ranking Guide

## Overview

This guide explains how to validate technical indicator effectiveness using actual training data and compare results with research findings.

**New Indicators Implemented:**
1. **Squeeze Pro** - #1 research-ranked indicator for swing trading
2. **RSI(63)** - Longer-period RSI (63-day, research-backed)
3. **ROC(63)** - Longer-period Rate of Change (63-day, research-backed)

**Validation Script:** `scripts/validate_indicator_ranking.py`

---

## Quick Start

### Validate Indicators on Single Symbol

```bash
cd /home/brennan/nac/nova_aetus
source venv/bin/activate  # If using venv

python scripts/validate_indicator_ranking.py --symbols AAPL --years 5 --top-n 30
```

### Validate Multiple Symbols (Recommended)

```bash
python scripts/validate_indicator_ranking.py \
    --symbols AAPL MSFT GOOGL AMZN NVDA \
    --years 5 \
    --top-n 30
```

**Output:**
- Console output with rankings and comparisons
- JSON report saved to `reports/indicator_ranking_YYYYMMDD_HHMMSS.json`

---

## What the Script Does

### 1. **Ranks Indicators by Importance**
- Trains XGBoost model with all features
- Extracts feature importance (gain-based)
- Ranks indicators by importance score
- Averages rankings across multiple symbols

### 2. **Compares with Research Findings**
- Checks if research-backed top indicators appear in top 30
- Calculates correlation between research rankings and actual rankings
- Validates if expected indicators perform well

### 3. **Validates Longer Periods**
- Compares RSI(14) vs RSI(63)
- Compares ROC(12) vs ROC(63)
- Determines if longer periods benefit swing trading

---

## Research Findings Reference

**Top-Performing Indicators** (from RESEARCH_SUMMARY.md):
1. **Squeeze_pro** - Highest importance (#1)
2. **PPO** - Percentage Price Oscillator (#2)
3. **MACD** - Most profitable, lowest-risk (#3)
4. **ROC63** - Rate of Change, 63-day (#4)
5. **RSI63** - RSI, 63-day (#5)

**Key Insights:**
- Longer periods (63, 252 days) often outperform shorter periods for swing trading
- Windowed features (multiple periods) improve performance
- Feature selection critical: Aim for top 20-30 from 100+ indicators

---

## New Indicators Implemented

### 1. Squeeze Pro (`squeeze_pro`)

**What it does:** Detects volatility squeeze when Bollinger Bands contract inside Keltner Channels, indicating low volatility period before potential breakouts.

**Components:**
- `squeeze`: Boolean (1 = squeeze active, 0 = no squeeze)
- `squeeze_pro`: Float (1.0 = high compression, 0.5 = medium, 0.0 = none)
- `squeeze_momentum`: Momentum histogram for breakout direction

**Formula:**
```
Squeeze ON = (BB_upper < KC_upper) AND (BB_lower > KC_lower)
Squeeze Pro = 1.0 if BB inside tight KC, 0.5 if inside medium KC, 0.0 otherwise
Momentum = EMA(Price - Price[12], 6)  # Smoothed price change
```

**Parameters:**
- BB period: 20
- BB stddev: 2.0
- KC period: 20
- KC multipliers: 1.0 (tight), 1.5 (medium), 2.0 (wide)
- Momentum: 12-period change, 6-period EMA smoothing

**Research Status:** #1 ranked indicator for swing trading

### 2. RSI(63) (`rsi_63`)

**What it does:** Relative Strength Index with 63-day period (instead of default 14-day).

**Why:** Research shows longer periods (63 days) outperform shorter periods for swing trading.

**Formula:** Same as RSI(14), but with 63-day period for Wilder's smoothing.

**Research Status:** #5 ranked indicator (with 63-day period)

### 3. ROC(63) (`roc_63`)

**What it does:** Rate of Change with 63-day period (instead of default 12-day).

**Why:** Research shows longer periods (63 days) outperform shorter periods for swing trading.

**Formula:**
```
ROC(63) = ((Close - Close[63]) / Close[63]) * 100
```

**Research Status:** #4 ranked indicator (with 63-day period)

---

## Example Output

```
============================================================
INDICATOR RANKING & VALIDATION
============================================================
Symbols: ['AAPL', 'MSFT', 'GOOGL']
Years: 5
Top N: 30
============================================================

Rank  Importance  Category                  Indicator Name
----- ----------  ------------------------  --------------------
1     0.023       Volatility Squeeze        squeeze_pro
2     0.019       Trend (MACD)              macd_histogram [R#3]
3     0.017       Momentum                  ppo [R#2]
4     0.016       Momentum                  roc_63 [R#4]
5     0.015       Momentum                  rsi_63 [R#5]
...
30    0.005       Normalized                price_sma50_pct_zscore

============================================================
COMPARING WITH RESEARCH FINDINGS
============================================================

Research Indicators Found: 5/5
  ✅ squeeze_pro     | Research: # 1 | Actual: #  1 | Diff:  +0 | Importance: 0.023
  ✅ ppo             | Research: # 2 | Actual: #  3 | Diff:  +1 | Importance: 0.019
  ✅ macd            | Research: # 3 | Actual: #  2 | Diff:  -1 | Importance: 0.020
  ✅ roc_63          | Research: # 4 | Actual: #  4 | Diff:  +0 | Importance: 0.016
  ✅ rsi_63          | Research: # 5 | Actual: #  5 | Diff:  +0 | Importance: 0.015

Research Correlation: 0.98
  ✅ Strong correlation - Research findings validated!

============================================================
VALIDATING LONGER-PERIOD INDICATORS (63-day)
============================================================

RSI 14 vs 63:
  ✅ RSI(14): Rank #12, Importance: 0.008
  ✅ RSI(63): Rank #5, Importance: 0.015
    → RSI(63) outperforms RSI(14) by 7 ranks

ROC 12 vs 63:
  ✅ ROC(12): Rank #15, Importance: 0.006
  ✅ ROC(63): Rank #4, Importance: 0.016
    → ROC(63) outperforms ROC(12) by 11 ranks
```

---

## Interpreting Results

### Research Correlation

- **> 0.7**: ✅ Strong correlation - Research findings validated!
- **0.4 - 0.7**: ⚠️ Moderate correlation - Some alignment with research
- **< 0.4**: ❌ Low correlation - Research findings need review

### Longer Period Validation

- **RSI(63) rank < RSI(14) rank**: ✅ Longer period outperforms - Use RSI(63)
- **ROC(63) rank < ROC(12) rank**: ✅ Longer period outperforms - Use ROC(63)
- **Opposite**: ❌ Shorter period better - Consider keeping both

### Expected Top 30 Indicators

Check if these research-backed indicators appear in top 30:
- ✅ `squeeze_pro` (should be #1 or top 3)
- ✅ `ppo` (should be top 5)
- ✅ `macd` or `macd_histogram` (should be top 5)
- ✅ `roc_63` (should be top 10)
- ✅ `rsi_63` (should be top 10)

---

## Using Results

### 1. Enable Feature Selection

After validation, enable top N feature selection in `config.toml`:

```toml
[ml]
feature_selection_top_n = 30  # Use top 30 features from validation
```

### 2. Review Indicator Rankings

Check `reports/indicator_ranking_*.json` for detailed rankings per symbol.

### 3. Update Feature Lists

If certain indicators consistently rank poorly, consider removing them from feature generation to reduce noise.

---

## Command Options

```bash
python scripts/validate_indicator_ranking.py [OPTIONS]
```

**Options:**
- `--symbols SYMBOL [SYMBOL ...]`: Symbols to analyze (default: AAPL MSFT GOOGL)
- `--years YEARS`: Years of historical data (default: 5)
- `--top-n N`: Number of top indicators to rank (default: 30)
- `--output FILE`: Output JSON file path (default: `indicator_ranking_YYYYMMDD_HHMMSS.json`)

**Examples:**

```bash
# Single symbol, 3 years of data
python scripts/validate_indicator_ranking.py --symbols AAPL --years 3

# Multiple symbols, top 50 indicators
python scripts/validate_indicator_ranking.py --symbols AAPL MSFT GOOGL AMZN NVDA --top-n 50

# Custom output file
python scripts/validate_indicator_ranking.py --output my_rankings.json
```

---

## Troubleshooting

### "No data fetched"

**Solution:** Check API keys in `.env` or verify symbol is valid.

### "Feature importance returns empty"

**Solution:** Ensure enough historical data (3+ years recommended).

### "GPU not available"

**Solution:** Script falls back to CPU automatically - will be slower but works.

---

## Next Steps

1. **Run validation** on multiple symbols to get aggregate rankings
2. **Compare results** with research findings
3. **Enable feature selection** in config if validation confirms top 30 approach
4. **Monitor model performance** after enabling feature selection
5. **Re-validate periodically** (quarterly) as market conditions change

---

**Last Updated:** January 2025
**Script:** `scripts/validate_indicator_ranking.py`
**Report Location:** `reports/indicator_ranking_*.json`
