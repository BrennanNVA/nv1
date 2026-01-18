# Post-Training Model Analysis: Institutional Best Practices
## What Quantitative Trading Firms Want to See After Model Training

**Research Date:** January 2025
**Purpose:** Reference guide for evaluating trained models using institutional-grade metrics and practices

---

## 🎯 Executive Summary

Institutional quant firms (RenTec, Two Sigma, etc.) evaluate models using a **comprehensive suite of metrics** that go beyond simple accuracy. They focus on:

1. **Risk-adjusted returns** (Sharpe, Calmar, Sortino)
2. **Drawdown analysis** (Max DD, duration, frequency, velocity)
3. **Feature importance & stability** (which indicators matter, do they stay important?)
4. **Overfitting checks** (walk-forward analysis, out-of-sample testing)
5. **Regime robustness** (does it work in bull/bear/volatile markets?)
6. **Trade-level metrics** (win rate, profit factor, expectancy)
7. **Benchmark comparison** (alpha, beta, information ratio)

**Key Insight:** No single metric is sufficient. Firms use a **composite approach** combining multiple metrics to assess model quality.

---

## 📊 1. Core Performance Metrics

### Risk-Adjusted Returns

**Sharpe Ratio** - Most common institutional metric
- **Formula:** `(Annualized Return - Risk-free Rate) / Annualized Volatility`
- **Target:** > 1.0 (decent), > 2.0 (strong) - RenTec's Medallion fund targets Sharpe > 2.0
- **Why:** Measures return per unit of total volatility
- **Limitation:** Assumes normal distributions, doesn't penalize downside specifically

**Sortino Ratio** - Better for asymmetric returns
- **Formula:** `(Annualized Return - Risk-free Rate) / Downside Volatility`
- **Target:** Higher than Sharpe ratio (penalizes only downside)
- **Why:** Better when upside is lumpy or volatile

**Calmar Ratio** - Focuses on worst-case loss
- **Formula:** `Annualized Return / Maximum Drawdown`
- **Target:** > 1.0 (strong), > 2.0 (excellent)
- **Why:** Directly ties return to worst-case loss - critical for capital preservation
- **Use Case:** Favored by firms with strict drawdown limits

**Information Ratio** - Relative to benchmark
- **Formula:** `(Strategy Return - Benchmark Return) / Tracking Error`
- **Target:** > 0.5 (good), > 1.0 (strong)
- **Why:** Measures consistent outperformance vs benchmark per unit of tracking error
- **Benchmark:** S&P 500, sector indices, or risk-free rate

### Absolute Returns

- **Total Return** - Cumulative gain/loss over period
- **Annualized Return** - Standardized for period comparison
- **Alpha** - Excess return over benchmark (after adjusting for beta)
- **Beta** - Sensitivity to market movements (should be < 1.0 for swing trading)

---

## 📉 2. Drawdown Analysis

Drawdowns are critical for institutional risk management - they affect capital preservation, investor confidence, and capital allocation.

### Key Metrics

**Maximum Drawdown (MDD)**
- **Definition:** Largest peak-to-trough decline in equity curve
- **Target:** < 10-15% for swing trading strategies
- **Why Critical:** Represents worst-case loss - affects capacity and investor patience
- **Example:** If equity peaks at $100k, drops to $85k, then MDD = 15%

**Average Drawdown**
- **Definition:** Average of all drawdowns within a period
- **Why:** Shows typical downside stress (not just worst case)

**Drawdown Duration**
- **Definition:** Time from peak to recovery to prior high
- **Target:** < 3-6 months for swing trading
- **Why Critical:** Long drawdowns = opportunity cost, psychological stress, investor outflows

**Drawdown Frequency & Velocity**
- **Frequency:** How often drawdowns occur
- **Velocity:** How fast losses accumulate (steep drops vs gradual declines)
- **Use Case:** Identify patterns - are drawdowns frequent but shallow, or rare but deep?

### Visualization

- **Equity Curve with Drawdown Overlay** - "Underwater" plot showing peaks and troughs
- **Drawdown Table** - List of largest drawdowns with dates, durations, recovery dates
- **Drawdown Distribution** - Quantiles of drawdown depth vs duration

### Risk Controls

- **Hard Limits:** Stop trading if drawdown > X% (e.g., 15%)
- **Soft Limits:** Reduce position sizing if drawdown > Y% (e.g., 10%)
- **Recovery Rules:** Only restore full leverage after recovery threshold met

---

## 🔍 3. Feature Importance & Interpretability

Understanding which indicators drive model predictions is critical for:
- **Interpretability** - Can explain why model makes decisions
- **Robustness** - Removing noise, focusing on signal
- **Overfitting Prevention** - If obscure features dominate, model may be overfit

### Methods

**XGBoost Feature Importance** (Gain-based)
- **What:** Tree-based importance using gain from each feature
- **Pros:** Fast, built into XGBoost
- **Use:** Quick ranking of which indicators matter most

**Permutation Importance**
- **What:** Randomly permute feature values, measure performance drop
- **Pros:** Model-agnostic, robust
- **Use:** Cross-validation of feature importance

**SHAP (Shapley Additive Explanations)**
- **What:** Game-theoretic approach measuring marginal contribution per feature
- **Pros:** Excellent for tree-based models, provides local + global importance
- **Use:** Deep dive into feature contributions

**Drop-Feature Tests**
- **What:** Remove feature, retrain, measure degradation
- **Pros:** Direct measure of feature utility
- **Cons:** Computationally expensive

### What to Look For

✅ **Good Signs:**
- Top features align with research (Squeeze_pro, MACD, RSI63, ROC63)
- Feature importance is stable across different time windows
- Economic logic - features make sense for swing trading

⚠️ **Warning Signs:**
- Obscure/unknown features dominate
- Feature importance unstable (changes drastically over time)
- Very similar features both in top 10 (redundancy)

### Action Items

- **Review top 20-30 features** - Do they align with research findings?
- **Check feature stability** - Same features matter across bull/bear markets?
- **Consider feature selection** - If 80+ indicators, top 30 often perform better
- **Remove redundant features** - If RSI(14) and RSI(21) both in top 10, pick one

---

## 🔄 4. Walk-Forward Analysis (WFA)

**Walk-forward analysis is the gold standard for institutional validation** - it tests model stability and prevents overfitting.

### What It Is

1. **Split data** into In-Sample (IS) training and Out-of-Sample (OOS) testing periods
2. **Train model** on IS data
3. **Test model** on OOS data (with same parameters)
4. **Roll forward** - Move window forward, repeat train→test cycles
5. **Aggregate** OOS results to simulate live performance

### Why It Matters

- **Prevents Overfitting** - Tests generalization to unseen data
- **Regime Adaptation** - Tests performance across different market conditions
- **Parameter Stability** - Do optimal parameters change drastically over time?
- **Realistic Performance** - OOS results simulate live trading better than in-sample

### Metrics to Track

For each walk-forward fold:
- **OOS Sharpe Ratio** - Primary metric
- **OOS Max Drawdown** - Worst-case loss in test period
- **OOS Calmar Ratio** - Return / Max DD for test period
- **IS vs OOS Comparison** - How much does performance degrade?

### What to Look For

✅ **Good Signs:**
- OOS Sharpe > 1.0 consistently across folds
- IS vs OOS degradation < 20% (e.g., IS Sharpe 1.5, OOS Sharpe 1.2)
- Parameters stable across time windows
- Performance consistent across bull/bear/volatile markets

⚠️ **Warning Signs:**
- OOS Sharpe much lower than IS Sharpe (> 30% degradation)
- Large variation in OOS metrics across folds
- Parameters change drastically over time
- Performance only good in specific market conditions

### Implementation

Your system uses walk-forward analysis when `use_walk_forward=True`:
- Trains on historical data
- Tests on forward periods
- Validates robustness across time

**Action:** Always enable walk-forward for production models.

---

## 🏛️ 5. Regime Robustness Testing

Models must perform across different market environments - not just one type of market.

### Market Regimes to Test

1. **Bull Markets** - Rising prices, low volatility
2. **Bear Markets** - Falling prices, high volatility
3. **High Volatility** - Large price swings (VIX > 30)
4. **Low Volatility** - Calm markets (VIX < 15)
5. **Crisis Periods** - Market crashes, extreme events (2008, 2020, etc.)
6. **Sideways Markets** - Range-bound, no clear trend

### Metrics to Track Per Regime

- Sharpe Ratio per regime
- Max Drawdown per regime
- Win Rate per regime
- Feature Importance per regime (do same features matter?)

### What to Look For

✅ **Good Signs:**
- Consistent performance across regimes (Sharpe > 1.0 in bull, bear, volatile)
- Same features important in different regimes
- Graceful degradation (works well in bull, okay in bear, not catastrophic)

⚠️ **Warning Signs:**
- Only performs well in bull markets
- Catastrophic failures in bear/volatile markets
- Different features dominate in each regime (model unstable)

### Action Items

- **Segment backtest by regime** - Analyze performance in bull vs bear periods
- **Stress test** - What happens in 2008-style crash?
- **Compare feature importance** - Stable across regimes?

---

## 📈 6. Trade-Level Metrics

Institutional firms analyze individual trades, not just aggregate performance.

### Core Metrics

**Win Rate**
- **Definition:** Percentage of profitable trades
- **Target:** > 45-55% for swing trading
- **Context:** Lower win rate acceptable if average win >> average loss

**Profit Factor**
- **Formula:** `Gross Profits / Gross Losses`
- **Target:** > 1.5 (strong), > 2.0 (excellent)
- **Why:** Measures ability to generate profits relative to losses

**Average Win vs Average Loss**
- **Target:** Average win should be 1.5-2x average loss
- **Use:** Risk-reward ratio assessment

**Risk-Reward Ratio**
- **Formula:** `Average Win / Average Loss`
- **Target:** > 1.5:1 minimum
- **Why:** Ensures profits can cover losses

**Expectancy**
- **Formula:** `(Win Rate × Average Win) - (Loss Rate × Average Loss)`
- **Target:** > 0 (positive expectancy required)
- **Why:** Expected value per trade

**Trade Frequency & Holding Period**
- **Frequency:** Trades per month/year
- **Holding Period:** Average days in position
- **Context:** More trades = higher costs (slippage, commissions)
- **Target:** 10-50 trades per year for swing trading

**Largest Win / Largest Loss**
- **Why:** Identify outliers - is performance driven by one huge win?
- **Action:** Ensure not over-reliant on single trades

---

## 🎭 7. Overfitting & Generalization Checks

Institutions are obsessed with avoiding overfitting - a model that looks great in backtest but fails live.

### Warning Signs of Overfitting

1. **IS vs OOS Degradation**
   - **Bad:** IS Sharpe 2.5, OOS Sharpe 0.8 (> 60% drop)
   - **Good:** IS Sharpe 1.8, OOS Sharpe 1.5 (< 20% drop)

2. **Parameter Sensitivity**
   - **Bad:** Performance drops sharply if parameters change slightly
   - **Good:** Performance stable across parameter ranges

3. **Feature Importance Instability**
   - **Bad:** Top features change drastically over time
   - **Good:** Same core features important across periods

4. **Regime-Specific Performance**
   - **Bad:** Only works in one market type
   - **Good:** Works across bull/bear/volatile

5. **Unrealistic Metrics**
   - **Bad:** Sharpe > 3.0, Win Rate > 80%, No drawdowns
   - **Good:** Realistic metrics (Sharpe 1.5-2.5, Win Rate 50-60%)

### Validation Techniques

- **Walk-Forward Analysis** - ✅ You have this
- **Purged Cross-Validation** - Prevents data leakage
- **Out-of-Sample Testing** - Hold back recent data, test on it
- **Monte Carlo Simulation** - Bootstrap trade selection, test robustness
- **Parameter Stability Testing** - Vary hyperparameters, check sensitivity

### Deflated Sharpe Ratio (DSR)

Accounts for multiple testing - if testing 100 strategies, some will look good by chance.

- **Your system:** Calculates DSR in validation
- **Target:** DSR > 0.95 (strong evidence strategy is real, not luck)
- **Action:** Always report DSR alongside Sharpe

---

## 📊 8. Benchmark Comparison

Institutional strategies are evaluated relative to benchmarks.

### Benchmarks to Compare Against

1. **S&P 500** - Market benchmark
2. **Sector Indices** - If trading specific sectors
3. **Risk-Free Rate** - Treasury bonds (for Sharpe ratio)
4. **Buy-and-Hold** - Simple baseline strategy
5. **Peer Strategies** - Other quant strategies in same space

### Metrics

**Alpha** - Excess return over benchmark (after adjusting for beta)
- **Target:** Positive alpha (> 0) indicates skill

**Beta** - Sensitivity to market movements
- **Target:** < 1.0 for swing trading (lower market exposure)

**Information Ratio** - Consistent outperformance vs benchmark
- **Target:** > 0.5 (good), > 1.0 (strong)

**Tracking Error** - How much strategy deviates from benchmark
- **Context:** Lower is not always better - need balance with alpha

---

## ✅ 9. Post-Training Checklist

Use this checklist after model training completes:

### Performance Metrics
- [ ] Sharpe Ratio > 1.0 (target > 1.5-2.0)
- [ ] Calmar Ratio > 1.0 (target > 2.0)
- [ ] Sortino Ratio > Sharpe Ratio
- [ ] Max Drawdown < 15% (target < 10%)
- [ ] Annualized Return > benchmark

### Feature Analysis
- [ ] Top 20-30 features align with research (Squeeze_pro, MACD, RSI63, ROC63)
- [ ] Feature importance stable across time windows
- [ ] Features make economic sense for swing trading
- [ ] No redundant features dominating (e.g., both RSI14 and RSI21 in top 10)

### Walk-Forward Validation
- [ ] Walk-forward enabled (`use_walk_forward=True`)
- [ ] OOS Sharpe > 1.0 across all folds
- [ ] IS vs OOS degradation < 20%
- [ ] Deflated Sharpe Ratio (DSR) > 0.95

### Drawdown Analysis
- [ ] Max Drawdown < 15%
- [ ] Average drawdown duration < 3-6 months
- [ ] Drawdown frequency acceptable (not too frequent)
- [ ] Drawdown velocity acceptable (not too steep)

### Trade-Level Metrics
- [ ] Win Rate > 45% (target 50-60%)
- [ ] Profit Factor > 1.5 (target > 2.0)
- [ ] Average Win > 1.5x Average Loss
- [ ] Positive expectancy
- [ ] Trade frequency reasonable (10-50 trades/year for swing)

### Robustness Checks
- [ ] Performance consistent across bull/bear markets
- [ ] Parameters stable across time windows
- [ ] No catastrophic failures in stress tests (2008, 2020 crashes)
- [ ] Feature importance stable across regimes

### Benchmark Comparison
- [ ] Positive alpha vs S&P 500
- [ ] Beta < 1.0 (lower market exposure)
- [ ] Information Ratio > 0.5
- [ ] Outperforms buy-and-hold baseline

---

## 📚 10. Reporting Template

After training, create a model evaluation report with:

### Executive Summary
- Model name, training date, symbol
- Key metrics (Sharpe, Calmar, Max DD, Win Rate)
- Overall assessment (Pass/Fail criteria)

### Performance Metrics
- Risk-adjusted returns (Sharpe, Sortino, Calmar, Information Ratio)
- Absolute returns (Total, Annualized, Alpha)
- Benchmark comparison

### Drawdown Analysis
- Maximum Drawdown with dates
- Drawdown duration table
- Equity curve with drawdown overlay

### Feature Importance
- Top 20-30 features ranked by importance
- Feature importance stability across time
- Comparison with research findings

### Walk-Forward Results
- OOS metrics per fold
- IS vs OOS comparison
- Parameter stability across folds
- Deflated Sharpe Ratio

### Trade Analysis
- Win rate, profit factor, expectancy
- Average win/loss, risk-reward ratio
- Trade frequency and holding periods
- Largest wins/losses

### Robustness Testing
- Performance by market regime (bull/bear/volatile)
- Stress test results (crisis periods)
- Parameter sensitivity analysis

### Recommendations
- Deploy to production? (Yes/No/With conditions)
- Suggested improvements
- Next steps (retrain, adjust parameters, etc.)

---

## 🎯 11. Institutional Thresholds & Targets

Based on research of RenTec, Two Sigma, and other quant firms:

### Minimum Acceptable Thresholds
- **Sharpe Ratio:** > 1.0 (minimum), > 1.5 (good), > 2.0 (strong)
- **Calmar Ratio:** > 1.0 (minimum), > 2.0 (strong)
- **Max Drawdown:** < 15% (acceptable), < 10% (strong)
- **Win Rate:** > 45% (minimum), > 50% (good)
- **Profit Factor:** > 1.5 (minimum), > 2.0 (strong)
- **DSR:** > 0.90 (minimum), > 0.95 (strong)

### Red Flags (Reject Model)
- Sharpe < 0.5
- Max Drawdown > 20%
- IS vs OOS degradation > 50%
- DSR < 0.85
- Only works in one market regime
- Catastrophic failures in stress tests

---

## 📖 References

1. Quant.fish - Metrics for Algorithmic Trading Strategies
2. QuantifiedStrategies.com - Trading Performance Analysis
3. Research papers on walk-forward analysis and deflated Sharpe ratio
4. Institutional quant firm practices (RenTec, Two Sigma)
5. Academic research on model validation for financial ML

---

**Last Updated:** January 2025
**Next Steps:** Use this checklist after training completes to evaluate model quality
