# RenTec-Style Strategic Roadmap

## What RenTec Would Do Next

Based on institutional quant firm patterns (RenTec, Two Sigma, D.E. Shaw), here's what to build next with your current toolkit.

## Current State Assessment ✅

**You Have:**
- ✅ Signal generation (Technical, Sentiment, Fundamental)
- ✅ Model training (XGBoost GPU, NPMM labeling)
- ✅ Confluence layer (Regime-aware signal combination)
- ✅ Risk management (Position sizing, Kelly Criterion)
- ✅ Execution (Alpaca integration, cost modeling)
- ✅ Monitoring (IC decay, drift detection, SHAP)
- ✅ Validation (Walk-forward, PBO, DSR)

**Missing (Critical Gaps):**
- ❌ **Portfolio Construction** - Currently 1 trade at a time, no portfolio optimization
- ❌ **Backtesting Engine** - No comprehensive historical simulation
- ❌ **Multi-Strategy Framework** - No way to run/compare multiple strategies
- ❌ **Performance Attribution** - Can't decompose returns by signal/regime
- ❌ **Alpha Research Pipeline** - No systematic way to test new ideas

---

## Phase 1: Portfolio Construction (Weeks 1-2) 🔴 **HIGHEST PRIORITY**

### Problem
Currently generating signals one-at-a-time. RenTec thinks in **portfolios** - simultaneously optimizing across all positions.

### What to Build

#### 1.1 Portfolio Optimizer
```python
# src/nova/strategy/portfolio_optimizer.py
class PortfolioOptimizer:
    """Mean-variance and risk parity portfolio construction."""

    def optimize_mean_variance(
        signals: Dict[str, float],  # Signal for each symbol
        returns_covariance: np.ndarray,
        risk_aversion: float = 1.0
    ) -> Dict[str, float]:  # Target weights

    def optimize_risk_parity(
        signals: Dict[str, float],
        returns_covariance: np.ndarray
    ) -> Dict[str, float]:  # Risk-parity weights

    def optimize_kelly_portfolio(
        signals: Dict[str, float],
        win_rates: Dict[str, float],
        avg_wins: Dict[str, float],
        avg_losses: Dict[str, float]
    ) -> Dict[str, float]:  # Kelly-optimal weights
```

**Benefits:**
- Diversification (not all-in on one signal)
- Risk-adjusted position sizing
- Correlation-aware (avoid correlated positions)

#### 1.2 Signal-to-Position Mapping
Convert confluence signals → portfolio weights → target positions

**Integration Point:**
- In `main.py` trading loop, collect all signals first
- Run portfolio optimizer once per iteration
- Execute rebalancing trades

---

## Phase 2: Backtesting Engine (Weeks 2-4) 🔴 **CRITICAL**

### Problem
Can't validate strategies before live trading. RenTec backtests everything extensively.

### What to Build

#### 2.1 Comprehensive Backtester
```python
# src/nova/backtesting/backtester.py
class Backtester:
    """Full historical simulation with realistic execution."""

    def backtest_strategy(
        strategy: Strategy,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_model: Optional[SlippageModel] = None
    ) -> BacktestResult:
        """Run full backtest with execution costs."""
```

**Features:**
- Realistic execution (slippage, commissions, market impact)
- Position tracking (open/close, P&L)
- Performance metrics (Sharpe, Sortino, max drawdown)
- Trade log (every trade with details)

#### 2.2 Strategy Comparison Framework
Run multiple strategies on same data, compare results:

```python
strategies = [
    Strategy(name="Technical Only", weights={"technical": 1.0}),
    Strategy(name="Confluence", weights={"technical": 0.4, "sentiment": 0.35}),
    Strategy(name="Regime-Aware", use_regime_weights=True),
]

results = backtester.compare_strategies(strategies, data)
```

**Benefits:**
- Validate before live trading
- A/B test signal combinations
- Find optimal parameters

---

## Phase 3: Multi-Strategy Framework (Weeks 4-6) 🟡 **HIGH VALUE**

### Problem
Single strategy = single point of failure. RenTec runs multiple strategies simultaneously.

### What to Build

#### 3.1 Strategy Manager
```python
# src/nova/strategy/strategy_manager.py
class StrategyManager:
    """Manage multiple strategies running in parallel."""

    def add_strategy(
        name: str,
        strategy: Strategy,
        capital_allocation: float  # % of capital
    ):
        """Register a strategy."""

    def run_all_strategies(self, market_data: Dict) -> Dict[str, Portfolio]:
        """Generate signals from all strategies."""

    def combine_portfolios(self, portfolios: Dict[str, Portfolio]) -> Portfolio:
        """Combine into single portfolio."""
```

**Use Cases:**
- Momentum strategy (technical-only)
- Mean-reversion strategy (contrarian signals)
- Sentiment-driven strategy (high sentiment weight)
- Each runs independently, combined at portfolio level

#### 3.2 Strategy Performance Tracking
Track each strategy's performance separately:
- Returns attribution per strategy
- Which strategy is working best right now
- Dynamic capital allocation based on performance

---

## Phase 4: Performance Attribution (Weeks 6-7) 🟡 **ANALYTICS**

### Problem
Can't tell which signals/regimes are driving returns. RenTec knows exactly.

### What to Build

#### 4.1 Attribution Engine
```python
# src/nova/analytics/performance_attribution.py
class PerformanceAttributor:
    """Decompose returns by signal, regime, time period."""

    def attribute_returns(
        trades: List[Trade],
        signals: List[ConfluenceSignal]
    ) -> AttributionReport:
        """Break down returns."""
```

**Metrics:**
- Returns by signal type (technical vs sentiment vs fundamental)
- Returns by regime (bullish vs bearish vs high vol)
- Returns by time period (monthly/quarterly)
- Win rate by signal strength
- Sharpe ratio by regime

**Dashboard Integration:**
- New tab: "Performance Attribution"
- Visualize: "Where is alpha coming from?"

---

## Phase 5: Alpha Research Pipeline (Weeks 7-10) 🟢 **SCALING**

### Problem
No systematic way to test new ideas. RenTec tests hundreds of ideas per year.

### What to Build

#### 5.1 Research Framework
```python
# src/nova/research/research_pipeline.py
class ResearchPipeline:
    """Systematic alpha research workflow."""

    def test_feature(
        new_feature: Feature,
        base_strategy: Strategy,
        test_period: Tuple[datetime, datetime]
    ) -> ResearchResult:
        """Test if new feature improves returns."""

    def test_signal_combination(
        signal_weights: Dict[str, float],
        historical_data: DataFrame
    ) -> ResearchResult:
        """Test signal weight combinations."""
```

**Workflow:**
1. Generate hypothesis: "RSI(21) better than RSI(14)?"
2. Create feature variant
3. Backtest vs baseline
4. Statistical significance test (t-test, bootstrap)
5. If significant, add to production

#### 5.2 Feature Engineering Lab
```python
# src/nova/research/feature_lab.py
class FeatureLab:
    """Generate and test new features."""

    def generate_candidate_features(
        base_features: List[str],
        transformations: List[Callable]
    ) -> List[Feature]:
        """Generate feature variants."""

    def evaluate_features(
        features: List[Feature],
        target: Series
    ) -> FeatureRanking:
        """Rank features by IC, correlation, etc."""
```

---

## Phase 6: Advanced Portfolio Techniques (Weeks 10-12) 🟢 **ADVANCED**

### What RenTec Would Add:

#### 6.1 Market Neutral Hedging
```python
# src/nova/strategy/hedging.py
class MarketNeutralHedger:
    """Hedge portfolio beta to market."""

    def calculate_hedge(
        portfolio: Portfolio,
        market_beta: float,
        hedge_instrument: str = "SPY"  # ETF for hedging
    ) -> Dict[str, float]:  # Hedge positions
```

**Use Case:**
- Long individual stocks
- Short market ETF to neutralize beta
- Capture alpha regardless of market direction

#### 6.2 Pairs Trading
```python
# src/nova/strategy/pairs_trading.py
class PairsTrader:
    """Statistical arbitrage via cointegrated pairs."""

    def find_pairs(
        symbols: List[str],
        lookback: int = 252
    ) -> List[Tuple[str, str, float]]:  # (symbol1, symbol2, correlation)
```

#### 6.3 Dynamic Rebalancing
```python
# src/nova/strategy/rebalancer.py
class DynamicRebalancer:
    """Rebalance portfolio based on signal changes."""

    def should_rebalance(
        current_positions: Portfolio,
        target_positions: Portfolio,
        threshold: float = 0.05  # 5% drift threshold
    ) -> bool:
```

---

## Implementation Priority

### 🔴 **Do First** (Next 2 weeks)
1. **Portfolio Optimizer** - Critical for moving from single trades to portfolios
2. **Backtesting Engine** - Must validate before live trading

### 🟡 **Do Second** (Weeks 3-6)
3. **Multi-Strategy Framework** - Diversify signal sources
4. **Performance Attribution** - Understand what's working

### 🟢 **Do Later** (Weeks 7+)
5. **Alpha Research Pipeline** - Scale systematic research
6. **Advanced Techniques** - Market neutral, pairs trading

---

## Quick Wins (This Week)

### 1. Add Portfolio View to Dashboard
Show current portfolio composition:
```python
# In dashboard/app.py
def show_portfolio(config):
    positions = get_all_positions()
    portfolio_value = sum(p.market_value for p in positions)

    # Portfolio composition chart
    fig = px.pie(positions, values='market_value', names='symbol')

    # Portfolio-level metrics
    portfolio_sharpe = calculate_portfolio_sharpe(positions)
    portfolio_beta = calculate_portfolio_beta(positions, market='SPY')
```

### 2. Simple Portfolio Optimization
Start with equal-weight allocation:
```python
# In main.py, after collecting signals:
if len(approved_signals) > 0:
    # Equal-weight portfolio (simple start)
    target_weight = 1.0 / len(approved_signals)
    for symbol, signal in approved_signals.items():
        target_value = current_equity * target_weight
        target_shares = target_value / current_price[symbol]
        # Rebalance if needed
```

### 3. Backtest Current Strategy
Create simple backtest script:
```python
# scripts/backtest_current_strategy.py
# Load historical data
# Run confluence layer on each day
# Track positions and P&L
# Generate performance report
```

---

## Expected Outcomes

### After Phase 1-2 (Portfolio + Backtesting):
- **Move from 1-trade to portfolio thinking**
- **Validate strategies before live trading**
- **Expected improvement: 20-30% better risk-adjusted returns**

### After Phase 3-4 (Multi-Strategy + Attribution):
- **Diversify across multiple alpha sources**
- **Understand what's driving returns**
- **Expected improvement: 10-15% additional returns from diversification**

### After Phase 5-6 (Research + Advanced):
- **Systematic alpha discovery**
- **Market-neutral strategies**
- **Expected improvement: 5-10% additional alpha from research**

---

## RenTec Principles Applied

1. **Portfolio-First Thinking**: Always optimize across all positions
2. **Rigorous Backtesting**: Never trade unvalidated strategies
3. **Diversification**: Multiple uncorrelated strategies
4. **Attribution**: Know exactly what's working
5. **Systematic Research**: Test everything, keep what works
6. **Risk Management**: Always hedge, never naked exposure

---

## Next Steps

1. **This Week**: Implement portfolio optimizer + simple backtester
2. **Week 2**: Integrate portfolio optimization into trading loop
3. **Week 3**: Build comprehensive backtesting framework
4. **Week 4**: Start multi-strategy experiments

Start with portfolio optimization - it's the biggest gap and highest impact.
