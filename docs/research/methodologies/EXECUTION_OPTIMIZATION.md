# Execution Optimization Methodology

## Overview

Execution cost modeling is critical for realistic backtesting and position sizing. This document describes the Almgren-Chriss framework and execution cost models implemented in Nova Aetus.

## Execution Costs

Execution costs can erode **20-40% of alpha** in high-frequency strategies. Components:

1. **Commission**: Fixed per-trade fee
2. **Spread Cost**: Half of bid-ask spread
3. **Market Impact**: Temporary price impact from order
4. **Slippage**: Difference between expected and actual execution price

## Almgren-Chriss Framework

The Almgren-Chriss framework minimizes execution cost + risk:

```
Cost = Market Impact + Risk Cost
```

### Market Impact

Market impact follows square-root law:

```
Impact = coefficient * sqrt(participation_rate) * price * size
```

Where:
- `participation_rate = order_size / avg_daily_volume`

### Optimal Execution Schedule

The framework calculates optimal trade schedule to minimize cost + risk:

```python
from nova.strategy.execution_cost import ExecutionCostModel

model = ExecutionCostModel()
schedule, total_cost = model.calculate_almgren_chriss_optimal(
    order_size=1000,
    current_price=100.0,
    volatility=0.20,
    risk_aversion=1.0,
    time_horizon=1.0  # days
)
```

## Slippage Models

Slippage increases with:
- Higher participation rate
- Higher urgency
- Market orders vs limit orders

### Square-Root Law

```
Slippage = sqrt(participation_rate) * base_rate
```

## VWAP/TWAP Execution

### VWAP (Volume-Weighted Average Price)

Executes proportionally to expected volume profile:

```python
schedule = model.calculate_vwap_schedule(
    order_size=1000,
    volume_profile=expected_volumes,
    time_steps=390  # minutes
)
```

### TWAP (Time-Weighted Average Price)

Equal distribution over time:

```python
schedule = model.calculate_twap_schedule(
    order_size=1000,
    time_steps=390
)
```

## Usage

```python
from nova.strategy.execution_cost import ExecutionCostModel, OrderType

# Initialize model
cost_model = ExecutionCostModel(
    commission_rate=0.001,  # 0.1%
    spread_bps=5.0,  # 5 basis points
)

# Estimate cost
cost = cost_model.estimate_cost(
    order_size=1000,
    current_price=100.0,
    avg_daily_volume=1000000,
    urgency=0.5,
    order_type=OrderType.MARKET
)

print(f"Total cost: ${cost.total_cost:.2f}")
print(f"Slippage: ${cost.slippage:.2f}")
print(f"Market impact: ${cost.market_impact:.2f}")
```

## Cost Attribution

Costs are attributed to P&L:

```python
attribution = cost_model.get_cost_attribution(
    pnl=1000.0,
    start_date=start_date,
    end_date=end_date
)

print(f"Gross P&L: ${attribution['gross_pnl']:.2f}")
print(f"Execution cost: ${attribution['total_execution_cost']:.2f}")
print(f"Net P&L: ${attribution['net_pnl']:.2f}")
print(f"Cost as % of gross: {attribution['cost_as_pct_of_gross']:.2f}%")
```

## Integration with Execution Engine

The execution engine automatically estimates costs:

```python
from nova.strategy.execution import ExecutionEngine

engine = ExecutionEngine(execution_cost_model=cost_model)

# Execute order with cost estimation
result = await engine.execute_order(
    symbol="AAPL",
    quantity=100,
    side="buy",
    order_type="market",
    current_price=150.0,
    avg_daily_volume=50000000,
    urgency=0.5
)

if "execution_cost_estimate" in result:
    print(f"Estimated cost: ${result['execution_cost_estimate']['estimated_cost']:.2f}")
```

## References

- Almgren, R., & Chriss, N. (2000). "Optimal execution of portfolio transactions"
- "Execution Costs in Financial Markets" (Academic research)
