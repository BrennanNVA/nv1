"""Execution cost modeling for realistic backtesting and position sizing.

Research-backed implementation based on:
- Almgren-Chriss optimal execution framework
- Square-root law for market impact
- Participation rate-based slippage models
- VWAP/TWAP execution strategies

Key features:
- Slippage estimation
- Market impact modeling
- Transaction cost tracking
- Cost attribution to P&L
- Integration with execution engine

Research Finding: Execution costs can erode 20-40% of alpha in high-frequency strategies.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order execution type."""

    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"


@dataclass
class ExecutionCost:
    """Execution cost breakdown."""

    total_cost: float  # Total cost in dollars
    slippage: float  # Slippage cost
    market_impact: float  # Market impact cost
    commission: float  # Commission/fees
    spread_cost: float  # Bid-ask spread cost
    timestamp: datetime
    order_size: float  # Shares
    order_value: float  # Dollar value
    participation_rate: float  # % of ADV

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_cost": self.total_cost,
            "slippage": self.slippage,
            "market_impact": self.market_impact,
            "commission": self.commission,
            "spread_cost": self.spread_cost,
            "timestamp": self.timestamp.isoformat(),
            "order_size": self.order_size,
            "order_value": self.order_value,
            "participation_rate": self.participation_rate,
        }


class ExecutionCostModel:
    """Model execution costs for realistic backtesting and position sizing.

    Implements:
    - Almgren-Chriss optimal execution
    - Square-root law for market impact
    - Participation rate-based slippage
    - Transaction cost tracking
    """

    def __init__(
        self,
        commission_rate: float = 0.001,  # 0.1% commission
        spread_bps: float = 5.0,  # 5 basis points spread
        market_impact_coefficient: float = 0.1,
        permanent_impact_coefficient: float = 0.05,
        volatility_annualized: float = 0.20,  # 20% annual vol
    ) -> None:
        """
        Initialize execution cost model.

        Args:
            commission_rate: Commission rate (default: 0.1%)
            spread_bps: Bid-ask spread in basis points (default: 5 bps)
            market_impact_coefficient: Temporary market impact coefficient
            permanent_impact_coefficient: Permanent market impact coefficient
            volatility_annualized: Annualized volatility for Almgren-Chriss
        """
        self.commission_rate = commission_rate
        self.spread_bps = spread_bps
        self.market_impact_coefficient = market_impact_coefficient
        self.permanent_impact_coefficient = permanent_impact_coefficient
        self.volatility_annualized = volatility_annualized

        # Track execution costs
        self.execution_history: list[ExecutionCost] = []

        logger.info(
            f"ExecutionCostModel initialized: commission={commission_rate:.4f}, "
            f"spread={spread_bps}bps, impact_coef={market_impact_coefficient}"
        )

    def estimate_cost(
        self,
        order_size: float,
        current_price: float,
        avg_daily_volume: float,
        urgency: float = 0.5,  # 0 = passive, 1 = aggressive
        order_type: OrderType = OrderType.MARKET,
    ) -> ExecutionCost:
        """
        Estimate execution cost for an order.

        Args:
            order_size: Number of shares to trade
            current_price: Current market price
            avg_daily_volume: Average daily volume (ADV)
            urgency: Execution urgency (0-1)
            order_type: Type of order

        Returns:
            ExecutionCost object with cost breakdown
        """
        order_value = order_size * current_price

        # Participation rate (% of ADV)
        participation_rate = order_size / avg_daily_volume if avg_daily_volume > 0 else 0.0

        # Commission
        commission = order_value * self.commission_rate

        # Spread cost (half spread, paid once)
        spread_cost = order_value * (self.spread_bps / 10000) * 0.5

        # Market impact (temporary)
        market_impact = self._calculate_market_impact(
            order_size=order_size,
            current_price=current_price,
            avg_daily_volume=avg_daily_volume,
            urgency=urgency,
            order_type=order_type,
        )

        # Slippage (based on participation rate and urgency)
        slippage = self._calculate_slippage(
            order_size=order_size,
            current_price=current_price,
            avg_daily_volume=avg_daily_volume,
            participation_rate=participation_rate,
            urgency=urgency,
            order_type=order_type,
        )

        total_cost = commission + spread_cost + market_impact + slippage

        cost = ExecutionCost(
            total_cost=total_cost,
            slippage=slippage,
            market_impact=market_impact,
            commission=commission,
            spread_cost=spread_cost,
            timestamp=datetime.now(),
            order_size=order_size,
            order_value=order_value,
            participation_rate=participation_rate,
        )

        # Store in history
        self.execution_history.append(cost)

        return cost

    def _calculate_market_impact(
        self,
        order_size: float,
        current_price: float,
        avg_daily_volume: float,
        urgency: float,
        order_type: OrderType,
    ) -> float:
        """
        Calculate market impact using square-root law and Almgren-Chriss.

        Market impact = coefficient * sqrt(participation_rate) * price * size

        Args:
            order_size: Number of shares
            current_price: Current price
            avg_daily_volume: Average daily volume
            urgency: Execution urgency
            order_type: Order type

        Returns:
            Market impact cost in dollars
        """
        if avg_daily_volume == 0:
            return 0.0

        participation_rate = order_size / avg_daily_volume

        # Square-root law: impact ~ sqrt(participation_rate)
        sqrt_participation = np.sqrt(participation_rate)

        # Adjust for urgency (more urgent = higher impact)
        urgency_factor = 0.5 + 0.5 * urgency

        # Adjust for order type
        if order_type == OrderType.MARKET:
            type_factor = 1.0  # Full impact
        elif order_type == OrderType.LIMIT:
            type_factor = 0.5  # Reduced impact (may not fill)
        elif order_type == OrderType.VWAP:
            type_factor = 0.7  # VWAP reduces impact
        elif order_type == OrderType.TWAP:
            type_factor = 0.6  # TWAP reduces impact
        else:
            type_factor = 1.0

        # Market impact
        impact = (
            self.market_impact_coefficient
            * sqrt_participation
            * current_price
            * order_size
            * urgency_factor
            * type_factor
        )

        return float(impact)

    def _calculate_slippage(
        self,
        order_size: float,
        current_price: float,
        avg_daily_volume: float,
        participation_rate: float,
        urgency: float,
        order_type: OrderType,
    ) -> float:
        """
        Calculate slippage based on participation rate and urgency.

        Slippage increases with:
        - Higher participation rate
        - Higher urgency
        - Market orders vs limit orders

        Args:
            order_size: Number of shares
            current_price: Current price
            avg_daily_volume: Average daily volume
            participation_rate: Participation rate (% of ADV)
            urgency: Execution urgency
            order_type: Order type

        Returns:
            Slippage cost in dollars
        """
        # Base slippage from participation rate
        # Square-root law: slippage ~ sqrt(participation_rate)
        base_slippage_rate = np.sqrt(participation_rate) * 0.01  # 1% base

        # Adjust for urgency
        urgency_slippage = urgency * 0.005  # Up to 0.5% additional

        # Adjust for order type
        if order_type == OrderType.MARKET:
            type_slippage = 0.002  # Market orders have slippage
        elif order_type == OrderType.LIMIT:
            type_slippage = 0.0  # Limit orders avoid slippage (but may not fill)
        elif order_type == OrderType.VWAP:
            type_slippage = 0.001  # VWAP reduces slippage
        elif order_type == OrderType.TWAP:
            type_slippage = 0.001  # TWAP reduces slippage
        else:
            type_slippage = 0.002

        total_slippage_rate = base_slippage_rate + urgency_slippage + type_slippage

        slippage = current_price * order_size * total_slippage_rate

        return float(slippage)

    def calculate_almgren_chriss_optimal(
        self,
        order_size: float,
        current_price: float,
        volatility: Optional[float] = None,
        risk_aversion: float = 1.0,
        time_horizon: float = 1.0,  # Days
    ) -> tuple[np.ndarray, float]:
        """
        Calculate optimal execution trajectory using Almgren-Chriss framework.

        Returns optimal trade schedule to minimize cost + risk.

        Args:
            order_size: Total shares to trade
            current_price: Current price
            volatility: Volatility (or use default)
            risk_aversion: Risk aversion parameter
            time_horizon: Time horizon in days

        Returns:
            Tuple of (trade_schedule, total_cost)
        """
        if volatility is None:
            volatility = self.volatility_annualized / np.sqrt(252)  # Daily vol

        # Almgren-Chriss parameters
        eta = self.market_impact_coefficient  # Temporary impact
        gamma = self.permanent_impact_coefficient  # Permanent impact
        lambda_param = risk_aversion  # Risk aversion

        # Optimal execution rate (simplified)
        # More complex implementation would solve the optimal control problem
        # For now, use exponential decay schedule
        n_steps = int(time_horizon * 390)  # 390 minutes per trading day
        t = np.linspace(0, time_horizon, n_steps)

        # Exponential decay schedule (aggressive early, passive later)
        decay_rate = lambda_param * volatility / eta
        schedule = order_size * np.exp(-decay_rate * t)
        schedule = np.diff(np.concatenate([[order_size], schedule]))
        schedule = np.maximum(schedule, 0)  # Ensure non-negative

        # Calculate total cost
        # Simplified: cost = market_impact + risk_cost
        market_impact_cost = eta * np.sum(schedule**2) / current_price
        risk_cost = lambda_param * volatility**2 * np.sum(schedule * t) / current_price

        total_cost = market_impact_cost + risk_cost

        return schedule, float(total_cost)

    def calculate_vwap_schedule(
        self,
        order_size: float,
        volume_profile: np.ndarray,  # Volume at each time step
        time_steps: int = 390,  # Minutes in trading day
    ) -> np.ndarray:
        """
        Calculate VWAP execution schedule.

        Args:
            order_size: Total shares to trade
            volume_profile: Expected volume at each time step
            time_steps: Number of time steps

        Returns:
            Trade schedule (shares per time step)
        """
        if len(volume_profile) != time_steps:
            # Interpolate if needed
            volume_profile = np.interp(
                np.linspace(0, len(volume_profile) - 1, time_steps),
                np.arange(len(volume_profile)),
                volume_profile,
            )

        # Normalize volume profile
        total_volume = np.sum(volume_profile)
        if total_volume == 0:
            # Equal distribution if no volume profile
            return np.full(time_steps, order_size / time_steps)

        # Allocate proportionally to volume
        schedule = order_size * volume_profile / total_volume

        return schedule

    def calculate_twap_schedule(
        self,
        order_size: float,
        time_steps: int = 390,
    ) -> np.ndarray:
        """
        Calculate TWAP (Time-Weighted Average Price) execution schedule.

        Equal distribution over time.

        Args:
            order_size: Total shares to trade
            time_steps: Number of time steps

        Returns:
            Trade schedule (shares per time step)
        """
        return np.full(time_steps, order_size / time_steps)

    def get_total_costs(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> dict[str, float]:
        """
        Get total execution costs for a time period.

        Args:
            start_date: Start date (None = all time)
            end_date: End date (None = all time)

        Returns:
            Dictionary with cost breakdown
        """
        costs = self.execution_history

        if start_date:
            costs = [c for c in costs if c.timestamp >= start_date]
        if end_date:
            costs = [c for c in costs if c.timestamp <= end_date]

        if not costs:
            return {
                "total_cost": 0.0,
                "total_slippage": 0.0,
                "total_market_impact": 0.0,
                "total_commission": 0.0,
                "total_spread_cost": 0.0,
                "total_value": 0.0,
                "num_trades": 0,
            }

        return {
            "total_cost": sum(c.total_cost for c in costs),
            "total_slippage": sum(c.slippage for c in costs),
            "total_market_impact": sum(c.market_impact for c in costs),
            "total_commission": sum(c.commission for c in costs),
            "total_spread_cost": sum(c.spread_cost for c in costs),
            "total_value": sum(c.order_value for c in costs),
            "num_trades": len(costs),
        }

    def get_cost_attribution(
        self, pnl: float, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> dict[str, float]:
        """
        Attribute costs to P&L.

        Args:
            pnl: Gross P&L before costs
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with cost attribution
        """
        costs = self.get_total_costs(start_date, end_date)
        total_cost = costs["total_cost"]
        net_pnl = pnl - total_cost

        return {
            "gross_pnl": pnl,
            "total_execution_cost": total_cost,
            "net_pnl": net_pnl,
            "cost_as_pct_of_gross": (total_cost / abs(pnl) * 100) if pnl != 0 else 0.0,
            "cost_as_pct_of_value": (
                (total_cost / costs["total_value"] * 100) if costs["total_value"] > 0 else 0.0
            ),
            **costs,
        }
