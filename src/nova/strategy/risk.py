"""Position sizing with Kelly Criterion, circuit breakers, and drawdown control.

Research-backed implementation based on:
- "Risk-Constrained Kelly Gambling" (Busseti, Ryu, Boyd, 2016)
- "Distributionally Robust Kelly Gambling" (Sun, Boyd, 2018)
- "Data-driven drawdown control with restart mechanism" (arXiv, 2023)
- SEC Rule 15c3-5 (Market Access Rule) for circuit breaker design

Key formulas:
- Kelly fraction: f* = (π * P - (1-π)) / (P - 1)
- Fractional Kelly: f = c * f* where c ∈ (0.25, 0.5) for safety
- Drawdown constraint: Prob(min wealth < α) ≤ β
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from ..core.config import RiskConfig

logger = logging.getLogger(__name__)


class CircuitBreakerLevel(Enum):
    """Circuit breaker severity levels."""

    NORMAL = "normal"
    WARN = "warn"
    SOFT_HALT = "soft_halt"
    HARD_HALT = "hard_halt"


@dataclass
class CircuitBreakerState:
    """Current state of circuit breaker."""

    level: CircuitBreakerLevel = CircuitBreakerLevel.NORMAL
    triggered_at: Optional[datetime] = None
    trigger_reason: str = ""
    error_count: int = 0
    consecutive_losses: int = 0
    daily_loss_realized: float = 0.0
    daily_loss_unrealized: float = 0.0

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed."""
        return self.level in [CircuitBreakerLevel.NORMAL, CircuitBreakerLevel.WARN]

    def reset(self) -> None:
        """Reset circuit breaker to normal state."""
        self.level = CircuitBreakerLevel.NORMAL
        self.triggered_at = None
        self.trigger_reason = ""


@dataclass
class KellyResult:
    """Result of Kelly criterion calculation."""

    optimal_fraction: float
    fractional_kelly: float
    position_size: int
    risk_amount: float
    win_probability: float
    payoff_ratio: float
    edge: float


class RiskManager:
    """Institutional-grade risk management with Kelly criterion and circuit breakers.

    Implements:
    - Kelly criterion position sizing (with fractional Kelly safety)
    - Multi-level circuit breaker system
    - Drawdown-constrained optimization
    - Real-time risk monitoring
    - ATR-based stop loss with trailing stops
    """

    def __init__(
        self,
        config: RiskConfig,
        kelly_fraction: float = 0.25,
    ) -> None:
        """
        Initialize risk manager.

        Args:
            config: Risk management configuration
            kelly_fraction: Kelly fraction multiplier (0.25-0.5 recommended)
        """
        self.config = config
        self.kelly_fraction = kelly_fraction

        # Equity tracking
        self.peak_equity: float = 0.0
        self.last_equity_check: Optional[datetime] = None
        self.starting_equity: float = 0.0

        # Circuit breaker state
        self.circuit_breaker = CircuitBreakerState()
        self.error_timestamps: deque = deque(maxlen=100)
        self.loss_streak: int = 0

        # Daily tracking
        self.daily_pnl: float = 0.0
        self.last_daily_reset: Optional[datetime] = None

        # Trade history for Kelly estimation
        self.trade_results: deque = deque(maxlen=100)

        logger.info(f"RiskManager initialized with Kelly fraction={kelly_fraction}")

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        capital: float,
    ) -> int:
        """
        Calculate position size based on risk per trade (fixed fractional).

        Args:
            entry_price: Entry price per share
            stop_loss: Stop loss price per share
            capital: Available capital

        Returns:
            Number of shares to purchase
        """
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share == 0:
            logger.warning("Risk per share is zero, returning minimum position")
            return 1

        max_risk_amount = capital * self.config.risk_per_trade_pct
        shares = int(max_risk_amount / risk_per_share)

        # Enforce maximum position size
        max_position_value = capital * self.config.max_position_size_pct
        max_shares_by_position = int(max_position_value / entry_price)
        shares = min(shares, max_shares_by_position)

        # Minimum 1 share
        shares = max(1, shares)

        logger.debug(
            f"Position size: {shares} shares "
            f"(risk: ${risk_per_share * shares:.2f}, "
            f"value: ${entry_price * shares:.2f})"
        )

        return shares

    # ========== KELLY CRITERION POSITION SIZING ==========

    def calculate_kelly_fraction(
        self,
        win_probability: float,
        payoff_ratio: float,
    ) -> float:
        """
        Calculate optimal Kelly fraction.

        Formula: f* = (p * b - q) / b
        Where: p = win probability, q = 1 - p, b = payoff ratio

        Args:
            win_probability: Probability of winning (0-1)
            payoff_ratio: Ratio of average win to average loss

        Returns:
            Optimal Kelly fraction (can be negative if negative edge)
        """
        if payoff_ratio <= 0:
            return 0.0

        q = 1 - win_probability

        # Kelly formula: f* = (p * b - q) / b
        kelly = (win_probability * payoff_ratio - q) / payoff_ratio

        return kelly

    def calculate_kelly_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        capital: float,
        win_probability: float,
        use_fractional: bool = True,
    ) -> KellyResult:
        """
        Calculate position size using Kelly criterion.

        Based on "Risk-Constrained Kelly Gambling" (Stanford, 2016).
        Uses fractional Kelly for safety against estimation error.

        Args:
            entry_price: Entry price per share
            stop_loss: Stop loss price
            take_profit: Take profit price
            capital: Available capital
            win_probability: Estimated win probability (from model)
            use_fractional: Apply fractional Kelly (recommended)

        Returns:
            KellyResult with position sizing details
        """
        # Calculate payoff ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        if risk <= 0:
            logger.warning("Invalid risk value for Kelly calculation")
            return KellyResult(
                optimal_fraction=0.0,
                fractional_kelly=0.0,
                position_size=0,
                risk_amount=0.0,
                win_probability=win_probability,
                payoff_ratio=0.0,
                edge=0.0,
            )

        payoff_ratio = reward / risk

        # Calculate Kelly fraction
        kelly = self.calculate_kelly_fraction(win_probability, payoff_ratio)

        # Calculate edge (expected value per unit risk)
        edge = win_probability * payoff_ratio - (1 - win_probability)

        # Apply fractional Kelly for safety
        if use_fractional:
            effective_kelly = kelly * self.kelly_fraction
        else:
            effective_kelly = kelly

        # Clamp to reasonable bounds
        effective_kelly = max(0.0, min(effective_kelly, self.config.max_position_size_pct))

        # Calculate position size
        if effective_kelly > 0:
            risk_amount = capital * effective_kelly
            shares = int(risk_amount / risk)

            # Apply position limits
            max_shares = int(capital * self.config.max_position_size_pct / entry_price)
            shares = min(shares, max_shares)
            shares = max(1, shares) if shares > 0 else 0
        else:
            risk_amount = 0.0
            shares = 0
            logger.info(f"Kelly suggests no position (negative edge: {edge:.3f})")

        result = KellyResult(
            optimal_fraction=kelly,
            fractional_kelly=effective_kelly,
            position_size=shares,
            risk_amount=risk_amount,
            win_probability=win_probability,
            payoff_ratio=payoff_ratio,
            edge=edge,
        )

        logger.debug(
            f"Kelly: f*={kelly:.3f}, f={effective_kelly:.3f}, " f"shares={shares}, edge={edge:.3f}"
        )

        return result

    def estimate_win_probability_from_history(self) -> float:
        """
        Estimate win probability from trade history.

        Returns:
            Estimated win probability (defaults to 0.5 if insufficient data)
        """
        if len(self.trade_results) < 10:
            return 0.5  # Default to neutral

        wins = sum(1 for pnl in self.trade_results if pnl > 0)
        return wins / len(self.trade_results)

    def estimate_payoff_ratio_from_history(self) -> float:
        """
        Estimate payoff ratio from trade history.

        Returns:
            Average win / average loss ratio
        """
        if len(self.trade_results) < 10:
            return 1.5  # Default assumption

        wins = [pnl for pnl in self.trade_results if pnl > 0]
        losses = [abs(pnl) for pnl in self.trade_results if pnl < 0]

        if not losses or not wins:
            return 1.5

        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)

        if avg_loss == 0:
            return 1.5

        return avg_win / avg_loss

    def record_trade_result(self, pnl: float) -> None:
        """
        Record a trade result for Kelly estimation.

        Args:
            pnl: Trade profit/loss
        """
        self.trade_results.append(pnl)

        # Update loss streak
        if pnl < 0:
            self.loss_streak += 1
            self.circuit_breaker.consecutive_losses += 1
        else:
            self.loss_streak = 0
            self.circuit_breaker.consecutive_losses = 0

        # Update daily P&L
        self.daily_pnl += pnl
        self.circuit_breaker.daily_loss_realized += pnl if pnl < 0 else 0

    def calculate_atr_stop_loss(
        self,
        entry_price: float,
        atr_value: float,
        is_long: bool = True,
    ) -> float:
        """
        Calculate ATR-based stop loss.

        Args:
            entry_price: Entry price
            atr_value: Current ATR value
            is_long: True for long position, False for short

        Returns:
            Stop loss price
        """
        multiplier = self.config.atr_stop_multiplier

        if is_long:
            stop_loss = entry_price - (atr_value * multiplier)
        else:
            stop_loss = entry_price + (atr_value * multiplier)

        logger.debug(
            f"ATR stop loss: ${stop_loss:.2f} " f"(ATR: {atr_value:.2f}, multiplier: {multiplier})"
        )

        return stop_loss

    def update_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        current_stop_loss: float,
        is_long: bool = True,
    ) -> float:
        """
        Update trailing stop loss.

        Args:
            current_price: Current market price
            entry_price: Original entry price
            current_stop_loss: Current stop loss price
            is_long: True for long position, False for short

        Returns:
            Updated stop loss price
        """
        if not self.config.trailing_stop_enabled:
            return current_stop_loss

        trailing_distance = current_price * self.config.trailing_stop_pct

        if is_long:
            new_stop_loss = current_price - trailing_distance
            # Trailing stop only moves up
            stop_loss = max(current_stop_loss, new_stop_loss)
        else:
            new_stop_loss = current_price + trailing_distance
            # Trailing stop only moves down
            stop_loss = min(current_stop_loss, new_stop_loss)

        if stop_loss != current_stop_loss:
            logger.debug(f"Trailing stop updated: ${stop_loss:.2f}")

        return stop_loss

    def check_drawdown(self, current_equity: float) -> bool:
        """
        Check if drawdown limit is exceeded.

        Args:
            current_equity: Current portfolio equity

        Returns:
            True if trading can continue, False if drawdown limit exceeded
        """
        now = datetime.now()

        # Check if we need to update peak equity
        if (
            self.last_equity_check is None
            or (now - self.last_equity_check).total_seconds()
            >= self.config.peak_equity_check_interval
        ):
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            self.last_equity_check = now

        if self.peak_equity == 0:
            return True

        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        if drawdown > self.config.max_drawdown_pct:
            logger.error(
                f"Maximum drawdown exceeded: {drawdown:.2%} "
                f"(limit: {self.config.max_drawdown_pct:.2%})"
            )
            return False

        return True

    def validate_portfolio_risk(
        self,
        current_positions_value: float,
        new_position_value: float,
        total_capital: float,
    ) -> bool:
        """
        Validate that adding a new position doesn't exceed portfolio risk limits.

        Args:
            current_positions_value: Value of current positions
            new_position_value: Value of new position to add
            total_capital: Total portfolio capital

        Returns:
            True if risk limits are within bounds
        """
        total_exposure = current_positions_value + new_position_value
        exposure_pct = total_exposure / total_capital if total_capital > 0 else 0.0

        if exposure_pct > self.config.max_portfolio_risk_pct:
            logger.warning(
                f"Portfolio risk limit exceeded: {exposure_pct:.2%} "
                f"(limit: {self.config.max_portfolio_risk_pct:.2%})"
            )
            return False

        return True

    def reset_peak_equity(self) -> None:
        """Reset peak equity tracking."""
        self.peak_equity = 0.0
        self.last_equity_check = None
        logger.debug("Peak equity reset")

    # ========== MULTI-LEVEL CIRCUIT BREAKER SYSTEM ==========

    def record_error(self, error_type: str = "general") -> None:
        """
        Record an error for circuit breaker monitoring.

        Args:
            error_type: Type of error for logging
        """
        now = datetime.now()
        self.error_timestamps.append(now)

        # Clean old errors outside window
        cutoff = now - timedelta(seconds=self.config.error_window_seconds)
        while self.error_timestamps and self.error_timestamps[0] < cutoff:
            self.error_timestamps.popleft()

        self.circuit_breaker.error_count = len(self.error_timestamps)

        logger.warning(
            f"Error recorded ({error_type}): {self.circuit_breaker.error_count} in window"
        )

        # Check if we should trigger circuit breaker
        self._check_circuit_breaker_triggers()

    def _check_circuit_breaker_triggers(self) -> None:
        """Check all circuit breaker trigger conditions."""
        now = datetime.now()

        # Level 3 (HARD_HALT) triggers
        if self.circuit_breaker.error_count >= self.config.max_errors_per_minute:
            self._trigger_circuit_breaker(
                CircuitBreakerLevel.HARD_HALT,
                f"Error rate exceeded: {self.circuit_breaker.error_count}/min",
            )
            return

        if self.circuit_breaker.consecutive_losses >= 5:
            self._trigger_circuit_breaker(
                CircuitBreakerLevel.HARD_HALT,
                f"Consecutive losses: {self.circuit_breaker.consecutive_losses}",
            )
            return

        # Level 2 (SOFT_HALT) triggers
        if (
            abs(self.circuit_breaker.daily_loss_realized)
            > self.starting_equity * self.config.max_drawdown_pct
        ):
            self._trigger_circuit_breaker(
                CircuitBreakerLevel.SOFT_HALT,
                f"Daily loss limit: ${abs(self.circuit_breaker.daily_loss_realized):.2f}",
            )
            return

        if self.circuit_breaker.consecutive_losses >= 3:
            self._trigger_circuit_breaker(
                CircuitBreakerLevel.SOFT_HALT,
                f"Loss streak warning: {self.circuit_breaker.consecutive_losses}",
            )
            return

        # Level 1 (WARN) triggers
        if self.circuit_breaker.error_count >= self.config.max_errors_per_minute // 2:
            self._trigger_circuit_breaker(
                CircuitBreakerLevel.WARN,
                f"Error rate warning: {self.circuit_breaker.error_count}/min",
            )
            return

    def _trigger_circuit_breaker(
        self,
        level: CircuitBreakerLevel,
        reason: str,
    ) -> None:
        """
        Trigger circuit breaker at specified level.

        Args:
            level: Circuit breaker level to trigger
            reason: Reason for triggering
        """
        # Only escalate, never de-escalate automatically
        if level.value >= self.circuit_breaker.level.value:
            self.circuit_breaker.level = level
            self.circuit_breaker.triggered_at = datetime.now()
            self.circuit_breaker.trigger_reason = reason

            logger.error(f"CIRCUIT BREAKER {level.value.upper()}: {reason}")

    def check_can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed under current circuit breaker state.

        Returns:
            Tuple of (can_trade, reason)
        """
        if not self.circuit_breaker.is_trading_allowed():
            return False, f"Circuit breaker active: {self.circuit_breaker.trigger_reason}"

        return True, "OK"

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """
        Get current circuit breaker status.

        Returns:
            Dictionary with circuit breaker state
        """
        return {
            "level": self.circuit_breaker.level.value,
            "is_trading_allowed": self.circuit_breaker.is_trading_allowed(),
            "triggered_at": (
                self.circuit_breaker.triggered_at.isoformat()
                if self.circuit_breaker.triggered_at
                else None
            ),
            "trigger_reason": self.circuit_breaker.trigger_reason,
            "error_count": self.circuit_breaker.error_count,
            "consecutive_losses": self.circuit_breaker.consecutive_losses,
            "daily_loss_realized": self.circuit_breaker.daily_loss_realized,
        }

    def reset_circuit_breaker(self, force: bool = False) -> bool:
        """
        Attempt to reset circuit breaker to normal.

        Args:
            force: Force reset even if conditions not met

        Returns:
            True if reset successful
        """
        if self.circuit_breaker.level == CircuitBreakerLevel.HARD_HALT and not force:
            logger.warning("Cannot auto-reset HARD_HALT - use force=True")
            return False

        self.circuit_breaker.reset()
        self.error_timestamps.clear()

        logger.info("Circuit breaker reset to NORMAL")
        return True

    def reset_daily_tracking(self) -> None:
        """Reset daily P&L and loss tracking (call at market open)."""
        now = datetime.now()

        # Only reset if it's a new day
        if self.last_daily_reset is None or self.last_daily_reset.date() != now.date():
            self.daily_pnl = 0.0
            self.circuit_breaker.daily_loss_realized = 0.0
            self.circuit_breaker.daily_loss_unrealized = 0.0
            self.last_daily_reset = now

            # Also consider resetting soft halt if from previous day
            if self.circuit_breaker.level == CircuitBreakerLevel.SOFT_HALT:
                self.reset_circuit_breaker(force=True)

            logger.info("Daily risk tracking reset")

    # ========== DRAWDOWN-CONSTRAINED POSITION SIZING ==========

    def calculate_drawdown_adjusted_size(
        self,
        base_position: int,
        current_equity: float,
    ) -> int:
        """
        Adjust position size based on current drawdown.

        Based on "Data-driven drawdown control" research.
        Reduces exposure as drawdown increases.

        Args:
            base_position: Original calculated position size
            current_equity: Current portfolio equity

        Returns:
            Adjusted position size
        """
        if self.peak_equity == 0:
            self.peak_equity = current_equity
            return base_position

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calculate current drawdown
        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        # Scale position based on drawdown
        # As drawdown approaches max, reduce position to near zero
        max_dd = self.config.max_drawdown_pct

        if drawdown >= max_dd:
            # At or beyond max drawdown - no new positions
            logger.warning(f"Max drawdown reached ({drawdown:.1%}), blocking new positions")
            return 0

        # Linear scaling: full position at 0% DD, zero at max DD
        scale_factor = 1 - (drawdown / max_dd)

        # Apply scale with floor at 50%
        scale_factor = max(0.5, scale_factor)

        adjusted = int(base_position * scale_factor)

        if adjusted < base_position:
            logger.debug(
                f"Drawdown adjustment: {base_position} -> {adjusted} "
                f"(DD: {drawdown:.1%}, scale: {scale_factor:.2f})"
            )

        return adjusted

    def set_starting_equity(self, equity: float) -> None:
        """
        Set starting equity for daily loss calculations.

        Args:
            equity: Starting equity value
        """
        self.starting_equity = equity
        if self.peak_equity == 0:
            self.peak_equity = equity
        logger.info(f"Starting equity set: ${equity:,.2f}")

    # ========== COMPREHENSIVE POSITION EVALUATION ==========

    def evaluate_trade(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        capital: float,
        win_probability: float,
        current_equity: float,
    ) -> dict[str, Any]:
        """
        Comprehensive trade evaluation with all risk checks.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            capital: Available capital
            win_probability: Model's win probability estimate
            current_equity: Current portfolio equity

        Returns:
            Dictionary with trade evaluation and recommended position
        """
        # Check if trading is allowed
        can_trade, reason = self.check_can_trade()

        if not can_trade:
            return {
                "approved": False,
                "reason": reason,
                "position_size": 0,
            }

        # Calculate Kelly-based position
        kelly_result = self.calculate_kelly_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            capital=capital,
            win_probability=win_probability,
        )

        # Check if edge is positive
        if kelly_result.edge <= 0:
            return {
                "approved": False,
                "reason": f"Negative edge: {kelly_result.edge:.3f}",
                "position_size": 0,
                "kelly_result": kelly_result,
            }

        # Apply drawdown adjustment
        adjusted_size = self.calculate_drawdown_adjusted_size(
            kelly_result.position_size,
            current_equity,
        )

        # Validate portfolio risk
        position_value = adjusted_size * entry_price
        current_exposure = 0  # Would come from portfolio tracking

        if not self.validate_portfolio_risk(current_exposure, position_value, capital):
            adjusted_size = int(adjusted_size * 0.5)  # Reduce by half
            logger.warning("Portfolio risk limit - reducing position")

        return {
            "approved": adjusted_size > 0,
            "reason": "OK" if adjusted_size > 0 else "Position too small",
            "position_size": adjusted_size,
            "kelly_result": kelly_result,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_per_share": abs(entry_price - stop_loss),
            "reward_per_share": abs(take_profit - entry_price),
            "risk_reward_ratio": (
                abs(take_profit - entry_price) / abs(entry_price - stop_loss)
                if abs(entry_price - stop_loss) > 0
                else 0
            ),
        }
