"""Portfolio optimization for swing trading (PDT-safe).

Research-backed implementation based on:
- Mean-Variance Optimization (Markowitz, 1952)
- Risk Parity (Equal Risk Contribution)
- Kelly Criterion Portfolio
- Minimum Variance Portfolio
- Black-Litterman Model (for expected returns)

Key features:
- Correlation-aware position sizing
- Diversification across positions
- Risk-adjusted portfolio construction
- Swing trading compatible (multi-day holds)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization method."""

    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    KELLY = "kelly"
    MIN_VARIANCE = "min_variance"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class PortfolioWeights:
    """Optimized portfolio weights."""

    weights: dict[str, float]  # Symbol -> weight (0 to 1)
    expected_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    method: str
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "weights": self.weights,
            "expected_return": self.expected_return,
            "portfolio_volatility": self.portfolio_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "method": self.method,
            "timestamp": self.timestamp.isoformat(),
        }


class PortfolioOptimizer:
    """Portfolio optimizer for swing trading strategies.

    Implements multiple optimization methods:
    - Mean-Variance: Maximize Sharpe ratio
    - Risk Parity: Equal risk contribution
    - Hierarchical Risk Parity: Robust clustering-based allocation
    - Kelly: Optimal growth
    - Minimum Variance: Minimize risk
    """

    def __init__(
        self,
        method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
        risk_aversion: float = 1.0,
        max_position_weight: float = 0.20,  # 20% max per position
        min_position_weight: float = 0.0,
        long_only: bool = True,
    ) -> None:
        """
        Initialize portfolio optimizer.

        Args:
            method: Optimization method
            risk_aversion: Risk aversion parameter (higher = more risk-averse)
            max_position_weight: Maximum weight per position (default: 20%)
            min_position_weight: Minimum weight per position
            long_only: Only long positions (no shorting)
        """
        self.method = method
        self.risk_aversion = risk_aversion
        self.max_position_weight = max_position_weight
        self.min_position_weight = min_position_weight
        self.long_only = long_only

        logger.info(
            f"PortfolioOptimizer initialized: method={method.value}, "
            f"max_weight={max_position_weight}, long_only={long_only}"
        )

    def optimize(
        self,
        signals: dict[str, float],  # Symbol -> signal strength (-1 to 1)
        returns_covariance: Optional[np.ndarray] = None,
        expected_returns: Optional[np.ndarray] = None,
        historical_returns: Optional[np.ndarray] = None,
        symbols: Optional[list[str]] = None,
        win_rates: Optional[dict[str, float]] = None,
        avg_wins: Optional[dict[str, float]] = None,
        avg_losses: Optional[dict[str, float]] = None,
    ) -> PortfolioWeights:
        """
        Optimize portfolio weights.

        Args:
            signals: Signal strengths for each symbol
            returns_covariance: Covariance matrix of returns (n_symbols x n_symbols)
            expected_returns: Expected returns vector (n_symbols)
            historical_returns: Historical returns matrix (n_periods x n_symbols)
            symbols: List of symbols (required if using covariance matrix)
            win_rates: Win rates for Kelly optimization
            avg_wins: Average win amounts for Kelly optimization
            avg_losses: Average loss amounts for Kelly optimization

        Returns:
            PortfolioWeights object
        """
        if not signals:
            raise ValueError("No signals provided")

        symbols_list = list(signals.keys())
        n_symbols = len(symbols_list)

        if n_symbols == 1:
            # Single position - weight = 1.0
            weights_dict = {symbols_list[0]: 1.0}
            return PortfolioWeights(
                weights=weights_dict,
                expected_return=0.0,
                portfolio_volatility=0.0,
                sharpe_ratio=0.0,
                method=self.method.value,
                timestamp=datetime.now(),
            )

        # Estimate covariance if not provided
        if returns_covariance is None:
            if historical_returns is not None:
                returns_covariance = np.cov(historical_returns.T)
            else:
                # Use simple correlation estimate from signals
                returns_covariance = self._estimate_covariance_from_signals(signals, symbols_list)

        # Estimate expected returns if not provided
        if expected_returns is None:
            expected_returns = self._estimate_returns_from_signals(signals, symbols_list)

        # Optimize based on method
        if self.method == OptimizationMethod.MEAN_VARIANCE:
            weights = self._optimize_mean_variance(
                expected_returns, returns_covariance, symbols_list
            )
        elif self.method == OptimizationMethod.RISK_PARITY:
            weights = self._optimize_risk_parity(returns_covariance, symbols_list)
        elif self.method == OptimizationMethod.HIERARCHICAL_RISK_PARITY:
            weights = self._optimize_hierarchical_risk_parity(returns_covariance, symbols_list)
        elif self.method == OptimizationMethod.KELLY:
            if win_rates and avg_wins and avg_losses:
                weights = self._optimize_kelly(
                    signals, win_rates, avg_wins, avg_losses, symbols_list
                )
            else:
                logger.warning(
                    "Kelly optimization requires win_rates/avg_wins/avg_losses, falling back to mean-variance"
                )
                weights = self._optimize_mean_variance(
                    expected_returns, returns_covariance, symbols_list
                )
        elif self.method == OptimizationMethod.MIN_VARIANCE:
            weights = self._optimize_min_variance(returns_covariance, symbols_list)
        elif self.method == OptimizationMethod.EQUAL_WEIGHT:
            weights = self._optimize_equal_weight(symbols_list)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns_covariance, weights)))
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0

        # Create weights dictionary
        weights_dict = {symbol: float(w) for symbol, w in zip(symbols_list, weights, strict=False)}

        return PortfolioWeights(
            weights=weights_dict,
            expected_return=float(portfolio_return),
            portfolio_volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe),
            method=self.method.value,
            timestamp=datetime.now(),
        )

    def _optimize_mean_variance(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        symbols: list[str],
    ) -> np.ndarray:
        """
        Mean-variance optimization (maximize Sharpe ratio).

        Maximizes: w^T * mu - lambda * w^T * Sigma * w
        where mu = expected returns, Sigma = covariance
        """
        n = len(symbols)

        # Objective: maximize Sharpe ratio = (w^T * mu) / sqrt(w^T * Sigma * w)
        # Or equivalently: maximize w^T * mu - lambda * w^T * Sigma * w
        def objective(w):
            portfolio_return = np.dot(w, expected_returns)
            portfolio_var = np.dot(w, np.dot(covariance, w))
            # Negative because we minimize
            return -(portfolio_return - self.risk_aversion * portfolio_var)

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]  # Weights sum to 1

        # Bounds
        bounds = [(self.min_position_weight, self.max_position_weight) for _ in range(n)]

        # Initial guess (equal weights)
        x0 = np.ones(n) / n

        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if result.success:
                return result.x
            else:
                logger.warning(f"Optimization failed: {result.message}, using equal weights")
                return np.ones(n) / n
        except Exception as e:
            logger.error(f"Mean-variance optimization error: {e}, using equal weights")
            return np.ones(n) / n

    def _optimize_risk_parity(
        self,
        covariance: np.ndarray,
        symbols: list[str],
    ) -> np.ndarray:
        """
        Risk parity optimization (equal risk contribution).

        Each position contributes equally to portfolio risk.
        """
        n = len(symbols)

        def objective(w):
            # Portfolio volatility
            portfolio_vol = np.sqrt(np.dot(w, np.dot(covariance, w)))

            # Risk contribution per asset
            risk_contributions = w * np.dot(covariance, w) / portfolio_vol

            # Minimize variance of risk contributions (equal risk contribution)
            mean_rc = np.mean(risk_contributions)
            variance_rc = np.sum((risk_contributions - mean_rc) ** 2)

            return variance_rc

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]  # Weights sum to 1

        # Bounds
        bounds = [(self.min_position_weight, self.max_position_weight) for _ in range(n)]

        # Initial guess (equal weights)
        x0 = np.ones(n) / n

        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if result.success:
                return result.x
            else:
                logger.warning(
                    f"Risk parity optimization failed: {result.message}, using equal weights"
                )
                return np.ones(n) / n
        except Exception as e:
            logger.error(f"Risk parity optimization error: {e}, using equal weights")
            return np.ones(n) / n

    def _optimize_kelly(
        self,
        signals: dict[str, float],
        win_rates: dict[str, float],
        avg_wins: dict[str, float],
        avg_losses: dict[str, float],
        symbols: list[str],
    ) -> np.ndarray:
        """
        Kelly Criterion portfolio optimization.

        Optimal growth portfolio based on win rates and payoff ratios.
        """
        n = len(symbols)
        weights = np.zeros(n)

        for i, symbol in enumerate(symbols):
            if symbol not in win_rates or symbol not in avg_wins or symbol not in avg_losses:
                continue

            win_rate = win_rates[symbol]
            avg_win = avg_wins[symbol]
            avg_loss = abs(avg_losses[symbol])

            if avg_loss == 0:
                continue

            # Kelly fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            # But we need to normalize across portfolio
            payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

            # Kelly percentage for this asset
            kelly_fraction = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio

            # Only long positions (if long_only)
            if self.long_only and kelly_fraction < 0:
                kelly_fraction = 0

            # Normalize by signal strength
            signal_strength = abs(signals.get(symbol, 0))
            weights[i] = max(0, kelly_fraction * signal_strength)

        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # Fall back to equal weights
            weights = np.ones(n) / n

        # Apply max position constraint
        weights = np.clip(weights, self.min_position_weight, self.max_position_weight)
        weights = weights / np.sum(weights)  # Renormalize

        return weights

    def _optimize_min_variance(
        self,
        covariance: np.ndarray,
        symbols: list[str],
    ) -> np.ndarray:
        """Minimum variance portfolio (minimize risk)."""
        n = len(symbols)

        def objective(w):
            return np.dot(w, np.dot(covariance, w))

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]  # Weights sum to 1

        # Bounds
        bounds = [(self.min_position_weight, self.max_position_weight) for _ in range(n)]

        # Initial guess (equal weights)
        x0 = np.ones(n) / n

        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if result.success:
                return result.x
            else:
                logger.warning(
                    f"Min variance optimization failed: {result.message}, using equal weights"
                )
                return np.ones(n) / n
        except Exception as e:
            logger.error(f"Min variance optimization error: {e}, using equal weights")
            return np.ones(n) / n

    def _optimize_equal_weight(self, symbols: list[str]) -> np.ndarray:
        """Equal weight portfolio (1/n)."""
        n = len(symbols)
        return np.ones(n) / n

    def _estimate_covariance_from_signals(
        self,
        signals: dict[str, float],
        symbols: list[str],
    ) -> np.ndarray:
        """
        Estimate covariance matrix from signals (simplified).

        Uses signal similarity as proxy for correlation.
        """
        n = len(symbols)
        covariance = np.eye(n) * 0.04  # Base volatility (20% annual)

        # Add correlation based on signal similarity
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i != j:
                    # Higher correlation if signals point in same direction
                    signal_corr = np.sign(signals[sym1]) * np.sign(signals[sym2])
                    covariance[i, j] = 0.04 * 0.5 * (1 + signal_corr * 0.5)  # 0.03 to 0.05

        return covariance

    def _estimate_returns_from_signals(
        self,
        signals: dict[str, float],
        symbols: list[str],
    ) -> np.ndarray:
        """
        Estimate expected returns from signals.

        Assumes signal strength maps to expected return magnitude.
        """
        returns = np.array([signals.get(sym, 0.0) for sym in symbols])

        # Scale to realistic expected returns (e.g., -10% to +10% annual)
        # For swing trading, assume 2-7 day holds, so daily expected return
        returns = returns * 0.001  # 0.1% per day per unit signal

        return returns

    def _optimize_hierarchical_risk_parity(
        self,
        covariance: np.ndarray,
        symbols: list[str],
    ) -> np.ndarray:
        """
        Hierarchical Risk Parity (HRP) optimization.

        Based on Lopez de Prado (2016) "Building Diversified Portfolios that Outperform
        Out of Sample". Uses hierarchical clustering to build robust portfolios that
        are less sensitive to estimation errors in covariance matrices.

        Steps:
        1. Compute correlation matrix from covariance
        2. Build hierarchical tree using correlation distances
        3. Quasi-diagonalize covariance matrix using tree structure
        4. Recursively allocate weights using inverse-variance weighting

        Args:
            covariance: Covariance matrix (n x n)
            symbols: List of symbols

        Returns:
            Optimal weights array (n)
        """
        n = len(symbols)

        if n == 1:
            return np.array([1.0])

        try:
            # Step 1: Compute correlation matrix
            std_dev = np.sqrt(np.diag(covariance))
            correlation = covariance / np.outer(std_dev, std_dev)

            # Step 2: Convert correlation to distance matrix
            # Distance = sqrt(0.5 * (1 - correlation))
            distance_matrix = np.sqrt(0.5 * (1 - correlation))

            # Step 3: Build hierarchical tree using single linkage
            condensed_distances = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_distances, method="single")

            # Step 4: Get order of leaves from dendrogram
            leaves_order = leaves_list(linkage_matrix)

            # Step 5: Quasi-diagonalize covariance matrix
            # Reorder covariance matrix according to hierarchical structure
            reordered_cov = covariance[np.ix_(leaves_order, leaves_order)]
            reordered_symbols = [symbols[i] for i in leaves_order]

            # Step 6: Recursively allocate weights
            weights = self._hrp_recursive_bisection(reordered_cov, list(range(n)))

            # Step 7: Map weights back to original symbol order
            final_weights = np.zeros(n)
            for i, orig_idx in enumerate(leaves_order):
                final_weights[orig_idx] = weights[i]

            # Apply constraints
            final_weights = np.clip(
                final_weights, self.min_position_weight, self.max_position_weight
            )
            final_weights = final_weights / np.sum(final_weights)  # Renormalize

            return final_weights

        except Exception as e:
            logger.error(f"Hierarchical Risk Parity optimization error: {e}, using equal weights")
            return np.ones(n) / n

    def _hrp_recursive_bisection(self, covariance: np.ndarray, indices: list[int]) -> np.ndarray:
        """
        Recursively bisect portfolio and allocate weights using inverse-variance.

        This is the core HRP algorithm that recursively splits the portfolio
        into two clusters and allocates weights inversely proportional to variance.

        Args:
            covariance: Covariance matrix (n x n)
            indices: List of indices for current cluster

        Returns:
            Weights array for the given indices
        """
        n = len(indices)

        if n == 1:
            return np.array([1.0])

        # Compute inverse-variance weights for current cluster
        sub_cov = covariance[np.ix_(indices, indices)]
        inv_var = 1.0 / np.diag(sub_cov)
        inv_var_weights = inv_var / np.sum(inv_var)

        # If cluster is small enough, return inverse-variance weights
        if n <= 2:
            return inv_var_weights

        # Find split point (bisect at middle)
        mid = n // 2
        left_indices = indices[:mid]
        right_indices = indices[mid:]

        # Recursively allocate to left and right clusters
        left_weights = self._hrp_recursive_bisection(covariance, left_indices)
        right_weights = self._hrp_recursive_bisection(covariance, right_indices)

        # Compute variance of left and right clusters
        left_cov = covariance[np.ix_(left_indices, left_indices)]
        right_cov = covariance[np.ix_(right_indices, right_indices)]

        left_var = np.sum(left_weights * np.dot(left_cov, left_weights))
        right_var = np.sum(right_weights * np.dot(right_cov, right_weights))

        # Allocate between clusters inversely proportional to variance
        if left_var == 0 and right_var == 0:
            alpha = 0.5
        elif left_var == 0:
            alpha = 1.0
        elif right_var == 0:
            alpha = 0.0
        else:
            alpha = 1.0 - left_var / (left_var + right_var)

        # Combine weights
        combined_weights = np.zeros(n)
        combined_weights[:mid] = alpha * left_weights
        combined_weights[mid:] = (1 - alpha) * right_weights

        return combined_weights
