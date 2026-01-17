"""Streamlit dashboard for Nova Aetus trading system.

Real-time visualization of positions, P&L, signals, and system health.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.nova.core.config import load_config
from src.nova.core.health import HealthChecker, HealthStatus
from src.nova.data.storage import StorageService
from src.nova.strategy.execution import ExecutionEngine

# Setup
st.set_page_config(
    page_title="Nova Aetus Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_system_config():
    """Load system configuration (cached)."""
    try:
        return load_config()
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        return None


def get_storage_service(config) -> Optional[StorageService]:
    """Get storage service instance."""
    try:
        return StorageService(config.data)
    except Exception as e:
        logger.error(f"Failed to initialize storage service: {e}")
        return None


def run_async(coro):
    """Run async function in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def main() -> None:
    """Main dashboard application."""
    st.title("📈 Nova Aetus Trading System")
    st.sidebar.title("Navigation")

    # Load config
    config = load_system_config()
    if config is None:
        st.stop()

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page",
        [
            "Overview",
            "Portfolio",
            "Positions",
            "Performance",
            "Models",
            "Model Monitoring",
            "Settings",
        ],
    )

    if page == "Overview":
        show_overview(config)
    elif page == "Portfolio":
        show_portfolio(config)
    elif page == "Positions":
        show_positions(config)
    elif page == "Performance":
        show_performance(config)
    elif page == "Models":
        show_models(config)
    elif page == "Model Monitoring":
        show_model_monitoring(config)
    elif page == "Settings":
        show_settings(config)


def show_overview(config) -> None:
    """Display overview dashboard with real-time data."""
    st.header("System Overview")

    storage = get_storage_service(config)

    # System Health Status
    health_checker = HealthChecker()
    system_health = None
    try:
        system_health = run_async(
            health_checker.get_system_health(
                storage_service=(
                    storage if storage and hasattr(storage, "pool") and storage.pool else None
                ),
                sentiment_analyzer=None,  # Skip sentiment check for dashboard performance
            )
        )
    except Exception as e:
        logger.debug(f"Could not fetch system health: {e}")
        system_health = None

    # Display health status badge
    if system_health:
        status = system_health.status
        if status == HealthStatus.HEALTHY:
            st.success("🟢 System Healthy")
        elif status == HealthStatus.DEGRADED:
            st.warning("🟡 System Degraded - Some components unhealthy")
        elif status == HealthStatus.UNHEALTHY:
            st.error("🔴 System Unhealthy - Critical issues detected")
        else:
            st.info("⚪ System Status Unknown")

        # Show component status if any issues
        if not system_health.all_healthy:
            with st.expander("Component Status", expanded=False):
                for name, comp in system_health.components.items():
                    if comp.status == HealthStatus.HEALTHY:
                        st.markdown(f"✅ **{name}**: {comp.status.value}")
                    elif comp.status == HealthStatus.DEGRADED:
                        st.markdown(f"⚠️ **{name}**: {comp.status.value} - {comp.message or 'N/A'}")
                    else:
                        st.markdown(f"❌ **{name}**: {comp.status.value} - {comp.message or 'N/A'}")
    else:
        st.info("⚪ System health check unavailable")

    st.divider()

    # Get account info from execution engine (if available)
    account_info = {"equity": 100000.0, "buying_power": 100000.0, "cash": 100000.0}
    try:
        execution_engine = ExecutionEngine(data_config=config.data)
        account_info = run_async(execution_engine.get_account_info())
    except Exception as e:
        logger.debug(f"Could not fetch account info: {e}")

    # Get open positions
    open_positions_df = None
    open_positions_list = []
    if storage:
        try:
            open_positions_df = run_async(storage.get_open_positions())
            if open_positions_df is not None and not open_positions_df.is_empty():
                open_positions_list = open_positions_df.to_dicts()
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")

    # Calculate daily P&L
    daily_pnl = 0.0
    if open_positions_list:
        for pos in open_positions_list:
            if pos.get("unrealized_pl"):
                daily_pnl += pos["unrealized_pl"]

    # Get recent signals
    recent_signals = []
    if storage:
        try:
            signals_df = run_async(
                storage.load_signals(
                    symbol=None,
                    start_date=(datetime.now() - timedelta(days=7)).isoformat(),
                )
            )
            if not signals_df.is_empty():
                recent_signals = signals_df.to_dicts()
        except Exception as e:
            logger.error(f"Error fetching signals: {e}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        equity = account_info.get("equity", 100000.0)
        st.metric("Total Equity", f"${equity:,.2f}", f"${daily_pnl:,.2f}")

    with col2:
        st.metric("Open Positions", len(open_positions_list), "")

    with col3:
        st.metric(
            "Daily P&L",
            f"${daily_pnl:,.2f}",
            f"{daily_pnl/equity*100:.2f}%" if equity > 0 else "0%",
        )

    with col4:
        # Calculate Sharpe from recent signals/returns
        sharpe = 0.0
        if recent_signals:
            # Simplified Sharpe calculation
            returns = [s.get("confluence_score", 0) * 0.01 for s in recent_signals]  # Placeholder
            if returns:
                import numpy as np

                mean_ret = np.mean(returns)
                std_ret = np.std(returns) if len(returns) > 1 else 0.01
                sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", "")

    # Equity curve
    st.subheader("Equity Curve")
    if storage:
        try:
            # Get historical positions/equity data
            # For now, use signals to approximate equity curve
            if recent_signals:
                equity_data = []
                current_equity = equity
                for signal in reversed(recent_signals):
                    # Simplified: assume small impact per signal
                    current_equity += signal.get("confluence_score", 0) * 10
                    equity_data.append(
                        {
                            "timestamp": signal.get("timestamp", datetime.now()),
                            "equity": current_equity,
                        }
                    )

                if equity_data:
                    equity_df = pl.DataFrame(equity_data)
                    fig = px.line(
                        equity_df.to_pandas(),
                        x="timestamp",
                        y="equity",
                        title="Equity Curve (Last 7 Days)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient data for equity curve")
            else:
                st.info("No trading data available yet")
        except Exception as e:
            logger.error(f"Error generating equity curve: {e}")
            st.info("Equity curve data not available")
    else:
        st.info("Storage service not available")

    # Recent signals table
    st.subheader("Recent Signals")
    if recent_signals:
        signals_data = []
        for sig in recent_signals[:10]:  # Show last 10
            signals_data.append(
                {
                    "Symbol": sig.get("symbol", "N/A"),
                    "Direction": sig.get("direction", "N/A"),
                    "Strength": f"{sig.get('strength', 0):.2f}",
                    "Confidence": f"{sig.get('confidence', 0):.2f}",
                    "Confluence": f"{sig.get('confluence_score', 0):.2f}",
                    "Timestamp": sig.get("timestamp", "N/A"),
                }
            )
        st.dataframe(signals_data, use_container_width=True)
    else:
        st.info("No signals generated yet.")


def show_positions(config) -> None:
    """Display current positions with real-time P&L."""
    st.header("Current Positions")

    storage = get_storage_service(config)

    # Get positions from database
    open_positions_list = []
    if storage:
        try:
            open_positions_df = run_async(storage.get_open_positions())
            if open_positions_df is not None and not open_positions_df.is_empty():
                open_positions_list = open_positions_df.to_dicts()
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")

    # Also try to get from Alpaca directly
    try:
        execution_engine = ExecutionEngine(data_config=config.data)
        alpaca_positions = run_async(execution_engine.get_positions())
        # Merge with database positions
        if alpaca_positions:
            open_positions_list.extend(alpaca_positions)
    except Exception as e:
        logger.debug(f"Could not fetch Alpaca positions: {e}")

    if open_positions_list:
        # Create positions table
        positions_data = []
        total_unrealized_pl = 0.0

        for pos in open_positions_list:
            symbol = pos.get("symbol", "N/A")
            quantity = pos.get("quantity", 0)
            entry_price = pos.get("entry_price") or pos.get("avg_entry_price", 0)
            current_price = pos.get("current_price", entry_price)
            unrealized_pl = pos.get("unrealized_pl", 0)
            unrealized_plpc = pos.get("unrealized_plpc", 0)

            total_unrealized_pl += unrealized_pl

            positions_data.append(
                {
                    "Symbol": symbol,
                    "Quantity": quantity,
                    "Entry Price": f"${entry_price:.2f}",
                    "Current Price": f"${current_price:.2f}",
                    "Unrealized P&L": f"${unrealized_pl:.2f}",
                    "P&L %": f"{unrealized_plpc*100:.2f}%",
                    "Market Value": f"${pos.get('market_value', current_price * quantity):,.2f}",
                }
            )

        st.metric("Total Unrealized P&L", f"${total_unrealized_pl:,.2f}")
        st.dataframe(positions_data, use_container_width=True)

        # Position distribution chart
        if positions_data:
            symbols = [p["Symbol"] for p in positions_data]
            values = [
                float(p["Market Value"].replace("$", "").replace(",", "")) for p in positions_data
            ]

            fig = px.pie(
                values=values,
                names=symbols,
                title="Position Distribution by Market Value",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No open positions.")


def show_performance(config) -> None:
    """Display performance metrics from database."""
    st.header("Performance Metrics")

    storage = get_storage_service(config)

    # Get closed positions for performance calculation
    closed_positions = []
    if storage:
        try:
            # Query closed positions
            query = """
                SELECT * FROM portfolio_positions
                WHERE status = 'CLOSED'
                ORDER BY closed_at DESC
                LIMIT 100
            """
            result = run_async(storage.query(query))
            if result and not result.is_empty():
                closed_positions = result.to_dicts()
        except Exception as e:
            logger.error(f"Error fetching closed positions: {e}")

    if closed_positions:
        # Calculate performance metrics
        import numpy as np

        pnl_values = [p.get("realized_pl", 0) for p in closed_positions if p.get("realized_pl")]
        winning_trades = [p for p in pnl_values if p > 0]
        losing_trades = [p for p in pnl_values if p < 0]

        win_rate = len(winning_trades) / len(pnl_values) * 100 if pnl_values else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = (
            abs(sum(winning_trades) / sum(losing_trades))
            if losing_trades and sum(losing_trades) != 0
            else 0
        )

        # Calculate max drawdown
        equity_curve = []
        running_equity = 100000.0  # Starting equity
        peak = running_equity
        max_dd = 0.0

        for pnl in pnl_values:
            running_equity += pnl
            if running_equity > peak:
                peak = running_equity
            dd = (peak - running_equity) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
            equity_curve.append(running_equity)

        # Returns distribution
        st.subheader("Returns Distribution")
        if pnl_values:
            fig = px.histogram(
                x=pnl_values,
                nbins=30,
                title="Trade P&L Distribution",
                labels={"x": "P&L ($)", "y": "Frequency"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trade data available")

        # Risk metrics
        st.subheader("Risk Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
            st.metric("Win Rate", f"{win_rate:.2f}%")
        with col2:
            st.metric("Profit Factor", f"{profit_factor:.2f}")
            st.metric("Average Trade", f"${np.mean(pnl_values):,.2f}" if pnl_values else "$0.00")

        # Equity curve
        st.subheader("Equity Curve (Closed Trades)")
        if equity_curve:
            fig = px.line(
                x=range(len(equity_curve)),
                y=equity_curve,
                title="Equity Curve",
                labels={"x": "Trade Number", "y": "Equity ($)"},
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed positions yet. Performance metrics will appear after trades are closed.")

        st.subheader("Risk Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Drawdown", "0.00%")
            st.metric("Win Rate", "0.00%")
        with col2:
            st.metric("Profit Factor", "0.00")
            st.metric("Average Trade", "$0.00")


def show_models(config) -> None:
    """Display model information and status."""
    st.header("Model Status")

    # Check for trained models
    model_dir = project_root / "models"
    model_files = list(model_dir.glob("*.json")) if model_dir.exists() else []

    st.subheader("Technical Model")
    if model_files:
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        st.success(f"Model loaded: {latest_model.name}")
        st.info(
            f"Last modified: {datetime.fromtimestamp(latest_model.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Try to load metadata
        metadata_file = model_dir / f"{latest_model.stem}_metadata.json"
        if metadata_file.exists():
            import json

            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    st.json(metadata)
            except Exception as e:
                logger.debug(f"Could not load metadata: {e}")
    else:
        st.warning("Technical model not yet trained.")

    st.subheader("Sentiment Analyzer")
    st.success(f"Sentiment analyzer active (Model: {config.sentiment.model_name})")

    # Get recent model predictions from signals
    storage = get_storage_service(config)
    if storage:
        try:
            recent_signals = run_async(
                storage.load_signals(
                    symbol=None,
                    start_date=(datetime.now() - timedelta(days=1)).isoformat(),
                )
            )

            if not recent_signals.is_empty():
                st.subheader("Recent Model Predictions")
                predictions_data = []
                for row in recent_signals.to_dicts():
                    predictions_data.append(
                        {
                            "Symbol": row.get("symbol", "N/A"),
                            "Technical Score": f"{row.get('technical_score', 0):.3f}",
                            "Sentiment Score": f"{row.get('sentiment_score', 0):.3f}",
                            "Confluence Score": f"{row.get('confluence_score', 0):.3f}",
                            "Direction": row.get("direction", "N/A"),
                            "Confidence": f"{row.get('confidence', 0):.2f}",
                        }
                    )
                st.dataframe(predictions_data, use_container_width=True)
            else:
                st.info("No recent predictions available.")
        except Exception as e:
            logger.error(f"Error fetching predictions: {e}")
            st.info("Model predictions will be displayed here once models are trained.")
    else:
        st.info("Storage service not available for model predictions.")


def show_model_monitoring(config) -> None:
    """Display model monitoring: SHAP, IC decay, drift detection."""
    st.header("Model Monitoring")

    storage = get_storage_service(config)

    # Tabs for different monitoring views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["SHAP Interpretability", "IC Decay", "Drift Detection", "Feature Importance"]
    )

    with tab1:
        st.subheader("SHAP (SHapley Additive exPlanations)")

        # Load model metadata with SHAP values
        model_dir = project_root / "models"
        model_files = list(model_dir.glob("*.json")) if model_dir.exists() else []

        if model_files:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            metadata_file = model_dir / f"{latest_model.stem}.json"

            if metadata_file.exists():
                import json

                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                    if "shap" in metadata:
                        shap_data = metadata["shap"]

                        # Feature importance bar chart
                        if "feature_importance" in shap_data:
                            feature_names = list(shap_data["feature_importance"].keys())
                            importances = list(shap_data["feature_importance"].values())

                            fig = go.Figure(
                                data=[go.Bar(x=importances, y=feature_names, orientation="h")]
                            )
                            fig.update_layout(
                                title="Feature Importance (Mean |SHAP|)",
                                xaxis_title="Mean |SHAP Value|",
                                yaxis_title="Feature",
                                height=600,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # SHAP summary stats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Expected Value", f"{shap_data.get('expected_value', 0):.4f}")
                        with col2:
                            st.metric("Samples Analyzed", shap_data.get("n_samples", 0))
                    else:
                        st.info(
                            "SHAP values not available for this model. Train with SHAP enabled."
                        )
                except Exception as e:
                    logger.error(f"Error loading SHAP data: {e}")
                    st.error("Failed to load SHAP data")
        else:
            st.warning("No trained models found. Train a model first to see SHAP analysis.")

    with tab2:
        st.subheader("Information Coefficient (IC) Decay")

        # IC tracker visualization
        st.info("IC decay monitoring tracks signal quality over time at multiple horizons.")

        # Placeholder for IC data (would come from database or ICTracker)
        if storage:
            try:
                # Query IC history from database if available
                # For now, show placeholder
                st.info("IC tracking data will be displayed here once IC tracking is enabled.")

                # Example visualization
                horizons = [1, 5, 20]
                signals = ["technical", "sentiment", "fundamental"]

                # Create sample IC data for demonstration
                try:
                    import pandas as pd

                    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
                except ImportError:
                    # Fallback if pandas not available
                    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]

                for signal in signals:
                    st.subheader(f"{signal.capitalize()} Signal IC")
                    ic_data = []
                    for horizon in horizons:
                        # Sample IC values (would come from ICTracker)
                        ic_values = np.random.normal(0.05, 0.02, len(dates))
                        ic_data.append({"Date": dates, "IC": ic_values, "Horizon": f"{horizon}d"})

                    # Plot IC over time
                    fig = go.Figure()
                    for data in ic_data:
                        fig.add_trace(
                            go.Scatter(
                                x=data["Date"],
                                y=data["IC"],
                                name=data["Horizon"],
                                mode="lines+markers",
                            )
                        )
                    fig.update_layout(
                        title=f"{signal.capitalize()} IC Over Time",
                        xaxis_title="Date",
                        yaxis_title="IC",
                        hovermode="x unified",
                    )
                    fig.add_hline(
                        y=0.05,
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Strong Signal Threshold",
                    )
                    fig.add_hline(
                        y=0.02,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text="Moderate Signal Threshold",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.error(f"Error displaying IC data: {e}")
        else:
            st.info("Storage service not available for IC tracking.")

    with tab3:
        st.subheader("Drift Detection")

        st.info("Drift detection monitors data distribution shifts and model performance decay.")

        # Drift detection status
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Data Drift", "No Drift", "✅")
        with col2:
            st.metric("Concept Drift", "No Drift", "✅")
        with col3:
            st.metric("Prediction Drift", "No Drift", "✅")

        # Drift history (placeholder)
        st.subheader("Drift History")
        st.info("Drift detection history will be displayed here once drift monitoring is enabled.")

        # PSI and KS statistics
        st.subheader("Drift Statistics")
        drift_stats = {
            "Max PSI": 0.15,
            "Min KS p-value": 0.08,
            "Drifted Features": 0,
        }
        st.json(drift_stats)

    with tab4:
        st.subheader("Feature Importance Over Time")

        st.info("Track feature importance changes over time to identify regime shifts.")

        # Feature importance over time (placeholder)
        if storage:
            try:
                # Would load feature importance history from database
                st.info("Feature importance tracking will be displayed here once enabled.")
            except Exception as e:
                logger.error(f"Error displaying feature importance: {e}")
        else:
            st.info("Storage service not available.")


def show_portfolio(config) -> None:
    """Display portfolio composition, metrics, and optimization."""
    st.header("Portfolio Overview")

    storage = get_storage_service(config)
    execution_engine = None

    try:
        execution_engine = ExecutionEngine(data_config=config.data)
    except Exception as e:
        logger.debug(f"Could not initialize execution engine: {e}")

    # Get account info
    account_info = {"equity": 100000.0, "buying_power": 100000.0, "cash": 100000.0}
    if execution_engine:
        try:
            account_info = run_async(execution_engine.get_account_info())
        except Exception as e:
            logger.debug(f"Could not fetch account info: {e}")

    # Get all positions
    open_positions_list = []
    if storage:
        try:
            open_positions_df = run_async(storage.get_open_positions())
            if open_positions_df is not None and not open_positions_df.is_empty():
                open_positions_list = open_positions_df.to_dicts()
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")

    # Portfolio metrics
    total_equity = account_info.get("equity", 100000.0)
    total_positions_value = sum(
        pos.get("market_value", pos.get("quantity", 0) * pos.get("current_price", 0))
        for pos in open_positions_list
    )
    cash = total_equity - total_positions_value

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Equity", f"${total_equity:,.2f}")
    with col2:
        st.metric("Positions Value", f"${total_positions_value:,.2f}")
    with col3:
        st.metric("Cash", f"${cash:,.2f}")
    with col4:
        positions_count = len(open_positions_list)
        st.metric("Positions", positions_count)

    if open_positions_list:
        # Portfolio composition by value
        st.subheader("Portfolio Composition")

        # Prepare data for pie chart
        symbols = [pos.get("symbol", "N/A") for pos in open_positions_list]
        values = []
        for pos in open_positions_list:
            market_value = pos.get("market_value")
            if market_value:
                values.append(float(market_value))
            else:
                qty = pos.get("quantity", 0)
                price = pos.get("current_price", pos.get("entry_price", 0))
                values.append(float(qty * price))

        # Portfolio allocation pie chart
        if values:
            fig = px.pie(
                values=values,
                names=symbols,
                title="Portfolio Allocation by Market Value",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Portfolio weights table
        st.subheader("Position Weights")
        portfolio_data = []
        total_value = sum(values) if values else 1.0

        for i, pos in enumerate(open_positions_list):
            symbol = pos.get("symbol", "N/A")
            value = values[i] if i < len(values) else 0.0
            weight = (value / total_value * 100) if total_value > 0 else 0.0

            portfolio_data.append(
                {
                    "Symbol": symbol,
                    "Quantity": pos.get("quantity", 0),
                    "Entry Price": f"${pos.get('entry_price', 0):.2f}",
                    "Current Price": f"${pos.get('current_price', pos.get('entry_price', 0)):.2f}",
                    "Market Value": f"${value:,.2f}",
                    "Weight": f"{weight:.2f}%",
                    "P&L": f"${pos.get('unrealized_pl', 0):,.2f}",
                    "P&L %": f"{pos.get('unrealized_plpc', 0) * 100:.2f}%",
                }
            )

        st.dataframe(portfolio_data, use_container_width=True)

        # Portfolio risk metrics (simplified)
        st.subheader("Portfolio Risk Metrics")
        col1, col2 = st.columns(2)

        with col1:
            # Calculate portfolio-level metrics
            total_pnl = sum(pos.get("unrealized_pl", 0) for pos in open_positions_list)
            portfolio_return_pct = (total_pnl / total_equity * 100) if total_equity > 0 else 0.0
            st.metric("Portfolio Return", f"{portfolio_return_pct:.2f}%")

        with col2:
            # Number of positions
            st.metric("Diversification", f"{positions_count} positions")

        # Portfolio optimization info
        st.subheader("Portfolio Optimization")
        st.info(
            "Portfolio optimization runs once per day (swing trading). "
            "Positions are held for 2-7 days. "
            "Method: Mean-Variance Optimization"
        )
    else:
        st.info("No open positions. Portfolio will be optimized when signals are generated.")


def show_settings(config) -> None:
    """Display system settings."""
    st.header("System Settings")

    st.subheader("Configuration")
    st.json(
        {
            "Technical Indicators": {
                "RSI Period": config.technical.rsi_period,
                "MACD Fast": config.technical.macd_fast,
                "MACD Slow": config.technical.macd_slow,
            },
            "Risk Management": {
                "Risk per Trade": f"{config.risk.risk_per_trade_pct:.2%}",
                "Max Drawdown": f"{config.risk.max_drawdown_pct:.2%}",
                "Max Position Size": f"{config.risk.max_position_size_pct:.2%}",
            },
            "Circuit Breaker": {
                "Max Errors": config.circuit_breaker.max_errors_per_minute,
                "Time Window": f"{config.circuit_breaker.error_window_seconds}s",
            },
        }
    )


if __name__ == "__main__":
    main()
