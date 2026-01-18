#!/usr/bin/env python3
"""
Training Data Review Script

Comprehensive analysis of training data quality, coverage, and readiness.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Load environment variables first
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")
load_dotenv()

sys.path.insert(0, str(project_root))

from src.nova.core.config import load_config
from src.nova.data.loader import DataLoader
from src.nova.data.storage import StorageService
from src.nova.features.technical import TechnicalFeatures


async def analyze_symbol_data(
    symbol: str,
    data_loader: DataLoader,
    storage: Optional[StorageService],
    config: Any,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, Any]:
    """Analyze training data for a single symbol."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {symbol}")
    print(f"{'='*60}")

    # Calculate date range
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    # else: end_date is already a datetime

    if start_date is None:
        # Use lookback_periods from config
        days_per_period = 1 if "day" in config.data.default_timeframe.lower() else 1 / 24
        start_date = end_date - timedelta(days=config.data.lookback_periods * days_per_period)
    elif isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    # else: start_date is already a datetime

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    results = {
        "symbol": symbol,
        "data_available": False,
        "source": None,
        "row_count": 0,
        "date_range": {"start": None, "end": None},
        "missing_values": {},
        "data_quality": {},
        "features_available": False,
        "feature_count": 0,
        "labels_available": False,
        "label_distribution": {},
        "recommendations": [],
    }

    # Try to load from database first
    df = None
    if storage:
        try:
            # Convert datetime to ISO string for storage.load_bars
            start_date_iso = (
                start_date.isoformat() if isinstance(start_date, datetime) else start_date
            )
            end_date_iso = end_date.isoformat() if isinstance(end_date, datetime) else end_date
            df = await storage.load_bars(
                symbol=symbol, start_date=start_date_iso, end_date=end_date_iso
            )
            if not df.is_empty():
                results["source"] = "database"
                results["data_available"] = True
        except Exception as e:
            print(f"  ⚠️  Database load failed: {e}")

    # If no database data, fetch from API
    if df is None or df.is_empty():
        try:
            df = await data_loader.fetch_historical_bars(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=config.data.default_timeframe,
            )
            if not df.is_empty():
                results["source"] = "api"
                results["data_available"] = True
        except Exception as e:
            print(f"  ❌ API fetch failed: {e}")
            results["recommendations"].append(f"Failed to fetch data: {e}")
            return results

    if df.is_empty():
        results["recommendations"].append("No data available for this symbol")
        return results

    # Basic data statistics
    results["row_count"] = len(df)
    if "timestamp" in df.columns:
        results["date_range"]["start"] = str(df["timestamp"].min())
        results["date_range"]["end"] = str(df["timestamp"].max())

    # Check for missing values
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            null_count = df[col].null_count()
            results["missing_values"][col] = {
                "count": null_count,
                "pct": (null_count / len(df)) * 100 if len(df) > 0 else 0,
            }

    # Data quality checks
    if "close" in df.columns:
        # Check for zero/negative prices
        zero_prices = int((df["close"] <= 0).sum())
        results["data_quality"]["zero_prices"] = zero_prices

        # Check for extreme price changes (potential data errors)
        if len(df) > 1:
            pct_changes = df["close"].pct_change().drop_nulls()
            if len(pct_changes) > 0:
                extreme_changes = (pct_changes.abs() > 0.5).sum()  # >50% change
                results["data_quality"]["extreme_changes"] = int(extreme_changes)
                max_change = pct_changes.abs().max()
                results["data_quality"]["max_change_pct"] = (
                    float(max_change) if max_change is not None else 0.0
                )

    # Check volume
    if "volume" in df.columns:
        zero_volume = int((df["volume"] == 0).sum())
        results["data_quality"]["zero_volume_days"] = zero_volume

    # Calculate features
    try:
        technical_features = TechnicalFeatures(config.technical)
        features_df = technical_features.calculate_ml_features(
            df,
            apply_ffd=True,
            apply_zscore=True,
        )

        if not features_df.is_empty():
            results["features_available"] = True
            results["feature_count"] = len(features_df.columns)

            # Check feature quality
            feature_nulls = {}
            for col in features_df.columns:
                if col not in ["timestamp", "label"]:
                    null_count = features_df[col].null_count()
                    if null_count > 0:
                        feature_nulls[col] = {
                            "count": null_count,
                            "pct": (null_count / len(features_df)) * 100,
                        }
            results["data_quality"]["feature_nulls"] = feature_nulls

            # Generate labels (NPMM methodology)
            from src.nova.models.trainer import ModelTrainer

            trainer = ModelTrainer(config.ml)
            labeled_df = trainer.labeler.generate_binary_labels(features_df, price_col="close")

            if "label" in labeled_df.columns:
                results["labels_available"] = True
                label_counts = labeled_df["label"].value_counts()
                # Convert Polars Series to dict, handling scalar values properly
                label_dist_dict = {}
                for row in label_counts.iter_rows(named=False):
                    key, value = row[0], row[1]
                    # Handle Polars scalar conversion
                    if hasattr(value, "item"):
                        label_dist_dict[str(key)] = int(value.item())
                    else:
                        label_dist_dict[str(key)] = int(value)
                results["label_distribution"] = label_dist_dict

                # Calculate label balance
                if len(label_dist_dict) >= 2:
                    label_values = list(label_dist_dict.values())
                    balance_ratio = (
                        min(label_values) / max(label_values) if max(label_values) > 0 else 0
                    )
                    results["data_quality"]["label_balance"] = balance_ratio

                    if balance_ratio < 0.3:
                        results["recommendations"].append(
                            "⚠️  Severe class imbalance detected - consider resampling or class weights"
                        )

    except Exception as e:
        print(f"  ⚠️  Feature calculation failed: {e}")
        results["recommendations"].append(f"Feature calculation error: {e}")

    # Generate recommendations
    if results["row_count"] < 100:
        results["recommendations"].append(
            f"❌ Insufficient data: {results['row_count']} rows (need at least 100)"
        )
    elif results["row_count"] < 252:
        results["recommendations"].append(
            f"⚠️  Limited data: {results['row_count']} rows (recommend 252+ for daily timeframe)"
        )

    if results["row_count"] < config.data.lookback_periods:
        results["recommendations"].append(
            f"⚠️  Data coverage below configured lookback ({config.data.lookback_periods} periods)"
        )

    # Check date coverage
    if results["date_range"]["start"] and results["date_range"]["end"]:
        start_dt = datetime.fromisoformat(results["date_range"]["start"].replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(results["date_range"]["end"].replace("Z", "+00:00"))
        days_covered = (end_dt - start_dt).days

        if days_covered < 365:
            results["recommendations"].append(f"⚠️  Less than 1 year of data ({days_covered} days)")
        elif days_covered < 756:
            results["recommendations"].append(
                f"⚠️  Less than 3 years of data ({days_covered} days) - recommended minimum for swing trading"
            )

    # Check for gaps in data
    if "timestamp" in df.columns and len(df) > 1:
        df_sorted = df.sort("timestamp")
        time_diffs = df_sorted["timestamp"].diff().drop_nulls()
        if len(time_diffs) > 0:
            # For daily data, expect ~1 day difference
            expected_diff = timedelta(days=1)
            large_gaps = (time_diffs > expected_diff * 2).sum()
            if large_gaps > 0:
                results["recommendations"].append(
                    f"⚠️  {large_gaps} large gaps detected in time series (potential missing data)"
                )

    return results


def print_analysis_report(results: list[dict[str, Any]]) -> None:
    """Print comprehensive analysis report."""
    print("\n" + "=" * 80)
    print("TRAINING DATA REVIEW REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().isoformat()}\n")

    # Summary statistics
    total_symbols = len(results)
    symbols_with_data = sum(1 for r in results if r["data_available"])
    symbols_with_features = sum(1 for r in results if r["features_available"])
    symbols_with_labels = sum(1 for r in results if r["labels_available"])

    print("SUMMARY")
    print("-" * 80)
    print(f"Total Symbols Analyzed: {total_symbols}")
    print(f"Symbols with Data: {symbols_with_data} ({symbols_with_data/total_symbols*100:.1f}%)")
    print(
        f"Symbols with Features: {symbols_with_features} ({symbols_with_features/total_symbols*100:.1f}%)"
    )
    print(
        f"Symbols with Labels: {symbols_with_labels} ({symbols_with_labels/total_symbols*100:.1f}%)"
    )
    print()

    # Per-symbol details
    print("PER-SYMBOL ANALYSIS")
    print("-" * 80)

    for result in results:
        symbol = result["symbol"]
        print(f"\n{symbol}:")
        print(f"  Data Source: {result.get('source', 'N/A')}")
        print(f"  Rows: {result['row_count']}")

        if result.get("date_range", {}).get("start"):
            print(f"  Date Range: {result['date_range']['start']} to {result['date_range']['end']}")

        if result["features_available"]:
            print(f"  Features: {result['feature_count']} calculated")
        else:
            print("  Features: ❌ Not available")

        if result["labels_available"]:
            label_dist = result["label_distribution"]
            print(f"  Labels: {label_dist}")
            if "label_balance" in result.get("data_quality", {}):
                balance = result["data_quality"]["label_balance"]
                print(
                    f"  Label Balance: {balance:.2f} ({'✅ Good' if balance > 0.3 else '⚠️  Imbalanced'})"
                )
        else:
            print("  Labels: ❌ Not available")

        # Data quality issues
        if result.get("missing_values"):
            missing = result["missing_values"]
            for col, info in missing.items():
                if info["count"] > 0:
                    print(f"  ⚠️  Missing {col}: {info['count']} ({info['pct']:.1f}%)")

        if result.get("data_quality", {}).get("zero_prices", 0) > 0:
            print(f"  ⚠️  Zero/Negative Prices: {result['data_quality']['zero_prices']}")

        if result.get("data_quality", {}).get("extreme_changes", 0) > 0:
            print(f"  ⚠️  Extreme Price Changes: {result['data_quality']['extreme_changes']}")

        # Recommendations
        if result.get("recommendations"):
            for rec in result["recommendations"]:
                print(f"  {rec}")

    # Overall recommendations
    print("\n" + "=" * 80)
    print("OVERALL RECOMMENDATIONS")
    print("=" * 80)

    # Check data coverage
    avg_rows = sum(r["row_count"] for r in results if r["data_available"]) / max(
        symbols_with_data, 1
    )
    if avg_rows < 252:
        print(f"❌ Average data rows per symbol: {avg_rows:.0f} (recommend 252+ for daily)")
    elif avg_rows < 756:
        print(f"⚠️  Average data rows per symbol: {avg_rows:.0f} (recommend 756+ for swing trading)")

    # Check feature availability
    if symbols_with_features < symbols_with_data:
        print(f"⚠️  {symbols_with_data - symbols_with_features} symbols missing features")

    # Check label availability
    if symbols_with_labels < symbols_with_features:
        print(f"⚠️  {symbols_with_features - symbols_with_labels} symbols missing labels")

    # Check for common issues
    symbols_with_issues = sum(
        1
        for r in results
        if r.get("recommendations") or r.get("data_quality", {}).get("extreme_changes", 0) > 0
    )
    if symbols_with_issues > 0:
        print(f"⚠️  {symbols_with_issues} symbols have data quality issues")

    print("\n✅ Review complete!")


async def main():
    """Main entry point."""
    print("=" * 80)
    print("NOVA AETUS - TRAINING DATA REVIEW")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    # Initialize components
    data_loader = DataLoader(config.data)
    storage = None

    try:
        storage = StorageService(config.data)
        await storage.connect()
        await storage.init_schema()
        print("✓ Database connected")
    except Exception as e:
        print(f"⚠️  Database not available: {e}")
        print("  Continuing with API-only data fetching...")

    # Analyze each symbol
    symbols = config.data.symbols
    print(f"\nAnalyzing {len(symbols)} symbols...")

    # Calculate date range
    end_date = datetime.now()
    days_per_period = 1 if "day" in config.data.default_timeframe.lower() else 1 / 24
    start_date = end_date - timedelta(days=config.data.lookback_periods * days_per_period)

    results = []
    for symbol in symbols:
        try:
            result = await analyze_symbol_data(
                symbol=symbol,
                data_loader=data_loader,
                storage=storage,
                config=config,
                start_date=start_date,
                end_date=end_date,
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "symbol": symbol,
                    "data_available": False,
                    "features_available": False,
                    "labels_available": False,
                    "row_count": 0,
                    "date_range": {"start": None, "end": None},
                    "missing_values": {},
                    "data_quality": {},
                    "error": str(e),
                    "recommendations": [f"Analysis failed: {e}"],
                }
            )

    # Print comprehensive report
    print_analysis_report(results)

    # Cleanup
    if storage:
        await storage.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
