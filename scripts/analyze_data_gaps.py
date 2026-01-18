#!/usr/bin/env python3
"""
Data Gap Analysis Script

Analyzes gaps in time series data, accounting for trading days vs calendar days.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

# Load environment variables
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")
load_dotenv()

sys.path.insert(0, str(project_root))

from src.nova.core.config import load_config
from src.nova.data.loader import DataLoader
from src.nova.data.storage import StorageService


def is_trading_day(date: datetime) -> bool:
    """Check if a date is a trading day (Monday-Friday, excluding major holidays)."""
    # Monday = 0, Friday = 4
    if date.weekday() > 4:  # Saturday or Sunday
        return False

    # Major US market holidays (simplified - could be expanded)
    month_day = (date.month, date.day)
    holidays = [
        (1, 1),  # New Year's Day
        (7, 4),  # Independence Day
        (12, 25),  # Christmas
        (12, 31),  # New Year's Eve (sometimes)
    ]

    if month_day in holidays:
        return False

    return True


def analyze_gaps(df: pl.DataFrame, symbol: str) -> dict:
    """Analyze gaps in time series data."""
    if df.is_empty() or "timestamp" not in df.columns:
        return {"error": "No data or missing timestamp column"}

    df_sorted = df.sort("timestamp")
    timestamps = df_sorted["timestamp"]

    # Calculate time differences
    time_diffs = timestamps.diff().drop_nulls()

    # Expected difference for daily data (1 trading day)
    # Account for weekends: 1-3 days is normal (Fri->Mon = 3 calendar days)
    normal_gap_max = timedelta(days=3)

    # Large gaps (> 3 days) indicate missing data
    large_gaps = []
    gap_details = []

    for i in range(1, len(timestamps)):
        prev_time = timestamps[i - 1]
        curr_time = timestamps[i]
        diff = curr_time - prev_time

        if diff > normal_gap_max:
            # Count expected trading days in this gap
            expected_trading_days = 0
            check_date = prev_time + timedelta(days=1)
            while check_date < curr_time:
                if is_trading_day(check_date):
                    expected_trading_days += 1
                check_date += timedelta(days=1)

            large_gaps.append(
                {
                    "gap_start": str(prev_time),
                    "gap_end": str(curr_time),
                    "calendar_days": diff.days,
                    "expected_trading_days": expected_trading_days,
                    "actual_trading_days": 0,  # No data for these days
                }
            )

    # Calculate overall statistics
    if len(time_diffs) > 0:
        avg_gap = time_diffs.mean()
        median_gap = time_diffs.median()
        max_gap = time_diffs.max()
    else:
        avg_gap = None
        median_gap = None
        max_gap = None

    # Count missing trading days
    if len(timestamps) > 0:
        start_date = timestamps.min()
        end_date = timestamps.max()

        # Count expected trading days in range
        expected_trading_days = 0
        check_date = start_date
        while check_date <= end_date:
            if is_trading_day(check_date):
                expected_trading_days += 1
            check_date += timedelta(days=1)

        actual_trading_days = len(df)
        missing_trading_days = expected_trading_days - actual_trading_days
    else:
        expected_trading_days = 0
        actual_trading_days = 0
        missing_trading_days = 0

    return {
        "symbol": symbol,
        "total_rows": len(df),
        "date_range": {
            "start": str(timestamps.min()) if len(timestamps) > 0 else None,
            "end": str(timestamps.max()) if len(timestamps) > 0 else None,
        },
        "gap_statistics": {
            "avg_gap_days": float(avg_gap.total_seconds() / 86400) if avg_gap else None,
            "median_gap_days": float(median_gap.total_seconds() / 86400) if median_gap else None,
            "max_gap_days": float(max_gap.total_seconds() / 86400) if max_gap else None,
        },
        "trading_days": {
            "expected": expected_trading_days,
            "actual": actual_trading_days,
            "missing": missing_trading_days,
            "coverage_pct": (
                (actual_trading_days / expected_trading_days * 100)
                if expected_trading_days > 0
                else 0
            ),
        },
        "large_gaps": large_gaps,
        "large_gap_count": len(large_gaps),
    }


async def main():
    """Main entry point."""
    print("=" * 80)
    print("DATA GAP ANALYSIS")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}\n")

    config = load_config()
    data_loader = DataLoader(config.data)
    storage = None

    try:
        storage = StorageService(config.data)
        await storage.connect()
        await storage.init_schema()
        print("✓ Database connected")
    except Exception as e:
        print(f"⚠️  Database not available: {e}")

    # Analyze first few symbols as sample
    symbols = config.data.symbols[:5]  # Analyze first 5 as sample

    print(f"\nAnalyzing gaps for {len(symbols)} symbols...\n")

    # Calculate date range
    end_date = datetime.now()
    days_per_period = 1 if "day" in config.data.default_timeframe.lower() else 1 / 24
    start_date = end_date - timedelta(days=config.data.lookback_periods * days_per_period)

    results = []
    for symbol in symbols:
        print(f"Analyzing {symbol}...")

        # Fetch data
        df = None
        if storage:
            try:
                df = await storage.load_bars(
                    symbol=symbol,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                )
            except Exception:
                pass

        if df is None or df.is_empty():
            try:
                df = await data_loader.fetch_historical_bars(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=config.data.default_timeframe,
                )
            except Exception as e:
                print(f"  ❌ Failed to fetch data: {e}")
                continue

        if df.is_empty():
            print("  ⚠️  No data available")
            continue

        # Analyze gaps
        gap_analysis = analyze_gaps(df, symbol)
        results.append(gap_analysis)

        print(f"  Rows: {gap_analysis['total_rows']}")
        print(f"  Expected trading days: {gap_analysis['trading_days']['expected']}")
        print(f"  Actual trading days: {gap_analysis['trading_days']['actual']}")
        print(f"  Missing trading days: {gap_analysis['trading_days']['missing']}")
        print(f"  Coverage: {gap_analysis['trading_days']['coverage_pct']:.1f}%")
        print(f"  Large gaps (>3 days): {gap_analysis['large_gap_count']}")

        if gap_analysis["large_gaps"]:
            print("  Large gap examples:")
            for gap in gap_analysis["large_gaps"][:3]:  # Show first 3
                print(
                    f"    - {gap['gap_start']} to {gap['gap_end']}: "
                    f"{gap['calendar_days']} calendar days, "
                    f"{gap['expected_trading_days']} expected trading days"
                )
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        avg_coverage = sum(r["trading_days"]["coverage_pct"] for r in results) / len(results)
        total_missing = sum(r["trading_days"]["missing"] for r in results)
        total_large_gaps = sum(r["large_gap_count"] for r in results)

        print(f"Average Coverage: {avg_coverage:.1f}%")
        print(f"Total Missing Trading Days: {total_missing}")
        print(f"Total Large Gaps: {total_large_gaps}")

        if avg_coverage < 95:
            print("\n⚠️  Low coverage detected - investigate data source")
        if total_large_gaps > 0:
            print("\n⚠️  Large gaps detected - may indicate data source issues")

    if storage:
        await storage.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
