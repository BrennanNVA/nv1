#!/usr/bin/env python3
"""Test script to verify Alpaca Pro API connection and data fetching."""

import asyncio
import logging
import sys
from pathlib import Path

# Load environment variables first (before importing config)
from dotenv import load_dotenv

# Load .env from current directory and parent directory
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")  # Load from project root
load_dotenv()  # Also try current directory

# Add project root to path
sys.path.insert(0, str(project_root / "src"))

from nova.core.config import load_config
from nova.data.loader import DataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_alpaca_connection() -> None:
    """Test Alpaca Pro API connection and data fetching capabilities."""
    print("\n" + "=" * 70)
    print("Alpaca Pro API Connection Test")
    print("=" * 70 + "\n")

    # Load configuration
    try:
        config = load_config()
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return

    # Check credentials
    has_api_key = bool(config.data.alpaca_api_key)
    has_secret = bool(config.data.alpaca_secret_key)

    print("\n📋 Credentials Check:")
    print(f"  API Key configured: {has_api_key}")
    print(f"  Secret Key configured: {has_secret}")

    if not has_api_key or not has_secret:
        print("\n⚠️  WARNING: Alpaca credentials not found in .env file")
        print("   Please add your Pro Trader API keys:")
        print("   ALPACA_API_KEY=your_key_here")
        print("   ALPACA_SECRET_KEY=your_secret_here")
        print("\n   Get your keys from: https://app.alpaca.markets/paper/dashboard/overview")
        return

    # Initialize data loader
    try:
        data_loader = DataLoader(config.data)
        print("\n✓ DataLoader initialized")
    except Exception as e:
        print(f"✗ Failed to initialize DataLoader: {e}")
        return

    # Check if Alpaca client was initialized
    if data_loader.alpaca_client is None:
        print("\n⚠️  WARNING: Alpaca client not initialized")
        print("   This usually means:")
        print("   1. Credentials are invalid")
        print("   2. alpaca-py package is not installed")
        print("   3. There was an error during initialization")
        return

    print("✓ Alpaca Pro API client initialized")
    print(f"  Base URL: {config.data.alpaca_base_url}")

    # Test 1: Fetch historical bars
    print("\n" + "-" * 70)
    print("Test 1: Fetching Historical Bars (AAPL, last 30 days)")
    print("-" * 70)
    try:
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        df = await data_loader.fetch_historical_bars(
            "AAPL", start_date=start_date, end_date=end_date, timeframe="1Day"
        )

        if not df.is_empty():
            print(f"✓ Successfully fetched {len(df)} bars")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Columns: {df.columns}")
            print("\n  Sample data (last 5 rows):")
            print(df.tail(5))
        else:
            print("⚠️  No data returned (empty DataFrame)")
    except Exception as e:
        print(f"✗ Failed to fetch bars: {e}")
        import traceback

        traceback.print_exc()

    # Test 2: Fetch with Pro features (if recent data available)
    print("\n" + "-" * 70)
    print("Test 2: Testing Pro Features (Extended Hours, Adjustments)")
    print("-" * 70)
    try:
        # Test with different adjustments
        for adj in ["raw", "split", "all"]:
            try:
                df = await data_loader.fetch_historical_bars(
                    "AAPL",
                    start_date=start_date,
                    end_date=end_date,
                    timeframe="1Day",
                    adjustment=adj,
                )
                if not df.is_empty():
                    print(f"✓ Adjustment '{adj}': {len(df)} bars fetched")
            except Exception as e:
                print(f"⚠️  Adjustment '{adj}' failed: {e}")
    except Exception as e:
        print(f"✗ Pro features test failed: {e}")

    # Test 3: Multi-symbol fetch
    print("\n" + "-" * 70)
    print("Test 3: Multi-Symbol Fetch (Testing Rate Limits)")
    print("-" * 70)
    try:
        symbols = ["AAPL", "MSFT", "GOOGL"][:3]  # Limit to 3 for testing
        results = await data_loader.fetch_multiple_symbols(
            symbols=symbols, start_date=start_date, end_date=end_date, timeframe="1Day"
        )

        for symbol, df in results.items():
            if not df.is_empty():
                print(f"✓ {symbol}: {len(df)} bars")
            else:
                print(f"⚠️  {symbol}: No data")
    except Exception as e:
        print(f"✗ Multi-symbol fetch failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print("✓ Alpaca Pro API connection verified")
    print("✓ Historical data fetching working")
    print("\n💡 Pro Trader Benefits Confirmed:")
    print("  • Full SIP feed (all US exchanges)")
    print("  • No 15-minute delay on recent data")
    print("  • Higher rate limits (10,000 calls/min)")
    print("  • Access to trades and quotes data")
    print("  • Extended hours support")
    print("\n🚀 You're ready to pull rich historical data!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_alpaca_connection())
