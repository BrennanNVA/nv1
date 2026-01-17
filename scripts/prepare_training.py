#!/usr/bin/env python3
"""
Data Readiness & Training Preparation Script

Checks data availability and prepares for model training.
Run this before training to ensure everything is ready.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Load environment variables first (before importing config)
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")  # Load from project root
load_dotenv()  # Also try current directory

sys.path.insert(0, str(project_root))

from src.nova.core.config import load_config
from src.nova.data.loader import DataLoader
from src.nova.data.storage import StorageService
from src.nova.features.technical import TechnicalFeatures


async def check_data_availability() -> dict:
    """Check if we can fetch data from available sources."""
    print("=" * 60)
    print("DATA AVAILABILITY CHECK")
    print("=" * 60)

    config = load_config()
    data_loader = DataLoader(config.data)

    results = {
        "alpaca_configured": bool(config.data.alpaca_api_key and config.data.alpaca_secret_key),
        "alpaca_available": data_loader.alpaca_client is not None,
        "yahooquery_available": False,
        "test_fetch_success": False,
        "test_symbol": None,
        "test_rows": 0,
    }

    # Check yahooquery availability
    try:
        from yahooquery import Ticker

        results["yahooquery_available"] = True
    except ImportError:
        results["yahooquery_available"] = False

    print("\n✓ Configuration loaded")
    print(f"  Symbols: {config.data.symbols}")
    print(f"  Timeframe: {config.data.default_timeframe}")
    print(f"  Lookback: {config.data.lookback_periods} periods")

    print("\n📊 Data Sources:")
    print(f"  Alpaca API configured: {results['alpaca_configured']}")
    print(f"  Alpaca client initialized: {results['alpaca_available']}")
    print(f"  yahooquery available: {results['yahooquery_available']}")

    # Test data fetch
    if results["alpaca_available"] or results["yahooquery_available"]:
        test_symbol = config.data.symbols[0] if config.data.symbols else "AAPL"
        print(f"\n🧪 Testing data fetch for {test_symbol}...")

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last 30 days

            df = await data_loader.fetch_historical_bars(
                symbol=test_symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=config.data.default_timeframe,
            )

            if not df.is_empty():
                results["test_fetch_success"] = True
                results["test_symbol"] = test_symbol
                results["test_rows"] = len(df)

                print(f"  ✓ Successfully fetched {len(df)} rows")
                print(f"  ✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"  ✓ Columns: {', '.join(df.columns)}")
                print("  ✓ Sample data:")
                print(df.head(3))
            else:
                print("  ✗ Fetch returned empty DataFrame")
        except Exception as e:
            print(f"  ✗ Fetch failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("\n⚠️  No data sources available!")
        print("  Please configure Alpaca API keys or install yahooquery")

    return results


async def check_database() -> dict:
    """Check database connectivity and schema."""
    print("\n" + "=" * 60)
    print("DATABASE CHECK")
    print("=" * 60)

    config = load_config()
    storage = StorageService(config.data)

    results = {
        "connected": False,
        "schema_initialized": False,
        "error": None,
    }

    try:
        await storage.connect()
        results["connected"] = True
        print("✓ Database connection successful")

        # Check if schema exists (try a simple query)
        try:
            await storage.query("SELECT 1")
            results["schema_initialized"] = True
            print("✓ Database schema accessible")
        except Exception as e:
            print(f"⚠️  Schema may not be initialized: {e}")
            print("  Run: python -m src.nova.main --init-db")

        await storage.disconnect()
    except Exception as e:
        results["error"] = str(e)
        print(f"✗ Database connection failed: {e}")
        print("  Ensure TimescaleDB is running: docker-compose up -d")

    return results


async def check_feature_generation() -> dict:
    """Test feature generation pipeline."""
    print("\n" + "=" * 60)
    print("FEATURE GENERATION CHECK")
    print("=" * 60)

    config = load_config()
    data_loader = DataLoader(config.data)
    technical_features = TechnicalFeatures(config.technical)

    results = {
        "success": False,
        "feature_count": 0,
        "error": None,
    }

    # Fetch sample data
    test_symbol = config.data.symbols[0] if config.data.symbols else "AAPL"
    print(f"\n🧪 Testing feature generation for {test_symbol}...")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.data.lookback_periods)

        df = await data_loader.fetch_historical_bars(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=config.data.default_timeframe,
        )

        if df.is_empty():
            results["error"] = "No data fetched"
            print("  ✗ No data available for feature generation")
            return results

        print(f"  ✓ Fetched {len(df)} rows")

        # Generate features
        features_df = technical_features.calculate_ml_features(
            df,
            apply_ffd=True,
            apply_zscore=True,
        )

        if features_df.is_empty():
            results["error"] = "Feature generation returned empty DataFrame"
            print("  ✗ Feature generation failed")
            return results

        results["success"] = True
        results["feature_count"] = len(features_df.columns)

        print(f"  ✓ Generated {len(features_df.columns)} features")
        print(f"  ✓ Feature rows: {len(features_df)}")
        print(f"  ✓ Sample features: {list(features_df.columns[:10])}")

    except Exception as e:
        results["error"] = str(e)
        print(f"  ✗ Feature generation failed: {e}")
        import traceback

        traceback.print_exc()

    return results


async def estimate_training_data() -> dict:
    """Estimate how much training data we can fetch."""
    print("\n" + "=" * 60)
    print("TRAINING DATA ESTIMATE")
    print("=" * 60)

    config = load_config()
    data_loader = DataLoader(config.data)

    results = {
        "symbols": config.data.symbols,
        "estimated_rows_per_symbol": 0,
        "total_estimated_rows": 0,
        "date_range": None,
    }

    if not config.data.symbols:
        print("⚠️  No symbols configured")
        return results

    # Estimate based on lookback periods
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.data.lookback_periods)

    results["date_range"] = {
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
    }

    # For daily data, estimate ~252 rows per year
    if config.data.default_timeframe == "1Day":
        estimated_rows = config.data.lookback_periods
    else:
        # Rough estimate for other timeframes
        estimated_rows = config.data.lookback_periods

    results["estimated_rows_per_symbol"] = estimated_rows
    results["total_estimated_rows"] = estimated_rows * len(config.data.symbols)

    print("\n📈 Training Data Estimate:")
    print(f"  Symbols: {len(config.data.symbols)}")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    print(f"  Estimated rows per symbol: ~{estimated_rows}")
    print(f"  Total estimated rows: ~{results['total_estimated_rows']}")
    print(f"  Timeframe: {config.data.default_timeframe}")

    return results


async def main():
    """Run all checks and provide summary."""
    print("\n" + "=" * 60)
    print("NOVA AETUS - TRAINING PREPARATION CHECK")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    # Run all checks
    data_check = await check_data_availability()
    db_check = await check_database()
    feature_check = await check_feature_generation()
    training_estimate = await estimate_training_data()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)

    all_ready = True

    if not data_check["test_fetch_success"]:
        print("\n❌ DATA FETCH: Not ready")
        print("   → Configure Alpaca API keys in .env")
        print("   → Or ensure yahooquery is installed: pip install yahooquery")
        all_ready = False
    else:
        print("\n✅ DATA FETCH: Ready")
        print(
            f"   → Successfully fetched {data_check['test_rows']} rows for {data_check['test_symbol']}"
        )

    if not db_check["connected"]:
        print("\n❌ DATABASE: Not ready")
        print("   → Start TimescaleDB: docker-compose up -d")
        print("   → Initialize schema: python -m src.nova.main --init-db")
        all_ready = False
    else:
        print("\n✅ DATABASE: Ready")
        if not db_check["schema_initialized"]:
            print("   ⚠️  Schema may need initialization")

    if not feature_check["success"]:
        print("\n❌ FEATURE GENERATION: Not ready")
        print(f"   → Error: {feature_check.get('error', 'Unknown')}")
        all_ready = False
    else:
        print("\n✅ FEATURE GENERATION: Ready")
        print(f"   → Generated {feature_check['feature_count']} features")

    print("\n" + "=" * 60)
    if all_ready:
        print("✅ SYSTEM READY FOR TRAINING")
        print("\nNext steps:")
        print("1. Run training script: python scripts/train_models.py")
        print("2. Or use training pipeline directly")
    else:
        print("⚠️  SYSTEM NOT FULLY READY")
        print("\nPlease address the issues above before training.")
    print("=" * 60 + "\n")

    return all_ready


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
