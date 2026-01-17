#!/usr/bin/env python3
"""
Model Training Script for Friday Deployment

Trains models on historical data using the research-backed best practices.
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nova.core.config import load_config
from src.nova.data.loader import DataLoader
from src.nova.data.storage import StorageService
from src.nova.models.training_pipeline import TrainingPipeline


async def train_models_for_deployment(
    symbols: Optional[list[str]] = None, years: Optional[int] = None, train_all: bool = False
):
    """Train models ready for Friday deployment."""
    print("=" * 60)
    print("NOVA AETUS - MODEL TRAINING FOR DEPLOYMENT")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Load configuration
    config = load_config()

    # Initialize components
    print("Initializing components...")
    data_loader = DataLoader(config.data)

    # Try to connect to database (optional - training can work without it)
    storage = None
    try:
        storage = StorageService(config.data)
        await storage.connect()
        await storage.init_schema()
        print("✓ Database connected")
    except Exception as e:
        print(f"⚠️  Database not available (training will continue): {e}")
        storage = None

    # Initialize training pipeline
    pipeline = TrainingPipeline(
        config=config,
        data_loader=data_loader,
        storage=storage,
    )

    # Determine training date range
    # Use command-line argument if provided, otherwise default to 3 years
    years_to_use = years if years is not None else 3
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_to_use * 365)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    print("\n📅 Training Date Range:")
    print(f"   Start: {start_date_str}")
    print(f"   End: {end_date_str}")
    print(f"   Duration: ~{years_to_use} years")

    # Get symbols from command-line argument if provided, otherwise from config, otherwise defaults
    # Check for --all flag first
    if train_all:
        symbols = config.data.symbols
        if not symbols:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]  # Default
            print(f"\n⚠️  No symbols in config, using defaults: {symbols}")
        else:
            print(f"\n📊 Training ALL symbols from config: {len(symbols)} symbols")
    elif symbols is None or len(symbols) == 0:
        symbols = config.data.symbols
        if not symbols:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]  # Default
            print(f"\n⚠️  No symbols in config or args, using defaults: {symbols}")
        else:
            print(f"\n📊 Training Symbols (from config): {len(symbols)} symbols")
    else:
        print(f"\n📊 Training Symbols (from command-line): {symbols}")

    # Train models
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    try:
        results = await pipeline.train_universe(
            symbols=symbols,
            start_date=start_date_str,
            end_date=end_date_str,
            use_walk_forward=True,  # Use walk-forward validation
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        # Print summary
        # train_universe returns a summary dict with "successful", "failed", etc.
        successful_count = results.get("successful", 0)
        total_symbols = results.get("total_symbols", 0)

        if successful_count > 0:
            print("\n✅ Training successful!")
            print(f"   Models trained: {successful_count}/{total_symbols}")
            print(f"   Models saved to: {pipeline.model_dir}")

            print("\n📊 Training Summary:")
            print(f"   Total symbols: {total_symbols}")
            print(f"   Successful: {successful_count}")
            print(f"   Failed: {results.get('failed', 0)}")

            if results.get("failed_symbols"):
                print(f"   Failed symbols: {results.get('failed_symbols')}")

            # Calculate average metrics from results
            all_results = results.get("results", [])
            if all_results:
                metrics_list = [
                    r.get("metrics", {})
                    for r in all_results
                    if r.get("success") and r.get("metrics")
                ]
                if metrics_list:
                    avg_accuracy = sum(m.get("accuracy", 0) for m in metrics_list) / len(
                        metrics_list
                    )
                    avg_precision = sum(m.get("precision", 0) for m in metrics_list) / len(
                        metrics_list
                    )
                    avg_recall = sum(m.get("recall", 0) for m in metrics_list) / len(metrics_list)
                    avg_f1 = sum(m.get("f1_score", 0) for m in metrics_list) / len(metrics_list)

                    print("\n📈 Average Metrics:")
                    print(f"   Accuracy: {avg_accuracy:.3f}")
                    print(f"   Precision: {avg_precision:.3f}")
                    print(f"   Recall: {avg_recall:.3f}")
                    print(f"   F1 Score: {avg_f1:.3f}")
        else:
            print("\n❌ Training failed!")
            print("   No models were successfully trained.")
            if results.get("failed_symbols"):
                print(f"   Failed symbols: {results.get('failed_symbols')}")
            return False

        # Cleanup
        if storage:
            await storage.disconnect()

        print("\n✅ Models ready for deployment!")
        print(f"   Model directory: {pipeline.model_dir}")
        print(f"   Completed: {datetime.now().isoformat()}")

        return True

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback

        traceback.print_exc()

        if storage:
            await storage.disconnect()

        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost models on historical data using NPMM labeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all symbols from config with default 5 years (recommended)
  python scripts/train_models.py --all

  # Train specific symbols with 5 years of data
  python scripts/train_models.py --symbols AAPL MSFT GOOGL --years 5

  # Train single symbol with 3 years (quick test)
  python scripts/train_models.py --symbols AAPL --years 3

  # Train all symbols with 10 years of data
  python scripts/train_models.py --all --years 10

Quick Commands:
  ./train AAPL           # Train AAPL (3 years)
  ./train all            # Train all symbols (5 years)
  ./train all 10         # Train all symbols (10 years)
        """,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Stock symbols to train on (overrides config.toml). Example: --symbols AAPL MSFT GOOGL. Use --all to train all symbols from config.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Train all symbols from config.toml (25 symbols)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Number of years of historical data to use (default: 3)",
    )

    args = parser.parse_args()

    # If --all is specified, ignore --symbols
    if args.all:
        success = asyncio.run(
            train_models_for_deployment(symbols=None, years=args.years, train_all=True)
        )
    else:
        success = asyncio.run(
            train_models_for_deployment(symbols=args.symbols, years=args.years, train_all=False)
        )
    sys.exit(0 if success else 1)
