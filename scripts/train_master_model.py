#!/usr/bin/env python3
"""
Train Master Ensemble Model - Renaissance-style unified model.

This script trains the master ensemble model that learns how to optimally combine
predictions from all individual symbol models, capturing cross-symbol patterns.

Usage:
    python scripts/train_master_model.py --symbols AAPL MSFT --years 2
    python scripts/train_master_model.py --all --years 3
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nova.core.config import load_config
from src.nova.data.loader import DataLoader
from src.nova.features.technical import TechnicalFeatures
from src.nova.models.master_ensemble import MasterEnsembleModel
from src.nova.models.predictor import ModelRegistry


async def train_master_model(
    symbols: list[str] = None,
    years: int = 2,
    meta_learner: str = "xgboost",
) -> bool:
    """Train master ensemble model on historical predictions."""
    print("=" * 60)
    print("NOVA AETUS - MASTER ENSEMBLE MODEL TRAINING")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Load configuration
    config = load_config()

    # Get symbols
    if symbols is None:
        symbols = config.data.symbols
        if not symbols:
            print("❌ No symbols configured")
            return False

    print(f"📊 Training master model for {len(symbols)} symbols")
    print(f"   Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}\n")

    # Initialize components
    data_loader = DataLoader(config.data)
    technical_features = TechnicalFeatures(config.technical)
    model_registry = ModelRegistry(project_root / "models")

    # Check that individual models exist
    available_symbols = model_registry.get_available_symbols()
    missing_symbols = [s for s in symbols if s not in available_symbols]
    if missing_symbols:
        print(f"⚠️  Warning: No individual models found for: {', '.join(missing_symbols)}")
        print("   Training master model only on symbols with individual models")
        symbols = [s for s in symbols if s in available_symbols]

    if not symbols:
        print("❌ No symbols with trained individual models found")
        return False

    print(f"✅ Training on {len(symbols)} symbols with individual models\n")

    # Initialize master model
    master_model = MasterEnsembleModel(
        model_dir=project_root / "models",
        meta_learner_type=meta_learner,
        use_cross_symbol_features=True,
    )

    # Prepare training data
    print("📅 Preparing training data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"   Duration: ~{years} years\n")

    training_data = []

    # For each symbol, collect historical predictions and outcomes
    for symbol in symbols:
        print(f"   Processing {symbol}...", end=" ", flush=True)

        try:
            # Fetch historical data
            df = await data_loader.fetch_historical_bars(
                symbol=symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                timeframe=config.data.default_timeframe,
            )

            if df.is_empty():
                print("❌ (no data)")
                continue

            # Calculate features
            features_df = technical_features.calculate_ml_features(
                df, apply_ffd=True, apply_zscore=True
            )

            if features_df.is_empty():
                print("❌ (no features)")
                continue

            # Get predictor for this symbol
            predictor = model_registry.get_predictor(symbol)
            if not predictor:
                print("❌ (no model)")
                continue

            # Collect predictions for each time step
            feature_names = technical_features.get_feature_names()
            feature_cols = [col for col in feature_names if col in features_df.columns]

            if not feature_cols:
                print("❌ (no feature cols)")
                continue

            # For each row (except last few for which we can't calculate future returns)
            # We'll use a rolling window approach
            window_size = 20  # Use last 20 days for each prediction
            min_samples = 50  # Need at least 50 samples

            if len(features_df) < min_samples:
                print("❌ (insufficient data)")
                continue

            symbol_predictions = []
            for i in range(window_size, len(features_df) - 5):  # Leave 5 days for future return
                # Get features for this time step
                window_features = features_df[i - window_size : i + 1]
                latest_features = window_features.tail(1).select(feature_cols)

                # Get prediction from individual model
                try:
                    pred_df = predictor.predict_with_confidence(latest_features)
                    score = float(pred_df["prediction"][0]) * 2 - 1
                    confidence = float(pred_df["confidence"][0])

                    # Calculate future return (target)
                    current_price = float(df["close"][i])
                    future_price = float(df["close"][i + 5])  # 5 days ahead
                    future_return = (future_price - current_price) / current_price

                    symbol_predictions.append(
                        {
                            "score": score,
                            "confidence": confidence,
                            "prediction": float(pred_df["prediction"][0]),
                            "future_return": future_return,
                            "timestamp": i,
                        }
                    )
                except Exception:
                    continue

            if len(symbol_predictions) < min_samples:
                print("❌ (insufficient predictions)")
                continue

            # Store for later aggregation
            training_data.append(
                {
                    "symbol": symbol,
                    "predictions": symbol_predictions,
                    "df": df,
                    "features_df": features_df,
                }
            )

            print(f"✅ ({len(symbol_predictions)} samples)")

        except Exception as e:
            print(f"❌ (error: {e})")
            continue

    if not training_data:
        print("\n❌ No training data collected")
        return False

    print(f"\n✅ Collected training data from {len(training_data)} symbols\n")

    # Aggregate training data: for each time step, collect predictions from all symbols
    print("🔄 Aggregating training examples...")
    aggregated_examples = []

    # Find common time range (use minimum length)
    min_length = min(len(td["predictions"]) for td in training_data)
    print(f"   Using {min_length} time steps (minimum across all symbols)\n")

    for t in range(min_length):
        # Collect predictions from all symbols at this time step
        individual_predictions = {}
        market_data_list = []

        for td in training_data:
            symbol = td["symbol"]
            pred_data = td["predictions"][t]

            individual_predictions[symbol] = {
                "score": pred_data["score"],
                "confidence": pred_data["confidence"],
                "prediction": pred_data["prediction"],
            }

            # Get market data for this symbol at this time step
            df = td["df"]
            features_df = td["features_df"]
            idx = pred_data["timestamp"]

            if idx < len(df) and idx < len(features_df):
                volatility = (
                    float(df["close"][max(0, idx - 20) : idx + 1].std() / df["close"][idx])
                    if idx > 0
                    else 0.015
                )
                trend_strength = (
                    float(features_df["adx"][idx])
                    if "adx" in features_df.columns and idx < len(features_df)
                    else 20.0
                )
                returns = (
                    df["close"]
                    .pct_change()
                    .drop_nulls()
                    .to_numpy()[max(0, idx - 20) : idx + 1]
                    .tolist()
                )

                market_data_list.append(
                    {
                        "volatility": volatility,
                        "trend_strength": trend_strength,
                        "returns": returns,
                    }
                )

        # Calculate aggregate market data
        if market_data_list:
            avg_volatility = sum(d["volatility"] for d in market_data_list) / len(market_data_list)
            avg_trend = sum(d["trend_strength"] for d in market_data_list) / len(market_data_list)
            all_returns = []
            for d in market_data_list:
                all_returns.extend(d.get("returns", []))
            aggregate_market_data = {
                "volatility": avg_volatility,
                "trend_strength": avg_trend,
                "returns": all_returns[-20:] if len(all_returns) >= 20 else all_returns,
            }
        else:
            aggregate_market_data = None

        # For each symbol, create a training example
        for td in training_data:
            symbol = td["symbol"]
            pred_data = td["predictions"][t]
            future_return = pred_data["future_return"]

            aggregated_examples.append(
                {
                    "individual_predictions": individual_predictions.copy(),
                    "market_data": aggregate_market_data,
                    "target": future_return,  # Target: future return for this symbol
                    "target_symbol": symbol,
                }
            )

    print(f"✅ Created {len(aggregated_examples)} training examples\n")

    # Train master model
    print("🎯 Training master ensemble model...")
    print(f"   Meta-learner: {meta_learner}")
    print(f"   Training examples: {len(aggregated_examples)}\n")

    # Group by target symbol for training
    # We'll train one master model that works for all symbols
    # (or we could train per-symbol, but unified is more Renaissance-like)
    metrics = master_model.train(aggregated_examples, target_symbol="ALL")

    print("\n📊 Training Metrics:")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   MAE: {metrics['mae']:.6f}")
    print(f"   Correlation: {metrics['correlation']:.4f}")
    print(f"   Features: {metrics['n_features']}")
    print(f"   Samples: {metrics['n_samples']}\n")

    # Save master model
    model_path = master_model.save()
    print(f"✅ Master model saved to: {model_path}")
    print(f"   Completed: {datetime.now().isoformat()}\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train master ensemble model that combines individual symbol models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to train on (default: all from config)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train on all symbols from config",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Years of historical data to use (default: 2)",
    )
    parser.add_argument(
        "--meta-learner",
        choices=["xgboost", "ridge", "linear"],
        default="xgboost",
        help="Meta-learner type (default: xgboost)",
    )

    args = parser.parse_args()

    symbols = args.symbols
    if args.all:
        symbols = None  # Will use all from config

    success = asyncio.run(train_master_model(symbols, args.years, args.meta_learner))
    sys.exit(0 if success else 1)
