#!/usr/bin/env python3
"""
Indicator Ranking & Validation Script

Ranks technical indicators by importance using actual training data,
then compares results with research findings to validate indicator effectiveness.

Usage:
    python scripts/validate_indicator_ranking.py [--symbols AAPL MSFT] [--years 5]
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Load environment variables
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")
sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl
import xgboost as xgb

from src.nova.core.config import load_config
from src.nova.data.loader import DataLoader
from src.nova.features.technical import TechnicalFeatures
from src.nova.models.trainer import ModelTrainer, NPMMLabeler


@dataclass
class IndicatorRanking:
    """Indicator ranking result."""

    feature_name: str
    importance: float
    rank: int
    category: str
    research_rank: Optional[int] = None


class IndicatorValidator:
    """Validates and ranks technical indicators against research findings."""

    # Research-backed top indicators (from RESEARCH_SUMMARY.md)
    RESEARCH_RANKINGS = {
        "squeeze_pro": 1,
        "ppo": 2,
        "macd": 3,
        "roc_63": 4,
        "rsi_63": 5,
    }

    # Expected high performers (based on research)
    EXPECTED_TOP_INDICATORS = [
        "squeeze_pro",
        "ppo",
        "macd",
        "macd_histogram",
        "roc_63",
        "rsi_63",
        "adx",
        "atr",
        "vwap_dev",
    ]

    def __init__(self, config):
        """Initialize validator."""
        self.config = config
        self.data_loader = DataLoader(config.data)
        self.technical_features = TechnicalFeatures(config.technical)
        self.trainer = ModelTrainer(config.ml)

    async def fetch_training_data(self, symbol: str, years: int = 5) -> pl.DataFrame:
        """Fetch historical data for training."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        print(f"Fetching {years} years of data for {symbol}...")
        df = await self.data_loader.fetch_historical_bars(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            timeframe="1Day",
        )

        if df.is_empty():
            raise ValueError(f"No data fetched for {symbol}")

        print(f"Fetched {len(df)} rows for {symbol}")
        return df

    def calculate_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate all technical features."""
        print("Calculating technical features...")
        features_df = self.technical_features.calculate_ml_features(
            df,
            apply_ffd=True,
            apply_zscore=True,
        )
        print(f"Generated {len(features_df.columns)} features for {len(features_df)} rows")
        return features_df

    def generate_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate NPMM labels for training."""
        print("Generating NPMM labels...")
        labeler = NPMMLabeler()
        labeled_df = labeler.generate_binary_labels(df, "close")

        # Filter valid labels
        valid_df = labeled_df.filter(pl.col("label") >= 0)

        labels = valid_df["label"].sum()
        total = len(valid_df)
        print(
            f"Generated {total} labeled samples: {labels} buy signals, {total - labels} sell signals"
        )

        return valid_df

    def rank_indicators(
        self, features_df: pl.DataFrame, labels: pl.Series, top_n: int = 30
    ) -> list[IndicatorRanking]:
        """
        Rank indicators by XGBoost feature importance.

        Args:
            features_df: DataFrame with features
            labels: Series with binary labels
            top_n: Number of top indicators to return

        Returns:
            List of IndicatorRanking objects sorted by importance
        """
        print(f"\nRanking indicators (top {top_n})...")

        # Convert to numpy for XGBoost
        feature_names = [
            col
            for col in features_df.columns
            if col not in ["symbol", "timestamp", "label", "target", "close"]
        ]
        X = features_df.select(feature_names).to_numpy()
        y = labels.to_numpy()

        # Train quick model for feature importance
        print("Training quick model for feature importance...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            device="cuda" if self.trainer.gpu_available else "cpu",
            tree_method="hist",
            random_state=42,
            eval_metric="logloss",
        )

        model.fit(X, y, verbose=False)

        # Get feature importance (gain-based)
        importance_dict = model.get_booster().get_score(importance_type="gain")

        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        # Map feature indices to names
        rankings = []
        for rank, (feature_key, importance) in enumerate(sorted_features[:top_n], start=1):
            # Feature keys are like "f0", "f1", etc. - map to actual names
            try:
                idx = int(feature_key.replace("f", ""))
                if idx < len(feature_names):
                    feature_name = feature_names[idx]
                else:
                    feature_name = feature_key
            except ValueError:
                feature_name = feature_key

            # Categorize indicator
            category = self._categorize_indicator(feature_name)

            # Check research ranking
            research_rank = self.RESEARCH_RANKINGS.get(
                feature_name.lower().replace("_zscore", "").replace("_ffd", "")
            )

            rankings.append(
                IndicatorRanking(
                    feature_name=feature_name,
                    importance=importance,
                    rank=rank,
                    category=category,
                    research_rank=research_rank,
                )
            )

        return rankings

    def _categorize_indicator(self, feature_name: str) -> str:
        """Categorize indicator by type."""
        name_lower = feature_name.lower()

        if "squeeze" in name_lower:
            return "Volatility Squeeze"
        elif any(x in name_lower for x in ["rsi", "roc", "ppo", "stoch", "williams"]):
            return "Momentum"
        elif "macd" in name_lower:
            return "Trend (MACD)"
        elif any(x in name_lower for x in ["adx", "di"]):
            return "Trend (ADX)"
        elif any(x in name_lower for x in ["sma", "ema", "ma"]):
            return "Moving Average"
        elif any(x in name_lower for x in ["bb", "bollinger"]):
            return "Volatility (BB)"
        elif any(x in name_lower for x in ["kc", "keltner"]):
            return "Volatility (KC)"
        elif any(x in name_lower for x in ["atr", "natr"]):
            return "Volatility (ATR)"
        elif any(x in name_lower for x in ["vwap", "obv", "volume"]):
            return "Volume"
        elif "return" in name_lower:
            return "Returns"
        elif "price" in name_lower or "cross" in name_lower:
            return "Price Pattern"
        elif "_ffd" in name_lower:
            return "Fractional Diff"
        elif "_zscore" in name_lower:
            return "Normalized"
        else:
            return "Other"

    def compare_with_research(self, rankings: list[IndicatorRanking]) -> dict[str, Any]:
        """
        Compare actual rankings with research findings.

        Returns:
            Dictionary with comparison results
        """
        print("\n" + "=" * 60)
        print("COMPARING WITH RESEARCH FINDINGS")
        print("=" * 60)

        # Find research indicators in actual rankings
        research_found = {}
        for ranking in rankings:
            base_name = ranking.feature_name.lower().replace("_zscore", "").replace("_ffd", "")
            # Check both exact match and if base name contains research indicator name
            for research_name, research_rank in self.RESEARCH_RANKINGS.items():
                if (
                    base_name == research_name
                    or base_name.endswith(f"_{research_name}")
                    or research_name in base_name
                ):
                    if research_name not in research_found:  # Only take first match
                        research_found[research_name] = {
                            "research_rank": research_rank,
                            "actual_rank": ranking.rank,
                            "importance": ranking.importance,
                            "category": ranking.category,
                            "feature_name": ranking.feature_name,
                        }
                        break

        # Check if expected top indicators are in top 30
        top_30_names = {
            r.feature_name.lower().replace("_zscore", "").replace("_ffd", "") for r in rankings[:30]
        }
        expected_found = {name: name in top_30_names for name in self.EXPECTED_TOP_INDICATORS}

        # Calculate correlation between research rank and actual rank
        research_correlation = None
        if len(research_found) >= 3:
            research_ranks = [v["research_rank"] for v in research_found.values()]
            actual_ranks = [v["actual_rank"] for v in research_found.values()]
            research_correlation = np.corrcoef(research_ranks, actual_ranks)[0, 1]

        results = {
            "research_indicators_found": research_found,
            "expected_indicators_in_top30": expected_found,
            "research_correlation": research_correlation,
            "total_indicators_ranked": len(rankings),
        }

        # Print summary
        print(f"\nResearch Indicators Found: {len(research_found)}/{len(self.RESEARCH_RANKINGS)}")
        for name, data in sorted(research_found.items(), key=lambda x: x[1]["research_rank"]):
            diff = data["actual_rank"] - data["research_rank"]
            status = "✅" if abs(diff) <= 5 else "⚠️" if abs(diff) <= 15 else "❌"
            print(
                f"  {status} {name:15} | Research: #{data['research_rank']:2} | Actual: #{data['actual_rank']:3} | "
                f"Diff: {diff:+3} | Importance: {data['importance']:.2f}"
            )

        print("\nExpected Indicators in Top 30:")
        for name, found in expected_found.items():
            status = "✅" if found else "❌"
            print(f"  {status} {name}")

        if research_correlation is not None:
            print(f"\nResearch Correlation: {research_correlation:.3f}")
            if research_correlation > 0.7:
                print("  ✅ Strong correlation - Research findings validated!")
            elif research_correlation > 0.4:
                print("  ⚠️  Moderate correlation - Some alignment with research")
            else:
                print("  ❌ Low correlation - Research findings need review")

        return results

    def validate_longer_periods(self, rankings: list[IndicatorRanking]) -> dict[str, Any]:
        """
        Validate if longer-period indicators (63-day) outperform shorter periods.

        Returns:
            Dictionary with validation results
        """
        print("\n" + "=" * 60)
        print("VALIDATING LONGER-PERIOD INDICATORS (63-day)")
        print("=" * 60)

        comparisons = {
            "rsi_14_vs_63": None,
            "roc_12_vs_63": None,
        }

        # Find RSI 14 vs 63 - check all rankings, not just top 30
        # RSI(14) might be "rsi", "rsi_zscore", etc.
        rsi_14 = None
        for r in rankings:
            name_lower = r.feature_name.lower()
            # Match "rsi" but not "rsi_63" or "rsi_..."
            if (
                name_lower == "rsi" or name_lower.startswith("rsi_")
            ) and "rsi_63" not in name_lower:
                if (
                    rsi_14 is None or r.importance > rsi_14.importance
                ):  # Take highest importance if multiple
                    rsi_14 = r

        rsi_63 = next((r for r in rankings if "rsi_63" in r.feature_name.lower()), None)

        if not rsi_14:
            print("\nRSI 14 vs 63:")
            print("  ⚠️  RSI(14) not found in top 30 rankings")
        if not rsi_63:
            print("\nRSI 14 vs 63:")
            print("  ⚠️  RSI(63) not found in top 30 rankings")

        if rsi_14 and rsi_63:
            comparisons["rsi_14_vs_63"] = {
                "rsi_14_rank": rsi_14.rank,
                "rsi_14_importance": rsi_14.importance,
                "rsi_63_rank": rsi_63.rank,
                "rsi_63_importance": rsi_63.importance,
                "longer_better": rsi_63.rank < rsi_14.rank,
            }

            status = "✅" if comparisons["rsi_14_vs_63"]["longer_better"] else "❌"
            print("\nRSI 14 vs 63:")
            print(f"  {status} RSI(14): Rank #{rsi_14.rank}, Importance: {rsi_14.importance:.2f}")
            print(
                f"  {'✅' if rsi_63.rank < rsi_14.rank else '❌'} RSI(63): Rank #{rsi_63.rank}, Importance: {rsi_63.importance:.2f}"
            )
            if rsi_63.rank < rsi_14.rank:
                print(f"    → RSI(63) outperforms RSI(14) by {rsi_14.rank - rsi_63.rank} ranks")

        # Find ROC 12 vs 63 - check all rankings
        # ROC(12) might be "roc_12", "roc_12_zscore", etc.
        roc_12 = None
        for r in rankings:
            name_lower = r.feature_name.lower()
            # Match "roc_12" but not "roc_63" or longer periods
            if (
                "roc_12" in name_lower
                and "roc_63" not in name_lower
                and "roc_252" not in name_lower
            ):
                if roc_12 is None or r.importance > roc_12.importance:
                    roc_12 = r

        roc_63 = next((r for r in rankings if "roc_63" in r.feature_name.lower()), None)

        if not roc_12:
            print("\nROC 12 vs 63:")
            print("  ⚠️  ROC(12) not found in top 30 rankings")
        if not roc_63:
            print("\nROC 12 vs 63:")
            print("  ⚠️  ROC(63) not found in top 30 rankings")

        if roc_12 and roc_63:
            comparisons["roc_12_vs_63"] = {
                "roc_12_rank": roc_12.rank,
                "roc_12_importance": roc_12.importance,
                "roc_63_rank": roc_63.rank,
                "roc_63_importance": roc_63.importance,
                "longer_better": roc_63.rank < roc_12.rank,
            }

            status = "✅" if comparisons["roc_12_vs_63"]["longer_better"] else "❌"
            print("\nROC 12 vs 63:")
            print(f"  {status} ROC(12): Rank #{roc_12.rank}, Importance: {roc_12.importance:.2f}")
            print(
                f"  {'✅' if roc_63.rank < roc_12.rank else '❌'} ROC(63): Rank #{roc_63.rank}, Importance: {roc_63.importance:.2f}"
            )
            if roc_63.rank < roc_12.rank:
                print(f"    → ROC(63) outperforms ROC(12) by {roc_12.rank - roc_63.rank} ranks")

        return comparisons


async def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate and rank technical indicators")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL"],
        help="Symbols to analyze (default: AAPL MSFT GOOGL)",
    )
    parser.add_argument(
        "--years", type=int, default=5, help="Years of historical data (default: 5)"
    )
    parser.add_argument(
        "--top-n", type=int, default=30, help="Top N indicators to rank (default: 30)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: indicator_ranking_<timestamp>.json)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("INDICATOR RANKING & VALIDATION")
    print("=" * 60)
    print(f"Symbols: {args.symbols}")
    print(f"Years: {args.years}")
    print(f"Top N: {args.top_n}")
    print("=" * 60)

    # Load configuration
    config = load_config()
    validator = IndicatorValidator(config)

    # Aggregate rankings across symbols
    all_rankings: dict[str, list[IndicatorRanking]] = {}
    all_comparisons = []
    all_validations = []

    for symbol in args.symbols:
        try:
            print(f"\n{'='*60}")
            print(f"Analyzing {symbol}")
            print(f"{'='*60}")

            # Fetch data
            df = await validator.fetch_training_data(symbol, years=args.years)

            # Calculate features
            features_df = validator.calculate_features(df)

            # Generate labels
            labeled_df = validator.generate_labels(features_df)

            # Get feature names and labels
            feature_cols = [
                col
                for col in labeled_df.columns
                if col not in ["symbol", "timestamp", "close", "label", "target"]
            ]
            X = labeled_df.select(feature_cols)
            y = labeled_df["label"]

            # Rank indicators - use top 100 for validation (to find RSI/ROC variants)
            all_rankings_list = validator.rank_indicators(X, y, top_n=100)
            # Keep top N for display
            rankings = all_rankings_list[: args.top_n]
            all_rankings[symbol] = rankings

            # Compare with research (use top N for display)
            comparison = validator.compare_with_research(rankings)
            all_comparisons.append({symbol: comparison})

            # Validate longer periods (use all 100 rankings to find both variants)
            validation = validator.validate_longer_periods(all_rankings_list)
            all_validations.append({symbol: validation})

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Aggregate rankings across symbols
    print("\n" + "=" * 60)
    print("AGGREGATED RANKINGS (Across All Symbols)")
    print("=" * 60)

    # Average importance across symbols
    feature_importance: dict[str, list[float]] = {}
    feature_ranks: dict[str, list[int]] = {}

    for symbol, rankings in all_rankings.items():
        for ranking in rankings:
            base_name = ranking.feature_name
            if base_name not in feature_importance:
                feature_importance[base_name] = []
                feature_ranks[base_name] = []
            feature_importance[base_name].append(ranking.importance)
            feature_ranks[base_name].append(ranking.rank)

    # Calculate averages and sort by importance
    avg_rankings = sorted(
        [
            IndicatorRanking(
                feature_name=name,
                importance=np.mean(imps),
                rank=int(np.mean(feature_ranks[name])),
                category=validator._categorize_indicator(name),
                research_rank=validator.RESEARCH_RANKINGS.get(
                    name.lower().replace("_zscore", "").replace("_ffd", "")
                ),
            )
            for name, imps in feature_importance.items()
        ],
        key=lambda x: x.importance,
        reverse=True,
    )

    # Re-assign ranks based on final sorted order
    for rank_idx, ranking in enumerate(avg_rankings, start=1):
        ranking.rank = rank_idx

    # Display top N
    print(f"\nTop {args.top_n} Indicators (Averaged Across Symbols):")
    print(f"{'Rank':<6} {'Importance':<12} {'Category':<25} {'Indicator Name'}")
    print("-" * 80)
    for i, ranking in enumerate(avg_rankings[: args.top_n], 1):
        research_marker = f" [R#{ranking.research_rank}]" if ranking.research_rank else ""
        print(
            f"{i:<6} {ranking.importance:<12.2f} {ranking.category:<25} {ranking.feature_name}{research_marker}"
        )

    # Final comparison
    print("\n" + "=" * 60)
    final_comparison = validator.compare_with_research(avg_rankings[: args.top_n])
    final_validation = validator.validate_longer_periods(
        avg_rankings
    )  # Use full rankings for validation

    # Save results
    output_file = (
        args.output or f"indicator_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    results = {
        "timestamp": datetime.now().isoformat(),
        "symbols": args.symbols,
        "years": args.years,
        "top_n": args.top_n,
        "rankings": [
            {
                "feature_name": r.feature_name,
                "importance": float(r.importance),
                "rank": r.rank,
                "category": r.category,
                "research_rank": r.research_rank,
            }
            for r in avg_rankings[: args.top_n]
        ],
        "comparison_with_research": final_comparison,
        "longer_period_validation": final_validation,
        "per_symbol_rankings": {
            symbol: [
                {
                    "feature_name": r.feature_name,
                    "importance": float(r.importance),
                    "rank": r.rank,
                    "category": r.category,
                }
                for r in rankings[: args.top_n]
            ]
            for symbol, rankings in all_rankings.items()
        },
    }

    output_path = project_root / "reports" / output_file
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
