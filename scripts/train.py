#!/usr/bin/env python3
"""
Unified Training Command - Train individual models or master ensemble model.

Usage:
    python scripts/train.py all              # Train all individual symbol models
    python scripts/train.py all --years 5    # Train all with 5 years of data
    python scripts/train.py master           # Train master ensemble model
    python scripts/train.py master --years 2 # Train master with 2 years of data
    python scripts/train.py AAPL MSFT        # Train specific symbols
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import training functions
from scripts.train_master_model import train_master_model
from scripts.train_models import train_models_for_deployment


def main():
    # Simple argument parsing - handle both positional and flag-based arguments
    args = sys.argv[1:]

    if not args:
        print(
            """
Usage:
  ./train all              # Train all individual symbol models (3 years)
  ./train all 5            # Train all individual models (5 years)
  ./train master           # Train master ensemble model (2 years)
  ./train master 3         # Train master model (3 years)
  ./train AAPL MSFT        # Train specific symbols (3 years)
  ./train AAPL MSFT 5      # Train specific symbols (5 years)
        """
        )
        sys.exit(1)

    # Check if last argument is a number (years)
    years = None
    meta_learner = "xgboost"

    if args and args[-1].isdigit():
        years = int(args[-1])
        args = args[:-1]  # Remove years from args
    elif args and args[-1] in ["--meta-learner", "-m"]:
        # Handle --meta-learner flag
        if len(args) >= 2:
            meta_learner = args[-1] if args[-1] in ["xgboost", "ridge", "linear"] else "xgboost"
            args = args[:-2]

    if not args:
        print("Error: No command specified")
        sys.exit(1)

    command = args[0].lower()

    # Handle 'all' command - train all individual models
    if command == "all":
        years_to_use = years if years is not None else 3
        print(f"🚀 Training all individual symbol models ({years_to_use} years)...\n")
        success = asyncio.run(
            train_models_for_deployment(symbols=None, years=years_to_use, train_all=True)
        )
        sys.exit(0 if success else 1)

    # Handle 'master' command - train master ensemble model
    elif command == "master":
        years_to_use = years if years is not None else 2
        print(f"🎯 Training master ensemble model ({years_to_use} years)...\n")
        success = asyncio.run(
            train_master_model(symbols=None, years=years_to_use, meta_learner=meta_learner)
        )
        sys.exit(0 if success else 1)

    # Handle symbol names - train specific individual models
    else:
        symbols = args  # All args are symbol names
        years_to_use = years if years is not None else 3
        print(
            f"🚀 Training individual models for: {', '.join(symbols)} ({years_to_use} years)...\n"
        )
        success = asyncio.run(
            train_models_for_deployment(symbols=symbols, years=years_to_use, train_all=False)
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
