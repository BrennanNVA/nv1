# Training Commands Quick Reference

## Unified Training Command

Use the `./train` script or `python scripts/train.py` for all training operations.

## Commands

### Train All Individual Models

Train individual models for all symbols in your config:

```bash
# Default: 3 years of data
./train all

# Custom years
./train all 5
```

**What it does:**
- Trains one model per symbol (24 models)
- Each model learns symbol-specific patterns
- Models saved to `models/` directory

### Train Master Ensemble Model

Train the master model that combines all individual models:

```bash
# Default: 2 years of data
./train master

# Custom years
./train master 3
```

**What it does:**
- Collects predictions from all individual symbol models
- Learns optimal combination weights
- Captures cross-symbol patterns (sector correlations)
- Saves as `models/master_model_{date}.json`

**Prerequisites:** Individual models must be trained first!

### Train Specific Symbols

Train individual models for specific symbols:

```bash
# Default: 3 years
./train AAPL MSFT GOOGL

# Custom years
./train AAPL MSFT 5
```

## Typical Workflow

1. **Train individual models first:**
   ```bash
   ./train all
   ```

2. **Then train master model:**
   ```bash
   ./train master
   ```

3. **Retrain periodically:**
   - Individual models: Monthly or when performance degrades
   - Master model: Quarterly or when adding new symbols

## Examples

```bash
# Train everything (individual + master)
./train all
./train master

# Train with more historical data
./train all 5      # 5 years for individual models
./train master 3   # 3 years for master model

# Train specific symbols
./train AAPL MSFT GOOGL
./train NVDA TSLA 5
```

## Python Script Usage

You can also use the Python script directly:

```bash
python scripts/train.py all
python scripts/train.py master
python scripts/train.py AAPL MSFT
```

## Model Files

After training, you'll find:

- **Individual models:** `models/{SYMBOL}_{DATE}.json` (e.g., `AAPL_20250117.json`)
- **Master model:** `models/master_model_{DATE}.json` (e.g., `master_model_20250117.json`)
- **Training reports:** `models/training_report_{TIMESTAMP}.json`

## Notes

- Individual models are required before training master model
- Master model automatically loads during trading if available
- Training time: Individual models ~10-30 min, Master model ~5-15 min
- More years = better accuracy but longer training time
