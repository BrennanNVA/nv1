# Quick Start Guide - Training Models

## 🚀 Quick Summary

**What is Model Training?** Training AI models to predict stock price movements using historical data.

**What does it do?**
1. Fetches historical stock data (or uses cached data from TimescaleDB)
2. Calculates 88+ technical indicators
3. Generates buy/sell signals using NPMM labeling
4. Trains XGBoost models with GPU acceleration (RTX 5070 Ti optimized)
5. Validates models with walk-forward optimization
6. Saves trained models for trading

**Why train models?** Models need to learn patterns from historical data to make accurate predictions. Without trained models, the system can't generate reliable trading signals.

**How long does it take?**
- **Single symbol:** 30-60 seconds (with GPU optimizations)
- **5 symbols:** 3-5 minutes
- **25 symbols (3 years):** 25-35 minutes (with database caching)
- **25 symbols (5 years):** 1-3 hours (full production)

---

## ⚡ Quick Commands (Start Here!)

### Option 1: Fast Training (3 years, recommended for quick testing)

```bash
# Recommended: Use the ./train script (easiest!)
./train all 3

# Or use Python script directly
python scripts/train_models.py --all --years 3
```

**Time:** ~25-35 minutes
**Best for:** Quick retraining, testing changes

### Option 2: Production Training (5 years, recommended for deployment)

```bash
# Recommended: Use the ./train script
./train all

# Or use Python script directly
python scripts/train_models.py --all --years 5
```

**Time:** 1-3 hours
**Best for:** Production deployment, monthly/quarterly retraining

### Quick Test (Single Symbol)

```bash
# Recommended: Use the ./train script
./train AAPL

# Or use Python script directly
python scripts/train_models.py --symbols AAPL --years 3
```

**Time:** 2-5 minutes
**Best for:** First-time testing, debugging

**Note:** The `./train` script is the easiest way - it handles environment setup automatically!

---

## 📋 Prerequisites

### Required
- ✅ Python 3.12+ installed
- ✅ Virtual environment activated
- ✅ Dependencies installed (`pip install -r requirements.txt`)
- ✅ GPU with CUDA (for fast training) - **RTX 5070 Ti recommended**

### Optional (for best performance)
- ✅ RMM installed (`pip install rmm`) - Faster GPU memory allocation
- ✅ TimescaleDB running (for data caching)

---

## 🎯 Step-by-Step: Training Your First Model

> **📋 For quick command cheat sheet, see:** `docs/guides/training/QUICK_COMMANDS.md`

### Step 1: Navigate to Project Directory

```bash
cd /home/brennan/nac/nova_aetus
```

### Step 2: Activate Virtual Environment

```bash
source venv/bin/activate
# You should see (venv) in your prompt
```

### Step 3: Verify GPU Availability

```bash
# Check if GPU is detected
nvidia-smi

# Should show your RTX 5070 Ti with CUDA version
```

### Step 4: Configure Training Settings (Optional)

Edit `config.toml` to customize training:

```toml
[ml]
# GPU optimization settings (enabled by default)
use_quantile_dmatrix = true  # 5x memory reduction
use_rmm = true  # Faster GPU memory allocation
gradient_sampling = false  # Enable for additional memory savings
feature_selection_top_n = null  # Set to 30 to use top 30 features

# Hyperparameter tuning
optuna_trials = 100  # Number of optimization trials
optuna_timeout = 3600  # Max time in seconds (1 hour)

# Training symbols
[data]
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
```

### Step 5: Train Models

**⭐ Recommended: Use the training script**

#### Quick Test (Single Symbol)

```bash
# Train single symbol (AAPL) with 3 years of data
python scripts/train_models.py --symbols AAPL --years 3

# Or use the shorthand script (if available)
./train AAPL
```

#### Train All Symbols (Production)

```bash
# Train all 25 symbols from config.toml with 3 years (Option 1: Fast)
python scripts/train_models.py --all --years 3

# Train all 25 symbols with 5 years (Option 2: Full, recommended for production)
python scripts/train_models.py --all --years 5

# Or use the shorthand script
./train all
./train all 3    # 3 years
./train all 5    # 5 years
```

#### Training Multiple Symbols

```bash
# Train specific symbols with custom years
python scripts/train_models.py --symbols AAPL MSFT GOOGL --years 3

# Or shorthand
./train AAPL MSFT GOOGL 3
```

**Available Options:**
- `--all` - Train all symbols from config.toml (25 symbols)
- `--symbols SYM1 SYM2 ...` - Train specific symbols
- `--years N` - Number of years of historical data (default: 3)
- `--help` - Show all options

**Example Commands:**
```bash
# Quick test: Single symbol, 3 years
python scripts/train_models.py --symbols AAPL --years 3

# Production: All symbols, 3 years (fast, Option 1)
python scripts/train_models.py --all --years 3

# Production: All symbols, 5 years (full, Option 2)
python scripts/train_models.py --all --years 5
```

**What happens:**
1. ✅ Fetches 4 years of historical data for AAPL
2. ✅ Calculates 88+ technical indicators
3. ✅ Generates NPMM labels (buy/sell signals)
4. ✅ Trains XGBoost model on GPU (RTX 5070 Ti)
5. ✅ Validates with walk-forward optimization
6. ✅ Saves model to `models/` directory

**Expected output:**
```
Starting universe training: 1 symbols from 2020-01-01 to 2024-12-31
Training model for AAPL (1/1)...
Preparing data for AAPL...
Calculating features for AAPL...
Generating NPMM labels for AAPL...
Training XGBoost model on GPU
Model training completed in 45.2s (samples=1000, features=88)
Model trained for AAPL: Accuracy=0.723, DSR=0.95, CV=0.715
Training complete: {'success': True, ...}
```

**Note:** Training now uses **database-first caching** - subsequent training runs will be 70-80% faster as data is loaded from TimescaleDB instead of fetching from the API.

---

## 🔧 Training Options

### Option 1: Quick Training (No Hyperparameter Tuning)

**Faster, uses default parameters:**

```python
result = await pipeline.train_universe(
    symbols=['AAPL'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    use_walk_forward=False  # Skip walk-forward validation
)
```

**Time:** ~30 seconds per symbol

### Option 2: Full Training (With Walk-Forward Validation)

**Recommended for production, validates model robustness:**

```python
result = await pipeline.train_universe(
    symbols=['AAPL'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    use_walk_forward=True  # Enable walk-forward validation
)
```

**Time:** ~60 seconds per symbol

### Option 3: Custom Date Range

**Train on specific time period:**

```python
result = await pipeline.train_universe(
    symbols=['AAPL'],
    start_date='2022-01-01',  # Custom start
    end_date='2024-12-31',    # Custom end
    use_walk_forward=True
)
```

### Option 4: Feature Selection

**Reduce features from 88+ to top 30 (faster training, less memory):**

Edit `config.toml`:
```toml
[ml]
feature_selection_top_n = 30  # Use top 30 features
```

**Benefits:**
- 60-70% memory reduction
- Faster training
- May improve accuracy (removes noise)

---

## 📊 Understanding Training Output

### Model Files

Trained models are saved to `models/` directory:

```
models/
├── AAPL_20250116.json          # Trained model
├── AAPL_20250116.json.metadata # Training metadata
├── AAPL_wfo_20250116.json      # Walk-forward optimized model
└── training_report_20250116_143022.json  # Training summary
```

### Training Metrics

**Accuracy:** Percentage of correct predictions (target: >60%)

**Precision:** Of predicted buys, how many were correct (target: >55%)

**Recall:** Of actual buys, how many were predicted (target: >50%)

**F1 Score:** Balance of precision and recall (target: >0.55)

**DSR (Deflated Sharpe Ratio):** Model robustness (target: >0.95)

**CV Score:** Cross-validation accuracy (target: >0.60)

### Example Output

```json
{
  "success": true,
  "symbol": "AAPL",
  "metrics": {
    "accuracy": 0.723,
    "precision": 0.681,
    "recall": 0.645,
    "f1_score": 0.662
  },
  "cv_results": {
    "mean_score": 0.715,
    "std_score": 0.032
  },
  "dsr": 0.97
}
```

---

## ⚡ GPU Optimization Features

### Automatic Optimizations (Enabled by Default)

1. **QuantileDMatrix:** 5x GPU memory reduction
2. **RMM (RAPIDS Memory Manager):** 5-10x faster memory allocation
3. **Async Pipeline:** Overlaps CPU preprocessing with GPU training
4. **Optuna Pruning:** Early termination of unpromising trials

### Expected Performance (RTX 5070 Ti)

- **Single symbol:** 30-60 seconds (vs 10-15 minutes on CPU)
- **5 symbols:** 3-5 minutes (vs 50-75 minutes on CPU)
- **With Optuna (100 trials):** 1-2 hours (vs 16-25 hours on CPU)

### Monitor GPU Usage

```bash
# Watch GPU utilization during training
watch -n 1 nvidia-smi
```

**What to look for:**
- GPU utilization: 80-100% (good)
- Memory usage: <14GB (with QuantileDMatrix)
- Temperature: <85°C (normal)

---

## 🐛 Troubleshooting

### Problem: GPU Not Detected

**Symptoms:**
```
Warning: GPU not available, falling back to CPU
```

**Solutions:**
1. Check CUDA installation: `nvidia-smi`
2. Verify XGBoost GPU support: `python -c "import xgboost; print(xgboost.__version__)"`
3. Install CUDA toolkit if missing
4. Check GPU drivers are up to date

### Problem: Out of Memory (OOM)

**Symptoms:**
```
CUDA out of memory
```

**Solutions:**
1. Enable QuantileDMatrix (already enabled by default)
2. Enable feature selection: `feature_selection_top_n = 30`
3. Enable gradient sampling: `gradient_sampling = true`
4. Reduce date range (less historical data)
5. Train one symbol at a time

### Problem: Training Too Slow

**Symptoms:**
- Training takes >5 minutes per symbol

**Solutions:**
1. Verify GPU is being used: Check logs for "Training XGBoost model on GPU"
2. Enable all optimizations in config.toml
3. Reduce Optuna trials: `optuna_trials = 50`
4. Use feature selection: `feature_selection_top_n = 30`
5. Check GPU utilization: `nvidia-smi`

### Problem: Low Model Accuracy

**Symptoms:**
- Accuracy <50%
- DSR <0.90

**Solutions:**
1. Use more historical data (3+ years)
2. Enable walk-forward validation
3. Increase Optuna trials for better hyperparameters
4. Check data quality (no missing values, sufficient samples)
5. Review feature selection (may need more features)

### Problem: No Data Fetched

**Symptoms:**
```
Error: No data fetched
```

**Solutions:**
1. Check Alpaca API keys in `.env` file
2. Verify symbol is valid (e.g., "AAPL" not "apple")
3. Check date range (markets closed on weekends)
4. Verify internet connection
5. Check Alpaca API status

---

## 📝 Best Practices

### 1. Regular Retraining

**When to retrain:**
- **Weekly:** For active trading
- **Monthly:** For swing trading
- **Quarterly:** For long-term strategies

**Why:** Market conditions change, models need to adapt

### 2. Use Walk-Forward Validation

**Always enable for production:**
```python
use_walk_forward=True
```

**Why:** Validates model robustness, prevents overfitting

### 3. Monitor Model Performance

**Check metrics after training:**
- Accuracy >60%
- DSR >0.95
- CV score >0.60

**If metrics are low:** Retrain with more data or different parameters

### 4. Feature Selection

**Start with all features, then optimize:**
1. Train with all 88+ features
2. Check feature importance
3. Retrain with top 30 features
4. Compare accuracy

### 5. GPU Memory Management

**Monitor GPU memory:**
```bash
nvidia-smi
```

**If memory issues:**
- Enable QuantileDMatrix (default)
- Enable feature selection
- Reduce date range

---

## 🎓 Advanced Usage

### Custom Training Script

Create your own training script:

```python
#!/usr/bin/env python3
"""Custom training script."""

import asyncio
from nova.core.config import load_config
from nova.data.loader import DataLoader
from nova.models.training_pipeline import TrainingPipeline

async def main():
    config = load_config()
    data_loader = DataLoader(config.data)
    pipeline = TrainingPipeline(config, data_loader)

    # Custom training
    result = await pipeline.train_universe(
        symbols=['AAPL', 'MSFT'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        use_walk_forward=True
    )

    print(f"Training complete: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Training Specific Symbols

```python
# Train only specific symbols
custom_symbols = ['AAPL', 'GOOGL', 'NVDA']
result = await pipeline.train_universe(
    symbols=custom_symbols,
    start_date='2020-01-01',
    end_date='2024-12-31',
    use_walk_forward=True
)
```

### Training Without Database

```python
# Training works without TimescaleDB
pipeline = TrainingPipeline(
    config=config,
    data_loader=data_loader,
    storage=None  # No database needed
)
```

---

## 📚 Additional Resources

- **Full Documentation:** `docs/guides/OPERATION_MANUAL.md`
- **Research:** `docs/research/training/TRAINING_OPTIMIZATION_RESEARCH_RTX5070TI_RYZEN7700X.md`
- **Implementation Details:** `docs/research/training/IMPLEMENTATION_SUMMARY.md`
- **Training Script:** `scripts/train_models.py`

---

## ✅ Quick Checklist

Before training:
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] GPU detected (`nvidia-smi`)
- [ ] Alpaca API keys configured (`.env` file)
- [ ] Config file updated (`config.toml`)

After training:
- [ ] Models saved to `models/` directory
- [ ] Training report generated
- [ ] Metrics reviewed (accuracy >60%, DSR >0.95)
- [ ] Models ready for trading

---

**Last Updated:** January 2025
**Optimized for:** NVIDIA RTX 5070 Ti + AMD Ryzen 7700x
