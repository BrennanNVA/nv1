# Training Scripts

Scripts for preparing and training models for deployment.

## Quick Start

### 1. Check System Readiness

```bash
cd /home/brennan/nac/nova_aetus
source venv/bin/activate  # If using venv
python scripts/prepare_training.py
```

This will check:
- ✅ Data source availability (Alpaca/yahooquery)
- ✅ Database connectivity
- ✅ Feature generation pipeline
- ✅ Training data estimates

### 2. Train Models

```bash
python scripts/train_models.py
```

This will:
- Fetch 3 years of historical data for configured symbols
- Generate technical features with fractional differencing
- Train XGBoost models using NPMM labeling
- Run walk-forward validation
- Save models to `models/` directory

### 3. Verify Models

```bash
ls -lh models/
```

You should see model files like:
- `AAPL_model.json` (XGBoost model)
- `AAPL_model_metadata.json` (training metadata)

## Configuration

Models are trained using symbols from `config.toml`:
```toml
[data]
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
default_timeframe = "1Day"
lookback_periods = 252
```

## Data Sources

The system will try:
1. **Alpaca Pro API** (if API keys configured in `.env`)
2. **yahooquery** (free fallback, no API keys needed)

To use Alpaca:
```bash
# Add to .env
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

## Training Time

- **Single symbol**: ~2-5 minutes
- **5 symbols**: ~10-25 minutes
- **With Optuna optimization**: +50-100% time

## Troubleshooting

### "No data fetched"
- Check internet connection
- Verify symbols are valid (e.g., "AAPL" not "apple")
- Try yahooquery fallback if Alpaca fails

### "Database connection failed"
- Start TimescaleDB: `docker-compose up -d`
- Initialize schema: `python -m src.nova.main --init-db`
- Training can work without database (won't cache data)

### "Feature generation failed"
- Ensure Polars is installed: `pip install polars`
- Check data has required columns (OHLCV)
- Verify enough historical data (need ~252 rows for indicators)

### "GPU not available"
- Training will fall back to CPU (slower but works)
- Check CUDA installation if GPU expected
- Verify XGBoost GPU support: `python -c "import xgboost; print(xgboost.__version__)"`

## Next Steps After Training

1. **Verify models**: Check `models/` directory has model files
2. **Test predictions**: Use `ModelPredictor` to test on new data
3. **Deploy**: Models will be automatically loaded by trading system
4. **Monitor**: Check dashboard for model predictions and performance
