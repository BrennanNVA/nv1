# Training Capacity & Swing Trading Guidelines Research
## Nova Aetus System - RTX 5070 Ti + Ryzen 7700x

**Research Date:** January 2025
**Hardware Configuration:**
- GPU: NVIDIA RTX 5070 Ti (16GB GDDR7 VRAM)
- CPU: AMD Ryzen 7 7700x (8 cores / 16 threads)
- System: Nova Aetus Swing Trading System

> ⚡ **For quick reference tables and configuration examples**, see: `docs/guides/training/TRAINING_CAPACITY_GUIDE.md`

---

## Executive Summary

Based on extensive research and your hardware specifications, here are the key findings:

### Ticker Capacity Recommendations
- **Conservative (Safe)**: **20-30 tickers** with 3-5 years of daily data
- **Moderate (Recommended)**: **50-75 tickers** with optimized settings
- **Aggressive (Maximum)**: **100-150 tickers** with external memory and careful tuning

### Training Frequency for Swing Trading
- **Recommended**: **Bi-weekly to monthly** retraining
- **Minimum**: **Weekly** retraining (if market conditions change rapidly)
- **Maximum**: **Quarterly** retraining (if performance remains stable)

### Data Requirements
- **Minimum**: 3 years of daily data (~750 trading days)
- **Recommended**: 5 years of daily data (~1,250 trading days)
- **Optimal**: 5-10 years for robust regime coverage

---

## 1. Ticker Capacity Analysis

### 1.1 GPU Memory Constraints (16GB VRAM)

#### Current Configuration Analysis
- **Features per ticker**: 88+ technical indicators
- **Data type**: float32 (4 bytes per value)
- **Current symbols**: 5 tickers
- **Lookback**: 252 periods (1 year daily data)

#### Memory Calculation Per Ticker

**Raw Data Size:**
- 252 days × 88 features × 4 bytes = **88.7 KB per ticker** (raw)
- With QuantileDMatrix compression (5x reduction): **~17.7 KB per ticker** (compressed)

**Training Memory Requirements:**
- QuantileDMatrix overhead: ~2-3x compressed size
- Gradient/hessian buffers: ~rows × features × 8 bytes
- Tree structure: ~trees × depth × nodes
- Working memory: ~20-30% additional overhead

**Per Ticker Estimate (3 years, 88 features):**
- Raw data: ~756 rows × 88 features × 4 bytes = **266 KB**
- Compressed (QuantileDMatrix): **~53 KB**
- Training overhead: **~150-200 KB per ticker**
- **Total per ticker: ~200-250 KB in GPU memory**

#### Capacity Estimates

| Scenario | Tickers | Years Data | Rows Total | VRAM Usage | Status |
|----------|---------|------------|------------|------------|--------|
| **Conservative** | 20 | 3 | ~15,120 | ~4-5 GB | ✅ Safe |
| **Moderate** | 50 | 3 | ~37,800 | ~9-12 GB | ✅ Recommended |
| **Aggressive** | 100 | 3 | ~75,600 | ~15-16 GB | ⚠️ Near limit |
| **Maximum** | 150 | 3 | ~113,400 | ~16+ GB | ⚠️ Requires external memory |

**Note:** These estimates assume:
- QuantileDMatrix enabled (5x memory reduction)
- Separate models per ticker (not multi-output)
- Tree depth: 6-8, n_estimators: 100-200
- No concurrent training (sequential processing)

### 1.2 CPU & System Memory Constraints

**AMD Ryzen 7 7700x Capabilities:**
- **8 cores / 16 threads**: Sufficient for data preprocessing while GPU trains
- **PCIe 4.0 x16**: ~64 GB/s bandwidth (adequate for GPU transfers)
- **System RAM**: Not specified, but recommend **32GB+** for:
  - TimescaleDB caching
  - Polars data processing
  - Multiple ticker data in memory

**Concurrent Processing:**
- Can preprocess 2-3 tickers while GPU trains 1 ticker
- Recommended: **Sequential training** (one ticker at a time) to maximize GPU utilization

### 1.3 TimescaleDB Caching Capacity

**Storage Capacity (Based on Research):**
- **Compression**: Up to 90× reduction with Hypercore columnstore
- **Daily bars**: ~100-200 bytes per row (compressed)
- **Per ticker per year**: ~25-50 KB (compressed)
- **100 tickers, 5 years**: ~12.5-25 MB (compressed)

**Practical Limits:**
- **Thousands of tickers** feasible with compression
- **Storage is NOT the bottleneck** - GPU memory is the constraint
- **Recommendation**: Cache all tickers you plan to train on

**Query Performance:**
- Continuous aggregates (1m, 1h, 1d) dramatically improve query speed
- Chunk size: Keep chunks fitting in ~25% of RAM
- For 32GB RAM: chunks should be <8GB each

---

## 2. Swing Trading Training Guidelines

### 2.1 Training Frequency Best Practices

**Research Findings:**
- Swing trading (2-7 day holding periods) requires less frequent retraining than day trading
- Concept drift in financial markets occurs gradually over weeks/months
- Over-training can lead to overfitting to noise

**Recommended Schedule:**

| Market Condition | Retraining Frequency | Rationale |
|------------------|---------------------|-----------|
| **Stable markets** | Monthly (every 4 weeks) | Gradual drift, stable patterns |
| **Normal volatility** | Bi-weekly (every 2 weeks) | Balanced adaptation vs. stability |
| **High volatility / Regime change** | Weekly | Rapid pattern shifts require faster adaptation |
| **Crisis / Extreme events** | Immediate (triggered) | Major market shifts need immediate response |

**Trigger-Based Retraining:**
Monitor these metrics and retrain if:
- **Performance drop**: >15% decrease in out-of-sample accuracy
- **Drift detection**: Statistical concept drift detected (ADWIN, DDM)
- **Error increase**: >20% increase in false positives/negatives
- **Market regime change**: Volatility regime shift detected

### 2.2 Data Requirements for Swing Trading

**Minimum Dataset Size:**
- **Trades needed**: 100-200 trades for basic reliability
- **Time span**: 3-5 years of daily data
- **Rows per ticker**: ~750-1,250 daily bars

**Optimal Dataset Size:**
- **Trades needed**: 200-400+ trades for strong confidence
- **Time span**: 5-10 years of daily data
- **Rows per ticker**: ~1,250-2,500 daily bars
- **Multiple regimes**: Bull, bear, sideways markets included

**For Your System:**
- **Current**: 252 periods (1 year) - **INSUFFICIENT**
- **Recommended**: 756-1,260 periods (3-5 years)
- **Optimal**: 1,260-2,520 periods (5-10 years)

**Calculation:**
- Swing trades: ~20-50 trades per year per ticker
- For 100 trades: Need 2-5 years minimum
- For 200 trades: Need 4-10 years recommended

### 2.3 Training Volume Guidelines

**Per Training Session:**

| Tickers | Training Time (GPU) | Training Time (CPU) | Recommended Batch |
|---------|---------------------|---------------------|-------------------|
| 5 | 3-5 minutes | 15-30 minutes | ✅ Current setup |
| 20 | 12-20 minutes | 60-120 minutes | ✅ Recommended |
| 50 | 30-50 minutes | 150-300 minutes | ⚠️ Long session |
| 100 | 60-100 minutes | 300-600 minutes | ⚠️ Overnight |

**With Optuna Hyperparameter Tuning:**
- Add 50-100% time overhead
- 5 tickers: 5-10 minutes → 10-20 minutes
- 20 tickers: 20-40 minutes → 40-80 minutes

**Recommendation:**
- **Daily training**: Max 20-30 tickers
- **Weekly training**: 50-75 tickers feasible
- **Monthly training**: 100+ tickers possible

### 2.4 Walk-Forward Validation Impact

**Memory & Time Impact:**
- Walk-forward splits data into train/test windows
- **Memory**: Similar per window (sequential processing)
- **Time**: 3-5× longer (multiple train/test cycles)

**Example:**
- Single model: 5 minutes
- Walk-forward (5 windows): 15-25 minutes

**Recommendation:**
- Use walk-forward for **final validation** (monthly)
- Skip walk-forward for **frequent retraining** (weekly/bi-weekly)

---

## 3. System Overload Prevention

### 3.1 GPU Memory Management

**Warning Signs:**
- GPU memory usage >90% (14.4 GB / 16 GB)
- CUDA out-of-memory errors
- Training slowdowns mid-process

**Prevention Strategies:**

1. **Use QuantileDMatrix** (already enabled)
   - Reduces memory by 5×
   - Pre-quantizes features before GPU transfer

2. **Enable Gradient-Based Sampling** (XGBoost 3.0+)
   ```toml
   gradient_sampling = true  # In config.toml
   ```
   - Additional 20-30% memory reduction
   - Slightly slower training

3. **Reduce max_bin** (if needed)
   ```toml
   max_bin = 256  # Default is 512
   ```
   - Reduces histogram memory
   - May slightly reduce model accuracy

4. **Feature Selection**
   ```toml
   feature_selection_top_n = 50  # Use top 50 features instead of 88
   ```
   - Reduces memory proportionally
   - May improve accuracy (removes noise)

5. **Sequential Training** (not concurrent)
   - Train one ticker at a time
   - Release GPU memory between tickers
   - Prevents memory accumulation

### 3.2 CPU & System Memory Management

**Warning Signs:**
- System RAM usage >80%
- Swap usage increasing
- TimescaleDB queries slowing

**Prevention Strategies:**

1. **Limit Concurrent Operations**
   - Train sequentially (one ticker at a time)
   - Preprocess next ticker while GPU trains current

2. **Database Connection Pooling**
   ```toml
   max_db_connections = 10  # Current setting is good
   ```

3. **Polars Streaming** (for large datasets)
   - Use `collect(streaming=True)` for datasets >RAM
   - Process in chunks

4. **TimescaleDB Compression**
   - Compress old data (>30 days)
   - Reduces memory footprint for queries

### 3.3 Training Schedule Optimization

**Recommended Workflow:**

**Daily (During Trading Hours):**
- No training (avoid system load during live trading)
- Only inference/prediction

**After Market Close:**
- Light retraining: 5-10 tickers (if triggered)
- Data updates and caching

**Weekly (Weekend):**
- Full retraining: 20-50 tickers
- Walk-forward validation for key tickers

**Monthly:**
- Complete retraining: All tickers (50-100+)
- Full walk-forward validation
- Hyperparameter optimization (Optuna)

---

## 4. Specific Recommendations for Your System

### 4.1 Immediate Recommendations

**Update `config.toml`:**

```toml
[data]
# Increase lookback for better training data
lookback_periods = 756  # 3 years (was 252)

# Expand symbol list gradually
symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "AMD", "INTC", "NFLX",
    "SPY", "QQQ", "DIA",  # Add indexes
    # Add more gradually up to 20-30
]

[ml]
# Enable gradient sampling for memory savings
gradient_sampling = true  # If XGBoost 3.0+

# Optional: Feature selection
# feature_selection_top_n = 50  # Uncomment if memory issues
```

### 4.2 Capacity Roadmap

**Phase 1 (Current - Safe):**
- **Tickers**: 5-10
- **Data**: 3 years (756 periods)
- **Training**: Weekly
- **Time**: 5-10 minutes

**Phase 2 (Recommended):**
- **Tickers**: 20-30
- **Data**: 5 years (1,260 periods)
- **Training**: Bi-weekly
- **Time**: 20-40 minutes

**Phase 3 (Advanced):**
- **Tickers**: 50-75
- **Data**: 5 years (1,260 periods)
- **Training**: Monthly
- **Time**: 60-120 minutes

**Phase 4 (Maximum):**
- **Tickers**: 100-150
- **Data**: 5 years (1,260 periods)
- **Training**: Monthly (overnight)
- **Time**: 2-4 hours
- **Requires**: External memory mode, careful tuning

### 4.3 Training Schedule Template

**Weekly Schedule:**
```
Monday:    Data update, no training
Tuesday:   Data update, no training
Wednesday: Data update, no training
Thursday:  Data update, no training
Friday:    Data update, no training
Saturday:  Full retraining (20-30 tickers), ~30-60 min
Sunday:    Validation, model review, deployment prep
```

**Monthly Schedule:**
```
Week 1-3:  Weekly retraining (20-30 tickers)
Week 4:    Full retraining (50-100 tickers) + Optuna tuning
           Walk-forward validation for all models
```

---

## 5. Monitoring & Alerts

### 5.1 Key Metrics to Monitor

**GPU Metrics:**
- VRAM usage: Should stay <14 GB (87.5%)
- GPU utilization: Should be 80-100% during training
- Temperature: Should stay <85°C

**System Metrics:**
- CPU usage: Should stay <80% average
- RAM usage: Should stay <80%
- Disk I/O: Monitor TimescaleDB query times

**Training Metrics:**
- Training time per ticker: Track for degradation
- Memory errors: Alert on CUDA OOM
- Model accuracy: Alert on >15% drop

### 5.2 Recommended Monitoring Setup

```python
# Add to training pipeline
import psutil
import GPUtil

def check_system_resources():
    """Check if system has capacity for training."""
    gpu = GPUtil.getGPUs()[0]
    ram = psutil.virtual_memory()

    if gpu.memoryUsed > 14 * 1024:  # 14 GB
        raise RuntimeError("GPU memory too high")
    if ram.percent > 80:
        raise RuntimeError("System RAM too high")

    return True
```

---

## 6. Research Sources & References

### Training Frequency
1. "When the World Changes: How Machine Learning Handles Concept Drift" (Medium)
2. "Machine Learning Trading" (SignalPilot Education)
3. "ML Model Retraining Strategies" (ML Journey)
4. "Can Machine Learning Be Used to Trade Profitably?" (TrendSpider)

### GPU Memory & XGBoost
1. XGBoost GPU Support Documentation
2. "Upscale XGBoost with QuantileDMatrix" (Medium)
3. NVIDIA Technical Blog: Multi-GPU XGBoost Training
4. XGBoost External Memory Tutorial

### TimescaleDB Capacity
1. TimescaleDB Financial Tick Data Tutorial
2. "Scaling PostgreSQL: 10 Billion Daily Records" (HackerNoon)
3. TimescaleDB Compression Documentation

### Swing Trading Data Requirements
1. "How Many Trades Are Enough?" (Medium)
2. "Sample Size in Trading Backtests" (BacktestBase)
3. "Importance of Sample Size" (PipUp)
4. Hurst Cycle Analysis Guidelines

---

## 7. Conclusion

### Key Takeaways

1. **Ticker Capacity**: Your system can comfortably handle **20-30 tickers** with current settings, up to **50-75 tickers** with optimization, and **100-150 tickers** with external memory.

2. **Training Frequency**: For swing trading, **bi-weekly to monthly** retraining is optimal. Weekly is acceptable if market conditions are volatile.

3. **Data Requirements**: Increase lookback from **252 to 756-1,260 periods** (3-5 years) for robust training. Current 1-year lookback is insufficient.

4. **System Limits**: GPU memory (16GB) is the primary constraint. CPU and storage are not bottlenecks with proper configuration.

5. **Best Practices**:
   - Use QuantileDMatrix (already enabled)
   - Train sequentially (one ticker at a time)
   - Monitor GPU memory usage
   - Schedule training during off-hours
   - Use walk-forward validation monthly, not weekly

### Next Steps

1. **Immediate**: Update `lookback_periods` to 756 (3 years)
2. **Short-term**: Gradually expand to 20-30 tickers
3. **Medium-term**: Optimize for 50-75 tickers
4. **Long-term**: Consider external memory for 100+ tickers

---

**Last Updated:** January 2025
**System:** Nova Aetus v1.0
**Hardware:** RTX 5070 Ti + Ryzen 7700x
