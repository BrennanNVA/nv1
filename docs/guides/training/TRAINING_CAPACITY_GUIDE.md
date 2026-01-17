# Training Capacity Quick Reference Guide
## Nova Aetus - RTX 5070 Ti System

**Quick answers to common questions about training capacity and scheduling.**

> 📚 **For detailed research analysis and methodology**, see: `docs/research/training/TRAINING_CAPACITY_RESEARCH.md`

---

## 🎯 How Many Tickers Can I Train?

| Scenario | Tickers | Training Time | Status |
|----------|---------|---------------|--------|
| **Current Setup** | 5-10 | 3-10 min | ✅ Safe |
| **Recommended** | 20-30 | 12-30 min | ✅ Optimal |
| **Advanced** | 50-75 | 30-60 min | ⚠️ Monitor |
| **Maximum** | 100-150 | 60-120 min | ⚠️ Requires tuning |

**Key Constraint:** GPU memory (16GB VRAM) - not CPU or storage.

---

## 📅 How Often Should I Retrain?

**For Swing Trading (2-7 day holds):**

| Market Condition | Frequency | When |
|------------------|-----------|------|
| **Stable** | Monthly | Every 4 weeks |
| **Normal** | Bi-weekly | Every 2 weeks ⭐ **Recommended** |
| **Volatile** | Weekly | Every week |
| **Crisis** | Immediate | Triggered by performance drop |

**Retrain immediately if:**
- Model accuracy drops >15%
- Market regime changes detected
- Error rate increases >20%

---

## 📊 How Much Data Do I Need?

**Minimum Requirements:**
- **3 years** of daily data (~750 trading days)
- **100-200 trades** for basic reliability

**Recommended:**
- **5 years** of daily data (~1,250 trading days)
- **200-400 trades** for strong confidence

**Current Setting:** `lookback_periods = 252` (1 year) - **TOO SHORT**

**Action Required:** Update to `lookback_periods = 756` (3 years minimum)

---

## ⚙️ Configuration Updates

### Update `config.toml`:

```toml
[data]
# Increase lookback for better training
lookback_periods = 756  # 3 years (was 252)

# Expand symbols gradually
symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "AMD", "INTC", "NFLX",
    # Add more up to 20-30 for recommended capacity
]

[ml]
# Enable gradient sampling for memory savings (XGBoost 3.0+)
gradient_sampling = true

# Optional: Reduce features if memory issues
# feature_selection_top_n = 50
```

---

## 🚨 System Overload Prevention

### GPU Memory Warnings

**Warning Signs:**
- GPU memory >14 GB (87.5% of 16 GB)
- CUDA out-of-memory errors
- Training slowdowns

**Solutions:**
1. ✅ QuantileDMatrix (already enabled) - 5× memory reduction
2. Enable `gradient_sampling = true` - Additional 20-30% savings
3. Reduce `max_bin = 256` (if needed)
4. Use `feature_selection_top_n = 50` (reduce from 88 features)
5. Train sequentially (one ticker at a time)

### System Memory Warnings

**Warning Signs:**
- RAM usage >80%
- Swap usage increasing
- Slow database queries

**Solutions:**
1. Limit concurrent operations
2. Use Polars streaming for large datasets
3. Compress old TimescaleDB data (>30 days)

---

## 📋 Recommended Training Schedule

### Weekly Schedule
```
Monday-Friday:  No training (live trading hours)
Saturday:       Full retraining (20-30 tickers), 30-60 min
Sunday:         Validation & deployment prep
```

### Monthly Schedule
```
Week 1-3:       Weekly retraining (20-30 tickers)
Week 4:         Full retraining (50-100 tickers) + Optuna
                Walk-forward validation for all models
```

---

## 🔢 Capacity Roadmap

### Phase 1: Current (Safe)
- **Tickers**: 5-10
- **Data**: 3 years
- **Time**: 5-10 min
- **Status**: ✅ Ready now

### Phase 2: Recommended
- **Tickers**: 20-30
- **Data**: 5 years
- **Time**: 20-40 min
- **Status**: ✅ Target for next month

### Phase 3: Advanced
- **Tickers**: 50-75
- **Data**: 5 years
- **Time**: 60-120 min
- **Status**: ⚠️ Requires optimization

### Phase 4: Maximum
- **Tickers**: 100-150
- **Data**: 5 years
- **Time**: 2-4 hours
- **Status**: ⚠️ Requires external memory

---

## 📈 Quick Calculations

### Estimate Training Time
```
Time per ticker: ~1-2 minutes (GPU)
Total time = Tickers × 1.5 minutes
```

### Estimate GPU Memory
```
Memory per ticker: ~200-250 KB (with QuantileDMatrix)
Total memory ≈ Tickers × 250 KB × 3 years
```

### Estimate Storage
```
Storage per ticker/year: ~25-50 KB (compressed)
100 tickers, 5 years: ~12.5-25 MB
Storage is NOT a bottleneck
```

---

## ✅ Checklist Before Training

- [ ] Updated `lookback_periods` to 756+ (3+ years)
- [ ] GPU memory available (<14 GB used)
- [ ] System RAM available (<80% used)
- [ ] QuantileDMatrix enabled (`use_quantile_dmatrix = true`)
- [ ] Training scheduled during off-hours
- [ ] Database connection pool configured
- [ ] Monitoring enabled (GPU temp, memory, etc.)

---

## 🔍 Monitoring Commands

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check System Resources
```bash
htop  # or top
```

### Check Training Progress
```bash
tail -f logs/training.log
```

---

## 📚 Full Research Document

For detailed analysis, see:
- `docs/research/training/TRAINING_CAPACITY_RESEARCH.md` - Complete research findings
- `docs/research/training/TRAINING_OPTIMIZATION_RESEARCH_RTX5070TI_RYZEN7700X.md` - GPU optimization

---

**Last Updated:** January 2025
**Quick Reference Version:** 1.0
