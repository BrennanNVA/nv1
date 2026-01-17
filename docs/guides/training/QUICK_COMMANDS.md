# Quick Training Commands

**Short commands for faster training workflow**

---

## 🚀 Ultra-Short Commands

### Single Symbol (Quick)
```bash
./train AAPL
```
**Trains:** AAPL with 3 years (default)
**Time:** 2-5 minutes

### Train All Symbols
```bash
./train all
```
**Trains:** All 25 symbols from config.toml with 5 years (default)
**Time:** 1-3 hours (depending on GPU)

### Custom Years
```bash
./train AAPL 5          # Single symbol, 5 years
./train all 10          # All symbols, 10 years
./train AAPL MSFT 5     # Multiple symbols, 5 years
```

---

## 📋 Standard Commands (Longer but More Control)

### Using the Launcher Script
```bash
# Single symbol
./START_TRAINING.sh AAPL 3

# All symbols
./START_TRAINING.sh all 5

# Multiple symbols
./START_TRAINING.sh AAPL MSFT GOOGL 5
```

### Direct Python Command
```bash
# Setup environment first
export PATH="$HOME/miniconda3/envs/nova_aetus/bin:$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nova_aetus

# Then run:
python scripts/train_models.py --symbols AAPL --years 3
python scripts/train_models.py --all --years 5  # All symbols
```

---

## ⚡ Command Comparison

| Goal | Shortest | Standard | Full Command |
|------|----------|----------|--------------|
| Train AAPL (3y) | `./train AAPL` | `./START_TRAINING.sh AAPL 3` | `python scripts/train_models.py --symbols AAPL --years 3` |
| Train All (5y) | `./train all` | `./START_TRAINING.sh all 5` | `python scripts/train_models.py --all --years 5` |
| Train Multiple | `./train AAPL MSFT 5` | `./START_TRAINING.sh AAPL MSFT 5` | `python scripts/train_models.py --symbols AAPL MSFT --years 5` |

---

## 🎯 Most Common Commands

```bash
# Quick test
./train AAPL

# Production (all symbols)
./train all

# Custom timeframe
./train all 10
```

---

## 📊 Training Time Estimates

| Command | Symbols | Time |
|---------|---------|------|
| `./train AAPL` | 1 | 2-5 min |
| `./train AAPL MSFT GOOGL` | 3 | 15-30 min |
| `./train all` | 25 | 1-3 hours |

---

## 🔄 Make It Even Shorter (Optional)

Add to your `~/.bashrc` for system-wide alias:

```bash
# Add to ~/.bashrc
alias train='cd /home/brennan/nac/nova_aetus && ./train'
```

Then reload and use:
```bash
source ~/.bashrc
train AAPL        # From anywhere!
train all
```

---

## ✅ What Each Command Does

- **`./train`** - No arguments = trains AAPL with 3 years (safest default)
- **`./train AAPL`** - Trains AAPL with 3 years
- **`./train AAPL 5`** - Trains AAPL with 5 years
- **`./train all`** - Trains all 25 symbols with 5 years
- **`./train all 10`** - Trains all 25 symbols with 10 years
- **`./train AAPL MSFT 5`** - Trains AAPL and MSFT with 5 years
