# Nova Aetus Documentation Summary

## 📚 Documentation Created

We've created comprehensive documentation for operating and viewing the Nova Aetus trading system:

### 1. **OPERATION_MANUAL.md** (Main Manual)
   **Purpose**: Complete operations guide for running the system

   **Contents**:
   - System overview and architecture
   - Prerequisites and setup instructions
   - Configuration guide
   - Running the system (trading loop, training, dashboard)
   - Model training workflows
   - Monitoring and maintenance
   - Troubleshooting guide
   - API reference
   - Best practices

   **Use When**:
   - Setting up the system for the first time
   - Understanding how components work together
   - Troubleshooting issues
   - Configuring the system

### 2. **DASHBOARD_GUIDE.md** (Dashboard Details)
   **Purpose**: Detailed guide for using the Streamlit dashboard

   **Location**: `docs/guides/dashboard/DASHBOARD_GUIDE.md`

   **Contents**:
   - Starting the dashboard
   - Page-by-page explanation
   - Interactive features
   - Data sources
   - Troubleshooting dashboard issues
   - Tips and best practices

   **Use When**:
   - Learning to navigate the dashboard
   - Understanding what each page shows
   - Troubleshooting dashboard issues

### 3. **QUICK_START_DASHBOARD.md** (Dashboard Quick Start)
   **Purpose**: Step-by-step instructions for first-time dashboard viewing

   **Location**: `docs/guides/dashboard/QUICK_START_DASHBOARD.md`

   **Contents**:
   - Quick summary at top
   - Prerequisites check
   - Step-by-step launch instructions
   - What you'll see on each page
   - Expected behavior
   - Troubleshooting

   **Use When**:
   - First time viewing the dashboard
   - Quick reference for launching and using dashboard
   - Step-by-step guidance

### 4. **QUICK_START_TRAINING.md** (Training Guide)
   **Location**: `docs/guides/training/QUICK_START_TRAINING.md`
   **Purpose**: Quick start guide for training ML models

   **Contents**:
   - Prerequisites and setup
   - Step-by-step training instructions
   - GPU optimization features (RTX 5070 Ti)
   - Training options and configurations
   - Understanding training output
   - Troubleshooting common issues
   - Best practices
   - Advanced usage examples

   **Use When**:
   - Training models for the first time
   - Need quick reference for training commands
   - Troubleshooting training issues
   - Understanding training metrics

## 🚀 Quick Start Commands

### View Dashboard
```bash
cd /home/brennan/nac/nova_aetus
./launch_dashboard.sh
```

### Train Models
```bash
cd /home/brennan/nac/nova_aetus
source venv/bin/activate
python scripts/train_models.py
```

### Or Manual Training
```bash
python -c "
from nova.core.config import load_config
from nova.data.loader import DataLoader
from nova.models.training_pipeline import TrainingPipeline
import asyncio

async def train():
    config = load_config()
    data_loader = DataLoader(config.data)
    pipeline = TrainingPipeline(config, data_loader)
    result = await pipeline.train_universe(
        symbols=['AAPL'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        use_walk_forward=True
    )
    print(f'Training complete: {result}')

asyncio.run(train())
"
```

### Install Dependencies (if needed)
```bash
pip install streamlit plotly polars pytomlpp xgboost optuna
```

## 📋 Documentation Structure

```
nova_aetus/
├── docs/
│   ├── guides/                  # User guides
│   │   ├── OPERATION_MANUAL.md  # Complete operations guide
│   │   ├── QUICK_START.md       # General quick start
│   │   ├── dashboard/           # Dashboard guides
│   │   │   ├── DASHBOARD_GUIDE.md   # Detailed dashboard usage
│   │   │   └── QUICK_START_DASHBOARD.md  # Quick start dashboard guide
│   │   ├── training/            # Training guides
│   │   │   ├── QUICK_START_TRAINING.md   # Complete training guide
│   │   │   ├── QUICK_COMMANDS.md         # Training command cheat sheet
│   │   │   └── TRAINING_CAPACITY_GUIDE.md # Training capacity guide
│   │   └── DOCUMENTATION_SUMMARY.md   # This file
│   ├── architecture/            # System architecture
│   │   └── SYSTEM_ARCHITECTURE.md
│   ├── reference/               # Reference documentation
│   │   ├── ALPACA_DATA_CAPABILITIES.md
│   │   ├── DATABASE_BACKUP.md
│   │   └── SECRETS_MANAGEMENT.md
│   ├── research/                # Research documentation
│   │   ├── training/            # Training research
│   │   ├── strategies/          # Trading strategies
│   │   └── best_practices/      # Best practices
│   └── development/             # Development docs
│       ├── versionlog.md
│       └── archive/             # Archived status docs
├── launch_dashboard.sh          # Dashboard launcher script
├── scripts/
│   └── train_models.py          # Training script
└── ...
```

## 🎯 What Each Document Covers

| Document | Setup | Running | Dashboard | Training | Troubleshooting |
|----------|-------|---------|-----------|----------|-----------------|
| OPERATION_MANUAL.md | ✅ | ✅ | ✅ | ✅ | ✅ |
| DASHBOARD_GUIDE.md | ⚠️ | ⚠️ | ✅ | ❌ | ✅ |
| QUICK_START_DASHBOARD.md | ✅ | ❌ | ✅ | ❌ | ✅ |
| QUICK_START_TRAINING.md | ✅ | ✅ | ❌ | ✅ | ✅ |

## 📖 Reading Order Recommendations

### For First-Time Setup:
1. `docs/guides/OPERATION_MANUAL.md` - Read "Prerequisites & Setup" section
2. `docs/guides/training/QUICK_START_TRAINING.md` - Train your first models
3. `docs/guides/dashboard/QUICK_START_DASHBOARD.md` - Follow step-by-step
4. `docs/guides/dashboard/DASHBOARD_GUIDE.md` - Learn dashboard features

### For Daily Operations:
1. `docs/guides/dashboard/QUICK_START_DASHBOARD.md` - Quick dashboard reference
2. `docs/guides/training/QUICK_START_TRAINING.md` - Quick training commands
3. `docs/guides/OPERATION_MANUAL.md` - Troubleshooting section as needed

### For Understanding the System:
1. `docs/architecture/SYSTEM_ARCHITECTURE.md` - System design
2. `docs/guides/OPERATION_MANUAL.md` - How components work together
3. `docs/guides/training/QUICK_START_TRAINING.md` - Training workflow details

## 🔍 Key Sections by Need

### "How do I start the dashboard?"
→ `docs/guides/dashboard/QUICK_START_DASHBOARD.md`

### "What does each dashboard page show?"
→ `docs/guides/dashboard/DASHBOARD_GUIDE.md` - Pages section

### "How do I train models?"
→ `docs/guides/training/QUICK_START_TRAINING.md` - Complete training guide
→ `docs/guides/OPERATION_MANUAL.md` - Model Training section (detailed)

### "How do I configure the system?"
→ **OPERATION_MANUAL.md** - Configuration section

### "Something's not working"
→ **OPERATION_MANUAL.md** - Troubleshooting section

### "How does the system work?"
→ **SYSTEM_ARCHITECTURE.md** + **OPERATION_MANUAL.md** - System Overview

## 💡 Tips

1. **Start with** `docs/guides/OPERATION_MANUAL.md` for comprehensive understanding
2. **Use** `docs/guides/dashboard/QUICK_START_DASHBOARD.md` as a quick reference
3. **Check** `docs/guides/dashboard/DASHBOARD_GUIDE.md` for detailed dashboard features
4. **Refer to** `docs/architecture/SYSTEM_ARCHITECTURE.md` for system design questions

## 🎬 Next Steps

1. ✅ **Read OPERATION_MANUAL.md** - Understand the system
2. ✅ **Launch Dashboard** - `./launch_dashboard.sh`
3. ✅ **Explore Pages** - Click through all dashboard pages
4. ✅ **Start Trading System** - `python -m nova.main` (separate terminal)
5. ✅ **Monitor Performance** - Watch dashboard populate with real data

---

**All documentation is ready!** Start with `OPERATION_MANUAL.md` for the complete guide, or `./launch_dashboard.sh` to see the dashboard in action! 🚀
