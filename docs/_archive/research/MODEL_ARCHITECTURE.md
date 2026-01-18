# Model Architecture: Individual vs Master Models

## What Institutional Funds Actually Do

### Renaissance Technologies Approach: Hierarchical Ensemble (The "Medallion Model")

**What Jim Simons Actually Built:**

Renaissance uses a **unified ensemble/master model** that combines many individual signals/models. This is the "major model" that gets refined over time.

**Architecture:**
```
Individual Models/Signals (per asset/sector/timeframe)
        ↓
Signal Validation & Filtering
        ↓
UNIFIED ENSEMBLE MODEL (The "Master Model")
        ↓
Risk & Capital Allocation
        ↓
Execution
```

**Key Insights:**
1. **Many Individual Models**: They have hundreds/thousands of individual signal models (per asset, per timeframe, per strategy)
2. **One Master Ensemble**: All signals feed into ONE unified model that learns how to combine them
3. **Cross-Asset Learning**: Improvements in one area (currencies) benefit others (equities) - unified framework
4. **Continuously Refined**: The master model is constantly retrained/refined as new signals are added/removed
5. **Integrated System**: Not siloed strategies - everything feeds into one system

**Why This Works:**
- Individual models capture symbol-specific patterns ✅
- Master model learns optimal combination weights ✅
- Cross-asset patterns improve all models ✅
- Single system easier to optimize/refine ✅

### Other Top Funds

**Two Sigma, Citadel, D.E. Shaw:**
- Similar hierarchical approach
- Individual models per asset/sector
- Ensemble/master model for combination
- Meta-learning to weight models dynamically

**AQR Capital:**
- Individual models with risk parity weighting
- Less emphasis on unified ensemble (more factor-based)

### Master Model (Less Common)

**Used by:** Some smaller funds, research teams, or when:
- Trading very similar assets (e.g., only tech stocks)
- Limited computational resources
- Research/prototyping phase

**Approach:**
- Single model trained on all symbols combined
- Learns general market patterns
- May include symbol as a feature

**Why:**
- Simpler to manage (one model)
- Faster training (one training run)
- Can learn cross-symbol patterns

**Trade-offs:**
- ✅ Simpler architecture
- ✅ Faster training
- ❌ Lower accuracy (averages across diverse symbols)
- ❌ Harder to debug symbol-specific issues
- ❌ Retraining affects all symbols

### Hybrid Approach (Advanced)

**Used by:** Large quant funds with 1000+ symbols

**Approach:**
- Individual models for major symbols (top 50-100)
- Sector models for smaller symbols (e.g., "tech_small_cap" model)
- Master model as fallback for new/rare symbols

**Why:**
- Balance between accuracy and resource usage
- Scales to large universes (1000+ symbols)

## Our Implementation: Renaissance-Style Hierarchical Ensemble ✅

**What We Have:**
- ✅ Individual models per symbol (24 models)
- ✅ `ModelRegistry` for symbol-specific loading
- ✅ **MASTER ENSEMBLE MODEL** - learns to combine individual predictions
- ✅ Cross-symbol pattern recognition (sector correlations, etc.)
- ✅ `ConfluenceLayer` for combining Technical + Fundamental + Sentiment signals
- ✅ Portfolio optimization for position sizing

**Architecture (Renaissance-Style):**
```
Individual Symbol Models (AAPL, MSFT, etc.) - 24 models
        ↓
MASTER ENSEMBLE MODEL (learns optimal combination) ⭐ NEW LAYER
├─ Collects predictions from all 24 models
├─ Learns cross-symbol patterns (tech sector correlations)
├─ Adapts weights based on market conditions
└─ Outputs improved technical scores per symbol
        ↓
ConfluenceLayer (combines improved Technical + Fundamental + Sentiment)
        ↓
Portfolio Optimizer (position sizing)
        ↓
Execution
```

**Master Model Features:**
1. **Meta-Learning**: Learns optimal weights for each symbol model
2. **Cross-Symbol Patterns**: Recognizes sector correlations (tech moves together)
3. **Market Regime Awareness**: Adapts weights based on volatility/trend
4. **Dynamic Adaptation**: Automatically down-weights underperforming models
5. **Continuous Refinement**: Can be retrained monthly/quarterly as new data arrives
- Train a meta-model that takes predictions from all 24 symbol models
- Learn optimal weights for combining predictions
- Continuously refine as new data comes in
- This would be the "major model" that gets refined over time (like Renaissance)

## Master Ensemble Model Implementation ✅

**Implementation:**

1. **`MasterEnsembleModel` Class** (`src/nova/models/master_ensemble.py`):
   - Collects predictions from all individual symbol models
   - Creates meta-features including:
     - Individual model predictions (one per symbol)
     - Model confidence scores
     - Cross-symbol statistics (mean, std, sector correlations)
     - Market regime indicators (volatility, trend strength)
   - Trains meta-model (XGBoost/Ridge/Linear) to learn optimal combination
   - Saves as `master_model_{date}.json`

2. **Training Script** (`scripts/train_master_model.py`):
   - Collects historical predictions from all symbol models
   - Creates training examples with future returns as targets
   - Trains master model on aggregated data
   - Usage: `python scripts/train_master_model.py --all --years 2`

3. **Integration** (`src/nova/main.py`):
   - Trading loop collects all individual predictions first
   - Master model improves predictions using cross-symbol patterns
   - Improved technical scores feed into ConfluenceLayer

**Benefits:**
- ✅ Learns which models work best in which market conditions
- ✅ Automatically down-weights underperforming models
- ✅ Captures cross-symbol patterns (e.g., tech sector moves together)
- ✅ Single "major model" that gets refined over time (Renaissance-style)

**Usage:**
```bash
# Train master model after training individual models
python scripts/train_master_model.py --all --years 2

# Master model will be automatically loaded during trading
# if master_model_*.json exists in models/ directory
```

## References

- "Advances in Financial Machine Learning" by Marcos López de Prado
- Renaissance Technologies: Unified ensemble system combining many individual signals
- "The Man Who Solved the Market" by Gregory Zuckerman (insights into Medallion architecture)
- Two Sigma: Hierarchical models with meta-learning
- AQR Capital: Individual models with risk parity weighting
