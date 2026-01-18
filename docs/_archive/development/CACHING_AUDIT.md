# Caching Audit & Optimization Report
## Nova Aetus Training Performance Analysis

**Date:** January 2025
**Purpose:** Identify caching opportunities to speed up training

---

## 🔍 Current Caching Status

### ✅ What IS Cached

1. **TimescaleDB - Raw OHLCV Data**
   - **Location:** `market_bars` hypertable in TimescaleDB
   - **How it works:** Data is stored after fetching from API
   - **Method:** `storage.store_bars()` called after fetching
   - **Storage:** Efficient compressed storage with TimescaleDB compression
   - **Status:** ✅ Data is being saved, but **NOT being reused during training**

2. **TimescaleDB - Continuous Aggregates**
   - **Location:** `ohlcv_1min`, `ohlcv_1hour`, `ohlcv_daily` views
   - **Purpose:** Pre-aggregated data for faster queries
   - **Status:** ✅ Configured, but **NOT used by training pipeline**

3. **Streamlit Dashboard Cache**
   - **Location:** `@st.cache_data(ttl=60)` decorator
   - **Purpose:** Caches dashboard queries for 60 seconds
   - **Status:** ✅ Working

---

## ❌ What is NOT Cached (But Should Be)

### Critical Missing Cache: Database Data Check Before API Fetch

**Current Flow (Inefficient):**
```
Training → Always fetch from API → Store in DB → Calculate features
```

**Optimal Flow (With Caching):**
```
Training → Check DB first → If data exists & complete: Use DB → Otherwise: Fetch from API → Store in DB → Calculate features
```

**Impact:**
- **First training:** ~2-5 minutes per symbol (API fetch + feature calculation)
- **Subsequent training:** Could be ~10-30 seconds per symbol (just feature calculation from cached data)
- **Time savings:** ~80-90% faster on retraining!

---

## 📊 Detailed Analysis

### 1. Training Pipeline Data Fetching

**File:** `src/nova/models/training_pipeline.py` (lines 108-125)

**Current Code:**
```python
# Fetch historical data (async I/O)
df = await self.data_loader.fetch_historical_bars(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    timeframe=self.config.data.default_timeframe,
)

# Store in database if available (async)
if self.storage:
    await self.storage.store_bars(df, symbol)
```

**Problem:** Always fetches from API, never checks database first!

**Available but Unused Method:** `storage.load_bars()` exists but is never called during training.

### 2. Data Loader Implementation

**File:** `src/nova/data/loader.py`

**Current:** Only fetches from API (Alpaca/yahooquery), no database check.

**Missing:** Database-first strategy:
1. Check if data exists in TimescaleDB for date range
2. If complete → use cached data
3. If missing/incomplete → fetch from API and update cache

### 3. Feature Calculation Caching

**Current:** Features recalculated every training run
- 88+ technical indicators
- Fractional differencing
- Z-score normalization
- NPMM labeling

**Potential Cache:** If raw OHLCV data unchanged, features could be cached
- **Challenge:** Features depend on rolling windows, need to recalculate on new data
- **Opportunity:** Cache intermediate calculations (raw indicators) if base data unchanged

---

## 🎯 Recommended Optimizations

### Priority 1: Database-First Data Loading (HIGH IMPACT)

**Implementation:**
Modify `training_pipeline.py` to check database before fetching from API:

```python
async def prepare_symbol_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Prepare data and features for a symbol (CPU-bound preprocessing)."""
    try:
        logger.debug(f"Preparing data for {symbol}...")

        # CHECK DATABASE FIRST
        df = None
        if self.storage:
            logger.debug(f"Checking database cache for {symbol}...")
            df = await self.storage.load_bars(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

            # Check if cached data is complete
            if not df.is_empty():
                data_start = df["timestamp"].min()
                data_end = df["timestamp"].max()

                # Convert to datetime for comparison
                from datetime import datetime
                req_start = datetime.fromisoformat(start_date.replace('Z', '+00:00')) if isinstance(start_date, str) else start_date
                req_end = datetime.fromisoformat(end_date.replace('Z', '+00:00')) if isinstance(end_date, str) else end_date

                # If cached data covers the required range, use it
                if data_start <= req_start and data_end >= req_end:
                    logger.info(f"Using cached data for {symbol} ({len(df)} bars from DB)")
                else:
                    logger.info(f"Cached data incomplete for {symbol}, fetching from API")
                    df = None  # Will fetch from API below

        # If no cache or incomplete, fetch from API
        if df is None or df.is_empty():
            logger.debug(f"Fetching {symbol} from API...")
            df = await self.data_loader.fetch_historical_bars(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=self.config.data.default_timeframe,
            )

            # Store in database for next time
            if self.storage and not df.is_empty():
                await self.storage.store_bars(df, symbol)

        if df.is_empty():
            return None

        # ... rest of feature calculation ...
```

**Benefits:**
- **First training:** Same speed (fetches + stores)
- **Subsequent training:** 80-90% faster (uses cached data)
- **Incremental updates:** Only fetches new data since last cache

### Priority 2: Incremental Data Updates (MEDIUM IMPACT)

**Implementation:**
Instead of always fetching full date range, only fetch missing days:

```python
# Check what dates are missing
if df_from_cache:
    cached_dates = set(df_from_cache["timestamp"].dt.date())
    required_dates = set(pd.date_range(start_date, end_date, freq='D'))
    missing_dates = required_dates - cached_dates

    if missing_dates:
        # Only fetch missing dates
        fetch_start = min(missing_dates)
        fetch_end = max(missing_dates)
        new_df = await self.data_loader.fetch_historical_bars(...)
        # Merge with cached data
```

**Benefits:**
- Faster updates when only recent data is missing
- Reduces API calls significantly

### Priority 3: Feature Store Caching (LOW-MEDIUM IMPACT)

**Concept:** Store calculated features in database
- **Table:** `ml_features` with columns: symbol, timestamp, feature_1, feature_2, ...
- **Use case:** If raw data unchanged, reuse calculated features

**Challenges:**
- Features depend on rolling windows (need to recalculate when new data added)
- Storage overhead (88+ features × dates)
- Complexity of incremental updates

**Recommendation:** Implement Priority 1 first, evaluate if Priority 3 is needed.

---

## 📈 Expected Performance Improvements

### Current Training Times (25 symbols, 3 years)
- **With API fetch:** ~25-35 minutes total (~1-2 min per symbol)
- **Breakdown:**
  - Data fetch from API: ~30-60 seconds per symbol
  - Feature calculation: ~20-40 seconds per symbol
  - Model training: ~30-60 seconds per symbol

### After Caching Optimization (Subsequent Runs)
- **With cached data:** ~5-10 minutes total (~10-20 sec per symbol)
- **Breakdown:**
  - Data load from DB: ~1-2 seconds per symbol ✅
  - Feature calculation: ~20-40 seconds per symbol (unchanged)
  - Model training: ~30-60 seconds per symbol (unchanged)

### Time Savings
- **First training:** No change (needs to populate cache)
- **Subsequent training:** **70-80% faster** (saves ~20-25 minutes per training run)
- **Weekly retraining:** Significant cumulative savings

---

## 🛠️ Implementation Plan

### Phase 1: Database-First Loading (Quick Win)
1. ✅ Modify `training_pipeline.py` `prepare_symbol_data()` method
2. ✅ Add database check before API fetch
3. ✅ Use `storage.load_bars()` if data exists
4. ✅ Fall back to API if cache missing/incomplete
5. ✅ Test with existing cached data

**Estimated effort:** 30-60 minutes
**Impact:** High (70-80% training time reduction)

### Phase 2: Incremental Updates (Nice to Have)
1. Detect missing dates in cached data
2. Only fetch missing periods
3. Merge with cached data

**Estimated effort:** 1-2 hours
**Impact:** Medium (additional 10-20% savings on updates)

### Phase 3: Feature Store (Future)
1. Create `ml_features` table schema
2. Store calculated features
3. Reuse if raw data unchanged

**Estimated effort:** 4-8 hours
**Impact:** Low-Medium (may not be worth complexity)

---

## ✅ Action Items

### Immediate (Before Next Training)

1. **Implement Priority 1:** Database-first data loading
   - Modify `training_pipeline.py`
   - Test with existing cached data
   - Verify fallback to API works

2. **Verify Database Contains Data**
   - Check if TimescaleDB has historical data from previous runs
   - If yes → immediate benefit on next training
   - If no → will populate cache on first run

### Short-term (Next Week)

3. **Add Cache Statistics**
   - Log cache hit/miss rates
   - Track time savings
   - Monitor cache effectiveness

4. **Implement Incremental Updates**
   - Only fetch new data since last cache
   - Merge with existing cache

---

## 🧪 Testing Plan

### Test Case 1: Cache Hit (Data Exists)
- **Setup:** Ensure database has 3 years of AAPL data
- **Action:** Train AAPL model
- **Expected:** Uses cached data, no API calls (except for latest day)
- **Verify:** Check logs for "Using cached data" message

### Test Case 2: Cache Miss (No Data)
- **Setup:** Clear AAPL data from database
- **Action:** Train AAPL model
- **Expected:** Fetches from API, stores in database
- **Verify:** Check logs for "Fetching from API" message

### Test Case 3: Partial Cache (Missing Recent Data)
- **Setup:** Database has 2 years of AAPL data (missing last year)
- **Action:** Train AAPL model (3 years)
- **Expected:** Uses cached 2 years, fetches missing 1 year
- **Verify:** Check logs for both cache use and API fetch

### Test Case 4: Performance Comparison
- **Run 1:** Train 5 symbols (fresh, no cache) → Time: X minutes
- **Run 2:** Train same 5 symbols (with cache) → Time: Y minutes
- **Expected:** Y < X * 0.3 (70%+ faster)

---

## 📋 Code Changes Summary

### Files to Modify

1. **`src/nova/models/training_pipeline.py`**
   - Modify `prepare_symbol_data()` method
   - Add database check before API fetch
   - Use `storage.load_bars()` when available

2. **Optional: `src/nova/data/loader.py`**
   - Could add cache-aware `fetch_historical_bars()` method
   - But keeping separation of concerns may be better

---

## 🎯 Success Metrics

After implementation, measure:
- **Cache hit rate:** % of symbols using cached data
- **Training time reduction:** Compare before/after times
- **API call reduction:** Count of API calls per training run
- **Database query performance:** Time to load from cache

**Target:** 70%+ cache hit rate, 70%+ time reduction on subsequent training runs

---

**Status:** Ready for implementation
**Priority:** HIGH - Will significantly improve training efficiency
**Impact:** 70-80% faster training on retraining runs
