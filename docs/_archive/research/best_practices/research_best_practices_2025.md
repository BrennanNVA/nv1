# Research-Backed Best Practices (2025 Deep Dive)

This document consolidates the latest research findings and best practices discovered through comprehensive web research for the Nova Aetus trading system.

## XGBoost GPU Optimization (2024-2025 Updates)

### QuantileDMatrix & External Memory Support

**Key Finding**: XGBoost 3.0+ introduces `ExtMemQuantileDMatrix` for datasets that don't fit in GPU memory.

**Best Practices**:
1. **Use QuantileDMatrix for Large Datasets**:
   - Use `QuantileDMatrix` when quantized representation fits in memory (faster)
   - Use `ExtMemQuantileDMatrix` only when data is too large for GPU memory
   - Accept I/O overhead with external memory (slower but enables >1TB datasets)

2. **Tree Method Selection**:
   - **Only `hist` tree method** supports QuantileDMatrix/ExtMemQuantileDMatrix
   - Never use `exact` or `approx` with external memory (not supported)
   - Set `tree_method="hist"` with `device="cuda"` for GPU acceleration

3. **Memory Management**:
   - Use **RMM (RAPIDS Memory Manager)** for better memory management:
     ```python
     import rmm
     mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource())
     rmm.mr.set_current_device_resource(mr)
     ```
   - Enable `extmem_single_page=True` to combine batches into single page (reduces PCIe overhead)
   - Use `gradient_based` subsampling: `subsample=0.2, sampling_method='gradient_based'`

4. **Performance Optimizations**:
   - Tune `max_quantile_batches` and `min_cache_page_bytes` based on host/device memory
   - Store cache on fast NVMe (not slow disks) - use OS caching when possible
   - Prefer PCIe5 or NVLink for host-to-device transfers (Grace Hopper systems ideal)

5. **Known Limitations**:
   - External memory features still experimental (watch for stability issues)
   - Memory fragmentation can occur - start from clean memory pool
   - Sparse datasets less efficient with quantile building
   - Inference/SHAP still requires large chunks in memory

**Performance Gains**: GPU training is **10-46x faster** than CPU. External memory enables training on >1TB datasets.

---

## Polars Streaming & Chunked Processing

### Streaming API Best Practices

**Key Finding**: Polars streaming engine outperforms in-memory by **2-7×** for large datasets, especially at scale SF-100+.

**Best Practices**:

1. **Enable Streaming**:
   ```python
   # Use lazy evaluation with streaming
   df = pl.scan_csv("large_file.csv")
   result = df.filter(...).select(...).collect(streaming=True)

   # Or use collect_batches for chunked processing
   for batch in df.collect_batches(chunk_size=10000):
       process_batch(batch)
   ```

2. **Chunk Size Tuning**:
   - Configure via `polars.Config.set_streaming_chunk_size()`
   - Default works well, but tune for string-heavy or many-column datasets
   - Smaller chunks = less memory but more overhead
   - Larger chunks = more efficient but higher peak memory

3. **Operations That Work Well**:
   - ✅ FILTER, SELECT, PROJECTION (high efficiency)
   - ✅ AGGREGATIONS (group_by) - many cases supported
   - ✅ Some JOINs (others trigger fallback)
   - ❌ SORT (limited streaming support, often fallback)
   - ❌ WINDOW/ROLLING operations (mostly not streaming, high memory)

4. **Detecting Fallbacks**:
   - Use `explain()` or `show_graph(engine="streaming")` to detect operations causing fallback
   - Fallbacks can cause unexpected memory spikes

5. **Best Practices**:
   - Start with `pl.scan_*` (lazy) - never use eager `read_csv()` on large data
   - Avoid global sorts and heavy window operations where possible
   - Use `map_batches()` for complex row-wise operations (avoids intermediate expansion)
   - Set `maintain_order=False` when order isn't essential (performance gain)

**Performance**: Streaming Polars beats in-memory Polars by **2.5×** at SF-10, remains performant at SF-100 where in-memory deteriorates.

---

## TimescaleDB Continuous Aggregates (Financial Data)

### Refresh Policy Best Practices

**Key Finding**: Proper refresh policy configuration is critical for real-time financial data with low latency requirements.

**Best Practices**:

1. **Bucket Granularity**:
   - Match bucket width to latency requirements:
     - Ticks → 1s buckets for second-level charts
     - 1 min buckets for default summaries
     - Hourly/daily for long-term metrics
   - Don't over-aggregate if not needed (costs storage and refresh work)

2. **Refresh Policy Parameters**:
   ```sql
   SELECT add_continuous_aggregate_policy('trades_per_minute',
     start_offset => INTERVAL '1 day',      -- Match retention policy
     end_offset   => INTERVAL '1 minute',    -- Exclude current bucket
     schedule_interval => INTERVAL '1 minute' -- Refresh frequency
   );
   ```
   - `schedule_interval`: Every 1 min for near-real-time, 5-15 min for mid-day, hourly for EOD
   - `end_offset`: Exclude current bucket to avoid frequent invalidations
   - `start_offset`: Match retention policy to avoid refreshing dropped data

3. **Real-Time vs Materialized Only**:
   - **Real-time aggregates** (`materialized_only = false`): For freshness (trading dashboards, alerts)
   - **Materialized only**: For stability/performance (historical summaries, EOD reports)
   - Note: Real-time aggregates disabled by default in TimescaleDB v2.13+ (must explicitly enable)

4. **Handling Late/Out-of-Order Data**:
   - Set up triggers or batching for late arrivals
   - Manual refresh for affected buckets: `refresh_continuous_aggregate('view', start, end)`
   - Ensure refresh policy covers backfill windows

5. **Data Retention Compatibility**:
   - Raw data retention policy must align with CAGG refresh `start_offset`
   - If raw data older than retention is dropped, CAGG covering that period will fail on refresh

6. **Performance Considerations**:
   - Avoid refreshing very large spans too often (target only new/changed buckets)
   - Limit columns in aggregation (only include what's needed)
   - Monitor locking and resource usage during refresh
   - Use UTC timestamps: `TIMESTAMP WITH TIME ZONE` to avoid timezone skew

**Example Configuration**:
```sql
-- 1-minute OHLCV aggregate
CREATE MATERIALIZED VIEW trades_per_minute
  WITH (timescaledb.continuous, materialized_only = false) AS
SELECT
  symbol,
  time_bucket('1 minute', time) AS bucket,
  first(price, time) AS open,
  max(price) AS high,
  min(price) AS low,
  last(price, time) AS close,
  sum(size) AS volume
FROM trades
GROUP BY symbol, bucket
WITH NO DATA;

-- Refresh every minute, exclude current minute
SELECT add_continuous_aggregate_policy('trades_per_minute',
  start_offset => INTERVAL '1 day',
  end_offset   => INTERVAL '1 minute',
  schedule_interval => INTERVAL '1 minute');
```

---

## Technical Indicator Importance Rankings (Research Findings)

### Top-Performing Indicators for Swing Trading

**Key Finding**: Recent research (2023-2025) identifies specific indicators that consistently rank highest in ML models.

**Top Indicators** (from Apple stock study, 13 years):
1. **Squeeze_pro** - Highest importance
2. **Percentage Price Oscillator (PPO)**
3. **Thermo**
4. **Decay**
5. **Archer On-Balance Volume**
6. **Bollinger Bands**
7. **Squeeze**
8. **Ichimoku**

**Top Indicators** (from Indonesian stocks LSTM study):
- **ROC63** (Rate of Change, 63-day period)
- **RSI63** (RSI, 63-day period)
- **MOM63** (Momentum, 63-day period)
- **MA252** (Moving Average, 252-day period)
- **EMA21** (Exponential MA, 21-day period)

**Key Insights**:
- **MACD** consistently ranks as most profitable and lowest-risk indicator (MACD-GRU model)
- **Longer periods** (63, 252 days) often outperform shorter periods for swing trading
- **Windowed features** (multiple periods: 21, 63, 252 days) improve performance

**Feature Selection Methods**:
1. **Random Forest Feature Importance** - Most common, effective for pruning indicator sets
2. **Correlation Filters** - Remove redundant features
3. **PCA / Autoencoders** - Extract lower-dimensional combinations
4. **Mutual Information** - Select features with predictive power

**Practical Implications**:
- Start with raw OHLCV as baseline
- Add indicators selectively (only if they improve out-of-sample performance)
- Use mix of indicator types: Trend (MA, MACD), Oscillators (RSI, Stochastic), Volatility (ATR, BB), Volume (OBV)
- Feature selection critical to avoid overfitting (aim for top 20-30 from 100+ indicators)
- Cross-validate rigorously across different market regimes

---

## asyncpg Connection Pooling Best Practices

### PostgreSQL/TimescaleDB Pool Configuration

**Key Finding**: Proper pool configuration prevents resource exhaustion and optimizes performance under high concurrency.

**Best Practices**:

1. **Pool Sizing**:
   ```python
   max_size ≈ floor((Postgres max_connections - 17) / num_app_instances) - buffer
   ```
   - TimescaleDB reserves 17 connections (12 superuser + 5 system)
   - Account for all app instances and external poolers (PgBouncer)
   - Keep `min_size=5-10` for startup performance

2. **Connection Lifetime**:
   ```python
   pool = await asyncpg.create_pool(
       dsn=dsn,
       min_size=5,
       max_size=50,
       max_queries=50000,  # Rotate after many queries
       max_inactive_connection_lifetime=300.0,  # Close idle after 5 min
       command_timeout=60.0,  # Prevent runaway queries
       connect_timeout=10.0,
   )
   ```

3. **Session State Management**:
   - Use `setup` hook to reset session-level settings (timezone, search_path, etc.)
   - Session state doesn't persist across releases automatically
   - Prepared statements persist but be careful with transaction-mode poolers

4. **Error Handling**:
   - Set `command_timeout` to prevent hanging queries
   - Use `connect_timeout` or `acquire(timeout=...)` to avoid pool exhaustion hangs
   - Failed connections are automatically replaced by asyncpg

5. **High Concurrency**:
   - Each worker process gets its own pool (total = `num_workers × max_size`)
   - Use short-lived connections: `async with pool.acquire() ...`
   - Fast acquire/release prevents connection hogging

6. **Monitoring**:
   - Track `pool.get_size()`, `pool.get_idle_size()`
   - Monitor query durations and wait times
   - Check Postgres `pg_stat_activity` for connection count

7. **Shutdown**:
   - Use `await pool.close()` for graceful shutdown
   - Waits for all checked-out connections to be released

**TimescaleDB Specifics**:
- Timescale provides its own pooler service (Session or Transaction mode)
- Transaction pooling better for frequent open/close patterns
- Account for Timescale reserved connections in sizing

---

## Prometheus & Grafana Observability Best Practices

### Production Monitoring Setup

**Key Finding**: Proper observability setup enables proactive issue detection and system optimization.

**Best Practices**:

1. **Prometheus Configuration**:
   - Scrape intervals: 15s-30s for high-frequency metrics, 1m for general metrics
   - Retention: 15 days for high-resolution, 1 year for aggregated data
   - Use recording rules for expensive queries
   - Set up alerting rules with appropriate thresholds

2. **Grafana Dashboards**:
   - Create dashboards for SRE Golden Signals: Latency, Traffic, Errors, Saturation
   - Use appropriate time ranges and refresh intervals
   - Implement alerting with notification channels (Discord, email, PagerDuty)
   - Use variables for dynamic filtering (symbol, time range, etc.)

3. **Metrics to Monitor**:
   - **Application**: Request rate, latency (p50, p95, p99), error rate
   - **Database**: Connection pool usage, query duration, transaction rate
   - **ML Pipeline**: Training time, inference latency, model accuracy
   - **Trading**: Signal generation rate, position count, P&L, drawdown
   - **System**: CPU, memory, disk I/O, network bandwidth

4. **Alerting Best Practices**:
   - Set up alerts for critical thresholds (circuit breaker, high error rate, drawdown limits)
   - Use multi-level alerts (WARN → CRITICAL)
   - Include context in alerts (symbol, error message, stack trace)
   - Test alerting channels regularly

5. **Performance Optimization**:
   - Use Prometheus recording rules for expensive queries
   - Implement metric cardinality limits (avoid high-cardinality labels)
   - Use aggregation where possible (sum, avg, rate functions)
   - Consider remote write for long-term storage (Thanos, Cortex)

---

## Summary of Key Research Findings

### Performance Optimizations
- **XGBoost GPU**: 10-46x speedup with proper configuration
- **Polars Streaming**: 2-7× faster than in-memory for large datasets
- **TimescaleDB CAGGs**: 100× faster queries vs raw data
- **asyncpg Pooling**: Proper sizing prevents connection exhaustion

### Critical Best Practices
1. Use `QuantileDMatrix` for large datasets (XGBoost 3.0+)
2. Enable Polars streaming for datasets > RAM
3. Configure TimescaleDB refresh policies for real-time data
4. Size connection pools correctly (account for TimescaleDB reserved connections)
5. Monitor SRE Golden Signals (Latency, Traffic, Errors, Saturation)

### Indicator Selection
- Top indicators: Squeeze_pro, PPO, MACD, ROC63, RSI63
- Longer periods (63, 252 days) often better for swing trading
- Feature selection critical (aim for top 20-30 from 100+ indicators)
- Start with raw OHLCV, add indicators selectively

### System Architecture
- Use lazy evaluation and streaming for large datasets
- Implement proper connection pooling and resource limits
- Set up comprehensive observability (Prometheus + Grafana)
- Configure circuit breakers with multi-level thresholds

---

## References

1. XGBoost GPU Documentation: https://xgboost.readthedocs.io/en/stable/gpu/
2. Polars Streaming Guide: https://pola-rs.github.io/polars-book/user-guide/concepts/streaming/
3. TimescaleDB Continuous Aggregates: https://docs.timescale.com/use-timescale/latest/continuous-aggregates/
4. asyncpg Documentation: https://magicstack.github.io/asyncpg/
5. Prometheus Best Practices: https://prometheus.io/docs/practices/
6. Research Papers:
   - "Feature selection and regression methods for stock price prediction using technical indicators" (2023)
   - "Selection of Trading Indicators Using Machine Learning and Stock Close Price Prediction with LSTM" (2025)
   - "Technical indicator empowered intelligent strategies to predict stock trading signals" (2024)

---

*Last Updated: January 2025*
*Research Period: 30-minute deep dive into latest best practices*
