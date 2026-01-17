# Research Summary: 30-Minute Deep Dive (January 2025)

## Overview

Conducted comprehensive web research to identify the absolute best practices for the Nova Aetus trading system. Research focused on performance optimization, production deployment, and latest methodologies.

## Research Areas Covered

1. ✅ XGBoost GPU Optimization (QuantileDMatrix, ExtMemQuantileDMatrix)
2. ✅ Polars Streaming API & Chunked Processing
3. ✅ TimescaleDB Continuous Aggregates & Refresh Policies
4. ✅ Technical Indicator Importance Rankings
5. ✅ asyncpg Connection Pooling Best Practices
6. ✅ Prometheus & Grafana Observability

## Key Discoveries

### Performance Optimizations

1. **XGBoost QuantileDMatrix** (XGBoost 3.0+)
   - Enables training on datasets >1TB
   - Use `QuantileDMatrix` when quantized representation fits in memory
   - Use `ExtMemQuantileDMatrix` only when data too large for GPU memory
   - Requires `tree_method="hist"` (only method supporting external memory)
   - Performance: 10-46× speedup with GPU vs CPU

2. **Polars Streaming API**
   - Streaming engine outperforms in-memory by **2-7×** for large datasets
   - Critical for datasets > RAM size
   - Use `collect(streaming=True)` or `collect_batches()`
   - Tune chunk size: `polars.Config.set_streaming_chunk_size()`
   - Operations: FILTER/SELECT work well, SORT/WINDOW often fallback

3. **TimescaleDB Continuous Aggregates**
   - Refresh policies critical for real-time financial data
   - Proper configuration: 100× faster queries vs raw data
   - Use `materialized_only = false` for real-time aggregates (v2.13+)
   - Match `start_offset` to retention policy
   - Exclude current bucket with `end_offset` to avoid invalidations

### Technical Indicator Rankings

**Top-Performing Indicators** (Research Findings):
1. **Squeeze_pro** - Highest importance (Apple stock study, 13 years)
2. **PPO** (Percentage Price Oscillator)
3. **MACD** - Most profitable and lowest-risk (consistently)
4. **ROC63** (Rate of Change, 63-day period)
5. **RSI63** (RSI, 63-day period)

**Key Insights**:
- Longer periods (63, 252 days) often outperform shorter periods for swing trading
- Feature selection critical: Aim for top 20-30 from 100+ indicators
- Start with raw OHLCV, add indicators selectively
- Use Random Forest feature importance for selection

### Connection Pooling

**asyncpg Best Practices**:
- Size pool: `max_size ≈ floor((max_connections - 17) / num_instances) - buffer`
- TimescaleDB reserves 17 connections (account for this)
- Use `max_queries=50000` to rotate connections
- Set `max_inactive_connection_lifetime=300` to close idle connections
- Use `setup` hook to reset session state (timezone, search_path)

### Observability

**Prometheus & Grafana**:
- Monitor SRE Golden Signals: Latency, Traffic, Errors, Saturation
- Use recording rules for expensive queries
- Set up multi-level alerts (WARN → CRITICAL)
- Implement proper retention policies (15 days high-res, 1 year aggregated)

## Documents Created

1. **`docs/research/best_practices/research_best_practices_2025.md`** (moved from `knowledge/`)
   - Comprehensive 200+ line document with all research findings
   - Code examples and configuration snippets
   - Performance benchmarks and trade-offs
   - Best practices for each component

2. **Updated `.cursorrules`**
   - Added reference to research document
   - Highlighted key performance gains
   - Updated knowledge base references

## Implementation Recommendations

### Immediate Actions

1. **XGBoost Optimization**:
   - Consider implementing `QuantileDMatrix` for large training datasets
   - Add RMM memory management for better GPU memory handling
   - Enable `extmem_single_page=True` if using external memory

2. **Polars Streaming**:
   - Enable streaming for datasets > RAM: `collect(streaming=True)`
   - Use `collect_batches()` for chunked processing
   - Inspect plans with `explain()` to detect fallbacks

3. **TimescaleDB CAGGs**:
   - Review refresh policies for optimal `schedule_interval`
   - Enable real-time aggregates: `materialized_only = false`
   - Ensure `start_offset` matches retention policy

4. **Connection Pooling**:
   - Review pool sizing: Account for TimescaleDB reserved connections
   - Add connection rotation: `max_queries=50000`
   - Implement idle connection cleanup: `max_inactive_connection_lifetime=300`

5. **Indicator Selection**:
   - Prioritize top indicators: Squeeze_pro, PPO, MACD, ROC63, RSI63
   - Implement feature selection (Random Forest importance)
   - Test longer periods (63, 252 days) for swing trading

### Future Enhancements

1. Implement Prometheus recording rules for expensive queries
2. Set up Grafana dashboards for SRE Golden Signals
3. Add RMM memory management for XGBoost GPU
4. Implement Polars streaming for large dataset processing
5. Optimize TimescaleDB refresh policies based on usage patterns

## Performance Targets

Based on research findings, system should achieve:

- **XGBoost Training**: 10-46× faster with GPU vs CPU
- **Polars Processing**: 2-7× faster with streaming vs in-memory
- **TimescaleDB Queries**: 100× faster with CAGGs vs raw data
- **Connection Pooling**: Zero connection exhaustion under high concurrency

## References

All research findings documented in:
- `docs/research/best_practices/research_best_practices_2025.md` - Full research document (moved from `knowledge/`)
- `.cursorrules` - Updated with key findings
- Original research sources cited in research document

**Note:** Research files have been reorganized. See `docs/research/README.md` for the new structure.

---

*Research Completed: January 2025*
*Duration: 30-minute comprehensive deep dive*
*Sources: XGBoost docs, Polars docs, TimescaleDB docs, asyncpg docs, Academic papers, Industry best practices*
