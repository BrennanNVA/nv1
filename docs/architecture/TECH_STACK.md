# Nova Aetus Technical Stack Reference

**Last Updated**: January 2025
**Version**: 2.0
**Status**: Production-Ready

---

## Executive Summary

Nova Aetus is built on a modern, institutional-grade tech stack optimized for swing trading. The architecture prioritizes:

1. **Performance**: GPU-accelerated ML training, Rust-based data processing
2. **Robustness**: Statistical validation, drift detection, circuit breakers
3. **Observability**: MCP integration, Discord alerts, comprehensive logging
4. **Research-Grade**: Advanced portfolio optimization, hierarchical ensemble models

---

## Core Technology Stack

### Language & Runtime
- **Python 3.12+**: Primary development language
- **Type Hints**: Strict typing with `typing` and `typing-extensions>=4.8.0`
- **Async/Await**: Full async I/O for data fetching, DB operations, notifications

### Data Processing Layer

#### Primary Data Engine
- **Polars>=1.0.0**: Rust-based DataFrame library
  - **Why**: 10-100x faster than Pandas for large datasets
  - **Use Case**: Feature engineering, technical indicators, data transformations
  - **Hardware**: Leverages SIMD and multi-threading

#### Data Validation
- **Pydantic>=2.0.0**: Runtime type validation and data modeling
  - **Why**: Ensures data integrity, prevents silent failures
  - **Use Case**: Config validation, feature schema enforcement

#### Data Sources
- **alpaca-py>=0.30.0**: Alpaca Markets API client
  - **Why**: Professional-grade market data, paper/live trading
  - **Features**: Real-time quotes, historical bars, account management

- **yahooquery>=2.4.0**: Yahoo Finance data fallback
  - **Why**: Free alternative for fundamental data, earnings, news
  - **Use Case**: Backup data source, fundamental analysis

---

## Machine Learning Stack

### Core ML Engine
- **XGBoost>=2.0.0**: Gradient boosting framework (GPU-enabled)
  - **Why**: Industry standard for tabular data, excellent performance
  - **Hardware**: Leverages Nvidia RTX 5070 Ti GPU for training
  - **Use Case**: Individual symbol models, master ensemble meta-learner

- **Optuna>=3.0.0**: Hyperparameter optimization framework
  - **Why**: Automated hyperparameter tuning, pruning, parallel trials
  - **Integration**: `optuna-integration[xgboost]>=0.5.0` for XGBoost-specific optimizations

- **scikit-learn>=1.3.0**: General-purpose ML utilities
  - **Why**: Preprocessing, metrics, cross-validation
  - **Use Case**: Feature scaling, model evaluation

### Model Interpretability
- **shap>=0.45.0**: SHAP (SHapley Additive exPlanations) values
  - **Why**: Explain model predictions, feature importance
  - **Use Case**: Post-training analysis, debugging model decisions

### Statistical Analysis
- **statsmodels>=0.14.0**: Statistical modeling and tests
  - **Why**: ADF tests for stationarity, time series analysis
  - **Use Case**: Optimal fractional differentiation parameter selection

---

## Sentiment Analysis

- **ollama>=0.1.0**: Local LLM inference server
  - **Why**: Privacy-preserving, no API costs, runs on local hardware
  - **Models**: Llama-3, Mistral (via Ollama)
  - **Hardware**: Leverages RTX 5070 Ti for inference
  - **Use Case**: News sentiment analysis, earnings call transcripts

---

## Portfolio Optimization

### Optimization Methods
1. **Mean-Variance Optimization** (Markowitz, 1952)
   - Maximizes Sharpe ratio
   - Risk-return tradeoff

2. **Risk Parity**
   - Equal risk contribution per asset
   - Diversification-focused

3. **Hierarchical Risk Parity (HRP)** ⭐ **NEW**
   - Clustering-based allocation (Lopez de Prado, 2016)
   - Robust to estimation errors
   - Uses `scipy.cluster.hierarchy` for dendrogram construction

4. **Kelly Criterion**
   - Optimal growth portfolio
   - Based on win rates and payoff ratios

5. **Minimum Variance**
   - Minimizes portfolio risk
   - Conservative allocation

6. **Equal Weight**
   - Simple 1/n allocation
   - Baseline method

### Implementation
- **scipy.optimize**: Constrained optimization (SLSQP)
- **scipy.cluster.hierarchy**: Hierarchical clustering for HRP
- **scipy.spatial.distance**: Distance matrices for HRP

---

## Performance Analytics

- **quantstats>=0.0.81** ⭐ **NEW**
  - **Why**: Professional portfolio analytics and tearsheets
  - **Features**:
    - Comprehensive performance metrics (Sharpe, Sortino, Calmar, etc.)
    - Visualizations (equity curves, drawdowns, monthly returns heatmaps)
    - Benchmark comparisons
    - HTML tearsheets for reporting
  - **Use Case**: Post-trade analysis, performance reporting, strategy evaluation

---

## Database & Storage

### Primary Database
- **TimescaleDB**: PostgreSQL extension for time-series data
  - **Why**: Optimized for time-series queries, compression, continuous aggregates
  - **Features**:
    - Hypercompression (90x compression ratios)
    - Continuous aggregates (1m, 1h, 1d)
    - Native time-series functions

### Database Drivers
- **asyncpg>=0.29.0**: Async PostgreSQL driver
  - **Why**: High-performance async database access
  - **Use Case**: Real-time data ingestion, query execution

- **psycopg2-binary>=2.9.0**: Synchronous PostgreSQL driver
  - **Why**: Compatibility, fallback for sync operations
  - **Use Case**: Migration scripts, one-off queries

---

## Dashboard & Visualization

- **Streamlit>=1.28.0**: Interactive web dashboard framework
  - **Why**: Rapid prototyping, Python-native, no frontend knowledge required
  - **Use Case**: Real-time monitoring, strategy performance, model diagnostics

- **Plotly>=5.17.0**: Interactive plotting library
  - **Why**: Professional-grade charts, zoom/pan, export capabilities
  - **Use Case**: Equity curves, drawdowns, feature distributions, correlation matrices

---

## Notifications & Alerts

- **discord-webhook>=1.0.0**: Discord webhook integration
  - **Why**: Real-time alerts, team collaboration, mobile notifications
  - **Use Case**: Trade signals, system health, error alerts

---

## Model Monitoring & Drift Detection

- **evidently>=0.4.0**: Comprehensive drift detection
  - **Why**: Data drift, concept drift, prediction drift detection
  - **Features**: Statistical tests, visualizations, automated reports
  - **Use Case**: Model health monitoring, retraining triggers

- **nannyml>=0.10.0**: Performance estimation without labels
  - **Why**: Detect model degradation in production without ground truth
  - **Use Case**: Real-time model monitoring, early warning system

---

## Feature Store

- **feast>=0.38.0**: Feature store infrastructure
  - **Why**: Centralized feature management, versioning, serving
  - **Use Case**: Feature reuse across models, online/offline feature serving

---

## Configuration & Environment

- **python-dotenv>=1.0.0**: Environment variable management
  - **Why**: Secure API key storage, environment-specific configs
  - **Use Case**: `.env` file parsing

- **pytomlpp>=1.0.0**: TOML configuration parser
  - **Why**: Human-readable config format, type-safe
  - **Use Case**: `config.toml` parsing for strategy parameters

---

## IDE Integration & Observability

- **mcp>=0.9.0**: Model Context Protocol
  - **Why**: Chat-queryable database access from Cursor IDE
  - **Use Case**: Diagnostics, ad-hoc queries, system exploration

---

## HTTP & API

- **aiohttp>=3.9.0**: Async HTTP client/server
  - **Why**: Non-blocking HTTP operations, health endpoints
  - **Use Case**: Health checks, API endpoints, webhook receivers

---

## Testing & Quality Assurance

- **pytest>=7.4.0**: Testing framework
- **pytest-asyncio>=0.21.0**: Async test support
- **pytest-cov>=4.1.0**: Coverage reporting
- **pytest-mock>=3.11.0**: Mocking utilities

---

## Utilities

- **typing-extensions>=4.8.0**: Extended type hints for older Python versions
- **watchdog>=3.0.0**: File system event monitoring
  - **Use Case**: Auto-reload on config changes, file watchers

---

## Hardware Optimization

### GPU Acceleration
- **Nvidia RTX 5070 Ti**: Primary GPU
  - **XGBoost**: GPU-accelerated training (`tree_method='gpu_hist'`)
  - **Ollama**: LLM inference acceleration
  - **Note**: RMM (RAPIDS Memory Manager) available via conda for advanced GPU memory management

### CPU
- **AMD Ryzen 7 7700X**: Multi-core CPU
  - **Polars**: Leverages multi-threading for parallel operations
  - **Optuna**: Parallel hyperparameter trials

---

## Architecture Patterns

### Data Flow
```
External Data Sources (Alpaca/Yahoo)
  ↓
Data Loader (async)
  ↓
TimescaleDB (storage)
  ↓
Feature Pipeline (Polars)
  ↓
ML Models (XGBoost GPU)
  ↓
Confluence Layer (signal aggregation)
  ↓
Portfolio Optimizer (HRP/Mean-Variance/etc.)
  ↓
Execution Engine
  ↓
Discord Alerts / Dashboard
```

### Model Architecture
- **Individual Symbol Models**: One XGBoost model per trading symbol
- **Master Ensemble Model**: Meta-learner that combines individual predictions
- **Hierarchical Ensemble**: Renaissance Technologies-inspired architecture

---

## Comparison with "Master Quant Stack Reference (2026)"

### Strengths of Current Stack
✅ **GPU Acceleration**: XGBoost + Ollama leverage RTX 5070 Ti
✅ **Modern Data Engine**: Polars outperforms Pandas significantly
✅ **Local LLM**: Privacy-preserving, no API costs
✅ **Statistical Rigor**: DSR, PSR, PBO validation
✅ **Hierarchical Ensemble**: Institutional-grade model architecture
✅ **Comprehensive Monitoring**: Evidently + NannyML for drift detection

### Recent Additions (2025)
⭐ **QuantStats**: Professional performance analytics
⭐ **Hierarchical Risk Parity**: Robust portfolio optimization
⭐ **Master Ensemble Model**: Cross-symbol learning

### Potential Future Enhancements
- **Vector Databases**: For semantic search of news/sentiment data
- **Real-time Streaming**: Kafka/Pulsar for live market data
- **Distributed Training**: Ray/Dask for multi-GPU model training
- **Advanced Feature Engineering**: AutoML feature selection
- **Reinforcement Learning**: For dynamic position sizing

---

## Version History

- **v2.0 (Jan 2025)**: Added QuantStats, Hierarchical Risk Parity
- **v1.0**: Initial production stack

---

## References

- [Lopez de Prado (2016)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678): "Building Diversified Portfolios that Outperform Out of Sample"
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Polars User Guide](https://pola-rs.github.io/polars-book/)
- [QuantStats Documentation](https://github.com/ranaroussi/quantstats)

---

## Maintenance Notes

- **Dependency Updates**: Review quarterly for security patches
- **GPU Drivers**: Keep CUDA toolkit updated for XGBoost compatibility
- **Database Backups**: Automated TimescaleDB backups recommended
- **Model Versioning**: Track model versions with timestamps in filenames
