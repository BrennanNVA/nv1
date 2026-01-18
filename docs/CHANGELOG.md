# Changelog

## 2025-01-17

### Master Model Implementation
- Added Master Ensemble Model that combines predictions from all 24 individual models
- Master model learns cross-symbol patterns and optimal combination weights
- Training script: `python scripts/train_master_model.py --all --years 2`

### Model Registry
- Fixed symbol-specific model loading bug
- Each symbol now uses its own trained model correctly
- Models loaded on-demand during trading

### Training Improvements
- Added database-first caching (70-80% faster subsequent training runs)
- Fixed model save bug (metadata no longer overwrites models)
- Added research-backed feature selection

### Portfolio Optimization
- Added Hierarchical Risk Parity (HRP) optimization method
- Integrated portfolio optimizer into trading loop
- Added portfolio dashboard page

---

## 2025-01-16

### Configuration Updates
- Expanded trading symbols from 12 to 25 (now 24 after removing BAC)
- Added sector diversification (Finance, Healthcare, Consumer, Energy, Industrial)

### Dashboard Improvements
- Added system health status indicator
- Added model monitoring page
- Fixed DataFrame display bugs

### Performance Optimizations
- Added Polars streaming support for large datasets
- Improved TimescaleDB connection pooling
- Real-time aggregates enabled for faster queries

---

## 2025-01-15

### Core Features
- Data fetcher implementation (Alpaca Pro API + yahooquery fallback)
- Individual stock model training pipeline
- Technical indicators (88+ indicators)
- Sentiment analysis (Ollama local LLM)
- Risk management (Kelly Criterion, circuit breakers)

### Infrastructure
- TimescaleDB setup with continuous aggregates
- Docker compose configuration
- MCP server integration for Cursor IDE
- Discord webhook notifications

---

For detailed history, see `docs/development/versionlog.md`
