# Alpaca Pro API Data Capabilities

## Overview

With your **Pro Trader subscription**, you have access to institutional-grade market data from Alpaca. This document outlines what data types are available and what's currently implemented in Nova Aetus.

---

## ✅ Currently Implemented

### 1. **Historical Bars (OHLCV)** ✅
**Status**: Fully implemented and tested

**What you get:**
- Open, High, Low, Close, Volume data
- Multiple timeframes: 1Min, 5Min, 15Min, 1Hour, 4Hour, 1Day, etc.
- Price adjustments: raw, split, dividend, or all
- Full SIP feed (all US exchanges, not just IEX)
- Historical data going back 7+ years (since ~2016)

**Usage:**
```python
from nova.data.loader import DataLoader
df = await data_loader.fetch_historical_bars(
    "AAPL",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    timeframe="1Day",
    adjustment="all"  # Adjust for splits and dividends
)
```

**Pro Benefits:**
- Full SIP feed (all exchanges, not just IEX)
- No 15-minute delay on recent data
- Higher rate limits (10,000 calls/min)

---

### 2. **Trades Data** ✅
**Status**: Method implemented, ready to use

**What you get:**
- Individual trade-level data (not just aggregated bars)
- Timestamp, price, size, exchange
- Useful for microstructure analysis, volume profile, order flow

**Usage:**
```python
trades_df = await data_loader.fetch_trades(
    "AAPL",
    start_date=datetime.now() - timedelta(days=1),
    end_date=datetime.now()
)
# Returns: timestamp, price, size, exchange
```

**Use Cases:**
- Volume-weighted average price (VWAP) calculations
- Trade size distribution analysis
- Exchange-level trade analysis
- Order flow imbalance detection

---

### 3. **Quotes Data** ✅
**Status**: Method implemented, ready to use

**What you get:**
- Bid/ask quotes with full depth
- Bid price, ask price, bid size, ask size
- Exchange information
- Useful for spread analysis, market depth, liquidity assessment

**Usage:**
```python
quotes_df = await data_loader.fetch_quotes(
    "AAPL",
    start_date=datetime.now() - timedelta(days=1),
    end_date=datetime.now()
)
# Returns: timestamp, bid_price, ask_price, bid_size, ask_size
```

**Use Cases:**
- Bid-ask spread analysis
- Market depth assessment
- Liquidity measurement
- NBBO (National Best Bid/Offer) tracking

---

## 🚀 Available but Not Yet Implemented

### 4. **Latest Bars (Real-time Snapshots)**
**What you get:**
- Most recent bar(s) without historical range
- Real-time or near real-time (depending on subscription)
- No adjustments applied

**Alpaca Endpoint:** `GET /v2/stocks/bars/latest?symbols=...`

**Potential Implementation:**
```python
async def fetch_latest_bars(self, symbols: list[str]) -> dict[str, pl.DataFrame]:
    """Fetch latest bar for symbols (real-time snapshot)."""
    # Implementation needed
```

---

### 5. **Latest Trades**
**What you get:**
- Most recent trade(s) per symbol
- Real-time or near real-time
- Includes exchange, conditions

**Alpaca Endpoint:** `GET /v2/stocks/{symbol}/trades/latest`

**Potential Implementation:**
```python
async def fetch_latest_trades(self, symbols: list[str]) -> dict[str, pl.DataFrame]:
    """Fetch latest trade for symbols."""
    # Implementation needed
```

---

### 6. **Latest Quotes**
**What you get:**
- Most recent quote(s) per symbol
- Real-time bid/ask with sizes
- Full NBBO via SIP feed (Pro)

**Alpaca Endpoint:** `GET /v2/stocks/{symbol}/quotes/latest`

**Potential Implementation:**
```python
async def fetch_latest_quotes(self, symbols: list[str]) -> dict[str, pl.DataFrame]:
    """Fetch latest quote for symbols."""
    # Implementation needed
```

---

### 7. **Snapshots (Combined Latest Data)**
**What you get:**
- Combines: latest trade, latest quote, 1-minute bar, daily bar, prior daily bar
- Single endpoint for comprehensive current state
- REST only (no WebSocket)

**Alpaca Endpoint:** `GET /v2/stocks/snapshots?symbols=...`

**Potential Implementation:**
```python
async def fetch_snapshots(self, symbols: list[str]) -> dict[str, dict]:
    """Fetch snapshot (latest trade, quote, bars) for symbols."""
    # Returns comprehensive current state per symbol
```

**Use Cases:**
- Real-time dashboard updates
- Current market state overview
- Quick health checks

---

### 8. **Historical News**
**What you get:**
- News articles going back to ~2015
- Provided by Benzinga
- Includes: headline, summary, author, symbols, content
- Can filter by symbols, date range

**Alpaca Endpoint:** `GET /v1beta1/news?symbols=...&start=...&end=...`

**Potential Implementation:**
```python
async def fetch_news(
    self,
    symbols: Optional[list[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 50
) -> pl.DataFrame:
    """Fetch historical news articles."""
    # Returns: timestamp, headline, summary, content, symbols, author, etc.
```

**Use Cases:**
- Sentiment analysis (already using Ollama, but could use Alpaca news)
- Event-driven trading signals
- News impact analysis
- Correlation between news and price movements

---

### 9. **Real-time News Stream (WebSocket)**
**What you get:**
- Real-time news feed via WebSocket
- Subscribe to `news` channel
- Same fields as historical news but streamed live

**Alpaca WebSocket:** `wss://stream.data.alpaca.markets/v1beta1/news`

**Potential Implementation:**
```python
async def stream_news(self, symbols: list[str], callback: Callable):
    """Stream real-time news for symbols."""
    # WebSocket implementation needed
```

---

### 10. **Options Data** 🎯
**What you get:**
- Historical options data (since Feb 2024)
- Trades, quotes, bars for option contracts
- OPRA feed (full, live) with Pro subscription
- Option chain/contracts metadata

**Alpaca Endpoints:**
- `GET /v2/options/contracts?underlying_symbols=...` - Option chain
- `GET /v2/options/bars` - Historical option bars
- `GET /v2/options/trades` - Historical option trades
- `GET /v2/options/quotes` - Historical option quotes

**Potential Implementation:**
```python
async def fetch_option_chain(
    self,
    underlying_symbol: str,
    expiration_date: Optional[datetime] = None
) -> pl.DataFrame:
    """Fetch option chain for underlying symbol."""
    # Returns: strike, expiration, type, open_interest, etc.

async def fetch_option_bars(
    self,
    option_symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "1Day"
) -> pl.DataFrame:
    """Fetch historical bars for option contract."""
```

**Use Cases:**
- Options trading strategies
- Implied volatility analysis
- Options flow analysis
- Greeks calculation

---

### 11. **Crypto Data**
**What you get:**
- Historical crypto bars, trades, quotes
- Data from Alpaca + Kraken
- Crypto bars include midpoint quote prices if no trades

**Alpaca Endpoints:** `/v1beta3/crypto/...`

**Potential Implementation:**
```python
async def fetch_crypto_bars(
    self,
    symbol: str,  # e.g., "BTCUSD"
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "1Day"
) -> pl.DataFrame:
    """Fetch historical crypto bars."""
```

---

### 12. **Real-time WebSocket Streams**
**What you get:**
- Real-time bars, trades, quotes via WebSocket
- Subscribe to multiple symbols simultaneously
- Lower latency than polling REST endpoints

**Alpaca WebSocket:** `wss://stream.data.alpaca.markets/v2/{feed}`

**Channels:**
- `bars` - Real-time bar updates
- `trades` - Real-time trade updates
- `quotes` - Real-time quote updates
- `dailyBars` - Daily bar updates
- `news` - Real-time news stream

**Potential Implementation:**
```python
async def stream_market_data(
    self,
    symbols: list[str],
    channels: list[str],  # ["bars", "trades", "quotes"]
    callback: Callable
):
    """Stream real-time market data via WebSocket."""
```

**Use Cases:**
- Live trading signals
- Real-time dashboard updates
- Low-latency execution
- Live monitoring

---

## 📊 Data Comparison: What You Get vs Free Tier

| Feature | Free Tier | Pro Trader (You) |
|---------|-----------|------------------|
| **Feed** | IEX only | Full SIP (all exchanges) |
| **Real-time Delay** | 15 minutes | No delay |
| **Rate Limits** | 200 calls/min | 10,000 calls/min |
| **Historical Bars** | ✅ 7+ years | ✅ 7+ years (better quality) |
| **Trades** | ✅ Limited | ✅ Full access |
| **Quotes** | ✅ IEX only | ✅ Full NBBO (SIP) |
| **Options** | Indicative (delayed) | OPRA (real-time) |
| **News** | ✅ Available | ✅ Available |
| **Extended Hours** | Limited | ✅ Full access |
| **WebSocket Streams** | Limited symbols | ✅ Unlimited |

---

## 🎯 Recommended Next Steps

### High Priority (Most Useful for Trading)
1. **Latest Bars/Snapshots** - Real-time market state
2. **Real-time WebSocket Streams** - Live data for active trading
3. **Historical News** - Sentiment analysis enhancement

### Medium Priority
4. **Options Data** - If you want to trade options
5. **Latest Trades/Quotes** - Real-time microstructure

### Lower Priority
6. **Crypto Data** - If expanding to crypto
7. **Real-time News Stream** - If you need live news

---

## 💡 Implementation Notes

### Current Architecture
- All data fetching is **async** (using `asyncio`)
- Data is returned as **Polars DataFrames** (fast, Rust-based)
- Automatic fallback to yahooquery if Alpaca fails
- All Alpaca SDK calls run in executor threads (non-blocking)

### Adding New Data Types
1. Add import for new Alpaca request types
2. Create async method in `DataLoader` class
3. Use `loop.run_in_executor()` for sync Alpaca SDK calls
4. Convert to Polars DataFrame
5. Add to test script

### Rate Limits
- **Pro Tier**: 10,000 calls/min
- Current implementation batches requests efficiently
- Consider caching for frequently accessed data

---

## 📚 Resources

- [Alpaca Market Data Docs](https://docs.alpaca.markets/docs/market-data)
- [Alpaca Python SDK](https://alpaca.markets/sdks/python/)
- [Alpaca WebSocket Streaming](https://docs.alpaca.markets/docs/streaming-market-data)

---

## Summary

**Currently Available:**
- ✅ Historical Bars (OHLCV) - **Fully working**
- ✅ Trades Data - **Method ready, tested**
- ✅ Quotes Data - **Method ready, tested**

**Can Be Added:**
- 🚀 Latest Bars/Snapshots
- 🚀 Real-time WebSocket Streams
- 🚀 Historical News
- 🚀 Options Data
- 🚀 Crypto Data

**Your Pro subscription gives you access to ALL of these!** 🎉
