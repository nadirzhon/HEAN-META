# Alternative Data Sources for Trading Edge

Comprehensive alternative data collection and analysis system for crypto trading, providing edge through:

- **Sentiment Analysis** (Twitter/Reddit via FinBERT)
- **On-Chain Metrics** (Glassnode/CryptoQuant)
- **Funding Rates** (Bybit, Binance, OKX)
- **News Sentiment** (NewsAPI/CryptoPanic)
- **Fear & Greed Index**

All data sources are cached in Redis and fully logged for production use.

## Features

✅ Real-time data collection from multiple sources  
✅ Redis caching for performance (configurable TTL)  
✅ Comprehensive logging and error handling  
✅ Health checks for all components  
✅ Feature engineering for ML pipelines  
✅ Composite sentiment scoring  

## Installation

```bash
pip install -r requirements.txt
```

### API Keys Required

Set the following environment variables:

```bash
# On-chain data
export GLASSNODE_API_KEY="your_key"
export CRYPTOQUANT_API_KEY="your_key"

# News sentiment
export NEWSAPI_KEY="your_key"
export CRYPTOPANIC_KEY="your_key"
```

Note: Funding rates and Fear & Greed Index work without API keys.

## Quick Start

### Complete Pipeline (Recommended)

```python
from hean.alternative_data import AlternativeDataPipeline

# Initialize pipeline
pipeline = AlternativeDataPipeline(
    redis_host="localhost",
    redis_port=6379,
    glassnode_api_key=os.getenv("GLASSNODE_API_KEY"),
    cryptoquant_api_key=os.getenv("CRYPTOQUANT_API_KEY"),
    newsapi_key=os.getenv("NEWSAPI_KEY"),
    cryptopanic_key=os.getenv("CRYPTOPANIC_KEY"),
)

# Get complete data snapshot
snapshot = pipeline.get_complete_snapshot(
    symbol="BTC",
    twitter_texts=["Bitcoin to the moon!", "BTC looking bearish"],
    reddit_texts=["Just bought more BTC", "Selling my stack"],
)

print(f"Overall Sentiment: {snapshot.overall_sentiment:.3f}")
print(f"Confidence: {snapshot.confidence_score:.1%}")

# Get features for ML model
features = pipeline.get_features_dict("BTC")
# Returns dict with keys like:
# - alt_twitter_sentiment
# - alt_exchange_netflow
# - alt_avg_funding_rate
# - alt_fear_greed_value
# - alt_overall_sentiment

# Human-readable report
print(pipeline.get_summary_report("BTC"))
```

### Individual Components

#### 1. Sentiment Engine (Twitter/Reddit)

```python
from hean.alternative_data import SentimentEngine

engine = SentimentEngine(redis_host="localhost")

# Analyze Twitter sentiment
tweets = [
    "Bitcoin breaking ATH! 🚀",
    "This is a bear trap, sell now",
    "HODL and DCA strategy working great"
]

twitter_sentiment = engine.get_twitter_sentiment("BTC", tweets)
print(f"Sentiment: {twitter_sentiment.sentiment:.3f}")
print(f"Confidence: {twitter_sentiment.confidence:.3f}")
print(f"Volume: {twitter_sentiment.volume}")

# Analyze Reddit sentiment
reddit_posts = [
    "Just bought the dip, feeling bullish",
    "Market looking weak, might crash soon"
]

reddit_sentiment = engine.get_reddit_sentiment("BTC", reddit_posts)

# Combined analysis
combined = engine.get_combined_sentiment(
    symbol="BTC",
    twitter_texts=tweets,
    reddit_texts=reddit_posts,
    twitter_weight=0.6,
    reddit_weight=0.4
)
print(f"Combined: {combined['combined_sentiment']:.3f}")
```

#### 2. On-Chain Data Collector

```python
from hean.alternative_data import OnChainDataCollector

collector = OnChainDataCollector(
    glassnode_api_key=os.getenv("GLASSNODE_API_KEY"),
    cryptoquant_api_key=os.getenv("CRYPTOQUANT_API_KEY"),
)

# Get all metrics
metrics = collector.get_metrics("BTC")

print(f"Exchange Netflow: {metrics.exchange_netflow:,.2f} BTC")
print(f"MVRV Ratio: {metrics.mvrv_ratio:.2f}")
print(f"Active Addresses: {metrics.active_addresses:,}")

# Individual metrics
flows = collector.get_exchange_flows("BTC")
mvrv = collector.get_mvrv_ratio("BTC")
active_addrs = collector.get_active_addresses("BTC")
```

**On-Chain Interpretation:**

- **Exchange Netflow**: Positive = coins flowing to exchanges (bearish), Negative = withdrawal (bullish)
- **MVRV Ratio**: >3.7 = overvalued (sell), <1.0 = undervalued (buy)
- **Active Addresses**: Increasing = network growing (bullish)

#### 3. Funding Rates Aggregator

```python
from hean.alternative_data import FundingRatesAggregator

aggregator = FundingRatesAggregator(redis_host="localhost")

# Get funding rates from all exchanges
rates = aggregator.get_all_funding_rates("BTC")

for exchange, rate_data in rates.items():
    print(f"{exchange}: {rate_data.funding_rate:.6f}")

# Average across exchanges
avg_rate = aggregator.get_average_funding_rate("BTC")

# Spread analysis
spread = aggregator.get_funding_rate_spread("BTC")
print(f"Average: {spread['average']:.6f}")
print(f"Spread: {spread['spread']:.6f}")

# Trading signal
signal = aggregator.analyze_funding_signal("BTC")
print(f"Signal: {signal['signal']}")  # bullish/bearish/neutral
print(f"Strength: {signal['strength']:.2f}")
```

**Funding Rate Interpretation:**

- **Positive rate** (>0.01%): Longs overheated → potential reversal down
- **Negative rate** (<-0.01%): Shorts overheated → potential reversal up
- **Near zero**: Balanced market

#### 4. News Sentiment

```python
from hean.alternative_data import Newssentiment

news = Newssentiment(
    newsapi_key=os.getenv("NEWSAPI_KEY"),
    cryptopanic_key=os.getenv("CRYPTOPANIC_KEY"),
)

# Get news sentiment (last 24 hours)
sentiment = news.get_news_sentiment("BTC", hours=24)

print(f"Sentiment: {sentiment['sentiment']:.3f}")
print(f"Articles: {sentiment['article_count']}")
print(f"Positive: {sentiment['positive_count']}")
print(f"Negative: {sentiment['negative_count']}")
```

#### 5. Fear & Greed Index

```python
from hean.alternative_data import FearGreedIndexCollector

fg = FearGreedIndexCollector(redis_host="localhost")

# Current index
current = fg.get_current_index()
print(f"Value: {current.value}/100")
print(f"Classification: {current.classification}")

# Trading signal (contrarian strategy)
signal = fg.get_trading_signal()
print(f"Signal: {signal['signal']}")  # strong_buy/buy/hold/sell/strong_sell

# Trend analysis
trend = fg.get_trend_analysis(days=7)
print(f"Trend: {trend['trend']}")  # increasing/decreasing/stable
print(f"Change: {trend['change']:+.0f}")
```

**Fear & Greed Interpretation (Contrarian):**

- **0-24 (Extreme Fear)**: Strong buy signal
- **25-49 (Fear)**: Buy signal
- **50 (Neutral)**: Hold
- **51-74 (Greed)**: Sell signal
- **75-100 (Extreme Greed)**: Strong sell signal

## Architecture

```
alternative_data/
├── __init__.py                      # Package exports
├── sentiment_engine.py              # FinBERT sentiment analysis
├── onchain_collector.py             # Glassnode/CryptoQuant integration
├── funding_rates_aggregator.py      # Multi-exchange funding rates
├── news_sentiment.py                # News aggregation & analysis
├── fear_greed_index.py              # Fear & Greed Index collector
├── alternative_data_pipeline.py     # Unified pipeline
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Health Checks

```python
# Check all components
health = pipeline.health_check()

for component, status in health.items():
    print(f"{component}: {status}")

# Output:
# sentiment_engine: {'model_loaded': True, 'redis_connected': True}
# onchain_collector: {'redis_connected': True, 'glassnode_configured': True, ...}
# funding_aggregator: {'redis_connected': True, 'binance_api': True, ...}
# ...
```

## Redis Caching

All data sources use Redis caching with configurable TTL:

- **Sentiment**: 5 minutes (300s)
- **On-chain**: 10 minutes (600s)
- **Funding rates**: 5 minutes (300s)
- **News**: 10 minutes (600s)
- **Fear & Greed**: 1 hour (3600s)

Disable caching:

```python
snapshot = pipeline.get_complete_snapshot("BTC", use_cache=False)
```

## Integration with HEAN Trading System

```python
# In your trading strategy
from hean.alternative_data import AlternativeDataPipeline

class MyStrategy:
    def __init__(self):
        self.alt_data = AlternativeDataPipeline(
            redis_host="localhost",
            redis_port=6379,
        )
    
    def get_signal(self, symbol):
        # Get alternative data features
        features = self.alt_data.get_features_dict(symbol)
        
        # Use in decision making
        if features['alt_overall_sentiment'] > 0.5:
            return "BUY"
        elif features['alt_overall_sentiment'] < -0.5:
            return "SELL"
        else:
            return "HOLD"
```

## Logging

All components use Python's `logging` module:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Logs will show:
# 2026-01-24 10:30:00 - hean.alternative_data.sentiment_engine - INFO - BTC twitter sentiment: 0.742 (confidence: 0.85, volume: 1500)
```

## Performance Considerations

- **FinBERT Model**: Loads on first use (~500MB), uses GPU if available
- **Batch Processing**: Sentiment analysis processes texts in batches for efficiency
- **Redis Caching**: Reduces API calls and improves response time
- **Parallel Requests**: Use `asyncio` for concurrent data collection (future enhancement)

## Error Handling

All components gracefully handle errors:

```python
# If API fails, returns default/neutral values
snapshot = pipeline.get_complete_snapshot("BTC")

# Check individual component health
if not pipeline.onchain_collector.health_check()['glassnode_configured']:
    logger.warning("Glassnode API not configured, using defaults")
```

## API Rate Limits

Be aware of API rate limits:

- **Glassnode**: ~10 req/min (free tier)
- **CryptoQuant**: Varies by plan
- **NewsAPI**: 100 req/day (free tier)
- **CryptoPanic**: 100 req/hour (free tier)
- **Funding rates**: Public APIs, no limits
- **Fear & Greed**: No limits

Use caching to minimize API calls!

## Roadmap

- [ ] Async data collection for better performance
- [ ] WebSocket support for real-time funding rates
- [ ] Additional on-chain metrics (NVT, Puell Multiple, etc.)
- [ ] Order book imbalance analysis
- [ ] Liquidation data integration
- [ ] Prometheus metrics export

## License

Part of HEAN-META trading system.
