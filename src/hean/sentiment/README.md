# sentiment -- Sentiment Analysis

Text sentiment analysis using FinBERT and social media aggregation from news, Reddit, and Twitter sources.

## Architecture

The `SentimentAnalyzer` uses the FinBERT model (ProsusAI/finbert) to classify financial text as bullish, bearish, or neutral. It runs asynchronously and produces `TextSentiment` results with label and confidence score. The `SentimentAggregator` combines sentiment from multiple sources (news, Twitter, Reddit) with configurable weights (default: news 50%, Twitter 30%, Reddit 20%) into a unified `SentimentSignal` with an action recommendation and trading confidence. Sentiment results are published as CONTEXT_UPDATE events with type `finbert_sentiment` and feed into the Oracle hybrid signal fusion at 20% weight.

## Key Classes

- `SentimentAnalyzer` (`analyzer.py`) -- Core FinBERT-based analyzer. Loads the ProsusAI/finbert model from HuggingFace. Provides `analyze(text)` returning a `TextSentiment` with label (bullish/bearish/neutral) and score. Requires the `transformers` library (optional dependency).
- `SentimentAggregator` (`aggregator.py`) -- Multi-source aggregation. Combines `NewsSentiment`, `TwitterSentiment`, and `RedditSentiment` with weighted averaging. Produces `SentimentSignal` with `should_trade` flag and `action` (BUY/SELL).
- `NewsSentiment` (`news_client.py`) -- Fetches and analyzes news headlines for crypto sentiment.
- `RedditSentiment` (`reddit_client.py`) -- Fetches and analyzes Reddit posts for crypto sentiment.
- `TwitterSentiment` (`twitter_client.py`) -- Fetches and analyzes Twitter posts for crypto sentiment.
- Data models (`models.py`) -- `SentimentLabel`, `SentimentSource`, `TextSentiment`, `SentimentScore`, `SentimentSignal`.

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| CONTEXT_UPDATE (type=finbert_sentiment) | Publishes | FinBERT sentiment analysis result |

## Configuration

Sentiment analysis does not have dedicated config flags in HEANSettings. It is activated when the SentimentStrategy is enabled or when the Oracle integration requests sentiment data. The FinBERT model requires the `transformers` library as an optional dependency.
