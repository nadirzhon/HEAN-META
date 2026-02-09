"""
Core sentiment analyzer using FinBERT

FinBERT - BERT model fine-tuned on financial texts
Best for analyzing trading-related sentiment
"""

import asyncio
import logging
from datetime import datetime

try:
    from transformers import (  # noqa: F401
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .models import SentimentLabel, SentimentSource, TextSentiment

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analyzer using FinBERT

    Usage:
        analyzer = SentimentAnalyzer()
        sentiment = await analyzer.analyze("Bitcoin price surging!")
        # Returns: TextSentiment(label="bullish", score=0.85, ...)
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize sentiment analyzer

        Args:
            model_name: HuggingFace model name (default: FinBERT)
        """
        self.model_name = model_name
        self._pipeline = None
        self._model = None
        self._tokenizer = None
        self._initialized = False

        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "transformers not installed. Install with: "
                "pip install transformers torch --break-system-packages"
            )

    async def initialize(self):
        """Load model (async to not block startup)"""
        if self._initialized:
            return

        if not TRANSFORMERS_AVAILABLE:
            logger.error("Cannot initialize: transformers not available")
            return

        try:
            logger.info(f"Loading sentiment model: {self.model_name}")

            # Load in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            self._pipeline = await loop.run_in_executor(
                None,
                lambda: pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=-1  # CPU (use 0 for GPU)
                )
            )

            self._initialized = True
            logger.info("Sentiment model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise

    async def analyze(
        self,
        text: str,
        source: SentimentSource = SentimentSource.TWITTER
    ) -> TextSentiment | None:
        """
        Analyze sentiment of text

        Args:
            text: text to analyze
            source: source of text

        Returns:
            TextSentiment or None if analysis fails
        """
        if not self._initialized:
            await self.initialize()

        if not self._pipeline:
            logger.warning("Model not initialized, cannot analyze")
            return None

        try:
            # Clean text
            text = self._preprocess(text)

            if len(text) < 10:
                logger.debug("Text too short to analyze")
                return None

            # Analyze in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._pipeline(text[:512])[0]  # Max 512 tokens
            )

            # Parse result
            label = self._parse_label(result['label'])
            score = self._convert_score(label, result['score'])

            return TextSentiment(
                text=text,
                label=label,
                score=score,
                confidence=result['score'],
                source=source,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return None

    def _preprocess(self, text: str) -> str:
        """Clean and prepare text"""
        # Remove URLs
        import re
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)

        # Remove mentions and hashtags (keep text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    def _parse_label(self, label: str) -> SentimentLabel:
        """Parse model output label to our enum"""
        label = label.lower()

        if label in ['positive', 'bullish']:
            return SentimentLabel.BULLISH
        elif label in ['negative', 'bearish']:
            return SentimentLabel.BEARISH
        else:
            return SentimentLabel.NEUTRAL

    def _convert_score(self, label: SentimentLabel, confidence: float) -> float:
        """
        Convert label + confidence to score (-1 to +1)

        Args:
            label: sentiment label
            confidence: model confidence (0 to 1)

        Returns:
            score: -1 (bearish) to +1 (bullish)
        """
        if label == SentimentLabel.BULLISH:
            return confidence
        elif label == SentimentLabel.BEARISH:
            return -confidence
        else:
            return 0.0

    async def analyze_batch(
        self,
        texts: list[str],
        source: SentimentSource = SentimentSource.TWITTER
    ) -> list[TextSentiment]:
        """
        Analyze multiple texts efficiently

        Args:
            texts: list of texts
            source: source of texts

        Returns:
            list of TextSentiment
        """
        # Process in parallel (but not too many at once)
        batch_size = 10
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.analyze(text, source) for text in batch],
                return_exceptions=True
            )

            # Filter out errors and None
            results.extend([
                r for r in batch_results
                if isinstance(r, TextSentiment)
            ])

        return results


# Singleton instance
_analyzer = None


async def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create sentiment analyzer singleton"""
    global _analyzer

    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
        await _analyzer.initialize()

    return _analyzer
