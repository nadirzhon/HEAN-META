#!/usr/bin/env python3
"""HEAN Brain Microservice - AI Decision Making.

Consumes physics data, runs Claude API for strategic analysis
and rule-based tactical decisions.
Subscribes to: physics:{symbol}
Publishes to: brain:signals, brain:analysis
"""
import asyncio
import json
import logging
import os
import signal
import sys
import time

import orjson
import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("brain")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
STRATEGIC_INTERVAL = int(os.getenv("STRATEGIC_INTERVAL", "30"))
MAX_STREAM_LENGTH = int(os.getenv("MAX_STREAM_LENGTH", "200"))

shutdown_event = asyncio.Event()

SYSTEM_PROMPT = """You are a crypto market analyst for the HEAN trading system.
You analyze market thermodynamic data (temperature, entropy, phase state) and prices
to produce short, actionable trading assessments.

Phase states: ICE (low volatility), WATER (normal), VAPOR (high volatility).
Temperature: market energy (higher = more volatile).
Entropy: market disorder (higher = more chaotic).

Rules:
- Respond ONLY with valid JSON, no markdown or extra text.
- JSON must have these fields:
  {
    "bias": "BULLISH" | "BEARISH" | "NEUTRAL",
    "confidence": 0.0-1.0,
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "summary": "one sentence market assessment",
    "signals": [{"symbol": "...", "action": "BUY"|"SELL"|"HOLD", "confidence": 0.0-1.0}]
  }
- Be conservative. Only suggest BUY/SELL when confidence > 0.7.
- High entropy + high temperature = HIGH risk, prefer HOLD.
- ICE phase with low entropy = stable, safe for positions.
- VAPOR phase = caution, reduce exposure."""


class TradingBrain:
    def __init__(self, redis_url: str, symbols: list[str]):
        self.redis_url = redis_url
        self.symbols = symbols
        self.redis = None
        self.anthropic_client = None
        self.latest_physics: dict[str, dict] = {}
        self.latest_prices: dict[str, float] = {}
        self.strategic_context: dict = {}
        self.last_strategic = 0.0
        self.signal_count = 0
        self.analysis_count = 0

        if ANTHROPIC_API_KEY:
            try:
                import anthropic
                self.anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
                logger.info("Anthropic Claude client initialized")
            except ImportError:
                logger.warning("anthropic package not available, running rules-only mode")

    async def connect_redis(self) -> None:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                self.redis = await aioredis.from_url(
                    self.redis_url, encoding="utf-8", decode_responses=False,
                    socket_connect_timeout=5, socket_keepalive=True,
                )
                await self.redis.ping()
                logger.info("Redis connected")
                return
            except Exception as e:
                logger.warning(f"Redis attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    raise

    async def strategic_analysis(self) -> dict | None:
        """Call Claude API for strategic market analysis."""
        if not self.anthropic_client:
            return None

        if not self.latest_physics:
            return None

        # Build market snapshot
        market_data = []
        for symbol in self.symbols:
            physics = self.latest_physics.get(symbol)
            if not physics:
                continue
            market_data.append(
                f"- {symbol}: price=${physics.get('price', 0):.2f}, "
                f"temp={physics.get('temperature', 0):.1f}, "
                f"entropy={physics.get('entropy', 0):.2f}, "
                f"phase={physics.get('phase', 'UNKNOWN')}, "
                f"volume={physics.get('volume', 0):.0f}"
            )

        if not market_data:
            return None

        prompt = f"Current market snapshot:\n" + "\n".join(market_data)

        try:
            response = await asyncio.wait_for(
                self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=512,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=15.0,
            )

            text = response.content[0].text.strip()
            # Parse JSON from response
            analysis = json.loads(text)
            self.analysis_count += 1
            logger.info(
                f"Claude analysis #{self.analysis_count}: "
                f"bias={analysis.get('bias')}, "
                f"risk={analysis.get('risk_level')}, "
                f"confidence={analysis.get('confidence')}"
            )
            return analysis

        except asyncio.TimeoutError:
            logger.warning("Claude API timeout (15s)")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Claude returned invalid JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return None

    async def tactical_decision(self, symbol: str, physics: dict) -> dict | None:
        """Rule-based tactical decision, enhanced by strategic context."""
        temperature = physics.get("temperature", 0)
        entropy = physics.get("entropy", 0)
        phase = physics.get("phase", "UNKNOWN")
        price = physics.get("price", 0)

        signal_type = None
        confidence = 0.0

        # Rule-based signals
        if phase == "WATER" and temperature > 400 and entropy < 3.0:
            signal_type = "BUY"
            confidence = 0.7
        elif phase == "VAPOR" and temperature > 800:
            signal_type = "HOLD"
            confidence = 0.9
        elif phase == "ICE" and entropy < 2.0:
            signal_type = "NEUTRAL"
            confidence = 0.5

        # Enhance with Claude strategic context
        if self.strategic_context:
            ctx_risk = self.strategic_context.get("risk_level", "MEDIUM")
            ctx_bias = self.strategic_context.get("bias", "NEUTRAL")

            # Check per-symbol signals from Claude
            for sig in self.strategic_context.get("signals", []):
                if sig.get("symbol") == symbol:
                    claude_action = sig.get("action")
                    claude_conf = sig.get("confidence", 0.0)
                    if claude_conf > 0.7 and claude_action in ("BUY", "SELL"):
                        signal_type = claude_action
                        confidence = min(claude_conf, 0.95)
                        break

            # Risk override: if Claude says HIGH risk, downgrade BUY to HOLD
            if ctx_risk == "HIGH" and signal_type == "BUY":
                signal_type = "HOLD"
                confidence = max(confidence - 0.2, 0.3)

            # Bias boost: if rules say BUY and Claude agrees BULLISH, boost confidence
            if signal_type == "BUY" and ctx_bias == "BULLISH":
                confidence = min(confidence + 0.1, 0.95)
            elif signal_type == "BUY" and ctx_bias == "BEARISH":
                confidence = max(confidence - 0.15, 0.3)

        if signal_type:
            source = "rules+claude" if self.strategic_context else "rules"
            return {
                "symbol": symbol, "signal": signal_type,
                "confidence": round(confidence, 3), "price": price,
                "temperature": temperature, "entropy": entropy,
                "phase": phase, "source": source,
                "timestamp": int(time.time() * 1000),
            }
        return None

    async def publish_signal(self, signal_data: dict) -> None:
        if not self.redis:
            return
        try:
            data_bytes = orjson.dumps(signal_data)
            await self.redis.xadd(
                "brain:signals", {"data": data_bytes},
                maxlen=MAX_STREAM_LENGTH, approximate=True,
            )
            self.signal_count += 1
            logger.info(
                f"Signal #{self.signal_count}: {signal_data['symbol']} "
                f"{signal_data['signal']} (conf={signal_data['confidence']:.2f}, "
                f"src={signal_data.get('source', 'rules')})"
            )
        except Exception as e:
            logger.error(f"Publish error: {e}")

    async def publish_analysis(self, analysis: dict) -> None:
        """Publish Claude analysis to Redis for UI consumption."""
        if not self.redis:
            return
        try:
            data_bytes = orjson.dumps(analysis)
            await self.redis.xadd(
                "brain:analysis", {"data": data_bytes},
                maxlen=100, approximate=True,
            )
            # Also store latest analysis as a key for quick reads
            await self.redis.set("brain:latest_analysis", data_bytes, ex=120)
        except Exception as e:
            logger.error(f"Publish analysis error: {e}")

    async def strategic_loop(self) -> None:
        """Periodic Claude strategic analysis."""
        # Wait for initial data to accumulate
        await asyncio.sleep(10)
        logger.info(f"Strategic analysis loop started (interval={STRATEGIC_INTERVAL}s)")

        while not shutdown_event.is_set():
            try:
                now = time.time()
                if now - self.last_strategic >= STRATEGIC_INTERVAL:
                    analysis = await self.strategic_analysis()
                    if analysis:
                        self.strategic_context = analysis
                        self.last_strategic = now
                        await self.publish_analysis(analysis)
                    else:
                        logger.debug("No analysis produced (no data or API unavailable)")

                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Strategic loop error: {e}")
                await asyncio.sleep(10)

    async def tactical_loop(self) -> None:
        """Main loop: consume physics data, produce tactical signals."""
        for symbol in self.symbols:
            stream_key = f"physics:{symbol}"
            try:
                await self.redis.xgroup_create(
                    stream_key, "brain-group", id="0", mkstream=True,
                )
            except Exception:
                pass

        logger.info(f"Tactical loop started for {self.symbols}")
        streams = {f"physics:{s}": ">" for s in self.symbols}

        while not shutdown_event.is_set():
            try:
                messages = await self.redis.xreadgroup(
                    "brain-group", "brain-consumer",
                    streams, count=5, block=1000,
                )

                for stream_name, stream_messages in messages:
                    symbol = stream_name.decode().split(":")[-1]
                    for msg_id, msg_data in stream_messages:
                        data_bytes = msg_data.get(b"data")
                        if data_bytes:
                            physics = orjson.loads(data_bytes)
                            self.latest_physics[symbol] = physics
                            self.latest_prices[symbol] = physics.get("price", 0)

                            signal_data = await self.tactical_decision(symbol, physics)
                            if signal_data:
                                await self.publish_signal(signal_data)

                        await self.redis.xack(stream_name, "brain-group", msg_id)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Tactical loop error: {e}")
                await asyncio.sleep(1)

    async def run(self) -> None:
        await self.connect_redis()

        mode = "Claude + Rules" if self.anthropic_client else "Rules-only"
        logger.info(f"Brain started: mode={mode}, symbols={self.symbols}")

        tasks = [asyncio.create_task(self.tactical_loop())]

        if self.anthropic_client:
            tasks.append(asyncio.create_task(self.strategic_loop()))
            logger.info(f"Claude strategic analysis enabled (every {STRATEGIC_INTERVAL}s)")
        else:
            logger.warning("No Anthropic API key - running rules-only mode")

        await asyncio.gather(*tasks)

        logger.info("Brain shutting down")
        if self.redis:
            await self.redis.close()


def signal_handler(sig, frame):
    logger.info(f"Signal {sig}, shutting down...")
    shutdown_event.set()


async def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    brain = TradingBrain(redis_url=REDIS_URL, symbols=SYMBOLS)
    try:
        await brain.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
