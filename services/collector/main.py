#!/usr/bin/env python3
"""HEAN Data Collector Microservice.

Connects to Bybit WebSocket and publishes market data to Redis Streams.
Subscribes to: Bybit WS (orderbook, trades, tickers)
Publishes to: market:{symbol} Redis Stream
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
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("collector")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
BYBIT_WS = os.getenv("BYBIT_WS", "wss://stream-testnet.bybit.com/v5/public/linear")
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
MAX_STREAM_LENGTH = int(os.getenv("MAX_STREAM_LENGTH", "1000"))

shutdown_event = asyncio.Event()


class MarketDataCollector:
    def __init__(self, redis_url: str, ws_url: str, symbols: list[str]):
        self.redis_url = redis_url
        self.ws_url = ws_url
        self.symbols = symbols
        self.redis = None
        self.message_count = 0
        self.last_heartbeat = time.time()

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

    async def subscribe_ws(self) -> None:
        topics = []
        for symbol in self.symbols:
            topics.extend([
                f"orderbook.50.{symbol}",
                f"publicTrade.{symbol}",
                f"tickers.{symbol}",
            ])

        subscribe_msg = {"op": "subscribe", "args": topics}

        while not shutdown_event.is_set():
            try:
                async with websockets.connect(self.ws_url, ping_interval=20) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"WebSocket connected, subscribed to {len(topics)} topics")

                    async for raw_msg in ws:
                        if shutdown_event.is_set():
                            break
                        await self.process_message(raw_msg)

            except websockets.ConnectionClosed:
                logger.warning("WebSocket disconnected, reconnecting in 3s...")
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)

    async def process_message(self, raw_msg: str) -> None:
        try:
            msg = orjson.loads(raw_msg)
        except Exception:
            return

        topic = msg.get("topic", "")
        data = msg.get("data")
        if not topic or not data:
            return

        # Extract symbol from topic
        parts = topic.split(".")
        if len(parts) < 2:
            return
        symbol = parts[-1]

        # Build normalized event
        event = {
            "topic": topic,
            "symbol": symbol,
            "data": data,
            "ts": msg.get("ts", int(time.time() * 1000)),
            "type": msg.get("type", "snapshot"),
        }

        # Publish to Redis Stream
        stream_key = f"market:{symbol}"
        data_bytes = orjson.dumps(event)

        try:
            await self.redis.xadd(
                stream_key, {"data": data_bytes},
                maxlen=MAX_STREAM_LENGTH, approximate=True,
            )
            self.message_count += 1

            if self.message_count % 1000 == 0:
                logger.info(f"Published {self.message_count} messages")

        except Exception as e:
            logger.error(f"Redis publish error: {e}")

    async def run(self) -> None:
        await self.connect_redis()
        logger.info(f"Collector started for {self.symbols}")
        await self.subscribe_ws()
        logger.info("Collector shutting down")
        if self.redis:
            await self.redis.close()


def signal_handler(sig, frame):
    logger.info(f"Signal {sig} received, shutting down...")
    shutdown_event.set()


async def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    collector = MarketDataCollector(
        redis_url=REDIS_URL, ws_url=BYBIT_WS, symbols=SYMBOLS,
    )
    try:
        await collector.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
