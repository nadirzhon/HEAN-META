#!/usr/bin/env python3
"""HEAN Physics Microservice.

Consumes market data from Redis, calculates T/S/phase, publishes results.
Subscribes to: market:{symbol}
Publishes to: physics:{symbol}
"""
import asyncio
import logging
import math
import os
import signal
import sys
import time
from collections import deque

import numpy as np
import orjson
import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("physics")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
LOOKBACK = int(os.getenv("LOOKBACK_WINDOW", "100"))
MAX_STREAM_LENGTH = int(os.getenv("MAX_STREAM_LENGTH", "500"))

shutdown_event = asyncio.Event()


class PhysicsCalculator:
    def __init__(self, redis_url: str, symbols: list[str], lookback: int = 100):
        self.redis_url = redis_url
        self.symbols = symbols
        self.lookback = lookback
        self.redis = None
        self.prices: dict[str, deque] = {s: deque(maxlen=lookback) for s in symbols}
        self.volumes: dict[str, deque] = {s: deque(maxlen=lookback) for s in symbols}
        self.calc_count = 0

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

    def calc_temperature(self, prices: list[float], volumes: list[float]) -> float:
        if len(prices) < 2:
            return 0.0
        p = np.array(prices)
        v = np.array(volumes)
        dp = np.diff(p)
        vm = v[1:]
        ke = float(np.sum((dp * vm) ** 2))
        return ke / len(prices)

    def calc_entropy(self, volumes: list[float]) -> float:
        if not volumes:
            return 0.0
        v = np.array(volumes, dtype=np.float64)
        v = v[v > 0]
        if len(v) == 0:
            return 0.0
        total = np.sum(v)
        if total == 0:
            return 0.0
        p = v / total
        return -float(np.sum(p * np.log(p)))

    def detect_phase(self, temperature: float, entropy: float) -> str:
        if temperature < 400 and entropy < 2.5:
            return "ICE"
        elif temperature >= 800 and entropy >= 3.5:
            return "VAPOR"
        else:
            return "WATER"

    async def process_symbol(self, symbol: str) -> None:
        prices = list(self.prices[symbol])
        volumes = list(self.volumes[symbol])

        if len(prices) < 10:
            return

        temperature = self.calc_temperature(prices, volumes)
        entropy = self.calc_entropy(volumes)
        phase = self.detect_phase(temperature, entropy)
        current_price = prices[-1] if prices else 0

        result = {
            "symbol": symbol,
            "price": current_price,
            "temperature": round(temperature, 4),
            "entropy": round(entropy, 4),
            "phase": phase,
            "timestamp": int(time.time() * 1000),
            "samples": len(prices),
        }

        stream_key = f"physics:{symbol}"
        data_bytes = orjson.dumps(result)

        try:
            await self.redis.xadd(
                stream_key, {"data": data_bytes},
                maxlen=MAX_STREAM_LENGTH, approximate=True,
            )
            self.calc_count += 1
            if self.calc_count % 100 == 0:
                logger.info(
                    f"[{symbol}] T={temperature:.2f} S={entropy:.2f} "
                    f"Phase={phase} (#{self.calc_count})"
                )
        except Exception as e:
            logger.error(f"Publish error: {e}")

    async def run(self) -> None:
        await self.connect_redis()

        for symbol in self.symbols:
            stream_key = f"market:{symbol}"
            try:
                await self.redis.xgroup_create(
                    stream_key, "physics-group", id="0", mkstream=True,
                )
            except Exception:
                pass

        logger.info(f"Physics engine started for {self.symbols}")

        streams = {f"market:{s}": ">" for s in self.symbols}

        while not shutdown_event.is_set():
            try:
                messages = await self.redis.xreadgroup(
                    "physics-group", "physics-consumer",
                    streams, count=10, block=1000,
                )

                for stream_name, stream_messages in messages:
                    symbol = stream_name.decode().split(":")[-1]
                    for msg_id, msg_data in stream_messages:
                        data_bytes = msg_data.get(b"data")
                        if data_bytes:
                            event = orjson.loads(data_bytes)
                            data = event.get("data", {})

                            # Extract price from different data types
                            if isinstance(data, list) and data:
                                price = float(data[0].get("p", 0))
                                volume = float(data[0].get("v", 0))
                            elif isinstance(data, dict):
                                price = float(data.get("lastPrice", data.get("p", 0)))
                                volume = float(data.get("volume24h", data.get("v", 1)))
                            else:
                                continue

                            if price > 0:
                                self.prices[symbol].append(price)
                                self.volumes[symbol].append(max(volume, 0.001))
                                await self.process_symbol(symbol)

                        await self.redis.xack(stream_name, "physics-group", msg_id)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(1)

        logger.info("Physics engine shutting down")
        if self.redis:
            await self.redis.close()


def signal_handler(sig, frame):
    logger.info(f"Signal {sig}, shutting down...")
    shutdown_event.set()


async def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    calc = PhysicsCalculator(redis_url=REDIS_URL, symbols=SYMBOLS, lookback=LOOKBACK)
    try:
        await calc.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
