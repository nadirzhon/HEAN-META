#!/usr/bin/env python3
"""HEAN Risk Management Microservice.

Applies iron-clad risk rules to trading signals.
Subscribes to: brain:signals
Publishes to: risk:approved
"""
import asyncio
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
logger = logging.getLogger("risk")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "300"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "2.0"))
MAX_WEEKLY_LOSS_PCT = float(os.getenv("MAX_WEEKLY_LOSS_PCT", "5.0"))
MAX_TRADE_RISK_PCT = float(os.getenv("MAX_TRADE_RISK_PCT", "1.0"))
MAX_STREAM_LENGTH = int(os.getenv("MAX_STREAM_LENGTH", "200"))

shutdown_event = asyncio.Event()


class RiskManager:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None
        self.capital = INITIAL_CAPITAL
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.last_day = time.strftime("%Y-%m-%d")
        self.last_week = time.strftime("%Y-W%W")
        self.approved_count = 0
        self.rejected_count = 0

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

    async def check_signal(self, sig: dict) -> tuple[bool, str]:
        # Reset periods
        today = time.strftime("%Y-%m-%d")
        week = time.strftime("%Y-W%W")
        if today != self.last_day:
            self.daily_pnl = 0.0
            self.last_day = today
        if week != self.last_week:
            self.weekly_pnl = 0.0
            self.last_week = week

        max_daily = self.capital * (MAX_DAILY_LOSS_PCT / 100)
        if self.daily_pnl < -max_daily:
            return False, f"Daily loss limit (${-self.daily_pnl:.2f}/${max_daily:.2f})"

        max_weekly = self.capital * (MAX_WEEKLY_LOSS_PCT / 100)
        if self.weekly_pnl < -max_weekly:
            return False, f"Weekly loss limit (${-self.weekly_pnl:.2f}/${max_weekly:.2f})"

        signal_type = sig.get("signal", "")
        if signal_type not in ("BUY", "SELL"):
            return False, f"Neutral signal: {signal_type}"

        return True, "APPROVED"

    async def run(self) -> None:
        await self.connect_redis()

        try:
            await self.redis.xgroup_create(
                "brain:signals", "risk-group", id="0", mkstream=True,
            )
        except Exception:
            pass

        logger.info("Risk manager started")

        while not shutdown_event.is_set():
            try:
                messages = await self.redis.xreadgroup(
                    "risk-group", "risk-consumer",
                    {"brain:signals": ">"}, count=10, block=1000,
                )

                for stream_name, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        data_bytes = msg_data.get(b"data")
                        if data_bytes:
                            sig = orjson.loads(data_bytes)
                            approved, reason = await self.check_signal(sig)

                            if approved:
                                enhanced = {
                                    **sig, "risk_approved": True,
                                    "risk_reason": reason,
                                    "capital": self.capital,
                                }
                                await self.redis.xadd(
                                    "risk:approved",
                                    {"data": orjson.dumps(enhanced)},
                                    maxlen=MAX_STREAM_LENGTH, approximate=True,
                                )
                                self.approved_count += 1
                                logger.info(
                                    f"APPROVED #{self.approved_count}: "
                                    f"{sig['symbol']} {sig['signal']}"
                                )
                            else:
                                self.rejected_count += 1
                                logger.warning(
                                    f"REJECTED #{self.rejected_count}: "
                                    f"{sig['symbol']} - {reason}"
                                )

                        await self.redis.xack(stream_name, "risk-group", msg_id)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(1)

        logger.info("Risk manager shutting down")
        if self.redis:
            await self.redis.close()


def signal_handler(sig, frame):
    logger.info(f"Signal {sig}, shutting down...")
    shutdown_event.set()


async def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    mgr = RiskManager(redis_url=REDIS_URL)
    try:
        await mgr.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
