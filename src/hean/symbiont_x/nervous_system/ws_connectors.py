"""
WebSocket Connectors - Подключения к рынку в реальном времени

Это "глаза и уши" организма - получает данные с биржи
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketClientProtocol = None

from .event_envelope import (
    CandleEvent,
    EventEnvelope,
    OrderbookEvent,
    TradeEvent,
    create_candle_envelope,
    create_orderbook_envelope,
    create_trade_envelope,
)

logger = logging.getLogger(__name__)


class BybitWSConnector:
    """
    WebSocket коннектор для Bybit

    Подключается к различным стримам и превращает сырые данные
    в EventEnvelope объекты
    """

    # Bybit WebSocket URLs
    WS_PUBLIC = "wss://stream.bybit.com/v5/public/linear"
    WS_PRIVATE = "wss://stream.bybit.com/v5/private"

    def __init__(
        self,
        symbols: list[str],
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("websockets library not installed. WebSocket functionality will be limited.")

        self.symbols = symbols
        self.api_key = api_key
        self.api_secret = api_secret

        # WebSocket connections
        self.ws_public: WebSocketClientProtocol | None = None
        self.ws_private: WebSocketClientProtocol | None = None

        # Event handlers
        self.event_handlers: dict[str, list[Callable]] = {
            'trade': [],
            'orderbook': [],
            'candle': [],
            'funding': [],
            'position': [],
            'order': [],
        }

        # Health tracking
        self.last_message_time = {}
        self.message_count = {}
        self.is_connected = False
        self.reconnect_attempts = 0

    async def connect(self):
        """Подключение ко всем нужным стримам"""
        logger.info(f"Connecting to Bybit WebSocket for symbols: {self.symbols}")

        try:
            # Connect public streams
            await self._connect_public()

            # Connect private streams if credentials provided
            if self.api_key and self.api_secret:
                await self._connect_private()

            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("✅ WebSocket connected successfully")

        except Exception as e:
            logger.error(f"❌ Failed to connect WebSocket: {e}")
            self.reconnect_attempts += 1
            await self._handle_reconnect()

    async def _connect_public(self):
        """Подключение к публичным стримам"""
        self.ws_public = await websockets.connect(self.WS_PUBLIC)

        # Subscribe to topics
        subscriptions = []

        for symbol in self.symbols:
            # Trades
            subscriptions.append(f"publicTrade.{symbol}")

            # Orderbook (50 levels)
            subscriptions.append(f"orderbook.50.{symbol}")

            # Klines (candles)
            subscriptions.append(f"kline.1.{symbol}")  # 1m candles
            subscriptions.append(f"kline.5.{symbol}")  # 5m candles

        # Send subscription message
        subscribe_msg = {
            "op": "subscribe",
            "args": subscriptions
        }

        await self.ws_public.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(subscriptions)} public topics")

        # Start listening
        asyncio.create_task(self._listen_public())

    async def _connect_private(self):
        """Подключение к приватным стримам"""
        self.ws_private = await websockets.connect(self.WS_PRIVATE)

        # Auth
        expires = int((time.time() + 5) * 1000)
        signature = self._generate_signature(expires)

        auth_msg = {
            "op": "auth",
            "args": [self.api_key, expires, signature]
        }

        await self.ws_private.send(json.dumps(auth_msg))

        # Subscribe to private topics
        subscriptions = [
            "position",  # Position updates
            "execution",  # Order fills
            "order",  # Order updates
        ]

        subscribe_msg = {
            "op": "subscribe",
            "args": subscriptions
        }

        await self.ws_private.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(subscriptions)} private topics")

        # Start listening
        asyncio.create_task(self._listen_private())

    async def _listen_public(self):
        """Слушает публичные сообщения"""
        try:
            async for message in self.ws_public:
                data = json.loads(message)

                # Update health
                topic = data.get('topic', '')
                self.last_message_time[topic] = time.time()
                self.message_count[topic] = self.message_count.get(topic, 0) + 1

                # Process message
                await self._process_public_message(data)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Public WebSocket connection closed")
            await self._handle_reconnect()
        except Exception as e:
            logger.error(f"Error in public listener: {e}")

    async def _listen_private(self):
        """Слушает приватные сообщения"""
        try:
            async for message in self.ws_private:
                data = json.loads(message)

                # Process message
                await self._process_private_message(data)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Private WebSocket connection closed")
            await self._handle_reconnect()
        except Exception as e:
            logger.error(f"Error in private listener: {e}")

    async def _process_public_message(self, data: dict[str, Any]):
        """Обработка публичных сообщений"""
        topic = data.get('topic', '')
        msg_type = data.get('type', '')

        # Pong/heartbeat
        if msg_type == 'pong':
            return

        # Data message
        if 'data' not in data:
            return

        payload = data['data']

        # Trade
        if 'publicTrade' in topic:
            await self._process_trade(payload, topic)

        # Orderbook
        elif 'orderbook' in topic:
            await self._process_orderbook(payload, topic)

        # Kline (candle)
        elif 'kline' in topic:
            await self._process_candle(payload, topic)

    async def _process_trade(self, data: list[dict], topic: str):
        """Обработка trade события"""
        # Extract symbol from topic (e.g., "publicTrade.BTCUSDT")
        symbol = topic.split('.')[-1]

        for trade_data in data:
            # Create trade event
            trade = TradeEvent(
                price=float(trade_data['p']),
                qty=float(trade_data['v']),
                side=trade_data['S'],  # "Buy" or "Sell"
                trade_id=trade_data['i'],
                is_block_trade=trade_data.get('BT', False)
            )

            # Create envelope
            envelope = create_trade_envelope(
                trade=trade,
                symbol=symbol,
                timestamp_ns=int(trade_data['T']) * 1_000_000,  # ms to ns
                source="bybit_ws"
            )

            # Emit to handlers
            await self._emit_event('trade', envelope)

    async def _process_orderbook(self, data: dict, topic: str):
        """Обработка orderbook события"""
        symbol = topic.split('.')[-1]

        # Create orderbook event
        orderbook = OrderbookEvent(
            bids=[(float(b[0]), float(b[1])) for b in data['b']],
            asks=[(float(a[0]), float(a[1])) for a in data['a']],
            update_id=data['u']
        )

        # Create envelope
        envelope = create_orderbook_envelope(
            orderbook=orderbook,
            symbol=symbol,
            timestamp_ns=int(data['ts']) * 1_000_000,
            source="bybit_ws"
        )

        # Emit to handlers
        await self._emit_event('orderbook', envelope)

    async def _process_candle(self, data: list[dict], topic: str):
        """Обработка candle события"""
        symbol = topic.split('.')[-1]
        timeframe = topic.split('.')[1]  # "1", "5", etc

        for candle_data in data:
            # Create candle event
            candle = CandleEvent(
                open=float(candle_data['open']),
                high=float(candle_data['high']),
                low=float(candle_data['low']),
                close=float(candle_data['close']),
                volume=float(candle_data['volume']),
                timeframe=f"{timeframe}m",
                is_confirmed=candle_data.get('confirm', False)
            )

            # Create envelope
            envelope = create_candle_envelope(
                candle=candle,
                symbol=symbol,
                timestamp_ns=int(candle_data['start']) * 1_000_000,
                source="bybit_ws"
            )

            # Emit to handlers
            await self._emit_event('candle', envelope)

    async def _process_private_message(self, data: dict[str, Any]):
        """Process private WebSocket messages (positions, orders, fills)."""
        topic = data.get('topic', '')
        msg_data = data.get('data', [])

        if topic == 'position':
            for item in msg_data:
                envelope = EventEnvelope(
                    event_type='position_update',
                    payload=item,
                    source='bybit_ws_private',
                    timestamp_ns=int(time.time() * 1_000_000_000),
                )
                await self._emit_event('position', envelope)

        elif topic == 'order':
            for item in msg_data:
                envelope = EventEnvelope(
                    event_type='order_update',
                    payload=item,
                    source='bybit_ws_private',
                    timestamp_ns=int(time.time() * 1_000_000_000),
                )
                await self._emit_event('order', envelope)

        elif topic == 'execution':
            for item in msg_data:
                envelope = EventEnvelope(
                    event_type='execution',
                    payload=item,
                    source='bybit_ws_private',
                    timestamp_ns=int(time.time() * 1_000_000_000),
                )
                await self._emit_event('execution', envelope)

    async def _emit_event(self, event_type: str, envelope: EventEnvelope):
        """Отправляет событие всем подписчикам"""
        handlers = self.event_handlers.get(event_type, [])

        for handler in handlers:
            try:
                await handler(envelope)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    def on_trade(self, handler: Callable):
        """Регистрирует обработчик trade событий"""
        self.event_handlers['trade'].append(handler)

    def on_orderbook(self, handler: Callable):
        """Регистрирует обработчик orderbook событий"""
        self.event_handlers['orderbook'].append(handler)

    def on_candle(self, handler: Callable):
        """Регистрирует обработчик candle событий"""
        self.event_handlers['candle'].append(handler)

    async def _handle_reconnect(self):
        """Обработка переподключения"""
        if self.reconnect_attempts > 10:
            logger.critical("❌ Max reconnect attempts reached. Stopping.")
            self.is_connected = False
            return

        # Exponential backoff
        wait_time = min(60, 2 ** self.reconnect_attempts)
        logger.info(f"Reconnecting in {wait_time}s (attempt {self.reconnect_attempts})")

        await asyncio.sleep(wait_time)
        await self.connect()

    def _generate_signature(self, expires: int) -> str:
        """Генерирует подпись для приватного WS"""
        import hashlib
        import hmac

        message = f"GET/realtime{expires}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    async def send_heartbeat(self):
        """Отправляет heartbeat для поддержания соединения"""
        while self.is_connected:
            try:
                if self.ws_public:
                    await self.ws_public.send(json.dumps({"op": "ping"}))

                if self.ws_private:
                    await self.ws_private.send(json.dumps({"op": "ping"}))

                await asyncio.sleep(20)  # Ping every 20s
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")

    async def disconnect(self):
        """Отключение от всех стримов"""
        self.is_connected = False

        if self.ws_public:
            await self.ws_public.close()

        if self.ws_private:
            await self.ws_private.close()

        logger.info("WebSocket disconnected")

    def get_health_status(self) -> dict[str, Any]:
        """Возвращает статус здоровья соединения"""
        now = time.time()

        health = {
            'is_connected': self.is_connected,
            'reconnect_attempts': self.reconnect_attempts,
            'topics': {}
        }

        for topic, last_time in self.last_message_time.items():
            age_seconds = now - last_time
            message_count = self.message_count.get(topic, 0)

            health['topics'][topic] = {
                'last_message_age_seconds': age_seconds,
                'message_count': message_count,
                'healthy': age_seconds < 60,  # Warning if no message in 60s
            }

        return health
