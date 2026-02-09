# HEAN — ПОШАГОВОЕ ТЕХНИЧЕСКОЕ ЗАДАНИЕ ДЛЯ AI-РАЗРАБОТЧИКА

## Инструкция: Как Пользоваться Этим Документом

> **Этот файл — полное ТЗ для Claude (или любого AI-кодера).**
> Ты открываешь новый чат с Claude, прикладываешь этот файл и говоришь:
> *«Прочитай этот документ. Начни с Фазы 1, Шаг 1. Напиши весь код. Дай мне файлы.»*
> Claude прочитает и напишет рабочий код. Ты копируешь файлы к себе и запускаешь.
> Когда Фаза 1 работает — возвращаешься и говоришь: *«Фаза 2.»*

---

# ОБЩИЕ ПРАВИЛА ДЛЯ AI-РАЗРАБОТЧИКА

Прочитай и запомни перед написанием любого кода:

```
ЯЗЫК БЭКЕНДА: Python 3.12+
СТИЛЬ: async/await везде, никаких sync блокирующих вызовов
ТИПИЗАЦИЯ: type hints обязательны для всех функций
ФРЕЙМВОРК: FastAPI + uvicorn
ДАННЫЕ: Polars (НЕ Pandas), DuckDB (НЕ SQLite)
JSON: orjson (НЕ стандартный json)
ЛОГИРОВАНИЕ: loguru (НЕ стандартный logging)
КОНФИГУРАЦИЯ: pydantic Settings + .env файл
БИРЖИ: ccxt async + websockets
AI: anthropic SDK для Claude API
ТЕСТЫ: pytest + pytest-asyncio

ЯЗЫК ФРОНТЕНДА: Swift 6 + SwiftUI
МИНИМАЛЬНАЯ iOS: 17.0
АРХИТЕКТУРА: @Observable (НЕ ObservableObject)
СЕТЬ: URLSession native (НЕ Alamofire)
ГРАФИКИ: Swift Charts

СТРУКТУРА ПАПОК:
hean/
├── server/          # Python бэкенд
├── ios/             # Swift iOS app
├── data/            # DuckDB + Parquet
├── .env             # Секреты
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

# ФАЗА 1: ГЛАЗА + МОЗГ (Сбор данных + Claude AI)

> **Результат фазы:** Запущенный сервер который:
> 1. Подключается к Bybit WebSocket
> 2. Получает ордербук, сделки, ликвидации в реальном времени
> 3. Сохраняет всё в DuckDB
> 4. Каждые 30 секунд отправляет снимок рынка в Claude API
> 5. Получает анализ и торговый сигнал
> 6. Отдаёт всё на iPhone через WebSocket

---

## Фаза 1, Шаг 1: Структура проекта и зависимости

**Задача:** Создать структуру папок и установить зависимости.

Создай файл `requirements.txt`:
```
# Ядро
fastapi>=0.115
uvicorn[standard]>=0.34
uvloop>=0.21
websockets>=14.0
pydantic>=2.10
pydantic-settings>=2.7
orjson>=3.10

# Биржи
ccxt>=4.4
pybit>=5.9

# Данные
polars>=1.20
numpy>=2.2
scipy>=1.15
duckdb>=1.2
pyarrow>=18.0

# AI
anthropic>=0.43

# Утилиты
loguru>=0.7
httpx>=0.28
python-dotenv>=1.0
```

Создай файл `.env`:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
BYBIT_API_KEY=xxxxxxxx
BYBIT_API_SECRET=xxxxxxxx
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=INFO
```

Создай структуру папок:
```
server/
├── __init__.py
├── main.py              # Точка входа FastAPI
├── config.py            # Настройки из .env
├── collectors/
│   ├── __init__.py
│   ├── bybit_ws.py      # WebSocket коллектор Bybit
│   └── models.py        # Pydantic модели данных с бирж
├── brain/
│   ├── __init__.py
│   ├── claude_client.py # Клиент Claude API
│   ├── snapshot.py      # Формирование снимка рынка
│   └── models.py        # Pydantic модели анализа
├── storage/
│   ├── __init__.py
│   └── database.py      # DuckDB менеджер
├── api/
│   ├── __init__.py
│   ├── ws_handler.py    # WebSocket для iPhone
│   └── routes.py        # REST эндпоинты
└── utils/
    ├── __init__.py
    └── helpers.py
```

---

## Фаза 1, Шаг 2: Конфигурация

**Задача:** Файл `server/config.py` — загрузка настроек из .env.

**Что должен делать:**
- Загружать все переменные из .env через pydantic-settings
- Валидировать что API ключи не пустые
- Предоставлять typed access ко всем настройкам

**Модель:**
```python
class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str
    
    # Bybit
    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    
    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    log_level: str = "INFO"
    
    # Trading
    symbol: str = "BTCUSDT"
    orderbook_depth: int = 50
    snapshot_interval_sec: int = 30
    
    # Paths
    db_path: str = "data/hean.db"
    
    model_config = SettingsConfigDict(env_file=".env")
```

Создай синглтон `settings = Settings()` для импорта из любого модуля.

---

## Фаза 1, Шаг 3: Модели данных

**Задача:** Файл `server/collectors/models.py` — Pydantic модели для данных с бирж.

**Нужны следующие модели:**

```python
class OrderBookLevel(BaseModel):
    price: float
    size: float

class OrderBookSnapshot(BaseModel):
    timestamp: datetime
    symbol: str
    bids: list[OrderBookLevel]  # отсортированы по цене DESC
    asks: list[OrderBookLevel]  # отсортированы по цене ASC
    
    @computed_field
    @property
    def mid_price(self) -> float:
        """Средняя цена между лучшим bid и ask"""
        
    @computed_field
    @property
    def spread(self) -> float:
        """Спред в процентах"""
        
    @computed_field
    @property    
    def bid_total(self) -> float:
        """Суммарный объём бидов"""
        
    @computed_field
    @property
    def ask_total(self) -> float:
        """Суммарный объём асков"""
        
    @computed_field
    @property
    def imbalance(self) -> float:
        """(bid_total - ask_total) / (bid_total + ask_total). 
        Диапазон [-1, 1]. Положительный = покупатели доминируют."""

class Trade(BaseModel):
    timestamp: datetime
    symbol: str
    price: float
    size: float
    side: Literal["Buy", "Sell"]
    
class Liquidation(BaseModel):
    timestamp: datetime
    symbol: str
    side: Literal["Buy", "Sell"]
    price: float
    size: float

class MarketState(BaseModel):
    """Агрегированное состояние рынка. Обновляется каждые 100мс."""
    timestamp: datetime
    symbol: str
    price: float
    orderbook: OrderBookSnapshot
    recent_trades: list[Trade]       # последние 100 сделок
    recent_liquidations: list[Liquidation]  # ликвидации за последний час
    
    # Агрегаты (вычисляются из данных выше)
    delta_1m: float = 0.0    # Buy volume - Sell volume за 1 мин
    delta_5m: float = 0.0    # Buy volume - Sell volume за 5 мин
    volume_1m: float = 0.0   # Общий объём за 1 мин
    big_trades_count: int = 0 # Сделки > 1 BTC за 5 мин
```

**Также** файл `server/brain/models.py`:

```python
class Force(BaseModel):
    name: str           # "gravitational_pull", "institutional_flow", etc.
    direction: str      # "UP", "DOWN", "NEUTRAL"
    strength: float     # 0.0 - 1.0
    description: str

class TradingSignal(BaseModel):
    timestamp: datetime
    direction: Literal["LONG", "SHORT", "NO_TRADE"]
    confidence: float           # 0.0 - 1.0
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    risk_reward: float | None = None
    reasoning: str              # Почему AI принял это решение

class BrainAnalysis(BaseModel):
    timestamp: datetime
    situation: str              # Описание ситуации человеческим языком
    forces: list[Force]         # Силы действующие на рынок
    signal: TradingSignal
    market_phase: Literal["ice", "ice_to_water", "water", "water_to_steam", "steam"]
    danger_level: float         # 0.0 - 1.0
```

---

## Фаза 1, Шаг 4: Bybit WebSocket Коллектор

**Задача:** Файл `server/collectors/bybit_ws.py` — подключение к Bybit и получение данных.

**Что должен делать:**
1. Подключиться к `wss://stream.bybit.com/v5/public/linear`
2. Подписаться на 3 топика:
   - `orderbook.50.BTCUSDT` — ордербук 50 уровней
   - `publicTrade.BTCUSDT` — все сделки
   - `liquidation.BTCUSDT` — ликвидации
3. Поддерживать соединение (ping/pong каждые 20 секунд)
4. При получении данных — парсить в Pydantic модели и складывать в очередь
5. При разрыве соединения — переподключаться через 3 секунды
6. Вести лог: сколько сообщений получено, средняя задержка

**Формат подписки Bybit v5:**
```json
{"op": "subscribe", "args": ["orderbook.50.BTCUSDT", "publicTrade.BTCUSDT", "liquidation.BTCUSDT"]}
```

**Формат ордербука Bybit v5:**
```json
{
    "topic": "orderbook.50.BTCUSDT",
    "type": "snapshot",  // или "delta"
    "data": {
        "s": "BTCUSDT",
        "b": [["94247.50", "2.345"], ...],  // bids: [price, size]
        "a": [["94248.00", "1.123"], ...],  // asks: [price, size]
        "u": 1234567  // update id
    },
    "ts": 1707312000000  // timestamp ms
}
```

**ВАЖНО:** Bybit шлёт первый снимок как `"type": "snapshot"`, потом дельты `"type": "delta"`. Нужно:
- При snapshot — заменить весь ордербук
- При delta — обновить только изменённые уровни (size=0 значит удалить уровень)

**Формат сделок Bybit v5:**
```json
{
    "topic": "publicTrade.BTCUSDT",
    "data": [
        {
            "s": "BTCUSDT",
            "S": "Buy",      // Buy или Sell
            "v": "0.123",    // volume
            "p": "94247.50", // price
            "T": 1707312000000  // timestamp ms
        }
    ]
}
```

**Формат ликвидаций Bybit v5:**
```json
{
    "topic": "liquidation.BTCUSDT",
    "data": {
        "symbol": "BTCUSDT",
        "side": "Sell",
        "price": "93200.00",
        "size": "0.567",
        "updatedTime": 1707312000000
    }
}
```

**Архитектура класса:**
```python
class BybitCollector:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.orderbook: OrderBookSnapshot | None = None
        self.trades: deque[Trade] = deque(maxlen=1000)
        self.liquidations: deque[Liquidation] = deque(maxlen=500)
        self._ws = None
        self._running = False
        
    async def start(self):
        """Запустить бесконечный цикл подключения"""
        
    async def _connect(self):
        """Подключиться и подписаться"""
        
    async def _listen(self):
        """Слушать сообщения, парсить, обновлять состояние"""
        
    async def _handle_orderbook(self, data: dict, msg_type: str):
        """Обработать snapshot или delta ордербука"""
        
    async def _handle_trade(self, data: list[dict]):
        """Обработать сделки"""
        
    async def _handle_liquidation(self, data: dict):
        """Обработать ликвидацию"""
        
    def get_market_state(self) -> MarketState:
        """Вернуть текущее агрегированное состояние рынка.
        Вычислить delta_1m, delta_5m, volume_1m, big_trades_count
        из self.trades за последние 1 и 5 минут."""
        
    async def _ping_loop(self):
        """Каждые 20 сек отправлять ping для поддержания соединения"""
```

---

## Фаза 1, Шаг 5: DuckDB хранение

**Задача:** Файл `server/storage/database.py` — менеджер базы данных.

**Что должен делать:**
1. При старте — создать таблицы если не существуют
2. Сохранять ордербук-снимки (каждые 5 секунд, не каждые 100мс — иначе БД раздуется)
3. Сохранять все сделки
4. Сохранять ликвидации
5. Сохранять анализы мозга (BrainAnalysis)
6. Сохранять торговые сигналы
7. Предоставлять методы для запросов

**Таблицы:**

```sql
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    timestamp TIMESTAMP,
    symbol VARCHAR,
    mid_price DOUBLE,
    spread DOUBLE,
    bid_total DOUBLE,
    ask_total DOUBLE,
    imbalance DOUBLE,
    top_bid_price DOUBLE,
    top_ask_price DOUBLE,
    top_bid_size DOUBLE,
    top_ask_size DOUBLE
);

CREATE TABLE IF NOT EXISTS trades (
    timestamp TIMESTAMP,
    symbol VARCHAR,
    price DOUBLE,
    size DOUBLE,
    side VARCHAR
);

CREATE TABLE IF NOT EXISTS liquidations (
    timestamp TIMESTAMP,
    symbol VARCHAR,
    side VARCHAR,
    price DOUBLE,
    size DOUBLE
);

CREATE TABLE IF NOT EXISTS brain_analyses (
    timestamp TIMESTAMP,
    situation TEXT,
    market_phase VARCHAR,
    danger_level DOUBLE,
    signal_direction VARCHAR,
    signal_confidence DOUBLE,
    signal_entry DOUBLE,
    signal_stop DOUBLE,
    signal_tp1 DOUBLE,
    signal_tp2 DOUBLE,
    signal_rr DOUBLE,
    reasoning TEXT
);
```

**Архитектура класса:**
```python
class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        
    async def initialize(self):
        """Создать директории, подключиться, создать таблицы"""
        
    def save_orderbook(self, snapshot: OrderBookSnapshot):
        """Сохранить снимок ордербука"""
        
    def save_trade(self, trade: Trade):
        """Сохранить сделку"""
        
    def save_trades_batch(self, trades: list[Trade]):
        """Сохранить пачку сделок (эффективнее)"""
        
    def save_liquidation(self, liq: Liquidation):
        """Сохранить ликвидацию"""
        
    def save_analysis(self, analysis: BrainAnalysis):
        """Сохранить анализ мозга"""
        
    def get_recent_trades(self, minutes: int = 5) -> list[dict]:
        """Получить сделки за последние N минут"""
        
    def get_liquidations_today(self) -> list[dict]:
        """Получить ликвидации за сегодня"""
        
    def get_stats(self) -> dict:
        """Получить статистику: кол-во записей в каждой таблице"""
```

**ВАЖНО:** DuckDB НЕ поддерживает async. Используй `asyncio.to_thread()` для вызовов из async контекста, или используй синхронный DuckDB в отдельном потоке.

---

## Фаза 1, Шаг 6: Claude Brain (Стратегический Мозг)

**Задача:** Файл `server/brain/claude_client.py` — клиент для анализа рынка через Claude.

**Что должен делать:**
1. Принимать MarketState
2. Формировать промпт с данными рынка
3. Отправлять в Claude API
4. Парсить ответ в BrainAnalysis
5. Возвращать структурированный результат

**Системный промпт для Claude (ВСТАВИТЬ ДОСЛОВНО):**

```
Ты — HEAN, термодинамический торговый движок. Ты анализируешь рынок как физическую систему.

ТВОИ ПРИНЦИПЫ:
1. Цена — следствие. Ты смотришь на ПРИЧИНЫ: ордерфлоу, ликвидности, поведение участников.
2. Ты классифицируешь участников: маркетмейкер, институционал, ритейл, бот, кит.
3. Ты измеряешь "температуру" (ликвидность/волатильность) и "энтропию" (хаос/порядок).
4. Ты торгуешь ТОЛЬКО когда физика рынка требует движения. Иначе — NO_TRADE.
5. Минимальный R:R = 1:2. Если не видишь — не торгуй.

ПРАВИЛА АНАЛИЗА:
- Дисбаланс ордербука > 0.3 = сильный сигнал
- Дельта отрицательная при росте = дивергенция (медвежий сигнал)
- Крупные сделки (>1 BTC) = институциональная активность
- Скопление ликвидаций = магнит для цены
- Funding rate > 0.05% = перегрев лонгов
- Long/Short ratio > 70/30 = опасность для доминирующей стороны

ФОРМАТ ОТВЕТА — ТОЛЬКО JSON, без markdown:
{
  "situation": "описание ситуации на рынке",
  "forces": [
    {"name": "имя_силы", "direction": "UP/DOWN/NEUTRAL", "strength": 0.0-1.0, "description": "описание"}
  ],
  "signal": {
    "direction": "LONG/SHORT/NO_TRADE",
    "confidence": 0.0-1.0,
    "entry_price": число или null,
    "stop_loss": число или null,
    "take_profit_1": число или null,
    "take_profit_2": число или null,
    "risk_reward": число или null,
    "reasoning": "почему принял это решение"
  },
  "market_phase": "ice/ice_to_water/water/water_to_steam/steam",
  "danger_level": 0.0-1.0
}
```

**Формирование пользовательского промпта** (файл `server/brain/snapshot.py`):

```python
def format_market_snapshot(state: MarketState) -> str:
    """Превращает MarketState в текстовый снимок для Claude.
    
    Формат:
    === СНИМОК РЫНКА {symbol} ===
    Время: {timestamp}
    Цена: {price}
    
    ОРДЕРБУК:
    - Спред: {spread}%
    - Дисбаланс: {imbalance} (>0 = покупатели, <0 = продавцы)
    - Суммарный bid: {bid_total} BTC
    - Суммарный ask: {ask_total} BTC
    - Крупнейшая стена bid: {price} @ {size} BTC
    - Крупнейшая стена ask: {price} @ {size} BTC
    
    ОРДЕРФЛОУ (последние 5 минут):
    - Дельта 1мин: {delta_1m} BTC (>0 = покупки, <0 = продажи)
    - Дельта 5мин: {delta_5m} BTC
    - Объём 1мин: {volume_1m} BTC
    - Крупных сделок (>1 BTC): {big_trades_count}
    - Последние 5 крупных: {описание каждой}
    
    ЛИКВИДАЦИИ (за час):
    - Long ликвидации: {total} BTC на сумму ${total_usd}
    - Short ликвидации: {total} BTC на сумму ${total_usd}
    
    ДОПОЛНИТЕЛЬНО:
    - Последние 3 анализа (если есть): краткое резюме предыдущих решений
    """
```

**Архитектура Claude клиента:**
```python
class HEANBrain:
    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        self.system_prompt = "..."  # системный промпт выше
        self.history: deque[BrainAnalysis] = deque(maxlen=100)
        
    async def analyze(self, state: MarketState) -> BrainAnalysis:
        """
        1. Сформировать снимок через format_market_snapshot(state)
        2. Отправить в Claude API с system_prompt
        3. Получить ответ
        4. Распарсить JSON в BrainAnalysis
        5. Сохранить в self.history
        6. Вернуть результат
        
        При ошибке парсинга — повторить запрос 1 раз.
        При ошибке API — логировать и вернуть NO_TRADE с пояснением.
        """
        
    def get_last_analyses(self, n: int = 3) -> list[BrainAnalysis]:
        """Последние N анализов для контекста"""
```

**ВАЖНО:** 
- Используй `anthropic.AsyncAnthropic` (async версию)
- Модель: `claude-sonnet-4-20250514` (баланс скорости и качества)
- `max_tokens=2000`
- `temperature=0.3` (нужна стабильность, не креативность)
- Парси JSON из ответа: Claude может обернуть в ```json```, нужно вычистить

---

## Фаза 1, Шаг 7: WebSocket для iPhone

**Задача:** Файл `server/api/ws_handler.py` — WebSocket сервер для отправки данных на iPhone.

**Что должен делать:**
1. Принимать WebSocket подключения на `/ws`
2. При подключении — отправлять текущее состояние
3. Каждые 500мс — отправлять обновление MarketState
4. При новом анализе мозга — отправлять BrainAnalysis
5. Поддерживать несколько подключений одновременно
6. При отключении клиента — чистить ресурсы

**Формат сообщений сервер → клиент:**

```json
{
    "type": "market_update",
    "data": {
        "price": 94247.50,
        "spread": 0.01,
        "imbalance": -0.23,
        "delta_1m": -1.45,
        "delta_5m": -4.82,
        "volume_1m": 12.34,
        "big_trades_count": 7,
        "bid_total": 124.5,
        "ask_total": 167.2
    }
}
```

```json
{
    "type": "brain_analysis",
    "data": {
        "situation": "...",
        "forces": [...],
        "signal": {...},
        "market_phase": "ice_to_water",
        "danger_level": 0.45
    }
}
```

```json
{
    "type": "system_status",
    "data": {
        "uptime_seconds": 3600,
        "ws_connected": true,
        "last_analysis": "2025-02-07T14:32:10Z",
        "trades_collected": 45230,
        "db_size_mb": 128.5
    }
}
```

**Архитектура:**
```python
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: dict):
        """Отправить всем подключённым клиентам.
        Использовать orjson для сериализации.
        При ошибке отправки — отключить клиента."""
```

---

## Фаза 1, Шаг 8: REST API

**Задача:** Файл `server/api/routes.py` — REST эндпоинты.

**Эндпоинты:**

```
GET /api/status
  → Статус системы: uptime, подключение к бирже, последний анализ, кол-во данных

GET /api/market
  → Текущий MarketState (последний снимок)

GET /api/brain/latest
  → Последний BrainAnalysis

GET /api/brain/history?limit=20
  → Последние N анализов

GET /api/trades/recent?minutes=5
  → Недавние сделки

GET /api/liquidations/today
  → Ликвидации за сегодня

GET /api/stats
  → Статистика БД: кол-во записей, размер, uptime
```

Все ответы — через `ORJSONResponse` (быстрее стандартного).

---

## Фаза 1, Шаг 9: Главный файл (оркестрация)

**Задача:** Файл `server/main.py` — точка входа, запуск всех компонентов.

**Что должен делать:**
1. Загрузить конфигурацию
2. Инициализировать DuckDB
3. Запустить Bybit WebSocket коллектор (в фоне)
4. Запустить Brain анализатор (каждые N секунд в фоне)
5. Запустить FastAPI сервер с WebSocket и REST
6. При остановке (SIGINT/SIGTERM) — корректно закрыть всё

**Логика:**
```python
app = FastAPI(title="HEAN Server", version="0.1.0")

@app.on_event("startup")
async def startup():
    # 1. Инициализировать БД
    app.state.db = Database(settings.db_path)
    await app.state.db.initialize()
    
    # 2. Запустить коллектор
    app.state.collector = BybitCollector(settings.symbol)
    asyncio.create_task(app.state.collector.start())
    
    # 3. Запустить мозг
    app.state.brain = HEANBrain(settings.anthropic_api_key)
    asyncio.create_task(brain_loop(app))
    
    # 4. Запустить периодическое сохранение в БД
    asyncio.create_task(db_save_loop(app))
    
    # 5. Запустить рассылку через WebSocket
    asyncio.create_task(ws_broadcast_loop(app))
    
    logger.info("HEAN Server started 🚀")

async def brain_loop(app):
    """Каждые snapshot_interval_sec секунд:
    1. Получить market_state из коллектора
    2. Если коллектор ещё не подключился — ждать
    3. Отправить в brain.analyze()
    4. Сохранить результат в БД
    5. Разослать через WebSocket
    """

async def db_save_loop(app):
    """Каждые 5 секунд:
    1. Сохранить текущий снимок ордербука
    2. Сохранить пачку новых сделок
    3. Сохранить новые ликвидации
    """

async def ws_broadcast_loop(app):
    """Каждые 500мс:
    1. Получить market_state из коллектора
    2. Сформировать market_update сообщение
    3. Разослать всем WebSocket клиентам
    """

# Подключить роуты
app.include_router(api_router)

# WebSocket эндпоинт
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Ждём сообщения от клиента (ping/pong или команды)
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Запуск
if __name__ == "__main__":
    uvicorn.run("server.main:app", host=settings.server_host, 
                port=settings.server_port, reload=False)
```

---

## Фаза 1, Шаг 10: Тестирование

**Задача:** Убедиться что всё работает.

**Тест 1: Запуск**
```bash
cd hean/
python -m server.main
```
Ожидаемый лог:
```
INFO | HEAN Server started 🚀
INFO | Connecting to Bybit WebSocket...
INFO | Subscribed to orderbook.50.BTCUSDT
INFO | Subscribed to publicTrade.BTCUSDT
INFO | Subscribed to liquidation.BTCUSDT
INFO | First orderbook snapshot received: mid_price=94247.50
INFO | Brain analysis #1: market_phase=ice, signal=NO_TRADE
```

**Тест 2: REST API**
```bash
curl http://localhost:8000/api/status
curl http://localhost:8000/api/market
curl http://localhost:8000/api/brain/latest
```

**Тест 3: WebSocket**
```python
# test_ws.py
import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        for _ in range(10):
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"Type: {data['type']}, Price: {data.get('data', {}).get('price', 'N/A')}")

asyncio.run(test())
```

---

# ФАЗА 2: ФИЗИКА (Температура, Энтропия, Фазы)

> **Результат фазы:** Сервер вычисляет T, S, определяет фазу рынка. 
> iPhone видит термометр и фазовые переходы.

---

## Фаза 2, Шаг 1: Модуль термодинамики

**Задача:** Создать `server/physics/` с модулями вычислений.

```
server/physics/
├── __init__.py
├── temperature.py    # Вычисление температуры рынка
├── entropy.py        # Вычисление энтропии
├── phase.py          # Определение фазы (лёд/вода/пар)
└── models.py         # Pydantic модели физики
```

**Файл `server/physics/models.py`:**
```python
class ThermodynamicState(BaseModel):
    timestamp: datetime
    temperature: float          # 0 - 2000+ (безразмерная)
    entropy: float              # 0.0 - 1.0
    phase: Literal["ice", "ice_to_water", "water", "water_to_steam", "steam"]
    phase_change_probability: float  # Вероятность фазового перехода
    energy: float               # Кинетическая энергия ордербука
    pressure: float             # "Давление" — дисбаланс × объём
```

**Файл `server/physics/temperature.py`:**

```python
def calculate_temperature(orderbook: OrderBookSnapshot, recent_trades: list[Trade]) -> float:
    """
    Температура рынка = мера активности и ликвидности.
    
    Формула:
    T = (KE / N) * scaling_factor
    
    где:
    KE (кинетическая энергия) = сумма по всем уровням ордербука:
        (изменение_цены_уровня × объём_на_уровне)²
        
    На практике (без истории уровней) используем приближение:
    KE = sum_i( (price_i - mid_price)² * size_i ) для всех уровней bid и ask
    
    N = количество ненулевых уровней
    
    scaling_factor = 1000 (для удобства чтения)
    
    Дополнительно учитываем объём сделок:
    trade_energy = sum( price * size ) за последние 60 секунд
    
    T_final = alpha * KE/N + beta * trade_energy
    где alpha=0.7, beta=0.3
    
    ВАЖНО: Хранить историю T за последние 60 минут для определения трендов.
    """
```

**Файл `server/physics/entropy.py`:**

```python
def calculate_entropy(orderbook: OrderBookSnapshot) -> float:
    """
    Энтропия рынка = мера распределения ордеров (хаос vs порядок).
    
    Формула (Shannon Entropy):
    S = -sum( p_i * log2(p_i) ) / log2(N)
    
    где:
    p_i = size_i / total_size  — доля объёма на уровне i
    N = количество уровней
    Делим на log2(N) для нормализации в [0, 1]
    
    Интерпретация:
    S → 1.0: ордера распределены равномерно → рынок в балансе, нет направления
    S → 0.0: ордера сконцентрированы на 1-2 уровнях → кто-то набирает позицию
    
    Считаем отдельно для bid и ask, потом среднее:
    S = (S_bid + S_ask) / 2
    
    ВАЖНО: 
    - Если size_i == 0, пропускаем (log(0) = -inf)
    - Хранить историю S за последние 60 минут
    - Резкое падение S (>20% за 5 мин) = АНОМАЛИЯ
    """
```

**Файл `server/physics/phase.py`:**

```python
def detect_phase(
    temperature_history: list[float],  # T за последние 30 мин (по 1 в минуту)
    entropy_history: list[float],      # S за последние 30 мин
    current_t: float,
    current_s: float
) -> tuple[str, float]:
    """
    Определение фазы рынка и вероятности фазового перехода.
    
    Фазы:
    
    ICE (боковик, замороженный рынок):
        T < median(T_history) AND S > 0.6
        Признаки: низкая волатильность, ордера равномерны, нет направления
        
    ICE_TO_WATER (сжатие перед движением):
        T < median AND S падает (diff_5min < -0.1)
        Признаки: T ещё низкая, но S резко падает → кто-то концентрирует ордера
        ЭТО САМЫЙ ВАЖНЫЙ МОМЕНТ — пружина сжимается
        
    WATER (нормальный тренд):
        T > median AND S в среднем диапазоне (0.3-0.7)
        Признаки: активное движение, ордера умеренно распределены
        
    WATER_TO_STEAM (разгон к каскаду):
        T растёт быстро (diff_5min > +20%) AND S падает
        Признаки: движение ускоряется, ордера исчезают с одной стороны
        
    STEAM (каскад/паника):
        T > 90th percentile AND (S < 0.2 OR S > 0.9)
        Признаки: экстремальная активность, либо полный хаос, либо одностороннее движение
    
    Вероятность фазового перехода:
        Считаем как скорость изменения S (|dS/dt|) нормализованную в [0, 1]
        Быстрое изменение энтропии = высокая вероятность перехода
    
    Returns: (phase_name, transition_probability)
    """
```

---

## Фаза 2, Шаг 2: Интеграция физики в сервер

**Задача:** Добавить физический движок в main loop.

**Изменения:**

1. В `server/main.py` добавить `PhysicsEngine`:
```python
class PhysicsEngine:
    def __init__(self):
        self.temperature_history: deque[float] = deque(maxlen=60)  # 60 мин
        self.entropy_history: deque[float] = deque(maxlen=60)
        self.current_state: ThermodynamicState | None = None
        
    def update(self, market_state: MarketState) -> ThermodynamicState:
        """
        1. Вычислить T из ордербука и сделок
        2. Вычислить S из ордербука
        3. Добавить в историю
        4. Определить фазу
        5. Вернуть ThermodynamicState
        """
```

2. В `brain_loop` — добавить ThermodynamicState в промпт Claude:
```
ФИЗИКА РЫНКА:
- Температура: {T} ({описание: холодный/тёплый/горячий/взрыв})
- Энтропия: {S} ({описание: порядок/баланс/хаос})
- Фаза: {phase}
- Вероятность фазового перехода: {prob}%
```

3. В WebSocket broadcast — добавить поля:
```json
{
    "type": "market_update",
    "data": {
        "price": 94247.50,
        "temperature": 847,
        "entropy": 0.34,
        "phase": "ice_to_water",
        "phase_change_prob": 0.72,
        ...
    }
}
```

---

# ФАЗА 3: РЕНТГЕН (Кластеризация Участников)

> **Результат фазы:** AI определяет КТО стоит за каждым ордером. 
> iPhone показывает: "ММ охотится на стопы", "Институционал набирает шорт".

---

## Фаза 3, Шаг 1: Классификатор участников

**Задача:** Создать `server/xray/` — модуль рентгена рынка.

```
server/xray/
├── __init__.py
├── classifier.py     # Классификация ордеров по типам
├── detector.py       # Детектор аномалий (стоп-хант, iceberg, спуфинг)
└── models.py         # Pydantic модели
```

**Модели:**
```python
class ParticipantType(str, Enum):
    MARKET_MAKER = "market_maker"
    INSTITUTIONAL = "institutional"
    RETAIL = "retail"
    BOT_ARBITRAGE = "bot_arbitrage"
    WHALE = "whale"

class ParticipantActivity(BaseModel):
    type: ParticipantType
    activity_level: float      # 0.0 - 1.0
    direction: str             # "buying", "selling", "neutral"
    description: str
    notable_actions: list[str] # Список заметных действий

class XRaySnapshot(BaseModel):
    timestamp: datetime
    participants: list[ParticipantActivity]
    dominant_player: ParticipantType
    anomalies: list[str]        # Обнаруженные аномалии
    stop_hunt_detected: bool
    stop_hunt_direction: str | None
```

**Логика классификации (эвристическая, без ML):**

```python
def classify_participants(
    orderbook: OrderBookSnapshot,
    trades: list[Trade],         # последние 500 сделок
    prev_orderbook: OrderBookSnapshot | None  # предыдущий снимок
) -> XRaySnapshot:
    """
    МАРКЕТМЕЙКЕР детектируется когда:
    - Симметричные ордера на bid и ask примерно одного размера
    - Частые отмены/перестановки (если есть prev_orderbook: уровни появляются/исчезают)
    - Спуфинг: крупный ордер появляется и исчезает за <5 сек
    
    ИНСТИТУЦИОНАЛ детектируется когда:
    - Крупные лимитные ордера на одном-двух уровнях (>5 BTC на одном уровне)
    - Iceberg: много сделок одного размера (~0.1 BTC) на одном уровне, но ордер не уменьшается
    - Устойчивое накопление: дельта стабильно в одну сторону за 15+ минут
    
    РИТЕЙЛ детектируется когда:
    - Маленькие маркет-ордера (<0.01 BTC)
    - Кластеры сделок на круглых числах (93000, 94000, 95000)
    - Покупки после роста, продажи после падения (мomentum chasing)
    
    БОТ-АРБИТРАЖНИК:
    - Серия одинаковых сделок с интервалом <100мс
    - Объёмы ровные (0.100, 0.200)
    
    КИТ:
    - Разовая сделка > 5 BTC маркет-ордером
    - Резкий сдвиг цены > 0.1% за одну сделку
    
    СТОП-ХАНТ:
    - Резкий импульс к круглому числу на низком объёме
    - Сразу после — разворот
    - Серия ликвидаций на уровне
    """
```

---

## Фаза 3, Шаг 2: Интеграция рентгена

В `brain_loop` добавить в промпт Claude:
```
РЕНТГЕН УЧАСТНИКОВ:
- Маркетмейкер: {описание активности}
- Институционал: {описание}
- Ритейл: {описание}
- Доминирующий: {тип}
- Аномалии: {список}
- Стоп-хант: {да/нет, направление}
```

В WebSocket broadcast добавить:
```json
{
    "type": "xray_update",
    "data": {
        "participants": [...],
        "dominant_player": "institutional",
        "anomalies": ["Стена 340 BTC на 95,200 снимается"],
        "stop_hunt_detected": true,
        "stop_hunt_direction": "up"
    }
}
```

---

# ФАЗА 4: iPHONE APP (SwiftUI)

> **Результат фазы:** Работающее iOS приложение которое:
> 1. Подключается к серверу по WebSocket
> 2. Показывает Dashboard с ценой, T, S, фазой
> 3. Показывает мышление AI в реальном времени
> 4. Показывает карту гравитации (ликвидации)
> 5. Показывает рентген участников

---

## Фаза 4, Шаг 1: Xcode проект

**Задача:** Создать iOS проект.

```
ios/HEAN/
├── HEANApp.swift
├── Core/
│   ├── Network/
│   │   ├── WebSocketManager.swift
│   │   └── APIClient.swift
│   ├── Models/
│   │   ├── MarketState.swift
│   │   ├── BrainAnalysis.swift
│   │   ├── ThermodynamicState.swift
│   │   └── XRaySnapshot.swift
│   └── Theme/
│       ├── Colors.swift
│       └── Fonts.swift
├── Features/
│   ├── Dashboard/
│   │   ├── DashboardView.swift
│   │   ├── TemperatureGauge.swift
│   │   ├── EntropyGauge.swift
│   │   └── SignalCard.swift
│   ├── Brain/
│   │   ├── BrainView.swift
│   │   └── ThoughtBubble.swift
│   ├── GravityMap/
│   │   └── GravityMapView.swift
│   ├── Players/
│   │   └── PlayersView.swift
│   └── Settings/
│       └── SettingsView.swift
└── Resources/
    └── Assets.xcassets
```

**Настройки проекта:**
- iOS 17.0+
- Swift 6
- Orientation: Portrait only
- Цветовая схема: Тёмная тема (тёмно-синий фон #0A0E27)

---

## Фаза 4, Шаг 2: Модели данных (Swift)

**Все модели — Codable для десериализации JSON с сервера.**

```swift
// MarketState.swift
struct MarketUpdate: Codable {
    let price: Double
    let spread: Double
    let imbalance: Double
    let delta1m: Double
    let delta5m: Double
    let volume1m: Double
    let bigTradesCount: Int
    let bidTotal: Double
    let askTotal: Double
    let temperature: Double
    let entropy: Double
    let phase: String
    let phaseChangeProb: Double
}

// BrainAnalysis.swift  
struct BrainAnalysis: Codable {
    let situation: String
    let forces: [Force]
    let signal: TradingSignal
    let marketPhase: String
    let dangerLevel: Double
}

struct Force: Codable {
    let name: String
    let direction: String
    let strength: Double
    let description: String
}

struct TradingSignal: Codable {
    let direction: String  // LONG, SHORT, NO_TRADE
    let confidence: Double
    let entryPrice: Double?
    let stopLoss: Double?
    let takeProfit1: Double?
    let takeProfit2: Double?
    let riskReward: Double?
    let reasoning: String
}
```

---

## Фаза 4, Шаг 3: WebSocket Manager

```swift
// WebSocketManager.swift
@Observable
class WebSocketManager {
    var isConnected = false
    var latestMarket: MarketUpdate?
    var latestBrain: BrainAnalysis?
    var latestXray: XRaySnapshot?
    var brainHistory: [BrainAnalysis] = []
    
    private var webSocket: URLSessionWebSocketTask?
    private let serverURL: URL
    
    init(serverURL: String = "ws://YOUR_SERVER_IP:8000/ws") {
        self.serverURL = URL(string: serverURL)!
    }
    
    func connect() async {
        // 1. Создать URLSessionWebSocketTask
        // 2. Resume
        // 3. Начать слушать в цикле
        // 4. При получении сообщения — десериализовать
        // 5. По полю "type" распределить:
        //    "market_update" → latestMarket
        //    "brain_analysis" → latestBrain, добавить в brainHistory
        //    "xray_update" → latestXray
        // 6. При разрыве — переподключиться через 3 сек
    }
    
    func disconnect() {
        // Закрыть соединение
    }
}
```

---

## Фаза 4, Шаг 4: Dashboard View

**Описание экрана:**
- Верхняя часть: символ + цена (крупно) + индикатор подключения
- Середина: два круговых gauge — Температура и Энтропия
- Под ними: фаза рынка (текстом + цвет)
- Карточка сигнала: направление, вход, стоп, TP, R:R, уверенность
- Нижняя часть: P&L сегодня (когда будет торговля)
- Tab bar: Dashboard, Brain, Map, Players, Settings

**Цвета:**
- Фон: #0A0E27 (тёмно-синий)
- Accent: #00D4FF (cyan)
- Danger: #FF3B5C (красный)
- Success: #00FF88 (зелёный)
- Warm: #FF9500 (оранжевый)
- T gauge: градиент от синего (холодный) до красного (горячий)
- S gauge: градиент от зелёного (порядок) до фиолетового (хаос)

**Используй Swift Charts** для gauge если подходит, или Custom View с `Canvas` / `Path` для кругового gauge.

---

## Фаза 4, Шаг 5: Brain View

**Описание экрана:**
- Скроллящаяся лента "мыслей" AI — как чат-лог
- Каждая мысль = карточка с:
  - Время
  - Текст ситуации
  - Список сил (с цветовой индикацией UP/DOWN)
  - Сигнал (если есть)
  - Фаза и уровень опасности
- Новые мысли появляются сверху с анимацией
- Пульсирующий индикатор "AI думает..." между анализами

---

## Фаза 4, Шаг 6: Gravity Map View

**Описание экрана:**
- Вертикальная шкала цен
- Горизонтальные бары — объём ордеров на каждом уровне
- Зелёные бары (bid/покупки) слева
- Красные бары (ask/продажи) справа
- Текущая цена — горизонтальная линия по центру
- Уровни скопления ликвидаций — подсвечены жёлтым пульсирующим цветом
- Анимированная стрелка показывающая "притяжение" цены

---

## Фаза 4, Шаг 7: Players View

**Описание экрана:**
- 5 карточек (одна для каждого типа участника)
- Каждая карточка: иконка + название + активность (progress bar) + направление + описание
- Карточка доминирующего игрока — увеличена и подсвечена
- Секция "Аномалии" внизу — красные плашки с текстом

---

# ФАЗА 5: ПЕРВАЯ КРОВЬ (Paper Trading)

> **Результат фазы:** HEAN торгует на тестовые деньги через Bybit Testnet.

---

## Фаза 5, Шаг 1: Торговый движок

**Задача:** Создать `server/trading/` — модуль торговли.

```
server/trading/
├── __init__.py
├── engine.py         # Торговый движок (открыть/закрыть позицию)
├── risk.py           # Железные правила риска
├── position.py       # Управление позицией
└── models.py
```

**Модели:**
```python
class Position(BaseModel):
    id: str
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    size: float
    leverage: int
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None
    opened_at: datetime
    status: Literal["open", "closed", "stopped"]
    pnl: float = 0.0
    close_reason: str | None = None

class RiskConfig(BaseModel):
    max_daily_loss_pct: float = 3.0      # -3% и стоп на сутки
    max_weekly_loss_pct: float = 7.0     # -7% и стоп до ПН
    max_trade_loss_pct: float = 1.0      # -1% на сделку
    max_consecutive_losses: int = 3       # 3 стопа → пауза 2ч
    min_risk_reward: float = 2.0         # Минимум R:R
    max_leverage: int = 20
    max_open_positions: int = 1          # Одна позиция за раз
```

**Risk Manager:**
```python
class RiskManager:
    def __init__(self, config: RiskConfig, deposit: float):
        self.config = config
        self.deposit = deposit
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.consecutive_losses = 0
        self.paused_until: datetime | None = None
        
    def can_trade(self) -> tuple[bool, str]:
        """Проверить все железные правила. 
        Вернуть (можно_ли_торговать, причина_если_нет)"""
        
    def calculate_position_size(self, entry: float, stop: float, leverage: int) -> float:
        """Рассчитать размер позиции по формуле:
        risk_amount = deposit * max_trade_loss_pct / 100
        price_diff_pct = abs(entry - stop) / entry
        position_size = risk_amount / (price_diff_pct * entry)
        """
        
    def record_trade(self, pnl: float):
        """Записать результат сделки, обновить счётчики"""
```

**Trading Engine:**
```python
class TradingEngine:
    def __init__(self, exchange, risk_manager: RiskManager):
        self.exchange = exchange  # ccxt async
        self.risk = risk_manager
        self.current_position: Position | None = None
        
    async def execute_signal(self, signal: TradingSignal) -> Position | None:
        """
        1. Проверить risk.can_trade()
        2. Если NO_TRADE — пропустить
        3. Рассчитать размер позиции
        4. Проверить R:R >= min_risk_reward
        5. Открыть позицию через exchange
        6. Поставить стоп-лосс и тейк-профит
        7. Вернуть Position
        """
        
    async def check_position(self, current_price: float):
        """Проверить текущую позицию:
        - Стоп сработал?
        - TP1 достигнут? → закрыть 30%, передвинуть стоп на безубыток
        - TP2 достигнут? → закрыть ещё 30%
        - Трейл для оставшихся 40%
        """
```

## Фаза 5, Шаг 2: Подключение к Bybit Testnet

```python
# Bybit Testnet для paper trading
exchange = ccxt.bybit({
    'apiKey': settings.bybit_api_key,
    'secret': settings.bybit_api_secret,
    'options': {'defaultType': 'linear'},
    'sandbox': True  # ← ТЕСТНЕТ
})
```

**ВАЖНО:** Сначала ТОЛЬКО testnet. Реальные деньги — после минимум 2 недель прибыльного paper trading.

---

# ФАЗА 6: ДЕМОН (Темпоральный стек + Мета-игра)

> **Результат фазы:** AI видит 5 уровней времени и принимает решения на основе всей картины.

## Фаза 6, Шаг 1: Темпоральный стек

**Задача:** `server/demon/temporal.py`

```python
class TemporalLevel(BaseModel):
    name: str
    timeframe: str
    trend: Literal["bullish", "bearish", "neutral"]
    strength: float
    key_levels: list[float]
    description: str

class TemporalStack(BaseModel):
    macro: TemporalLevel    # дни-недели
    session: TemporalLevel  # часы
    tactical: TemporalLevel # минуты
    execution: TemporalLevel # секунды
    micro: TemporalLevel    # миллисекунды
    
    alignment_score: float  # насколько уровни согласованы (0-1)
    # 1.0 = все уровни в одном направлении = сильнейший сигнал
```

## Фаза 6, Шаг 2: Межрыночные корреляции

**Задача:** `server/demon/cross_market.py`

Подключить дополнительный WebSocket к Binance для ETH:
```python
# Отслеживать задержку между BTC и ETH
# BTC двигается → через N мс → ETH двигается
# Если знаем N, можем войти в ETH ДО движения

class CrossMarketAnalyzer:
    def __init__(self):
        self.btc_moves: deque = deque(maxlen=1000)
        self.eth_moves: deque = deque(maxlen=1000)
        self.lag_history: deque[float] = deque(maxlen=100)
        
    def record_move(self, symbol: str, price_change: float, timestamp: float):
        """Записать движение"""
        
    def estimate_lag(self) -> float | None:
        """Оценить текущую задержку BTC → ETH в миллисекундах"""
        
    def predict_eth_move(self, btc_move: float) -> tuple[float, float]:
        """Предсказать движение ETH на основе движения BTC.
        Вернуть (expected_move, confidence)"""
```

---

# КАК ИСПОЛЬЗОВАТЬ ЭТОТ ДОКУМЕНТ

## Порядок работы

```
1. Открой новый чат с Claude
2. Прикрепи этот файл
3. Скажи: "Прочитай документ. Напиши полный код для Фазы 1, Шаг 1."
4. Claude даст тебе файлы — скопируй их в папку проекта
5. Скажи: "Теперь Шаг 2" — и так далее
6. Когда Фаза 1 готова — запусти, проверь что работает
7. Вернись: "Фаза 2, Шаг 1"
8. И так до конца

ЕСЛИ ЧТО-ТО НЕ РАБОТАЕТ:
- Скопируй ошибку
- Вставь в чат
- Claude починит

ЕСЛИ ХОЧЕШЬ ИЗМЕНИТЬ ЧТО-ТО:
- Опиши что хочешь изменить
- Claude переделает
```

## Чеклист готовности каждой фазы

```
☐ ФАЗА 1: Глаза + Мозг
  ☐ Шаг 1: Структура проекта создана
  ☐ Шаг 2: config.py работает, .env читается
  ☐ Шаг 3: Модели данных созданы
  ☐ Шаг 4: Bybit WebSocket подключается, данные идут
  ☐ Шаг 5: DuckDB сохраняет данные
  ☐ Шаг 6: Claude анализирует рынок каждые 30 сек
  ☐ Шаг 7: WebSocket для iPhone работает
  ☐ Шаг 8: REST API отвечает
  ☐ Шаг 9: main.py запускается, всё вместе работает
  ☐ Шаг 10: Тесты пройдены

☐ ФАЗА 2: Физика
  ☐ Шаг 1: T, S, фазы вычисляются
  ☐ Шаг 2: Интегрировано в сервер и WebSocket

☐ ФАЗА 3: Рентген
  ☐ Шаг 1: Классификатор участников работает
  ☐ Шаг 2: Интегрировано в сервер

☐ ФАЗА 4: iPhone App
  ☐ Шаг 1: Xcode проект создан
  ☐ Шаг 2: Модели данных Swift
  ☐ Шаг 3: WebSocket Manager подключается
  ☐ Шаг 4: Dashboard показывает данные
  ☐ Шаг 5: Brain View работает
  ☐ Шаг 6: Gravity Map показывает ордербук
  ☐ Шаг 7: Players View показывает участников

☐ ФАЗА 5: Paper Trading
  ☐ Шаг 1: Торговый движок + Risk Manager
  ☐ Шаг 2: Bybit Testnet подключён
  ☐ 2 недели прибыльного paper trading ✓

☐ ФАЗА 6: Демон
  ☐ Шаг 1: Темпоральный стек
  ☐ Шаг 2: Межрыночные корреляции
```

---

*Документ: HEAN AI-Developer Blueprint v1.0 | Февраль 2026*
*Каждая фаза самодостаточна. Каждый шаг — отдельная задача для AI.*
