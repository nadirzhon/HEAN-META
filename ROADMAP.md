# HEAN SYMBIONT X - Roadmap дальнейших доработок

**Дата:** 2026-01-29
**Текущий статус:** ✅ Базовая структура готова (33% - Implementation complete)

---

## 📊 Текущее состояние

### ✅ Что сделано (Phase 1 - COMPLETE):
1. **Структура кода** - 35/35 файлов реализовано
2. **Синтаксис** - 0 ошибок, 8,494 строк кода
3. **Архитектура** - 8 компонентов полностью спроектированы
4. **Docker setup** - Dockerfile + requirements.txt готовы
5. **Тестовые скрипты** - simple_test.py, test_symbiont.py
6. **Опциональные импорты** - websockets, numpy с fallback
7. **Документация** - README, Implementation Complete, Testing Report

### ⏳ Что требует доработки:
- **Тестирование:** 0% (нет unit/integration тестов)
- **Интеграция:** 0% (нет подключения к реальному API)
- **Production-ready:** нет мониторинга, CI/CD, дашборда

---

## 🎯 Phase 2: Установка зависимостей и функциональное тестирование

**Приоритет:** 🔴 КРИТИЧНО
**Время:** 1-2 часа
**Статус:** ⏳ TODO

### Задачи:

#### 2.1. Установка зависимостей
```bash
# Создать virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установить зависимости
pip install -r requirements.txt
```

**Зависимости для установки:**
- `pydantic>=2.0.0` - валидация данных
- `websockets>=12.0` - WebSocket подключение к Bybit
- `numpy>=1.24.0` - математические вычисления
- `pandas>=2.0.0` - обработка временных рядов
- `pybit>=5.6.0` - Bybit API client
- `aiohttp>=3.9.0` - async HTTP requests

#### 2.2. Запуск test_symbiont.py
```bash
cd /path/to/HEAN
python test_symbiont.py
```

**Ожидаемый результат:**
- ✅ Все 14 тестов должны пройти
- ✅ Все компоненты должны импортироваться
- ✅ Базовая функциональность работает

**Если тесты падают:**
- Проверить версии Python (требуется 3.10+)
- Проверить установку всех зависимостей
- Исправить найденные баги

---

## 🧪 Phase 3: Unit Testing (создание тестов для каждого компонента)

**Приоритет:** 🔴 КРИТИЧНО
**Время:** 2-3 дня
**Статус:** ⏳ TODO

### 3.1. Создать структуру тестов

```bash
mkdir -p tests
mkdir -p tests/nervous_system
mkdir -p tests/regime_brain
mkdir -p tests/genome_lab
mkdir -p tests/adversarial_twin
mkdir -p tests/capital_allocator
mkdir -p tests/immune_system
mkdir -p tests/decision_ledger
mkdir -p tests/execution_kernel
```

### 3.2. Написать unit тесты для каждого компонента

#### A. Nervous System Tests (`tests/nervous_system/`)
**Файлы для создания:**
- `test_event_envelope.py` - тесты EventEnvelope
- `test_ws_connectors.py` - тесты WebSocket коннекторов
- `test_health_sensors.py` - тесты сенсоров здоровья

**Что тестировать:**
- ✅ Создание EventEnvelope с корректными данными
- ✅ Сериализация/десериализация событий
- ✅ Mock WebSocket подключение (без реального API)
- ✅ Health metrics calculation
- ✅ Обнаружение аномалий в health sensors

**Пример теста:**
```python
# tests/nervous_system/test_event_envelope.py
import pytest
from hean.symbiont_x.nervous_system import EventEnvelope, EventType

def test_event_envelope_creation():
    """Тест создания envelope"""
    envelope = EventEnvelope(
        event_type=EventType.MARKET_DATA,
        symbol="BTCUSDT",
        data={"price": 50000}
    )
    assert envelope.symbol == "BTCUSDT"
    assert envelope.data["price"] == 50000
    assert envelope.timestamp_ns > 0

def test_event_serialization():
    """Тест сериализации в JSON"""
    envelope = EventEnvelope(
        event_type=EventType.MARKET_DATA,
        symbol="BTCUSDT",
        data={"price": 50000}
    )
    json_data = envelope.to_dict()
    assert "timestamp_ns" in json_data
    assert json_data["symbol"] == "BTCUSDT"
```

#### B. Regime Brain Tests (`tests/regime_brain/`)
**Файлы для создания:**
- `test_features.py` - тесты feature extraction
- `test_classifier.py` - тесты классификатора режимов
- `test_regime_types.py` - тесты типов режимов

**Что тестировать:**
- ✅ Feature extraction из market data
- ✅ Классификация режима (TREND_UP, RANGE, etc.)
- ✅ Обновление состояния режима
- ✅ Историческое окно данных
- ✅ Edge cases (недостаточно данных, NaN values)

#### C. Genome Lab Tests (`tests/genome_lab/`)
**Файлы для создания:**
- `test_genome_types.py` - тесты структуры генома
- `test_mutation_engine.py` - тесты мутаций
- `test_crossover.py` - тесты скрещивания
- `test_evolution_engine.py` - тесты эволюции

**Что тестировать:**
- ✅ Создание random genome
- ✅ Валидация генов (bounds, constraints)
- ✅ Мутации (point, gaussian, swap, etc.)
- ✅ Crossover (single-point, two-point, uniform)
- ✅ Selection (tournament, roulette, rank-based)
- ✅ Эволюция поколений (fitness improvement)

**Пример теста:**
```python
# tests/genome_lab/test_mutation_engine.py
import pytest
from hean.symbiont_x.genome_lab import create_random_genome, MutationEngine

def test_point_mutation():
    """Тест точечной мутации"""
    genome = create_random_genome("TestStrategy")
    original_gene_value = genome.genes["entry_threshold"]

    mutation_engine = MutationEngine()
    mutated = mutation_engine.mutate(genome, mutation_rate=1.0)

    # После мутации хотя бы один ген должен измениться
    assert mutated.genes != genome.genes

def test_mutation_rate():
    """Тест зависимости от mutation_rate"""
    genome = create_random_genome("TestStrategy")
    mutation_engine = MutationEngine()

    # С mutation_rate=0 не должно быть изменений
    mutated = mutation_engine.mutate(genome, mutation_rate=0.0)
    assert mutated.genes == genome.genes

    # С mutation_rate=1.0 должны быть изменения
    mutated = mutation_engine.mutate(genome, mutation_rate=1.0)
    assert mutated.genes != genome.genes
```

#### D. Adversarial Twin Tests (`tests/adversarial_twin/`)
**Что тестировать:**
- ✅ Создание test worlds (Replay, Paper, MicroReal)
- ✅ Симуляция ордеров в paper world
- ✅ Стресс-тесты (flash crash, liquidity drain, etc.)
- ✅ Survival score calculation
- ✅ Backtesting на исторических данных

#### E. Capital Allocator Tests (`tests/capital_allocator/`)
**Что тестировать:**
- ✅ Создание Portfolio
- ✅ Добавление/удаление стратегий
- ✅ Расчёт Sharpe ratio, drawdown
- ✅ Darwinian allocation (survival-weighted)
- ✅ Rebalancing logic
- ✅ Capital constraints (min/max allocation)

#### F. Immune System Tests (`tests/immune_system/`)
**Что тестировать:**
- ✅ Constitution validation (check_trade_allowed)
- ✅ Reflex system (auto-stop на аномалии)
- ✅ Circuit breakers (halt trading при критических событиях)
- ✅ Risk limits (max position size, max leverage)

#### G. Decision Ledger Tests (`tests/decision_ledger/`)
**Что тестировать:**
- ✅ Запись решений в ledger
- ✅ Append-only свойство (невозможность изменить прошлое)
- ✅ Replay решений из ledger
- ✅ Анализ (success rate, win/loss ratio)
- ✅ Экспорт в JSON/CSV

#### H. Execution Kernel Tests (`tests/execution_kernel/`)
**Что тестировать:**
- ✅ Создание OrderRequest
- ✅ Валидация ордеров
- ✅ Mock execution (без реального API)
- ✅ Order lifecycle (pending → filled/cancelled)
- ✅ Error handling (insufficient funds, invalid symbol)

### 3.3. Настроить pytest

**Создать `pytest.ini`:**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

**Создать `conftest.py`:**
```python
# tests/conftest.py
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_market_data():
    """Fixture с примерными рыночными данными"""
    return {
        'symbol': 'BTCUSDT',
        'price': 50000.0,
        'volume': 1000.0,
        'timestamp': 1234567890
    }

@pytest.fixture
def sample_genome():
    """Fixture с примерным геномом"""
    from hean.symbiont_x.genome_lab import create_random_genome
    return create_random_genome("TestStrategy")
```

### 3.4. Запустить тесты

```bash
# Установить pytest
pip install pytest pytest-asyncio pytest-cov

# Запустить все тесты
pytest tests/

# Запустить с coverage
pytest --cov=src/hean/symbiont_x --cov-report=html tests/

# Запустить конкретный компонент
pytest tests/genome_lab/
```

**Целевой результат:**
- ✅ Coverage > 80% для всех компонентов
- ✅ Все тесты проходят
- ✅ 0 критических багов

---

## 🔌 Phase 4: Integration Testing (интеграция с Bybit API)

**Приоритет:** 🟡 ВЫСОКИЙ
**Время:** 3-5 дней
**Статус:** ⏳ TODO

### 4.1. Создать Bybit Testnet аккаунт

**Шаги:**
1. Зарегистрироваться на https://testnet.bybit.com
2. Создать API ключи (Read + Write permissions)
3. Записать ключи в `.env` файл

**Создать `.env`:**
```bash
# Bybit Testnet credentials
BYBIT_API_KEY=your_testnet_api_key_here
BYBIT_API_SECRET=your_testnet_api_secret_here
BYBIT_TESTNET=true

# Trading configuration
INITIAL_CAPITAL=10000
SYMBOLS=BTCUSDT,ETHUSDT
```

### 4.2. Реализовать реальное WebSocket подключение

**Доработать `ws_connectors.py`:**

```python
# src/hean/symbiont_x/nervous_system/ws_connectors.py

class BybitWSConnector:
    """Real Bybit WebSocket connector"""

    async def connect_real(self):
        """Подключение к реальному Bybit WebSocket"""
        url = "wss://stream-testnet.bybit.com/v5/public/linear"

        async with websockets.connect(url) as ws:
            # Subscribe to ticker
            subscribe_msg = {
                "op": "subscribe",
                "args": [f"tickers.{self.symbol}"]
            }
            await ws.send(json.dumps(subscribe_msg))

            # Receive messages
            async for message in ws:
                data = json.loads(message)
                await self._process_message(data)
```

### 4.3. Реализовать REST API для ордеров

**Создать `src/hean/symbiont_x/execution_kernel/bybit_client.py`:**

```python
from pybit.unified_trading import HTTP

class BybitRESTClient:
    """Bybit REST API client"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.client = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )

    def place_order(self, symbol: str, side: str, qty: float, order_type: str = "Market"):
        """Разместить ордер"""
        result = self.client.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType=order_type,
            qty=qty
        )
        return result

    def get_position(self, symbol: str):
        """Получить текущую позицию"""
        result = self.client.get_positions(
            category="linear",
            symbol=symbol
        )
        return result
```

### 4.4. Интеграционные тесты

**Создать `tests/integration/test_bybit_integration.py`:**

```python
import pytest
import asyncio
from hean.symbiont_x import HEANSymbiontX

@pytest.mark.asyncio
async def test_real_websocket_connection():
    """Тест реального подключения к Bybit WS"""
    config = {
        'symbols': ['BTCUSDT'],
        'testnet': True,
        # ... остальные параметры
    }

    symbiont = HEANSymbiontX(config)
    await symbiont.nervous_system.connect()

    # Подождать несколько секунд
    await asyncio.sleep(5)

    # Проверить, что получили market data
    assert symbiont.nervous_system.last_event is not None

@pytest.mark.asyncio
async def test_place_order_on_testnet():
    """Тест размещения ордера на testnet"""
    # ВНИМАНИЕ: тест будет размещать реальный ордер на testnet!
    config = {
        'symbols': ['BTCUSDT'],
        'testnet': True,
        'bybit_api_key': os.getenv('BYBIT_API_KEY'),
        'bybit_api_secret': os.getenv('BYBIT_API_SECRET')
    }

    symbiont = HEANSymbiontX(config)

    # Разместить минимальный тестовый ордер
    order = await symbiont.execution_kernel.execute_order(
        symbol='BTCUSDT',
        side='Buy',
        quantity=0.001  # Минимальный размер
    )

    assert order.order_id is not None
    assert order.status in ['Filled', 'PartiallyFilled']
```

**Запуск интеграционных тестов:**
```bash
# ВАЖНО: требуется .env с Testnet credentials
pytest tests/integration/ --testnet
```

### 4.5. Paper Trading тесты

**Создать режим paper trading:**
- Использовать реальный market data stream
- Симулировать исполнение ордеров локально
- Вести учёт виртуального баланса
- Логировать все сделки как "PAPER"

---

## 📈 Phase 5: Production-Ready Features

**Приоритет:** 🟢 СРЕДНИЙ
**Время:** 1-2 недели
**Статус:** ⏳ TODO

### 5.1. Добавить ML модели в Regime Brain

**Текущее состояние:**
- Regime Brain использует rule-based классификацию
- Нет машинного обучения

**Что добавить:**

#### A. Supervised Learning для классификации режимов

**Создать `src/hean/symbiont_x/regime_brain/ml_classifier.py`:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class MLRegimeClassifier:
    """ML-based режим классификатор"""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.trained = False

    def train(self, X_train, y_train):
        """Обучить модель на исторических данных"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.trained = True

    def predict(self, features: dict) -> MarketRegime:
        """Предсказать режим"""
        if not self.trained:
            raise RuntimeError("Model not trained")

        X = self._features_to_array(features)
        X_scaled = self.scaler.transform([X])
        prediction = self.model.predict(X_scaled)[0]
        return MarketRegime(prediction)

    def save(self, path: str):
        """Сохранить модель"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)

    def load(self, path: str):
        """Загрузить модель"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.trained = True
```

**Подготовка обучающих данных:**

```python
# scripts/prepare_training_data.py

import pandas as pd

def label_historical_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Разметить исторические данные вручную или через эвристики

    Input: OHLCV данные
    Output: OHLCV + regime_label
    """

    # Пример простой разметки
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()

    # Heuristic labeling
    def label_regime(row):
        if abs(row['returns']) > 0.02:  # High volatility
            return 'HIGH_VOL'
        elif row['returns'] > 0.005:  # Strong uptrend
            return 'TREND_UP'
        elif row['returns'] < -0.005:  # Strong downtrend
            return 'TREND_DOWN'
        else:
            return 'RANGE'

    df['regime'] = df.apply(label_regime, axis=1)
    return df

# Использование:
# 1. Скачать исторические данные с Bybit
# 2. Разметить режимы
# 3. Обучить ML модель
# 4. Сохранить модель
```

#### B. Online Learning (обновление модели в реальном времени)

**Использовать incremental learning:**
```python
from sklearn.linear_model import SGDClassifier

class OnlineRegimeClassifier:
    """Режим классификатор с online learning"""

    def __init__(self):
        self.model = SGDClassifier(loss='log_loss')
        self.buffer = []

    def partial_fit(self, X, y):
        """Обновить модель на новых данных"""
        self.model.partial_fit(X, y, classes=[0, 1, 2, 3, 4])

    def update_from_buffer(self):
        """Обновить модель из буфера"""
        if len(self.buffer) >= 100:
            X, y = zip(*self.buffer)
            self.partial_fit(X, y)
            self.buffer.clear()
```

### 5.2. Rust Execution Microkernel (для ultra-low latency)

**Зачем:**
- Python имеет GIL (Global Interpreter Lock)
- Для HFT нужна submillisecond latency
- Rust даёт zero-cost abstractions

**Создать Rust microservice:**

```bash
# Создать новый Rust проект
cargo new --lib execution_microkernel_rs
cd execution_microkernel_rs
```

**`src/lib.rs`:**
```rust
use pyo3::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

#[pyclass]
struct FastExecutor {
    orders: Vec<Order>,
}

#[pymethods]
impl FastExecutor {
    #[new]
    fn new() -> Self {
        FastExecutor {
            orders: Vec::new(),
        }
    }

    fn place_order(&mut self, symbol: String, side: String, qty: f64) -> PyResult<String> {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        // Ultra-fast order creation
        let order = Order {
            symbol,
            side,
            qty,
            timestamp_ns,
        };

        self.orders.push(order);
        Ok(format!("Order placed at {}", timestamp_ns))
    }
}

#[pymodule]
fn execution_microkernel_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FastExecutor>()?;
    Ok(())
}
```

**Интеграция с Python:**
```python
# src/hean/symbiont_x/execution_kernel/fast_executor.py

try:
    from execution_microkernel_rs import FastExecutor
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

class HybridExecutor:
    """Гибридный executor (Python + Rust)"""

    def __init__(self):
        if RUST_AVAILABLE:
            self.rust_executor = FastExecutor()
            print("✅ Using Rust microkernel for ultra-low latency")
        else:
            self.rust_executor = None
            print("⚠️  Rust microkernel not available, using Python")

    async def execute_order(self, order):
        if self.rust_executor:
            # Use Rust for critical path
            result = self.rust_executor.place_order(
                order.symbol,
                order.side,
                order.quantity
            )
            return result
        else:
            # Fallback to Python
            return await self._execute_order_python(order)
```

### 5.3. Web Dashboard UI

**Создать monitoring dashboard:**

```bash
mkdir -p dashboard
cd dashboard
npm init -y
npm install react react-dom next.js recharts
```

**Dashboard features:**
1. **Live Market Data** - real-time ticker, orderbook
2. **Regime Monitor** - текущий режим рынка + история
3. **Strategy Population** - список живых стратегий + survival scores
4. **Portfolio View** - allocation, PnL, drawdown
5. **Decision Ledger Viewer** - история решений с поиском
6. **System Health** - CPU, memory, latency, event rates
7. **Alerts** - circuit breakers, anomalies, critical events

**Технологии:**
- **Frontend:** React + Next.js + TailwindCSS
- **Charts:** Recharts или Plotly.js
- **WebSocket:** для real-time updates
- **Backend API:** FastAPI endpoint в SYMBIONT

**Создать `src/hean/symbiont_x/api/dashboard_api.py`:**
```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
async def get_status():
    """Получить статус системы"""
    return {
        "status": "running",
        "uptime": symbiont.get_uptime(),
        "total_trades": symbiont.get_total_trades()
    }

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для real-time updates"""
    await websocket.accept()

    while True:
        # Отправлять updates каждую секунду
        data = {
            "timestamp": time.time(),
            "regime": symbiont.regime_brain.current_regime,
            "portfolio_value": symbiont.capital_allocator.get_portfolio_value(),
            "active_strategies": len(symbiont.genome_lab.population)
        }
        await websocket.send_json(data)
        await asyncio.sleep(1)
```

### 5.4. Мониторинг и логирование

**A. Structured Logging с JSON**

```python
# src/hean/symbiont_x/utils/logging.py

import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON formatter для structured logging"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        return json.dumps(log_data)

# Setup logger
logger = logging.getLogger("hean.symbiont_x")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

**B. Prometheus Metrics**

```python
# pip install prometheus-client

from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
trades_total = Counter('symbiont_trades_total', 'Total number of trades')
trade_latency = Histogram('symbiont_trade_latency_seconds', 'Trade execution latency')
portfolio_value = Gauge('symbiont_portfolio_value', 'Current portfolio value')
active_strategies = Gauge('symbiont_active_strategies', 'Number of active strategies')

# Start metrics server
start_http_server(9090)

# В коде:
trades_total.inc()  # Increment trade counter
trade_latency.observe(0.123)  # Record latency
portfolio_value.set(15000.0)  # Set gauge value
```

**C. Grafana Dashboard**

```yaml
# docker-compose.yml для мониторинга

version: '3.8'

services:
  symbiont:
    build: .
    ports:
      - "9090:9090"  # Prometheus metrics
      - "8000:8000"  # Dashboard API
    environment:
      - BYBIT_API_KEY=${BYBIT_API_KEY}
      - BYBIT_API_SECRET=${BYBIT_API_SECRET}

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9091:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### 5.5. CI/CD Pipeline

**A. GitHub Actions**

**`.github/workflows/test.yml`:**
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ --cov=src/hean/symbiont_x --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

**B. Docker Build Pipeline**

**`.github/workflows/docker.yml`:**
```yaml
name: Docker Build

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Build Docker image
      run: docker build -t hean-symbiont-x:${{ github.sha }} .

    - name: Test Docker image
      run: |
        docker run hean-symbiont-x:${{ github.sha }} python simple_test.py

    - name: Push to registry
      if: startsWith(github.ref, 'refs/tags/')
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push hean-symbiont-x:${{ github.sha }}
```

### 5.6. Backtesting Engine

**Создать полноценный backtesting framework:**

**`src/hean/symbiont_x/backtesting/backtest_engine.py`:**

```python
import pandas as pd
from typing import List
from datetime import datetime

class BacktestEngine:
    """Движок для backtesting стратегий"""

    def __init__(self, historical_data: pd.DataFrame, initial_capital: float = 10000):
        self.data = historical_data
        self.initial_capital = initial_capital
        self.results = []

    def run_backtest(self, genome: StrategyGenome) -> BacktestResult:
        """Запустить backtest для одной стратегии"""

        capital = self.initial_capital
        position = 0
        trades = []

        for idx, row in self.data.iterrows():
            # Simulate strategy decision
            decision = self._evaluate_strategy(genome, row)

            if decision == "BUY" and position == 0:
                # Open long position
                position = capital / row['close']
                capital = 0
                trades.append({
                    'type': 'BUY',
                    'price': row['close'],
                    'timestamp': row['timestamp']
                })

            elif decision == "SELL" and position > 0:
                # Close position
                capital = position * row['close']
                position = 0
                trades.append({
                    'type': 'SELL',
                    'price': row['close'],
                    'timestamp': row['timestamp']
                })

        # Calculate metrics
        final_value = capital + (position * self.data.iloc[-1]['close'])
        return_pct = ((final_value - self.initial_capital) / self.initial_capital) * 100

        return BacktestResult(
            genome=genome,
            trades=trades,
            final_value=final_value,
            return_pct=return_pct,
            sharpe_ratio=self._calculate_sharpe(trades),
            max_drawdown=self._calculate_max_drawdown(trades)
        )

    def run_population_backtest(self, population: List[StrategyGenome]) -> List[BacktestResult]:
        """Backtest всей популяции"""
        return [self.run_backtest(genome) for genome in population]
```

**Использование:**
```python
# scripts/run_backtest.py

import pandas as pd
from hean.symbiont_x.backtesting import BacktestEngine
from hean.symbiont_x.genome_lab import create_random_genome

# Load historical data
df = pd.read_csv('historical_data/BTCUSDT_1h.csv')

# Create backtest engine
engine = BacktestEngine(df, initial_capital=10000)

# Create test population
population = [create_random_genome(f"Strategy_{i}") for i in range(100)]

# Run backtest
results = engine.run_population_backtest(population)

# Sort by Sharpe ratio
results.sort(key=lambda x: x.sharpe_ratio, reverse=True)

# Print top 10
print("Top 10 strategies:")
for i, result in enumerate(results[:10]):
    print(f"{i+1}. {result.genome.name}: Sharpe={result.sharpe_ratio:.2f}, Return={result.return_pct:.2f}%")
```

### 5.7. Дополнительные фичи

#### A. Multi-symbol Trading
- Одновременная торговля несколькими парами
- Корреляционный анализ
- Cross-symbol арбитраж

#### B. Advanced Risk Management
- Portfolio-level risk limits
- Correlation-based position sizing
- Dynamic leverage adjustment
- VaR (Value at Risk) monitoring

#### C. Market Making Strategies
- Bid-ask spread capture
- Liquidity provision
- Inventory management

#### D. News Sentiment Analysis
- Подключить news feed API
- NLP для sentiment analysis
- Event-driven trading triggers

#### E. Multi-exchange Support
- Поддержка Binance, OKX, etc.
- Cross-exchange arbitrage
- Unified API interface

---

## 📦 Phase 6: Deployment & Operations

**Приоритет:** 🟢 СРЕДНИЙ
**Время:** 1 неделя
**Статус:** ⏳ TODO

### 6.1. Production Deployment Checklist

**Infrastructure:**
- [ ] Cloud provider (AWS/GCP/Azure)
- [ ] Kubernetes cluster или EC2/VPS
- [ ] Load balancer для dashboard API
- [ ] PostgreSQL для ledger persistence
- [ ] Redis для caching
- [ ] S3/Cloud Storage для backups

**Security:**
- [ ] Encrypted API keys (AWS Secrets Manager / HashiCorp Vault)
- [ ] SSL/TLS certificates
- [ ] Firewall rules
- [ ] VPN для админ-доступа
- [ ] 2FA для критических операций

**Monitoring:**
- [ ] Prometheus + Grafana
- [ ] Error tracking (Sentry)
- [ ] Uptime monitoring (Pingdom/UptimeRobot)
- [ ] Log aggregation (ELK stack / CloudWatch)
- [ ] Alerting (PagerDuty / Slack notifications)

**Backups:**
- [ ] Daily ledger backups
- [ ] Genome population snapshots
- [ ] Configuration backups
- [ ] Disaster recovery plan

### 6.2. Kubernetes Deployment

**`k8s/deployment.yaml`:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hean-symbiont-x
spec:
  replicas: 1  # For stateful trading bot
  selector:
    matchLabels:
      app: symbiont
  template:
    metadata:
      labels:
        app: symbiont
    spec:
      containers:
      - name: symbiont
        image: hean-symbiont-x:latest
        env:
        - name: BYBIT_API_KEY
          valueFrom:
            secretKeyRef:
              name: bybit-credentials
              key: api-key
        - name: BYBIT_API_SECRET
          valueFrom:
            secretKeyRef:
              name: bybit-credentials
              key: api-secret
        ports:
        - containerPort: 9090  # Metrics
        - containerPort: 8000  # API
        volumeMounts:
        - name: data
          mountPath: /app/data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: symbiont-data
```

---

## 📅 Временная оценка (timeline)

| Phase | Описание | Время | Приоритет |
|-------|----------|-------|-----------|
| ✅ Phase 1 | Базовая структура | DONE | 🔴 |
| Phase 2 | Установка зависимостей + функциональное тестирование | 1-2 часа | 🔴 |
| Phase 3 | Unit Testing (80%+ coverage) | 2-3 дня | 🔴 |
| Phase 4 | Integration Testing (Bybit Testnet) | 3-5 дней | 🟡 |
| Phase 5a | ML models для Regime Brain | 3-5 дней | 🟢 |
| Phase 5b | Rust Execution Microkernel | 3-5 дней | 🟢 |
| Phase 5c | Web Dashboard | 5-7 дней | 🟢 |
| Phase 5d | Monitoring + Logging | 2-3 дня | 🟡 |
| Phase 5e | CI/CD Pipeline | 1-2 дня | 🟡 |
| Phase 5f | Backtesting Engine | 3-5 дней | 🟡 |
| Phase 6 | Production Deployment | 5-7 дней | 🟢 |

**Всего:** ~30-45 дней разработки

---

## 🎯 Минимальная критическая цепочка (MVP для production)

Если нужно быстро запустить в production, вот минимум:

### 🔴 КРИТИЧНО (нельзя запускать без этого):

1. **Phase 2** - Установка зависимостей ✅
2. **Phase 3** - Unit тесты (хотя бы 50% coverage) ✅
3. **Phase 4** - Integration с Bybit Testnet ✅
4. **Phase 4** - Paper trading минимум 1 неделя ✅
5. **Phase 5d** - Basic logging и мониторинг ✅

### 🟡 ВАЖНО (желательно иметь):

6. **Phase 5f** - Backtesting на 6+ месяцах данных
7. **Phase 5c** - Simple dashboard для мониторинга
8. **Phase 5d** - Alerts для критических событий

### 🟢 ОПЦИОНАЛЬНО (можно добавить позже):

9. **Phase 5a** - ML models
10. **Phase 5b** - Rust microkernel
11. **Phase 6** - Kubernetes deployment

---

## 📝 Следующий конкретный шаг (ACTION ITEMS)

### ⚡ ЧТО ДЕЛАТЬ ПРЯМО СЕЙЧАС:

1. **Установить зависимости:**
   ```bash
   cd /path/to/HEAN
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Запустить test_symbiont.py:**
   ```bash
   python test_symbiont.py
   ```

   **Ожидается:** все 14 тестов проходят ✅

3. **Зарегистрироваться на Bybit Testnet:**
   - Перейти на https://testnet.bybit.com
   - Создать аккаунт
   - Сгенерировать API ключи
   - Сохранить в `.env` файл

4. **Создать первый unit тест:**
   ```bash
   mkdir -p tests/genome_lab
   # Создать test_genome_types.py (пример выше)
   pytest tests/genome_lab/test_genome_types.py
   ```

5. **Запустить backtesting на исторических данных:**
   - Скачать historical data с Bybit (последние 6 месяцев)
   - Создать backtest_engine.py
   - Запустить backtest на 100 random стратегиях
   - Проверить, есть ли profitable стратегии

---

## 💰 Оценка затрат

### Development:
- **Разработчик (senior Python/Rust):** $80-150/час × 300 часов = $24,000 - $45,000

### Infrastructure (monthly):
- **VPS/Cloud (для production):** $50-200/месяц
- **Monitoring (Grafana Cloud, Sentry):** $50-100/месяц
- **Database (PostgreSQL managed):** $30-100/месяц
- **ИТОГО:** ~$150-400/месяц

### Trading Capital:
- **Minimum для тестирования:** $1,000 (Bybit Testnet - бесплатно)
- **Recommended для production:** $10,000 - $50,000

---

## ✅ Критерии готовности к production

Система готова к production, если:

- [ ] ✅ Все unit тесты проходят (coverage > 80%)
- [ ] ✅ Integration тесты с Bybit Testnet проходят
- [ ] ✅ Paper trading 1+ недель показывает положительный Sharpe ratio
- [ ] ✅ Backtesting на 6+ месяцах показывает профит
- [ ] ✅ Immune System корректно срабатывает на аномалии
- [ ] ✅ Circuit breakers останавливают торговлю при критических событиях
- [ ] ✅ Decision Ledger сохраняет все решения
- [ ] ✅ Monitoring и alerts настроены
- [ ] ✅ Нет memory leaks (stress test 24+ часов)
- [ ] ✅ Latency < 100ms для decision-making
- [ ] ✅ API credentials зашифрованы
- [ ] ✅ Backup strategy реализована

---

## 🚨 Важные предупреждения

### ⚠️ РИСКИ:

1. **Финансовые риски:**
   - Trading bot может потерять весь капитал
   - ОБЯЗАТЕЛЬНО начинать с Testnet
   - ОБЯЗАТЕЛЬНО использовать paper trading минимум 1 неделю
   - НИКОГДА не торговать деньгами, которые не можете потерять

2. **Технические риски:**
   - Bugs в коде могут привести к потере денег
   - Network issues могут пропустить critical events
   - API rate limits могут заблокировать торговлю

3. **Регуляторные риски:**
   - Проверить легальность algo trading в вашей юрисдикции
   - Налоги на crypto trading
   - KYC/AML compliance

### ✅ BEST PRACTICES:

1. **Начинать маленько:**
   - Testnet → Paper trading → Micro-real ($100) → Real

2. **Постепенное увеличение капитала:**
   - Не вкладывать весь капитал сразу
   - Увеличивать только после стабильной прибыльности

3. **Мониторинг 24/7:**
   - Настроить alerts для критических событий
   - Проверять систему минимум раз в день

4. **Kill switch:**
   - Иметь способ мгновенно остановить бота
   - Закрыть все позиции одной командой

---

## 📚 Дополнительные ресурсы

### Documentation:
- [ ] API reference для каждого компонента
- [ ] Architecture diagrams
- [ ] Deployment guide
- [ ] Troubleshooting guide

### Learning Resources:
- **Bybit API:** https://bybit-exchange.github.io/docs/v5/intro
- **Algorithmic Trading:** "Advances in Financial Machine Learning" by Marcos López de Prado
- **Genetic Algorithms:** "An Introduction to Genetic Algorithms" by Melanie Mitchell
- **Trading Systems:** "Building Winning Algorithmic Trading Systems" by Kevin Davey

---

## 🎉 Заключение

**Текущий статус:**
- ✅ Implementation: 100% (8,494 строк кода)
- ⏳ Testing: 0% (следующий шаг)
- ⏳ Integration: 0% (после тестирования)
- ⏳ Production: 0% (после интеграции)

**Следующий критический шаг:**
→ **Phase 2: Установить зависимости и запустить test_symbiont.py**

**Временная оценка до production:**
- **Минимальный MVP:** 2-3 недели (Phase 2-4 + basic monitoring)
- **Полноценная система:** 6-8 недель (все phases)

**Рекомендация:**
Начать с **критической цепочки** (Phase 2-4), протестировать на Testnet 1-2 недели, и только потом переходить к production с минимальным капиталом ($100-1000).

---

*Документ создан: 2026-01-29*
*Версия: 1.0*
*Статус: 📋 ROADMAP READY*
