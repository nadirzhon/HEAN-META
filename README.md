# HEAN-Bybit

Production-grade, modular, event-driven crypto trading research system focused on Bybit.

**⚠️ WARNING: This is a research system. Paper trading is the default. Live trading is locked behind `LIVE_CONFIRM=YES` environment variable. Use at your own risk.**

## Features

- **Event-driven architecture** with async-first design
- **Paper trading by default** with realistic simulation (fees, slippage)
- **Multiple strategies**: Funding harvester, Basis arbitrage, Impulse engine
- **Agent generation**: LLM-powered automatic trading agent generation
- **Auto-improvement system**: Self-learning catalyst that optimizes strategies and capital allocation
- **Risk management**: Drawdown limits, position sizing, killswitch
- **Portfolio accounting**: Equity tracking, PnL, profit reinvestment
- **Backtesting**: Synthetic regime generation and metrics
- **Strict typing**: Full type hints, mypy-ready
- **Production-ready**: Structured logging, health checks, observability

## Quickstart

### Installation

```bash
# Install dependencies
make install

# Or manually:
pip install -e ".[dev]"
```

### Configuration

Edit `.env` (template included in repo) and adjust settings.

### Development Mode (Recommended)

Start the full development environment (API + Frontend + Monitoring):

```bash
make dev
```

This starts:
- **API**: http://localhost:8000 (FastAPI backend)
- **Command Center**: http://localhost:3000 (Trading Command Center UI)
- **Prometheus**: http://localhost:9091 (Metrics)
- **Grafana**: http://localhost:3001 (Dashboards, admin/admin)

### Running in Paper Mode (CLI)

```bash
# Run the system (paper trading by default)
make run

# Or directly:
python -m hean.main run
```

The system will:
- Start with a simulated price feed for BTCUSDT and ETHUSDT
- Run strategies in paper mode
- Print status every 10 seconds (equity, daily PnL, drawdown)
- Simulate fills, fees, and slippage

### Order decision telemetry (debugging zero trades)

- WS topic `order_decisions` + Redis key `hean:state:state:order_decisions` carry CREATE/SKIP/REJECT with `reason_code`.
- Fire a smoke signal: `POST /orders/test` (paper only) to verify orders/positions populate and metrics update.
- Inspect DecisionMemory blocks: `GET /risk/decision-memory/blocks` (empty = not blocking).
- Logs include `[ORDER_DECISION] …` for quick grep when diagnosing why a signal was skipped/rejected.

### Trading Command Center

Access the full-featured Trading Command Center at http://localhost:3000 (after running `make dev`).

**Features**:
- **Dashboard**: Real-time metrics, event feed, health panel
- **Trading**: Positions and orders management with actions
- **Strategies**: View, enable/disable, and configure strategies
- **Analytics**: Performance summary, blocked signals analysis, job queue
- **Risk**: Risk status, limits, gate inspector
- **Logs**: Real-time log stream with filtering and search
- **Settings**: System configuration (secrets masked), live trading checklist

**Real-time Features**:
- WebSocket topics for live updates
- Auto-updating metrics and status (polling)
- Command palette (Ctrl+K) for quick actions
- Theme toggle (light/dark)

**Security**:
- All live trading actions require double confirmation
- Secrets are masked in UI
- Request IDs for traceability

See [docs/UI.md](docs/UI.md) for detailed documentation.

### API Usage

The FastAPI backend provides REST endpoints for all operations:

```bash
# Health check
curl http://localhost:8000/health

# Start engine
curl -X POST http://localhost:8000/engine/start \
  -H "Content-Type: application/json" \
  -d '{"confirm_phrase": null}'

# Get positions
curl http://localhost:8000/positions

# Get orders
curl http://localhost:8000/orders
```

See [docs/API.md](docs/API.md) for complete API documentation.

### New Endpoints (AFO-Director)

- `GET /trading/why`: Comprehensive diagnostics explaining why trading may have stopped. Returns engine state, killswitch, last activity timestamps, top reason codes, profit capture state, execution quality, and multi-symbol status.
- `GET /system/changelog/today`: Get today's improvements/changelog from git log or changelog_today.json. Returns `available: false` if git/changelog not available (no fiction).

### Smoke Test

Run the smoke test to verify all features:

```bash
./scripts/smoke_test.sh
```

The smoke test checks:
- REST endpoints (`/telemetry/ping`, `/telemetry/summary`, `/trading/why`, `/portfolio/summary`)
- WebSocket connection and subscription
- Engine control (pause/resume)
- Multi-symbol support

**Important**: Only rebuild Docker after smoke test PASSES. If smoke fails, fix issues first, then rebuild.

### Running with Docker Compose

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for detailed Docker instructions.

### Auto-Deploy to Mac via Self-Hosted Runner

Automatically deploy the project to your Mac on every push to `main` using GitHub Actions and a self-hosted runner.

#### How It Works

1. **Self-Hosted Runner** runs on your Mac as a background service
2. **GitHub Actions workflow** triggers on push to `main`
3. **Deployment** executes: `docker compose up -d --build --remove-orphans`
4. **Verification** checks health endpoints and shows logs

#### One-Time Setup

Install and configure the GitHub self-hosted runner on your Mac:

```bash
# 1. Get installation commands from GitHub
# Go to: Settings → Actions → Runners → New self-hosted runner

# 2. Download and extract runner (use commands from GitHub)
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-osx-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-osx-x64-2.311.0.tar.gz
tar xzf ./actions-runner-osx-x64-2.311.0.tar.gz

# 3. Configure with the CRITICAL LABEL "nadir-mac"
./config.sh --url https://github.com/nadirzhon/HEAN-META --token YOUR_TOKEN --labels "nadir-mac"

# 4. Install as a service (auto-start on boot)
./svc.sh install
./svc.sh start

# 5. Verify it's running
./svc.sh status
```

**Important:** The workflow targets the `nadir-mac` label specifically. Make sure to include this label during configuration.

See **[scripts/runner_setup_macos.md](scripts/runner_setup_macos.md)** for detailed step-by-step instructions.

#### Testing the Auto-Deploy

Make a test commit to trigger deployment:

```bash
git checkout main
git commit --allow-empty -m "Test auto-deploy"
git push origin main
```

Then:
1. Go to **Actions** tab in GitHub
2. Watch the "Auto-Deploy to Mac (Self-Hosted Runner)" workflow run
3. Check your Mac: `docker ps` should show updated containers
4. Access the app: http://localhost:8000 (API), http://localhost:3000 (UI)

#### Manual Deployment (Local Testing)

Test the deployment locally without GitHub Actions:

```bash
./scripts/deploy_local.sh           # Normal deploy
./scripts/deploy_local.sh --rebuild # Force rebuild (no cache)
```

This script mimics the GitHub Actions workflow for debugging.

#### Debugging Auto-Deploy

**If workflow doesn't trigger:**
- Check runner status in GitHub: Settings → Actions → Runners
- Ensure runner is online (green dot)
- Check runner logs: `~/actions-runner/_diag/Runner_*.log`

**If deployment fails:**
- View GitHub Actions run logs in the Actions tab
- Check Docker is running: `docker ps`
- Check port conflicts: `lsof -i :8000`, `lsof -i :3000`, `lsof -i :6379`
- Run locally for debugging: `./scripts/deploy_local.sh`

**Common issues:**
- **Runner offline:** Restart with `cd ~/actions-runner && ./svc.sh restart`
- **Docker not found:** Ensure Docker Desktop is running
- **Port conflicts:** Stop conflicting services or change ports in `docker-compose.yml`
- **Permission errors:** Ensure your user has Docker access

#### Security Features

- ✅ Only deploys on push to `main` (never on `pull_request` from forks)
- ✅ Uses concurrency control to prevent overlapping deploys
- ✅ Targets specific runner label (`nadir-mac`) for isolation
- ✅ 30-minute timeout prevents stuck deployments
- ✅ Health checks verify successful deployment

#### Workflow Features

The auto-deploy workflow includes:
- System diagnostics (Docker version, disk usage)
- Container health checks (API and UI endpoints)
- Deployment logs (last 200 lines)
- Automatic cleanup of old Docker images
- Deployment summary with commit info

#### Quick Checklist

After setup, verify everything works:

- [ ] Runner installed and running: `~/actions-runner/svc.sh status`
- [ ] Runner shows "Idle" in GitHub Settings → Actions → Runners
- [ ] Test commit triggers workflow: `git commit --allow-empty -m "test" && git push`
- [ ] Workflow completes successfully (green checkmark in Actions tab)
- [ ] Containers running on Mac: `docker ps | grep hean`
- [ ] API accessible: `curl http://localhost:8000/health`
- [ ] UI accessible: Open http://localhost:3000 in browser

**Status:** ✅ PASS (all checks green) or ❌ FAIL (check logs and troubleshooting guide)

---

### Backtesting

```bash
# Run backtest for 30 days
python -m hean.main backtest --days 30

# Custom days
python -m hean.main backtest --days 7
```

### Agent Generation

Generate trading agents using LLM prompts:

```bash
# Install LLM dependencies (optional)
pip install -e ".[llm]"

# Set API key
export OPENAI_API_KEY="your-key"  # or ANTHROPIC_API_KEY

# Generate initial agent
python generate_agent.py initial -o my_agent.py

# Generate multiple agents
python generate_agent.py initial --count 10 -o generated_agents/

# Evolve agent based on best/worst performers
python generate_agent.py evolution \
  --best-agents "Agent1: PF=2.5" \
  --worst-agents "Agent2: PF=0.8" \
  --market-conditions "High volatility" \
  --performance-metrics "Avg PF: 1.5" \
  -o evolved_agent.py
```

See [AGENT_GENERATION.md](AGENT_GENERATION.md) for full documentation.

### Auto-Improvement System

The system includes an autonomous improvement catalyst that:
- Monitors performance every 30 minutes
- Identifies problems and generates improvements via LLM
- Optimizes strategy parameters automatically
- Redistributes capital between strategies intelligently
- Generates daily performance reports

The catalyst starts automatically when running the system. See [AUTO_IMPROVEMENT_SYSTEM.md](AUTO_IMPROVEMENT_SYSTEM.md) for details.

### Process Factory (Experimental)

The Process Factory is an extension layer that provides process-based capital allocation and automation beyond traditional trading strategies. It enables:

- **Process-based workflows**: Define processes (not just strategies) that can include trading, earning, campaigns, data collection, and more
- **Capital routing**: Automatic allocation across reserve/active/experimental buckets
- **Process lifecycle management**: Automatic kill/keep/scale decisions based on measured performance
- **Leverage-of-Process Engine**: Self-amplifying loops (automation leverage, data leverage, access leverage)
- **Environment scanning**: Read-only scanning of Bybit environment for opportunities
- **AI-powered process generation**: Generate new process definitions using LLMs (OpenAI/Anthropic)

**Status**: Experimental, disabled by default. Enable with `process_factory_enabled=true` in config.

#### Quick Start

```bash
# Enable Process Factory in config
export PROCESS_FACTORY_ENABLED=true

# Scan environment
python -m hean.main process scan

# Plan daily capital allocation
python -m hean.main process plan --capital 1000

# Run a process (sandbox mode)
python -m hean.main process run --process-id p1_capital_parking --mode sandbox

# Generate daily report
python -m hean.main process report

# Evaluate portfolio (replay last 30 days)
python -m hean.main process evaluate --days 30
```

#### Production Checklist

Before enabling Process Factory in production:

1. **Enable Safely**:
   ```bash
   # Start with Process Factory disabled (default)
   # Only enable after thorough testing
   export PROCESS_FACTORY_ENABLED=true
   ```

2. **Run Sandbox First**:
   ```bash
   # Always test in sandbox mode first
   python -m hean.main process run --process-id <process_id> --mode sandbox
   
   # Verify results before enabling live mode
   python -m hean.main process report
   ```

3. **Interpret Reports**:
   - **Top Contributors (Net)**: Processes with highest net contribution (after fees/costs)
   - **Profit Illusion List**: Processes that appear profitable (gross) but lose money (net)
   - **Portfolio Health Score**: Stability, concentration risk, churn rate
   - **Kill/Scale Suggestions**: Recommendations from evaluation

4. **Add New Processes Correctly**:
   - Use `process scan` to get environment snapshot
   - Use OpenAI factory to generate process (with quality scoring)
   - Test in sandbox mode first
   - Monitor for profit illusion (gross positive, net negative)
   - Use `process evaluate` to get recommendations

5. **Safety Settings**:
   - `process_factory_allow_actions=false` (default): Bybit actions disabled
   - `dry_run=true` (default): Dry run mode enabled

#### Troubleshooting: If You See No Orders

If Process Factory is enabled but you're not seeing any orders, follow these steps:

1. **Check Configuration Flags**:
   ```bash
   # Verify flags are printed at startup
   python -m hean.main run
   # Look for:
   # - PROCESS_FACTORY_ENABLED: true
   # - PROCESS_FACTORY_ALLOW_ACTIONS: true
   # - DRY_RUN: false
   ```

2. **Run Execution Smoke Test**:
   ```bash
   # Enable required flags first
   export PROCESS_FACTORY_ENABLED=true
   export PROCESS_FACTORY_ALLOW_ACTIONS=true
   export DRY_RUN=false
   
   # Run smoke test
   python -m hean.main process exec-smoke-test
   ```

3. **Interpret Common Failures**:
   - **"not_enabled" error**: Set `PROCESS_FACTORY_ALLOW_ACTIONS=true` and `DRY_RUN=false`
   - **"min_notional" error**: Increase `EXECUTION_SMOKE_TEST_NOTIONAL_USD` (default 5, minimum usually 5-10 USD)
   - **"validation_error"**: Check that all required flags are set
   - **Network/API errors**: Check `BYBIT_API_KEY` and `BYBIT_API_SECRET` are set correctly

4. **Check "Why Not Trading" Diagnostics**:
   When running `python -m hean.main run`, look for log messages like:
   ```
   ⚠ Trade blocked: BTCUSDT (strategy_id)
     Reasons: dry_run, process_factory_allow_actions_false
     Suggested fixes:
       - Set DRY_RUN=false to allow real orders
       - Set PROCESS_FACTORY_ALLOW_ACTIONS=true
   ```

5. **Minimum Order Sizing**:
   - Bybit typically requires minimum 5 USD notional value
   - For BTCUSDT at $50,000, minimum quantity is ~0.0001 BTC
   - Ensure your capital allocation meets minimum requirements

6. **Enable Process Factory Actions** (if using Process Factory execution):
   ```bash
   export PROCESS_FACTORY_ALLOW_ACTIONS=true
   export DRY_RUN=false  # Required for real orders
   ```
   
   **Note**: `DRY_RUN=false` requires explicit configuration - it defaults to `true` for safety.
   - Only enable `process_factory_allow_actions=true` if you have existing execution interfaces
   - Never enable UI automation or credential handling

6. **Monitoring**:
   - Check `process report` daily
   - Review profit illusion list weekly
   - Use `process evaluate` monthly for portfolio health

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for detailed implementation notes.

#### Built-in Processes

The system includes 6 starter processes:
- **P1: Capital Parking** - Earn-like placeholder for capital allocation
- **P2: Funding Monitor** - Monitor funding rates and suggest allocations (read-only)
- **P3: Fee/Slippage Monitor** - Track execution quality metrics
- **P4: Campaign Checklist** - Generate checklists for campaign/airdrop participation
- **P5: Execution Optimizer** - Analyze and suggest execution improvements
- **P6: Opportunity Scanner** - Scan for new opportunities across trading, earn, campaigns

#### Extending with New Processes

Create a new process file in `src/hean/process_factory/processes/`:

```python
# my_process.py
from hean.process_factory.schemas import ProcessDefinition, ProcessType, ...

def get_process_definition() -> ProcessDefinition:
    return ProcessDefinition(
        id="my_process",
        name="My Process",
        type=ProcessType.DATA,
        description="...",
        actions=[...],
        ...
    )
```

Then register it in the registry. Processes are automatically discovered and validated.

#### Key Concepts

- **Process vs Strategy**: Processes are broader workflows that can include trading, manual tasks, data collection, and more
- **Measurement**: Every process must be measurable (capital_delta, time_hours, ROI, etc.)
- **Safety**: Processes include safety policies (max capital, risk factors, manual approval requirements)
- **Sandboxing**: All processes can run in sandbox mode for testing
- **Human Tasks**: Processes can include HUMAN_TASK steps for things requiring manual intervention

See the Process Factory source code in `src/hean/process_factory/` for more details.

### Testing

```bash
# Run all tests
make test

# Or directly:
pytest

# With coverage
pytest --cov=src/hean --cov-report=html
```

### Linting

```bash
# Run linters
make lint

# Or directly:
ruff check src/
mypy src/
```

## Safety Notes

1. **Paper trading is the default** - No real orders are placed unless explicitly enabled
2. **Live trading requires `LIVE_CONFIRM=YES`** - Even then, order placement is disabled by default
3. **Killswitch protection** - System automatically stops trading on:
   - Max daily drawdown exceeded
   - Repeated errors
   - Extreme volatility/spread
4. **Risk limits** - Configurable per-trade risk, position limits, leverage caps

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Web UI)                        │
│  Dashboard | Strategies | Risk | Orders | Positions | Logs      │
└──────────────────────────────┬──────────────────────────────────┘
                                │ HTTP/REST
┌──────────────────────────────▼──────────────────────────────────┐
│                      FastAPI Backend (API)                        │
│  /health | /engine/* | /positions | /orders | /metrics           │
└──────────────────────────────┬──────────────────────────────────┘
                                │
┌──────────────────────────────▼──────────────────────────────────┐
│                    Engine Facade (Orchestration)                 │
│  Unified interface for TradingSystem + ProcessFactory            │
└──────────────────────────────┬──────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                               │
┌───────▼────────┐                            ┌─────────▼────────┐
│ TradingSystem  │                            │ ProcessFactory   │
│ (Main Engine)  │                            │ (Extension)      │
└───────┬────────┘                            └─────────┬───────┘
        │                                               │
        └───────────────────────┬───────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     Event Bus          │
                    │  (Async Event System)  │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                         │
┌───────▼────────┐    ┌─────────▼────────┐    ┌─────────▼────────┐
│  Strategies    │    │   Risk Mgmt      │    │   Execution      │
│  - Funding     │    │   - Limits        │    │   - Router       │
│  - Basis       │    │   - Killswitch   │    │   - Paper/Live   │
│  - Impulse     │    │   - PositionSize │    │   - Reconcile    │
└───────┬────────┘    └─────────┬────────┘    └─────────┬───────┘
        │                       │                         │
        └───────────────────────┼─────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Portfolio           │
                    │   - Accounting        │
                    │   - Allocation        │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Exchange (Bybit)    │
                    │   - HTTP API          │
                    │   - WebSocket         │
                    └───────────────────────┘
```

All components communicate via an async event bus. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Project Structure

```
src/hean/
├── main.py              # CLI entrypoint
├── config.py            # Configuration (Pydantic v2)
├── logging.py           # Structured logging
├── core/                # Event bus, clock, types
├── exchange/            # Exchange client protocol + Bybit implementation
├── execution/           # Order routing, paper broker
├── risk/                # Risk limits, position sizing, killswitch
├── portfolio/           # Accounting, allocation, rebalancing
├── strategies/          # Trading strategies
├── agent_generation/    # LLM-powered agent generation
├── backtest/            # Backtesting engine
└── observability/       # Metrics, health checks
```

## Environment Variables

See `.env.example` for all available options. Key variables:

- `LIVE_CONFIRM`: Must be `YES` to enable live trading (default: not set)
- `INITIAL_CAPITAL`: Starting capital in USDT (default: 10000)
- `MAX_DAILY_DRAWDOWN_PCT`: Maximum daily drawdown percentage (default: 5.0)
- `MAX_TRADE_RISK_PCT`: Maximum risk per trade (default: 1.0)

### AFO-Director Features

**Profit Capture** (disabled by default):
- `PROFIT_CAPTURE_ENABLED=false`: Enable profit capture (default: false)
- `PROFIT_CAPTURE_TARGET_PCT=20`: Target percentage growth to trigger (default: 20%)
- `PROFIT_CAPTURE_TRAIL_PCT=10`: Trail percentage drawdown to trigger (default: 10%)
- `PROFIT_CAPTURE_MODE=full|partial`: Capture mode - full closes all, partial reduces exposure (default: full)
- `PROFIT_CAPTURE_AFTER_ACTION=pause|continue`: Action after capture - pause stops trading, continue with reduced risk (default: pause)
- `PROFIT_CAPTURE_CONTINUE_RISK_MULT=0.25`: Risk multiplier when continuing after capture (default: 0.25 = 25%)

**Multi-Symbol Support**:
- `MULTI_SYMBOL_ENABLED=false`: Enable multi-symbol scanning (default: false)
- `SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,TONUSDT"`: List of symbols to scan (default: 10 symbols)

**WebSocket Topics**:
- `order_decisions`: Real-time ORDER_DECISION events with reason_codes, gating_flags, market_regime, advisory
- `ai_catalyst`: AI Catalyst events (AGENT_STATUS, AGENT_STEP) - see `/system/changelog/today` for today's improvements

## Docker

```bash
# Build and run
docker-compose up

# Or build manually
docker build -t hean .
docker run --env-file .env hean
```

## Development

```bash
# Install in development mode
make install

# Run tests
make test

# Lint and type check
make lint

# Run the system (CLI mode)
make run

# Start development environment (API + Frontend + Monitoring)
make dev
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System architecture and data flows
- [API Documentation](docs/API.md) - Complete API reference with examples
- [Assumptions](docs/ASSUMPTIONS.md) - Design decisions and assumptions

## Bybit Integration ✅

**Полная интеграция с Bybit завершена!**

### Что работает:
- ✅ HTTP API клиент (размещение ордеров, получение данных)
- ✅ Public WebSocket (реальные тики, order book)
- ✅ Private WebSocket (обновления ордеров, позиций, executions)
- ✅ Real-time price feed с Bybit
- ✅ Автоматическое исполнение ордеров
- ✅ Поддержка testnet и mainnet
- ✅ Автоматическое переподключение

### Быстрый старт:

1. Получите API ключи на https://testnet.bybit.com/ (для теста)
2. Добавьте в `.env`:
   ```bash
   BYBIT_API_KEY=your-key
   BYBIT_API_SECRET=your-secret
   BYBIT_TESTNET=true
   LIVE_CONFIRM=YES
   TRADING_MODE=live
   ```
3. Протестируйте: `python test_bybit_connection.py`
4. Запустите: `python -m hean.main run`

См. [BYBIT_INTEGRATION_COMPLETE.md](BYBIT_INTEGRATION_COMPLETE.md) и [BYBIT_SETUP_GUIDE.md](BYBIT_SETUP_GUIDE.md) для деталей.

## Acknowledgments

This project was developed with the assistance of **Claude AI** (Anthropic). Claude helped with architecture design, code implementation, testing, and documentation throughout the development process.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for more details.

## License

MIT
