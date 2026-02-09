"""Configuration management using Pydantic v2."""

import json
import os
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_list_env(value: Any) -> Any:
    """Parse list values from env (JSON array, comma-separated, or single item)."""
    if value is None:
        return value
    if isinstance(value, str):
        if value.strip() == "":
            return value
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items if items else value
    return value


def _find_env_file() -> str:
    """Find .env file: check project root first, then CWD."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        return env_path
    return ".env"


class HEANSettings(BaseSettings):
    """Main configuration for HEAN system."""

    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        enable_decoding=False,
        env_ignore_empty=True,
    )

    # Absolute+ / Meta-Learning controls (disabled by default for safety in containerized runs)
    absolute_plus_enabled: bool = Field(
        default=False,
        description="Enable Absolute+ meta-learning / patching systems",
    )
    meta_learning_auto_patch: bool = Field(
        default=False,
        description="Allow meta-learning engine to patch source code (requires writable volume)",
    )
    meta_learning_rate: int = Field(
        default=0,
        description="Simulation rate hint for meta-learning engine",
    )
    meta_learning_max_workers: int = Field(
        default=0,
        description="Max concurrent simulations for meta-learning",
    )

    # Environment
    environment: Literal["development", "production"] = Field(
        default="development",
        description="Application environment (development or production)"
    )

    # Trading Mode (BYBIT TESTNET ONLY - NO PAPER TRADING)
    live_confirm: str = Field(default="YES", description="Must be 'YES' to enable live trading")
    trading_mode: Literal["live"] = Field(default="live", description="Trading mode - ALWAYS LIVE (testnet)")

    # Capital Management
    initial_capital: float = Field(default=300.0, gt=0, description="Initial capital in USDT")
    reinvest_rate: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Profit reinvestment rate (legacy, use smart_reinvest_base_rate)",
    )
    cash_reserve_rate: float = Field(
        default=0.2, ge=0, le=1, description="Cash reserve rate (not allocated)"
    )
    capital_allocation_mode: str = Field(
        default="adaptive",
        description="Capital allocation mode: 'equal' (equal split), 'adaptive' (performance-based)",
    )
    force_equal_allocation: bool = Field(
        default=False,
        description="Force equal capital allocation regardless of performance (1/N for N strategies)",
    )

    # Profit Targets
    profit_target_daily_usd: float = Field(
        default=100.0, gt=0, description="Daily profit target in USD (minimum $100/day)"
    )

    # Smart Reinvestment
    smart_reinvest_base_rate: float = Field(
        default=0.85,
        ge=0,
        le=1,
        description="Base reinvestment rate for smart reinvestor (default 85%)",
    )
    smart_reinvest_min_reserve_pct: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="Minimum reserve percentage to always maintain (default 10%)",
    )
    smart_reinvest_drawdown_threshold: float = Field(
        default=15.0, gt=0, description="Drawdown threshold for reducing reinvestment (default 15%)"
    )

    # Risk Management
    max_daily_drawdown_pct: float = Field(
        default=15.0,
        gt=0,
        le=100,
        description="Maximum daily drawdown percentage (Testnet: 15% allows normal trading activity)",
    )
    max_trade_risk_pct: float = Field(
        default=1.0,
        gt=0,
        le=100,
        description="Maximum risk per trade percentage (HEAN v2 Iron Rule: -1% per trade)",
    )
    max_open_positions: int = Field(
        default=10, gt=0, description="Maximum number of open positions (anti-runaway guard)"
    )
    max_open_orders: int = Field(
        default=20,
        gt=0,
        description="Maximum number of open orders (prevents runaway order creation)",
    )
    max_hold_seconds: int = Field(
        default=900,
        gt=0,
        description="Maximum time to hold a position before force-closing (anti-stuck TTL, default 15m)",
    )
    position_monitor_check_interval: int = Field(
        default=30,
        gt=0,
        description="How often to check for stale positions (seconds, default 30s)",
    )
    position_monitor_enabled: bool = Field(
        default=True,
        description="Enable automatic force-closing of stale positions",
    )

    # Liquidity Sweep Detector Strategy
    liquidity_sweep_enabled: bool = Field(
        default=True,
        description="Enable Liquidity Sweep Detector strategy for catching institutional sweeps",
    )
    liquidity_sweep_threshold_pct: float = Field(
        default=0.003,
        description="Sweep threshold as decimal (0.003 = 0.3%)",
    )
    liquidity_sweep_cooldown_minutes: int = Field(
        default=15,
        ge=5,
        description="Cooldown between trades per symbol (minutes)",
    )

    max_concurrent_risk_pct: float = Field(
        default=20.0,
        gt=0,
        le=100,
        description="Maximum concurrent risk percentage across all positions",
    )
    max_leverage: float = Field(
        default=5.0, gt=0, le=100, description="Maximum leverage (used intelligently)"
    )

    # Multi-Level Protection
    max_strategy_loss_pct: float = Field(
        default=7.0,
        gt=0,
        le=100,
        description="Maximum loss per strategy as % of initial capital (default 7%)",
    )
    max_hourly_loss_pct: float = Field(
        default=15.0,
        gt=0,
        le=100,
        description="Maximum loss per hour as % of initial capital (default 15%)",
    )
    consecutive_losses_limit: int = Field(
        default=3, ge=1, description="Number of consecutive losses before pause (HEAN v2 Iron Rule: 3)"
    )
    consecutive_losses_cooldown_hours: int = Field(
        default=2, gt=0, description="Hours to pause after consecutive losses (HEAN v2 Iron Rule: 2h)"
    )

    # Minimum Risk:Reward Ratio (HEAN v2 Iron Rule #5)
    min_risk_reward_ratio: float = Field(
        default=2.0,
        gt=0,
        description="Minimum Risk:Reward ratio to enter a trade (HEAN v2 Iron Rule: 1:2)",
    )

    # Deposit Protection
    deposit_protection_active: bool = Field(
        default=True,
        description="Enable deposit protection (never allow equity below initial capital)",
    )
    killswitch_drawdown_pct: float = Field(
        default=30.0,
        gt=0,
        description="Killswitch drawdown percentage from initial capital (Testnet: 30% to prevent premature stops)",
    )

    # Capital Preservation Mode
    capital_preservation_drawdown_threshold: float = Field(
        default=12.0,
        gt=0,
        description="Drawdown % to activate capital preservation mode (default 12%)",
    )
    capital_preservation_pf_threshold: float = Field(
        default=0.9,
        gt=0,
        description="PF threshold to activate capital preservation mode (default 0.9)",
    )
    capital_preservation_consecutive_losses_threshold: int = Field(
        default=3,
        ge=1,
        description="Consecutive losses to activate capital preservation mode (default 3)",
    )

    # Smart Leverage
    min_edge_for_leverage_2x: float = Field(
        default=25.0, ge=0, description="Minimum edge in bps for 2x leverage (default 25)"
    )
    min_edge_for_leverage_3x: float = Field(
        default=35.0, ge=0, description="Minimum edge in bps for 3x leverage (default 35)"
    )
    min_edge_for_leverage_4x: float = Field(
        default=50.0, ge=0, description="Minimum edge in bps for 4x leverage (default 50)"
    )
    min_pf_for_leverage: float = Field(
        default=1.2, gt=0, description="Minimum PF for leverage > 1x (default 1.2)"
    )
    max_leverage_on_drawdown_10pct: float = Field(
        default=2.0, gt=0, description="Max leverage when drawdown 10-15% (default 2.0)"
    )

    # Strategy Settings
    funding_harvester_enabled: bool = Field(
        default=True, description="Enable funding harvester strategy"
    )
    basis_arbitrage_enabled: bool = Field(
        default=True, description="Enable basis arbitrage strategy"
    )
    impulse_engine_enabled: bool = Field(default=True, description="Enable impulse engine strategy")

    # Dormant Strategies (AFO-Director Phase 5 - disabled by default until tested)
    hf_scalping_enabled: bool = Field(
        default=False, description="Enable high-frequency scalping strategy (40-60 trades/day, 0.2-0.4% TP)"
    )
    enhanced_grid_enabled: bool = Field(
        default=False, description="Enable enhanced grid trading strategy (range-bound markets only)"
    )
    momentum_trader_enabled: bool = Field(
        default=False, description="Enable momentum trader strategy (trend following)"
    )
    inventory_neutral_mm_enabled: bool = Field(
        default=False, description="Enable inventory-neutral market making strategy"
    )
    correlation_arb_enabled: bool = Field(
        default=False, description="Enable correlation arbitrage strategy"
    )
    rebate_farmer_enabled: bool = Field(
        default=False, description="Enable rebate farmer strategy (maker fee capture)"
    )
    sentiment_strategy_enabled: bool = Field(
        default=False, description="Enable sentiment-based trading strategy"
    )

    # Trading Symbols
    trading_symbols: list[str] = Field(
        default=[
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "DOTUSDT", "LTCUSDT", "AVAXUSDT", "LINKUSDT",
            "BCHUSDT", "MATICUSDT", "ATOMUSDT", "ETCUSDT", "BNBUSDT",
            "FILUSDT", "NEARUSDT", "OPUSDT", "ARBUSDT", "APTUSDT",
            "SUIUSDT", "TRXUSDT", "XLMUSDT", "UNIUSDT", "AAVEUSDT",
            "MKRUSDT", "INJUSDT", "RNDRUSDT", "SEIUSDT", "RUNEUSDT",
            "ICPUSDT", "ALGOUSDT", "EOSUSDT", "FTMUSDT", "GALAUSDT",
            "SANDUSDT", "AXSUSDT", "CHZUSDT", "CRVUSDT", "KAVAUSDT",
            "GMXUSDT", "SNXUSDT", "ZILUSDT", "DYDXUSDT", "COMPUSDT",
            "1INCHUSDT", "LDOUSDT", "NEOUSDT", "XTZUSDT", "APEUSDT",
        ],
        description="List of trading symbols to monitor and trade (e.g., ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])",
    )

    # Multi-Symbol Support (AFO-Director feature)
    multi_symbol_enabled: bool = Field(
        default=False,
        description="Enable multi-symbol scanning and trading (default False). When enabled, scans all symbols in SYMBOLS list.",
    )
    symbols: list[str] = Field(
        default=[
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "TONUSDT",
        ],
        description="List of symbols for multi-symbol scanning (default: 10 symbols). Used when MULTI_SYMBOL_ENABLED=true.",
    )

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v: Any) -> Any:
        """Parse symbols from env, handling empty strings and JSON."""
        if v is None or v == "":
            return [
                "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
                "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "TONUSDT",
            ]
        return parse_list_env(v)

    # Paper mode data source
    paper_use_live_feed: bool = Field(
        default=False,
        description="Use Bybit public market data in paper mode (instead of synthetic feed). Default False to guarantee ticks in PAPER.",
    )

    # Triangular arbitrage (Phase 2 Profit Doubling: Optimized for Bybit)
    triangular_arb_enabled: bool = Field(
        default=True,
        description="Enable triangular arbitrage scanner",
    )
    triangular_fee_buffer: float = Field(
        default=0.0006,  # Optimized: 0.06% (Bybit maker: -0.01%, taker: 0.055%)
        ge=0,
        description="Fee buffer ratio for triangular arbitrage scanning",
    )
    triangular_min_profit_bps: float = Field(
        default=3.0,  # Optimized: Lower threshold = more opportunities (0.03%)
        ge=0,
        description="Minimum profit in bps required for triangular arbitrage",
    )

    @field_validator("trading_symbols", mode="before")
    @classmethod
    def parse_trading_symbols(cls, v: Any) -> Any:
        """Parse trading_symbols from env, handling empty strings and JSON."""
        if v is None or v == "":
            return ["BTCUSDT", "ETHUSDT"]  # Default value
        return parse_list_env(v)

    @field_validator("impulse_allowed_hours", mode="before")
    @classmethod
    def parse_impulse_allowed_hours(cls, v: Any) -> Any:
        """Parse impulse_allowed_hours from env values."""
        if v == "":
            return None
        return parse_list_env(v)

    # Impulse Engine Specific
    impulse_max_attempts_per_day: int = Field(
        default=120,
        gt=0,
        description="Maximum attempts per day for impulse engine (increased for higher frequency)",
    )
    impulse_cooldown_after_losses: int = Field(
        default=3, ge=0, description="Cooldown period after losses"
    )
    impulse_cooldown_minutes: int = Field(
        default=2,
        ge=1,
        description="Cooldown minutes between trades (reduced from 5 to 2 for more trades)",
    )
    impulse_max_spread_bps: int = Field(
        default=12,
        gt=0,
        description="Maximum spread in basis points for impulse engine (no-trade zone, increased from 8 to 12 for more opportunities). "
        "In Aggressive Mode (DEBUG_MODE=True), this is effectively doubled via multiplier.",
    )
    impulse_max_volatility_spike: float = Field(
        default=0.02,
        gt=0,
        description="Maximum volatility spike threshold (no-trade zone, increased from 0.01 to 0.02 for more opportunities)",
    )
    impulse_max_time_in_trade_sec: int = Field(
        default=300,
        gt=0,
        description="Maximum time in trade in seconds (force exit, increased from 200 to 300 to give trades more time)",
    )
    impulse_maker_edge_threshold_bps: float = Field(
        default=3.0, ge=0, description="Minimum maker edge in basis points (skip if below)"
    )
    impulse_allowed_hours: list[str] | None = Field(
        default=["07:00-11:00", "12:00-16:00", "17:00-21:00"],
        description="Allowed trading hours in UTC (list of time ranges as strings, e.g., ['08:00-12:00', '13:00-17:00'])",
    )
    impulse_vol_expansion_ratio: float = Field(
        default=1.03,
        gt=0,
        description="Required volatility expansion ratio (short-term / long-term, reduced from 1.08 to 1.03 for more opportunities)",
    )

    # Impulse regime / sizing controls
    impulse_allow_normal: bool = Field(
        default=True,
        description="Allow impulse engine to trade in NORMAL regime (with reduced size multiplier)",
    )
    impulse_normal_size_multiplier: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Size multiplier for impulse trades in NORMAL regime (<= 1.0)",
    )
    volatility_hard_block_percentile: float = Field(
        default=99.0,
        ge=0,
        le=100,
        description="Percentile of recent volatility above which trades are hard-blocked (P99 - extreme levels only)",
    )

    # Bybit API (for future live trading)
    bybit_api_key: str = Field(default="", description="Bybit API key")
    bybit_api_secret: str = Field(default="", description="Bybit API secret")
    bybit_testnet: bool = Field(default=True, description="Use Bybit testnet (default True for safety)")

    # Safety: require explicit confirmation for LIVE trading
    require_live_confirm: bool = Field(
        default=False,
        description="Require explicit LIVE confirmation (set to True after all smoke tests PASS)",
    )

    # LLM API Keys (for agent generation and catalyst)
    gemini_api_key: str = Field(default="", description="Google Gemini API key for agent generation and catalyst")

    # Claude Brain
    anthropic_api_key: str = Field(default="", description="Anthropic API key for Claude Brain analysis")
    brain_analysis_interval: int = Field(default=30, gt=5, description="Brain analysis interval in seconds")
    brain_enabled: bool = Field(default=True, description="Enable Claude Brain analysis module")

    # API Authentication (CRITICAL: Enable in production!)
    api_auth_enabled: bool = Field(
        default=False,
        description="Enable API authentication (set to True in production)",
    )
    api_auth_key: str = Field(
        default="",
        description="API key for authentication (generate with: python -c 'import secrets; print(secrets.token_hex(32))')",
    )
    jwt_secret: str = Field(
        default="",
        description="JWT secret for token signing (auto-generated if not set)",
    )
    ws_allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
        description="Allowed origins for WebSocket connections (CORS)",
    )

    @field_validator("ws_allowed_origins", mode="before")
    @classmethod
    def parse_ws_allowed_origins(cls, v: Any) -> Any:
        """Parse ws_allowed_origins from env values."""
        if v is None or v == "":
            return ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"]
        return parse_list_env(v)

    # Observability
    log_level: str = Field(default="INFO", description="Logging level")
    health_check_port: int = Field(default=8080, gt=0, le=65535, description="Health check port")
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode (bypasses safety checks, verbose logging). WARNING: Only for development!",
    )

    # Redis (distributed event bus / AFO streaming)
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis URL for AFO event streaming and shared state (reads from REDIS_URL env var, defaults to redis://redis:6379/0)",
    )

    # Backtest
    backtest_initial_capital: float = Field(
        default=10000.0, gt=0, description="Initial capital for backtesting"
    )
    backtest_maker_fee: float = Field(
        default=0.00005,
        ge=0,
        le=0.01,
        description="Maker fee for backtesting (default 0.005% = 0.5 bps, reduced from 0.01%)",
    )
    backtest_taker_fee: float = Field(
        default=0.0003,
        ge=0,
        le=0.01,
        description="Taker fee for backtesting (default 0.03% = 3 bps, reduced from 0.06%)",
    )

    # Execution Policy - MAKER-ONLY for small capital!
    maker_first: bool = Field(
        default=True,
        description="Use maker-first execution (post-only limit orders) - ALWAYS TRUE!",
    )
    maker_ttl_ms: int = Field(
        default=150,  # Phase 2 Optimization: 150ms (was 8000ms) for HFT execution
        gt=0,
        description="Time-to-live for maker orders in milliseconds (Phase 2: 150ms optimal for HFT, +5-15 bps per execution)",
    )
    allow_taker_fallback: bool = Field(
        default=True,
        description="Allow taker fallback if maker order expires and edge is still positive (smart fallback with edge check)",
    )
    maker_price_offset_bps: int = Field(
        default=1,
        ge=0,
        description="Price offset in basis points for maker orders (best bid/ask ± offset, reduced to 1 bps for better fill rate)",
    )
    maker_fill_tolerance_bps: float = Field(
        default=0.5,
        ge=0,
        description="Tolerance in bps for maker order fills - allows fills when price is within this range of limit (default 0.5 bps)",
    )

    # Execution Edge Estimator (optimized for small capital)
    min_edge_bps_normal: float = Field(
        default=5.0,
        ge=0,
        description="Minimum edge in bps for NORMAL regime (lowered for more trades)",
    )
    min_edge_bps_impulse: float = Field(
        default=8.0,
        ge=0,
        description="Minimum edge in bps for IMPULSE regime (lowered for more trades)",
    )
    min_edge_bps_range: float = Field(
        default=3.0,
        ge=0,
        description="Minimum edge in bps for RANGE regime (lowered for more trades)",
    )

    # Income streams - enable/disable and budgets
    stream_funding_enabled: bool = Field(default=True, description="Enable funding income stream")
    stream_maker_rebate_enabled: bool = Field(
        default=True, description="Enable maker rebate income stream"
    )
    stream_basis_enabled: bool = Field(default=True, description="Enable basis hedge income stream")
    stream_volatility_enabled: bool = Field(
        default=True, description="Enable volatility harvest income stream"
    )

    stream_funding_capital_pct: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Capital allocation fraction for funding income stream",
    )
    stream_maker_rebate_capital_pct: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Capital allocation fraction for maker rebate income stream",
    )
    stream_basis_capital_pct: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="Capital allocation fraction for basis hedge income stream",
    )
    stream_volatility_capital_pct: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Capital allocation fraction for volatility harvest income stream",
    )

    stream_funding_max_positions: int = Field(
        default=2,
        ge=0,
        description="Maximum concurrent positions for funding income stream",
    )
    stream_maker_rebate_max_positions: int = Field(
        default=10,
        ge=0,
        description="Maximum concurrent positions for maker rebate income stream",
    )
    stream_basis_max_positions: int = Field(
        default=4,
        ge=0,
        description="Maximum concurrent positions for basis hedge income stream",
    )
    stream_volatility_max_positions: int = Field(
        default=3,
        ge=0,
        description="Maximum concurrent positions for volatility harvest income stream",
    )

    # Strategy / Decision Memory
    memory_window_trades: int = Field(
        default=30,
        gt=0,
        description="Number of trades to look back for rolling profit factor in memory layers",
    )
    regime_blacklist_duration_days: int = Field(
        default=7, gt=0, description="Duration in days to blacklist a regime after repeated losses"
    )
    drawdown_threshold_pct: float = Field(
        default=15.0,
        gt=0,
        le=100,
        description="Strategy-level drawdown threshold percentage to trigger cooldown",
    )
    pf_penalty_threshold: float = Field(
        default=1.0, gt=0, description="Profit factor threshold below which penalties apply"
    )

    # Decision Memory specific
    memory_loss_streak: int = Field(
        default=3,
        ge=1,
        description="Consecutive losses in a context before temporarily blocking it",
    )
    memory_block_hours: int = Field(
        default=24,
        gt=0,
        description="Number of hours to block a bad context after loss streak/drawdown",
    )
    memory_drawdown_threshold_pct: float = Field(
        default=12.0,
        gt=0,
        le=100,
        description="Context-level drawdown threshold percentage to trigger cooldown block",
    )

    # Process Factory (experimental extension layer)
    process_factory_enabled: bool = Field(
        default=False,
        description="Enable Process Factory extension layer (experimental, disabled by default)",
    )
    process_factory_allow_actions: bool = Field(
        default=False,
        description="Allow Bybit actions from Process Factory (disabled by default, requires explicit enable)",
    )

    # Dry Run - REMOVED (No paper trading - Bybit testnet only)
    dry_run: bool = Field(
        default=False,
        description="DEPRECATED - No longer used. System uses Bybit testnet for all trading.",
    )

    # Execution Smoke Test
    execution_smoke_test_enabled: bool = Field(
        default=False,
        description="Enable execution smoke test (disabled by default)",
    )
    execution_smoke_test_symbol: str = Field(
        default="BTCUSDT",
        description="Symbol for smoke test (default BTCUSDT)",
    )
    execution_smoke_test_notional_usd: float = Field(
        default=5.0,
        gt=0,
        description="Notional value in USD for smoke test order (default 5)",
    )
    execution_smoke_test_side: Literal["BUY", "SELL"] = Field(
        default="BUY",
        description="Side for smoke test order (default BUY)",
    )
    execution_smoke_test_mode: Literal["PLACE_CANCEL"] = Field(
        default="PLACE_CANCEL",
        description="Smoke test mode (default PLACE_CANCEL: place and immediately cancel)",
    )

    # Paper Trade Assist - REMOVED (No paper trading mode)
    paper_trade_assist: bool = Field(
        default=False,
        description="DEPRECATED - No longer used. System uses Bybit testnet only.",
    )

    # Profit Capture (AFO-Director feature)
    profit_capture_enabled: bool = Field(
        default=False,
        description="Enable profit capture feature (default OFF). Automatically locks profits when target is reached.",
    )
    profit_capture_target_pct: float = Field(
        default=20.0,
        gt=0,
        description="Profit capture target percentage (default 20%). Triggers when equity grows by this % from start.",
    )
    profit_capture_trail_pct: float = Field(
        default=10.0,
        gt=0,
        description="Profit capture trail percentage (default 10%). Triggers when drawdown from peak reaches this %.",
    )
    profit_capture_mode: Literal["partial", "full"] = Field(
        default="full",
        description="Profit capture mode: 'full' closes all positions and cancels orders, 'partial' reduces exposure.",
    )
    profit_capture_after_action: Literal["pause", "continue"] = Field(
        default="pause",
        description="Action after profit capture: 'pause' stops trading, 'continue' continues with reduced risk.",
    )
    profit_capture_continue_risk_mult: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="Risk multiplier when continuing after profit capture (default 0.25 = 25% of normal risk).",
    )
    paper_trade_assist_micro_trade_interval_sec: int = Field(
        default=60,
        gt=0,
        description="Interval in seconds for fallback micro-trades when PAPER_TRADE_ASSIST=true (default 60)",
    )
    paper_trade_assist_micro_trade_notional_usd: float = Field(
        default=10.0,
        gt=0,
        description="Notional value in USD for fallback micro-trades (default 10)",
    )
    paper_trade_assist_micro_trade_tp_pct: float = Field(
        default=0.3,
        gt=0,
        le=5.0,
        description="Take profit percentage for micro-trades (default 0.3%)",
    )
    paper_trade_assist_micro_trade_sl_pct: float = Field(
        default=0.3,
        gt=0,
        le=5.0,
        description="Stop loss percentage for micro-trades (default 0.3%)",
    )
    paper_trade_assist_micro_trade_max_time_min: int = Field(
        default=5,
        gt=0,
        description="Maximum time in minutes for micro-trade before forced exit (default 5)",
    )

    # Phase 5: Statistical Arbitrage & Anti-Fragile Architecture
    phase5_correlation_engine_enabled: bool = Field(
        default=True,
        description="Enable Correlation Engine for pair trading (Phase 5)",
    )
    phase5_safety_net_enabled: bool = Field(
        default=True,
        description="Enable Global Safety Net (Black Swan Protection) (Phase 5)",
    )
    phase5_kelly_criterion_enabled: bool = Field(
        default=True,
        description="Enable Kelly Criterion for position sizing (Phase 5)",
    )
    phase5_kelly_fractional: float = Field(
        default=0.25,
        ge=0.1,
        le=1.0,
        description="Fractional Kelly to use (0.25 = quarter Kelly, recommended) (Phase 5)",
    )
    phase5_self_healing_enabled: bool = Field(
        default=True,
        description="Enable Self-Healing Middleware for system monitoring (Phase 5)",
    )
    phase5_correlation_min_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum correlation threshold for pair trading (default 0.7 = 70%) (Phase 5)",
    )
    phase5_correlation_gap_threshold: float = Field(
        default=2.0,
        gt=0,
        description="Price gap threshold in standard deviations for pair trading (default 2.0) (Phase 5)",
    )
    phase5_entropy_spike_threshold: float = Field(
        default=3.0,
        gt=1.0,
        description="Market entropy spike threshold (300% = 3.0x baseline) for safety net (Phase 5)",
    )
    phase5_emergency_size_multiplier: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Position size multiplier during emergency (80% reduction = 0.2x) (Phase 5)",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate trading mode after initialization."""
        # CRITICAL SAFETY CHECK: Prevent LIVE trading without explicit confirmation
        is_attempting_live = (
            not self.dry_run
            and self.live_confirm == "YES"
            and not self.bybit_testnet
        )

        if is_attempting_live and not self.require_live_confirm:
            raise ValueError(
                "🚨 LIVE TRADING BLOCKED 🚨\n"
                "LIVE trading mode requires REQUIRE_LIVE_CONFIRM=true.\n"
                "This safety check ensures all smoke tests have PASSED before enabling real trading.\n"
                f"Current config: DRY_RUN={self.dry_run}, LIVE_CONFIRM={self.live_confirm}, "
                f"BYBIT_TESTNET={self.bybit_testnet}, REQUIRE_LIVE_CONFIRM={self.require_live_confirm}\n"
                "To enable LIVE trading:\n"
                "1. Run all smoke tests and verify PASS\n"
                "2. Set REQUIRE_LIVE_CONFIRM=true in backend.env\n"
                "3. Set BYBIT_TESTNET=false, DRY_RUN=false, LIVE_CONFIRM=YES"
            )

        # Check environment for LIVE_CONFIRM if not already set
        live_confirm_env = os.getenv("LIVE_CONFIRM", "")
        if live_confirm_env == "YES" and self.trading_mode == "paper":
            # Update trading mode if LIVE_CONFIRM is set
            object.__setattr__(self, "trading_mode", "live")
            object.__setattr__(self, "live_confirm", "YES")

        # Auto-set live_confirm to YES if trading_mode is live and dry_run is False
        if self.trading_mode == "live" and not self.dry_run and self.live_confirm != "YES":
            object.__setattr__(self, "live_confirm", "YES")

        # Set dry_run default from environment if not explicitly set
        # DRY_RUN env var: "true"/"1"/"yes" -> True, "false"/"0"/"no" -> False, unset -> True (default)
        dry_run_env = os.getenv("DRY_RUN", "").lower()
        if dry_run_env and dry_run_env not in ("true", "1", "yes", "false", "0", "no"):
            # Only override if explicitly set in env and value is valid
            from hean.logging import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Invalid DRY_RUN value: {dry_run_env!r}. Ignoring.")
        elif dry_run_env in ("false", "0", "no"):
            object.__setattr__(self, "dry_run", False)

        # Paper trade assist is deprecated - no validation needed
        # System always uses Bybit testnet for safety

    @property
    def is_live(self) -> bool:
        """Check if system is in live trading mode (Always True - Bybit testnet)."""
        return True  # Always live mode with Bybit testnet


# Global settings instance
settings = HEANSettings()
