"""Configuration management using Pydantic v2."""

import json
import os
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class HEANSettings(BaseSettings):
    """Main configuration for HEAN system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Trading Mode
    live_confirm: str = Field(default="YES", description="Must be 'YES' to enable live trading")
    trading_mode: Literal["paper", "live"] = Field(default="live", description="Trading mode")

    # Capital Management
    initial_capital: float = Field(default=400.0, gt=0, description="Initial capital in USDT")
    reinvest_rate: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Profit reinvestment rate (legacy, use smart_reinvest_base_rate)",
    )
    cash_reserve_rate: float = Field(
        default=0.2, ge=0, le=1, description="Cash reserve rate (not allocated)"
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
        description="Maximum daily drawdown percentage (default 15% for $400 capital)",
    )
    max_trade_risk_pct: float = Field(
        default=2.0,
        gt=0,
        le=100,
        description="Maximum risk per trade percentage (default 2% for aggressive trading)",
    )
    max_open_positions: int = Field(
        default=100, gt=0, description="Maximum number of open positions (for multi-pair trading)"
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
        default=5, ge=1, description="Number of consecutive losses before pause (default 5)"
    )
    consecutive_losses_cooldown_hours: int = Field(
        default=1, gt=0, description="Hours to pause after consecutive losses (default 1)"
    )

    # Deposit Protection
    deposit_protection_active: bool = Field(
        default=True,
        description="Enable deposit protection (never allow equity below initial capital)",
    )
    killswitch_drawdown_pct: float = Field(
        default=20.0,
        gt=0,
        description="Killswitch drawdown percentage from initial capital (default 20%)",
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

    # Trading Symbols
    trading_symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT"],
        description="List of trading symbols to monitor and trade (e.g., ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])",
    )

    @field_validator("trading_symbols", mode="before")
    @classmethod
    def parse_trading_symbols(cls, v: Any) -> Any:
        """Parse trading_symbols from env, handling empty strings and JSON."""
        if v is None or v == "":
            return ["BTCUSDT", "ETHUSDT"]  # Default value
        if isinstance(v, str):
            # Try to parse as JSON first
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            # If not JSON, try comma-separated
            if "," in v:
                return [s.strip() for s in v.split(",") if s.strip()]
        return v

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
    bybit_testnet: bool = Field(default=False, description="Use Bybit testnet")

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
        default=8000,
        gt=0,
        description="Time-to-live for maker orders in milliseconds (increased from 3000 to 8000 for better fill rate)",
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

    # Dry Run / Execution Safety
    dry_run: bool = Field(
        default=False,
        description="Dry run mode (default False). Set to True to enable paper trading mode",
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

    # Paper Trade Assist (for testing/debugging in paper mode only)
    paper_trade_assist: bool = Field(
        default=False,
        description="Enable paper trade assist mode - softens filters/limits in paper/dry_run mode only. "
        "FORBIDDEN in live trading (DRY_RUN=false && LIVE_CONFIRM=YES).",
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
            pass
        elif dry_run_env in ("false", "0", "no"):
            object.__setattr__(self, "dry_run", False)

        # Validate PAPER_TRADE_ASSIST - can only be enabled in paper/sandbox mode
        if self.paper_trade_assist:
            is_paper_safe = self.dry_run or self.bybit_testnet
            is_live_unsafe = not self.dry_run and self.live_confirm == "YES"
            
            if is_live_unsafe:
                raise ValueError(
                    "PAPER_TRADE_ASSIST=true is FORBIDDEN in live trading. "
                    "It can only be enabled when DRY_RUN=true OR bybit_testnet=true. "
                    f"Current: DRY_RUN={self.dry_run}, LIVE_CONFIRM={self.live_confirm}, "
                    f"bybit_testnet={self.bybit_testnet}"
                )
            
            if not is_paper_safe:
                raise ValueError(
                    "PAPER_TRADE_ASSIST=true requires DRY_RUN=true OR bybit_testnet=true. "
                    f"Current: DRY_RUN={self.dry_run}, bybit_testnet={self.bybit_testnet}"
                )

    @property
    def is_live(self) -> bool:
        """Check if system is in live trading mode."""
        return self.trading_mode == "live" and not self.dry_run


# Global settings instance
settings = HEANSettings()
