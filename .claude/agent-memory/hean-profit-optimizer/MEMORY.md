# HEAN Profit Optimizer Memory

## Key Fee & Cost Facts (from source)
- Maker fee: -0.01% (rebate) — `backtest_maker_fee = 0.00005` (0.005% in config, but Bybit testnet = -0.01%)
- Taker fee: 0.03% in config (`backtest_taker_fee = 0.0003`), real Bybit = 0.06%
- Funding: 0.01%/8h typical, paid 3x/day = 0.03%/day on notional
- `maker_first = True` — system attempts maker orders by default
- `maker_ttl_ms = 150` — aggressive 150ms TTL, high expiry rate likely
- `allow_taker_fallback = True` — expired makers fall back to taker (costs 0.06%)
- `maker_price_offset_bps = 1` — 1 bps inside BBO, good fill probability

## Capital & Risk (from config defaults)
- `initial_capital = 300 USDT`
- `max_trade_risk_pct = 1.0%` — $3 max risk per trade
- `max_leverage = 3.0x` — capped at 3x (SmartLeverage can reach 4x with edge)
- `max_open_positions = 10`
- `max_hold_seconds = 900` — 15 min max hold, forced exit
- `killswitch_drawdown_pct = 30%` — $90 max total loss from capital
- `consecutive_losses_limit = 3` — 2h pause after 3 consecutive losses
- Position size floor: `min_notional_usd = 100.0` — $100 min order (Bybit mainnet req)
- Kelly: `fractional_kelly = 0.25` (quarter Kelly), adaptive 0.15-0.50

## Strategy Activity Rates
- ImpulseEngine: max 120 attempts/day, 2 min cooldown, 3-filter cascade (spread/vol/time)
  - Time windows: 07-11, 12-16, 17-21 UTC (9 hours active/day)
  - Filter rejection: 70-95% of signals blocked
  - Net trades: likely 2-8/day per symbol pair
- FundingHarvester: max 6 signals/day, 4h cooldown, 2 symbols (BTC+ETH)
  - Minimum funding threshold: 0.01%
  - Uses `prefer_maker=True`
- BasisArbitrage: max 4 signals/day, 2h cooldown
- HFScalpingStrategy: targets 40-60 trades/day, 30s cooldown, 4 symbols
  - TP=25bps, SL=15bps, min_move=10bps
- RebateFarmer: passive, refresh every 5 min, 1% equity per side
- Other 6 strategies (MomentumTrader, CorrelationArb, EnhancedGrid, InventoryNeutralMM, LiquiditySweep, SentimentStrategy): all enabled

## Execution Architecture (router_bybit_only.py)
- maker_first path → SmartLimitExecutor → post-only limit @ bid-1bps
- TTL 150ms → if expired → MakerRetryQueue → taker fallback (if edge > 2bps net)
- OFI imbalance detected → overrides to taker immediately with 1.5x size
- No paper trading; all orders to Bybit testnet

## Critical Sizing Constraint
- `min_notional_usd = 100.0` is hardcoded in PositionSizer for both `calculate_size` and `calculate_size_v2`
- With $300 capital, max 3 simultaneous $100 min positions = 100% capital deployed
- `max_trade_risk_pct = 1%` → $3 risk per trade → at 2% stop = $150 notional (0.5x leverage)
- Kelly not active until 10+ trades per strategy (returns 1.0 multiplier until then)

## Profit Decomposition Reference (see detailed analysis below)
- See: `profit_decomposition_300.md`
