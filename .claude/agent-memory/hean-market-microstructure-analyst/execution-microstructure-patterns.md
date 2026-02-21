# Execution-Microstructure Pattern Notes

## OFI Python Fallback Critical Flaw
The `_calculate_ofi_python()` method computes:
  ofi_value = (total_bid_volume - total_ask_volume) / total_volume
This is a STATIC snapshot metric, not true OFI. True OFI requires:
  OFI_t = sum_i [ (delta_BidQty_i if BidPx unchanged) + (BidQty_i if BidPx improved) - (AskQty_i if AskPx unchanged) ... ]
The Python fallback should be labeled "book_imbalance" not OFI.

## Random Entry Benchmark Formula
For phantom/orphan positions as alpha benchmark:
  E[PnL] for random entry = -spread_bps - taker_fee_bps (both sides)
  For BTCUSDT: approx -7 bps (fee) - 2 bps (half spread) = -9 bps per side
  Round trip: -14 to -20 bps depending on spread regime
  Any strategy beating -14 bps on a risk-adjusted basis has positive alpha vs random

## Market Order Timeout as Liquidity Signal
  fill_latency = t_fill - t_order_placed
  liquidity_score = 1 / (1 + fill_latency_seconds)  # normalized [0,1]
  If fill_latency > 5s: spread likely >10 bps or depth insufficient
  If fill_latency > 15s: potential liquidity crisis or connectivity issue

## Close Retry Mean Reversion Formula
  For a losing position closing:
  mean_reversion_wait = sigma * sqrt(tau) / mu_reversion  (Ornstein-Uhlenbeck)
  Where sigma = realized vol, tau = holding time, mu_reversion = speed of mean reversion
  Optimal wait only beneficial if: E[price_improvement] > carry_cost + risk_of_further_loss
  In practice: only wait if OFI flipping from adverse to favorable direction
