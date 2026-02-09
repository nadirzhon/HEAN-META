use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use parking_lot::RwLock;
use dashmap::DashMap;
use serde::{Serialize, Deserialize};
use thiserror::Error;

// ============================================================================
// TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_position_value: f64,
    pub max_daily_loss: f64,
    pub max_order_size: f64,
    pub max_leverage: f64,
    pub max_position_count: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Order {
    pub order_id: u64,
    pub symbol_id: u16,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OrderSide {
    Buy = 0,
    Sell = 1,
}

#[derive(Clone, Copy)]
pub struct Position {
    pub quantity: f64,
    pub avg_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Fill {
    pub symbol_id: u16,
    pub quantity: f64,  // positive for buy, negative for sell
    pub price: f64,
}

#[derive(Debug, Error)]
pub enum RiskError {
    #[error("Order too large: {0}")]
    OrderTooLarge(f64),
    #[error("Position limit exceeded")]
    PositionLimitExceeded,
    #[error("Daily loss exceeded")]
    DailyLossExceeded,
    #[error("Leverage exceeded")]
    LeverageExceeded,
    #[error("Too many positions")]
    TooManyPositions,
}

// ============================================================================
// POSITION TRACKER (Lock-Free)
// ============================================================================

pub struct PositionTracker {
    positions: Arc<DashMap<u16, Position>>,
    daily_pnl: Arc<RwLock<f64>>,
    total_realized_pnl: AtomicI64,  // * 100 for precision
    equity: AtomicU64,  // * 100 for precision
}

impl PositionTracker {
    pub fn new(initial_equity: f64) -> Self {
        Self {
            positions: Arc::new(DashMap::new()),
            daily_pnl: Arc::new(RwLock::new(0.0)),
            total_realized_pnl: AtomicI64::new(0),
            equity: AtomicU64::new((initial_equity * 100.0) as u64),
        }
    }

    /// Check if order passes all risk checks (< 10μs)
    pub fn check_risk(&self, order: &Order, limits: &RiskLimits) -> Result<(), RiskError> {
        let start = std::time::Instant::now();

        // 1. Check order size (< 100ns)
        let order_value = order.quantity * order.price;
        if order_value > limits.max_order_size {
            return Err(RiskError::OrderTooLarge(order_value));
        }

        // 2. Check position limit (< 500ns)
        let position_value = self.calculate_position_value(order.symbol_id);
        if position_value > limits.max_position_value {
            return Err(RiskError::PositionLimitExceeded);
        }

        // 3. Check daily loss (< 200ns)
        let daily_pnl = *self.daily_pnl.read();
        if daily_pnl < -limits.max_daily_loss {
            return Err(RiskError::DailyLossExceeded);
        }

        // 4. Check leverage (< 300ns)
        let total_exposure = self.calculate_total_exposure();
        let equity = self.get_equity();
        if total_exposure / equity > limits.max_leverage {
            return Err(RiskError::LeverageExceeded);
        }

        // 5. Check position count
        if self.positions.len() >= limits.max_position_count {
            return Err(RiskError::TooManyPositions);
        }

        let elapsed = start.elapsed();
        debug_assert!(elapsed.as_micros() < 10, "Risk check too slow: {:?}", elapsed);

        Ok(())
    }

    #[inline(always)]
    fn calculate_position_value(&self, symbol_id: u16) -> f64 {
        self.positions
            .get(&symbol_id)
            .map(|pos| pos.quantity.abs() * pos.avg_price)
            .unwrap_or(0.0)
    }

    fn calculate_total_exposure(&self) -> f64 {
        self.positions
            .iter()
            .map(|entry| {
                let pos = entry.value();
                pos.quantity.abs() * pos.avg_price
            })
            .sum()
    }

    fn get_equity(&self) -> f64 {
        self.equity.load(Ordering::Relaxed) as f64 / 100.0
    }

    /// Update position after fill (lock-free for most operations)
    pub fn update_position(&self, fill: Fill, current_price: f64) {
        self.positions
            .entry(fill.symbol_id)
            .and_modify(|pos| {
                let old_qty = pos.quantity;
                let new_qty = old_qty + fill.quantity;

                // Calculate realized PnL if closing/reducing position
                if old_qty.signum() != new_qty.signum() || new_qty.abs() < old_qty.abs() {
                    let closed_qty = if new_qty.signum() != old_qty.signum() {
                        old_qty.abs()
                    } else {
                        old_qty.abs() - new_qty.abs()
                    };

                    let pnl = if old_qty > 0.0 {
                        // Closing long
                        closed_qty * (fill.price - pos.avg_price)
                    } else {
                        // Closing short
                        closed_qty * (pos.avg_price - fill.price)
                    };

                    pos.realized_pnl += pnl;

                    // Update daily PnL
                    let mut daily_pnl = self.daily_pnl.write();
                    *daily_pnl += pnl;

                    // Update total realized PnL (atomic)
                    self.total_realized_pnl.fetch_add(
                        (pnl * 100.0) as i64,
                        Ordering::Relaxed
                    );
                }

                // Update position
                if new_qty.abs() < 1e-8 {
                    // Position closed
                    pos.quantity = 0.0;
                    pos.avg_price = 0.0;
                    pos.unrealized_pnl = 0.0;
                } else {
                    // Update average price
                    if old_qty.signum() == fill.quantity.signum() {
                        // Adding to position
                        pos.avg_price = (old_qty * pos.avg_price + fill.quantity * fill.price)
                            / new_qty;
                    }
                    pos.quantity = new_qty;

                    // Update unrealized PnL
                    pos.unrealized_pnl = if new_qty > 0.0 {
                        new_qty * (current_price - pos.avg_price)
                    } else {
                        new_qty.abs() * (pos.avg_price - current_price)
                    };
                }
            })
            .or_insert(Position {
                quantity: fill.quantity,
                avg_price: fill.price,
                unrealized_pnl: 0.0,
                realized_pnl: 0.0,
            });
    }

    /// Get current position for symbol
    pub fn get_position(&self, symbol_id: u16) -> Option<Position> {
        self.positions.get(&symbol_id).map(|p| *p)
    }

    /// Get all positions
    pub fn get_all_positions(&self) -> Vec<(u16, Position)> {
        self.positions
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect()
    }

    /// Get daily PnL
    pub fn get_daily_pnl(&self) -> f64 {
        *self.daily_pnl.read()
    }

    /// Reset daily PnL (call at EOD)
    pub fn reset_daily_pnl(&self) {
        let mut daily_pnl = self.daily_pnl.write();
        *daily_pnl = 0.0;
    }

    /// Get total realized PnL
    pub fn get_total_realized_pnl(&self) -> f64 {
        self.total_realized_pnl.load(Ordering::Relaxed) as f64 / 100.0
    }
}

// ============================================================================
// RISK ENGINE
// ============================================================================

pub struct RiskEngine {
    tracker: Arc<PositionTracker>,
    limits: Arc<RwLock<RiskLimits>>,
}

impl RiskEngine {
    pub fn new(limits: RiskLimits, initial_equity: f64) -> Self {
        Self {
            tracker: Arc::new(PositionTracker::new(initial_equity)),
            limits: Arc::new(RwLock::new(limits)),
        }
    }

    /// Pre-trade risk check (< 10μs)
    pub async fn pre_trade_check(&self, order: &Order) -> Result<(), RiskError> {
        let limits = self.limits.read();
        self.tracker.check_risk(order, &limits)
    }

    /// Post-trade update
    pub fn on_fill(&self, fill: Fill, current_price: f64) {
        self.tracker.update_position(fill, current_price);
    }

    /// Get position tracker
    pub fn tracker(&self) -> &Arc<PositionTracker> {
        &self.tracker
    }

    /// Update risk limits
    pub fn update_limits(&self, new_limits: RiskLimits) {
        let mut limits = self.limits.write();
        *limits = new_limits;
    }
}

// ============================================================================
// MAIN SERVER
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("🛡️  Starting Risk Engine...");

    let limits = RiskLimits {
        max_position_value: 100_000.0,
        max_daily_loss: 10_000.0,
        max_order_size: 50_000.0,
        max_leverage: 10.0,
        max_position_count: 50,
    };

    let engine = Arc::new(RiskEngine::new(limits, 100_000.0));

    tracing::info!("✅ Risk Engine ready!");

    // Demo: Check some orders
    let demo_order = Order {
        order_id: 1,
        symbol_id: 0,
        side: OrderSide::Buy,
        quantity: 1.0,
        price: 45000.0,
    };

    match engine.pre_trade_check(&demo_order).await {
        Ok(_) => tracing::info!("✅ Order passed risk checks"),
        Err(e) => tracing::error!("❌ Order failed risk check: {}", e),
    }

    // Simulate a fill
    engine.on_fill(
        Fill {
            symbol_id: 0,
            quantity: 1.0,
            price: 45000.0,
        },
        45100.0,
    );

    // Check position
    if let Some(pos) = engine.tracker().get_position(0) {
        tracing::info!(
            "📊 Position: qty={}, avg_price={}, unrealized_pnl={}",
            pos.quantity,
            pos.avg_price,
            pos.unrealized_pnl
        );
    }

    // Keep running
    tokio::signal::ctrl_c().await?;
    tracing::info!("Shutting down Risk Engine...");

    Ok(())
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_risk_checks() {
        let limits = RiskLimits {
            max_position_value: 100_000.0,
            max_daily_loss: 10_000.0,
            max_order_size: 50_000.0,
            max_leverage: 10.0,
            max_position_count: 50,
        };

        let engine = RiskEngine::new(limits, 100_000.0);

        // Valid order
        let order = Order {
            order_id: 1,
            symbol_id: 0,
            side: OrderSide::Buy,
            quantity: 1.0,
            price: 45000.0,
        };

        assert!(engine.pre_trade_check(&order).await.is_ok());

        // Order too large
        let large_order = Order {
            order_id: 2,
            symbol_id: 0,
            side: OrderSide::Buy,
            quantity: 10.0,
            price: 10_000.0,
        };

        assert!(engine.pre_trade_check(&large_order).await.is_err());
    }

    #[test]
    fn test_position_update() {
        let tracker = PositionTracker::new(100_000.0);

        // Open long position
        tracker.update_position(
            Fill {
                symbol_id: 0,
                quantity: 1.0,
                price: 45000.0,
            },
            45000.0,
        );

        let pos = tracker.get_position(0).unwrap();
        assert_eq!(pos.quantity, 1.0);
        assert_eq!(pos.avg_price, 45000.0);

        // Close position with profit
        tracker.update_position(
            Fill {
                symbol_id: 0,
                quantity: -1.0,
                price: 46000.0,
            },
            46000.0,
        );

        let pos = tracker.get_position(0).unwrap();
        assert_eq!(pos.quantity, 0.0);
        assert_eq!(pos.realized_pnl, 1000.0);
    }

    #[test]
    fn test_performance() {
        let tracker = PositionTracker::new(100_000.0);
        let limits = RiskLimits {
            max_position_value: 100_000.0,
            max_daily_loss: 10_000.0,
            max_order_size: 50_000.0,
            max_leverage: 10.0,
            max_position_count: 50,
        };

        let order = Order {
            order_id: 1,
            symbol_id: 0,
            side: OrderSide::Buy,
            quantity: 1.0,
            price: 45000.0,
        };

        // Benchmark: should be < 10μs
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = tracker.check_risk(&order, &limits);
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() / 1000;

        println!("Average risk check: {}μs", avg_us);
        assert!(avg_us < 10, "Risk check too slow: {}μs", avg_us);
    }
}
