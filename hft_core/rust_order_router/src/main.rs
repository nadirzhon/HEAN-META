use std::sync::Arc;
use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::time::Instant;
use serde::{Serialize, Deserialize};
use thiserror::Error;

// ============================================================================
// TYPES & STRUCTURES
// ============================================================================

/// Order structure - zero-copy design
#[repr(C)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Order {
    pub order_id: u64,
    pub symbol_id: u16,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub timestamp_ns: u64,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum OrderSide {
    Buy = 0,
    Sell = 1,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct OrderResponse {
    pub order_id: u64,
    pub status: OrderStatus,
    pub filled_quantity: f64,
    pub average_price: f64,
    pub timestamp_ns: u64,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum OrderStatus {
    Accepted,
    Filled,
    PartiallyFilled,
    Rejected,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct RiskLimits {
    pub max_position_value: f64,
    pub max_daily_loss: f64,
    pub max_order_size: f64,
    pub max_leverage: f64,
}

#[derive(Debug, Error)]
pub enum OrderError {
    #[error("Invalid quantity")]
    InvalidQuantity,
    #[error("Invalid price")]
    InvalidPrice,
    #[error("Risk limit exceeded")]
    RiskLimitExceeded,
    #[error("Exchange error: {0}")]
    ExchangeError(String),
    #[error("Network error: {0}")]
    NetworkError(String),
}

// ============================================================================
// SYMBOL MAPPING
// ============================================================================

static SYMBOLS: once_cell::sync::Lazy<Vec<&str>> = once_cell::sync::Lazy::new(|| {
    vec![
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "MATICUSDT",
        "DOTUSDT",
        "AVAXUSDT",
        "ATOMUSDT",
    ]
});

pub fn symbol_to_id(symbol: &str) -> Option<u16> {
    SYMBOLS.iter().position(|&s| s == symbol).map(|i| i as u16)
}

pub fn id_to_symbol(id: u16) -> Option<&'static str> {
    SYMBOLS.get(id as usize).copied()
}

// ============================================================================
// ORDER ROUTER
// ============================================================================

pub struct OrderRouter {
    active_orders: Arc<DashMap<u64, Order>>,
    risk_limits: Arc<RwLock<RiskLimits>>,
    exchange_clients: Vec<ExchangeClient>,
    metrics_tx: crossbeam::channel::Sender<MetricsEvent>,
}

#[derive(Debug)]
struct MetricsEvent {
    event_type: String,
    latency_us: u64,
    timestamp: u64,
}

impl OrderRouter {
    pub fn new(
        risk_limits: RiskLimits,
        metrics_tx: crossbeam::channel::Sender<MetricsEvent>,
    ) -> Self {
        // Initialize with one exchange client (can be expanded)
        let exchange_clients = vec![ExchangeClient::new("bybit".to_string())];

        Self {
            active_orders: Arc::new(DashMap::new()),
            risk_limits: Arc::new(RwLock::new(risk_limits)),
            exchange_clients,
            metrics_tx,
        }
    }

    pub async fn route_order(&self, mut order: Order) -> Result<OrderResponse, OrderError> {
        let start = Instant::now();

        // 1. Validate order (< 1μs)
        self.validate_order(&order)?;

        // 2. Risk check (< 10μs) - read without blocking
        {
            let limits = self.risk_limits.read();
            if !self.check_risk(&order, &limits) {
                return Err(OrderError::RiskLimitExceeded);
            }
        }

        // 3. Select best exchange (< 5μs)
        let exchange = self.select_exchange(&order);

        // 4. Send to exchange (< 50μs in simulation)
        let response = exchange.send_order(order).await?;

        // 5. Update state (lock-free)
        self.active_orders.insert(order.order_id, order);

        let elapsed = start.elapsed();

        // Send metrics
        let _ = self.metrics_tx.try_send(MetricsEvent {
            event_type: "order_routing".to_string(),
            latency_us: elapsed.as_micros() as u64,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        });

        tracing::info!(
            "Order routed: id={}, symbol={:?}, latency={}μs",
            order.order_id,
            id_to_symbol(order.symbol_id),
            elapsed.as_micros()
        );

        Ok(response)
    }

    #[inline(always)]
    fn validate_order(&self, order: &Order) -> Result<(), OrderError> {
        if order.quantity <= 0.0 {
            return Err(OrderError::InvalidQuantity);
        }
        if order.price <= 0.0 {
            return Err(OrderError::InvalidPrice);
        }
        Ok(())
    }

    #[inline(always)]
    fn check_risk(&self, order: &Order, limits: &RiskLimits) -> bool {
        let order_value = order.quantity * order.price;
        if order_value > limits.max_order_size {
            return false;
        }
        // Add more risk checks here
        true
    }

    fn select_exchange(&self, _order: &Order) -> &ExchangeClient {
        // Smart routing logic could go here
        &self.exchange_clients[0]
    }

    pub fn get_order(&self, order_id: u64) -> Option<Order> {
        self.active_orders.get(&order_id).map(|o| *o)
    }

    pub fn cancel_order(&self, order_id: u64) -> bool {
        self.active_orders.remove(&order_id).is_some()
    }
}

// ============================================================================
// EXCHANGE CLIENT
// ============================================================================

pub struct ExchangeClient {
    name: String,
    order_counter: Arc<parking_lot::Mutex<u64>>,
}

impl ExchangeClient {
    pub fn new(name: String) -> Self {
        Self {
            name,
            order_counter: Arc::new(parking_lot::Mutex::new(0)),
        }
    }

    pub async fn send_order(&self, order: Order) -> Result<OrderResponse, OrderError> {
        // Simulate exchange latency (20-50μs in production)
        tokio::time::sleep(tokio::time::Duration::from_micros(30)).await;

        let mut counter = self.order_counter.lock();
        *counter += 1;

        // Simulate successful order
        Ok(OrderResponse {
            order_id: order.order_id,
            status: OrderStatus::Filled,
            filled_quantity: order.quantity,
            average_price: order.price,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        })
    }
}

// ============================================================================
// MAIN SERVER
// ============================================================================

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("🚀 Starting Order Router...");

    // Pin to CPU core 0 for minimum latency
    if let Some(core_ids) = core_affinity::get_core_ids() {
        if !core_ids.is_empty() {
            core_affinity::set_for_current(core_ids[0]);
            tracing::info!("📌 Pinned to CPU core 0");
        }
    }

    // Create metrics channel
    let (metrics_tx, metrics_rx) = crossbeam::channel::unbounded();

    // Create order router
    let risk_limits = RiskLimits {
        max_position_value: 100_000.0,
        max_daily_loss: 10_000.0,
        max_order_size: 50_000.0,
        max_leverage: 10.0,
    };

    let router = Arc::new(OrderRouter::new(risk_limits, metrics_tx));

    // Spawn metrics collector
    tokio::task::spawn_blocking(move || {
        let mut total_orders = 0u64;
        let mut total_latency_us = 0u64;

        while let Ok(event) = metrics_rx.recv() {
            total_orders += 1;
            total_latency_us += event.latency_us;

            if total_orders % 1000 == 0 {
                let avg_latency = total_latency_us / total_orders;
                tracing::info!(
                    "📊 Metrics: {} orders, avg latency: {}μs",
                    total_orders,
                    avg_latency
                );
            }
        }
    });

    // ZeroMQ server for receiving orders from Python
    tracing::info!("🔌 Starting ZeroMQ server on tcp://127.0.0.1:5555");

    let context = zeromq::Context::new();
    let mut socket = zeromq::PullSocket::new(context);
    socket
        .bind("tcp://127.0.0.1:5555")
        .await
        .expect("Failed to bind socket");

    tracing::info!("✅ Order Router ready! Waiting for orders...");

    // Main event loop
    loop {
        match socket.recv().await {
            Ok(msg) => {
                // Deserialize order
                let order_data = msg.into_vec();
                match bincode::deserialize::<Order>(&order_data[0]) {
                    Ok(order) => {
                        let router = router.clone();
                        tokio::spawn(async move {
                            match router.route_order(order).await {
                                Ok(resp) => {
                                    tracing::debug!("✅ Order executed: {:?}", resp);
                                }
                                Err(e) => {
                                    tracing::error!("❌ Order failed: {:?}", e);
                                }
                            }
                        });
                    }
                    Err(e) => {
                        tracing::error!("Failed to deserialize order: {}", e);
                    }
                }
            }
            Err(e) => {
                tracing::error!("Failed to receive message: {}", e);
            }
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_validation() {
        let (tx, _rx) = crossbeam::channel::unbounded();
        let router = OrderRouter::new(
            RiskLimits {
                max_position_value: 100_000.0,
                max_daily_loss: 10_000.0,
                max_order_size: 50_000.0,
                max_leverage: 10.0,
            },
            tx,
        );

        let valid_order = Order {
            order_id: 1,
            symbol_id: 0,
            side: OrderSide::Buy,
            quantity: 1.0,
            price: 45000.0,
            timestamp_ns: 0,
        };

        assert!(router.validate_order(&valid_order).is_ok());

        let invalid_order = Order {
            order_id: 2,
            symbol_id: 0,
            side: OrderSide::Buy,
            quantity: -1.0,
            price: 45000.0,
            timestamp_ns: 0,
        };

        assert!(router.validate_order(&invalid_order).is_err());
    }

    #[test]
    fn test_symbol_mapping() {
        assert_eq!(symbol_to_id("BTCUSDT"), Some(0));
        assert_eq!(symbol_to_id("ETHUSDT"), Some(1));
        assert_eq!(id_to_symbol(0), Some("BTCUSDT"));
        assert_eq!(symbol_to_id("INVALID"), None);
    }
}
