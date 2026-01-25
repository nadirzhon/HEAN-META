use actix_web::{get, web, App, HttpResponse, HttpServer, Responder};
use actix_cors::Cors;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

mod websocket;
mod api;
mod indicators;

use websocket::binance::BinanceWsClient;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<(f64, f64)>,  // (price, quantity)
    pub asks: Vec<(f64, f64)>,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub timestamp: i64,
}

// Shared state
pub struct AppState {
    pub tickers: Arc<Mutex<HashMap<String, Ticker>>>,
    pub orderbooks: Arc<Mutex<HashMap<String, OrderBook>>>,
    pub klines: Arc<Mutex<HashMap<String, Vec<Kline>>>>,
}

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "service": "market-data-service",
        "version": "1.0.0"
    }))
}

#[get("/api/v1/ticker/{symbol}")]
async fn get_ticker(
    symbol: web::Path<String>,
    data: web::Data<AppState>,
) -> impl Responder {
    let tickers = data.tickers.lock().unwrap();

    match tickers.get(&symbol.to_string()) {
        Some(ticker) => HttpResponse::Ok().json(ticker),
        None => HttpResponse::NotFound().json(serde_json::json!({
            "error": "Symbol not found"
        })),
    }
}

#[get("/api/v1/orderbook/{symbol}")]
async fn get_orderbook(
    symbol: web::Path<String>,
    data: web::Data<AppState>,
) -> impl Responder {
    let orderbooks = data.orderbooks.lock().unwrap();

    match orderbooks.get(&symbol.to_string()) {
        Some(orderbook) => HttpResponse::Ok().json(orderbook),
        None => HttpResponse::NotFound().json(serde_json::json!({
            "error": "Symbol not found"
        })),
    }
}

#[get("/api/v1/klines/{symbol}")]
async fn get_klines(
    symbol: web::Path<String>,
    data: web::Data<AppState>,
) -> impl Responder {
    let klines = data.klines.lock().unwrap();

    match klines.get(&symbol.to_string()) {
        Some(klines) => HttpResponse::Ok().json(klines),
        None => HttpResponse::NotFound().json(serde_json::json!({
            "error": "Symbol not found"
        })),
    }
}

#[get("/api/v1/indicators/{symbol}")]
async fn get_indicators(
    symbol: web::Path<String>,
    data: web::Data<AppState>,
) -> impl Responder {
    let klines = data.klines.lock().unwrap();

    match klines.get(&symbol.to_string()) {
        Some(klines_data) => {
            // Calculate indicators
            let closes: Vec<f64> = klines_data.iter().map(|k| k.close).collect();

            if closes.len() < 14 {
                return HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Insufficient data for indicators"
                }));
            }

            let sma_20 = indicators::sma(&closes, 20);
            let rsi_14 = indicators::rsi(&closes, 14);

            HttpResponse::Ok().json(serde_json::json!({
                "symbol": symbol.to_string(),
                "sma_20": sma_20,
                "rsi_14": rsi_14,
                "timestamp": chrono::Utc::now().timestamp_millis()
            }))
        }
        None => HttpResponse::NotFound().json(serde_json::json!({
            "error": "Symbol not found"
        })),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    info!("🚀 Starting Market Data Service...");

    // Create shared state
    let state = web::Data::new(AppState {
        tickers: Arc::new(Mutex::new(HashMap::new())),
        orderbooks: Arc::new(Mutex::new(HashMap::new())),
        klines: Arc::new(Mutex::new(HashMap::new())),
    });

    // Start WebSocket clients (background tasks)
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    for symbol in symbols {
        let state_clone = state.clone();
        let symbol_str = symbol.to_string();

        tokio::spawn(async move {
            info!("📡 Starting WebSocket for {}", symbol_str);
            match BinanceWsClient::start(&symbol_str, state_clone).await {
                Ok(_) => info!("✅ WebSocket for {} completed", symbol_str),
                Err(e) => warn!("❌ WebSocket for {} failed: {}", symbol_str, e),
            }
        });
    }

    info!("🌐 Starting HTTP server on 0.0.0.0:8080");

    // Start HTTP server
    HttpServer::new(move || {
        let cors = Cors::permissive();

        App::new()
            .wrap(cors)
            .app_data(state.clone())
            .service(health)
            .service(get_ticker)
            .service(get_orderbook)
            .service(get_klines)
            .service(get_indicators)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
