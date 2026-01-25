use actix_web::web;
use futures::{SinkExt, StreamExt};
use log::{error, info, warn};
use serde_json::Value;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::{AppState, Kline, Ticker};

pub struct BinanceWsClient;

impl BinanceWsClient {
    pub async fn start(symbol: &str, state: web::Data<AppState>) -> anyhow::Result<()> {
        let symbol_lower = symbol.to_lowercase();

        // Binance WebSocket URL for ticker stream
        let url = format!(
            "wss://stream.binance.com:9443/ws/{}@ticker",
            symbol_lower
        );

        info!("📡 Connecting to Binance WebSocket: {}", url);

        loop {
            match connect_async(&url).await {
                Ok((ws_stream, _)) => {
                    info!("✅ Connected to Binance WebSocket for {}", symbol);

                    let (mut _write, mut read) = ws_stream.split();

                    // Read messages
                    while let Some(msg) = read.next().await {
                        match msg {
                            Ok(Message::Text(text)) => {
                                if let Err(e) = Self::handle_message(&text, symbol, &state) {
                                    warn!("Failed to handle message: {}", e);
                                }
                            }
                            Ok(Message::Close(_)) => {
                                warn!("WebSocket closed for {}", symbol);
                                break;
                            }
                            Ok(Message::Ping(data)) => {
                                // Respond to ping with pong
                                if let Err(e) = _write.send(Message::Pong(data)).await {
                                    error!("Failed to send pong: {}", e);
                                }
                            }
                            Err(e) => {
                                error!("WebSocket error for {}: {}", symbol, e);
                                break;
                            }
                            _ => {}
                        }
                    }

                    warn!("WebSocket stream ended for {}, reconnecting...", symbol);
                }
                Err(e) => {
                    error!("Failed to connect to WebSocket: {}", e);
                }
            }

            // Wait before reconnecting
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            info!("Reconnecting to WebSocket for {}...", symbol);
        }
    }

    fn handle_message(
        text: &str,
        symbol: &str,
        state: &web::Data<AppState>,
    ) -> anyhow::Result<()> {
        let data: Value = serde_json::from_str(text)?;

        // Parse Binance 24hr ticker data
        // Example: {"e":"24hrTicker","E":1234567890,"s":"BTCUSDT","c":"45000.00",...}

        if data["e"] == "24hrTicker" {
            let price = data["c"]
                .as_str()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let volume = data["v"]
                .as_str()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let timestamp = data["E"].as_i64().unwrap_or(0);

            let ticker = Ticker {
                symbol: symbol.to_string(),
                price,
                volume,
                timestamp,
            };

            // Update shared state
            let mut tickers = state.tickers.lock().unwrap();
            tickers.insert(symbol.to_string(), ticker);

            info!(
                "📊 Updated ticker for {}: ${:.2} (vol: {:.2})",
                symbol, price, volume
            );
        }

        Ok(())
    }

    pub async fn start_kline_stream(
        symbol: &str,
        interval: &str,
        state: web::Data<AppState>,
    ) -> anyhow::Result<()> {
        let symbol_lower = symbol.to_lowercase();

        // Binance WebSocket URL for kline stream
        let url = format!(
            "wss://stream.binance.com:9443/ws/{}@kline_{}",
            symbol_lower, interval
        );

        info!("📡 Connecting to Binance Kline WebSocket: {}", url);

        loop {
            match connect_async(&url).await {
                Ok((ws_stream, _)) => {
                    info!("✅ Connected to Binance Kline WebSocket for {}", symbol);

                    let (mut _write, mut read) = ws_stream.split();

                    while let Some(msg) = read.next().await {
                        match msg {
                            Ok(Message::Text(text)) => {
                                if let Err(e) = Self::handle_kline_message(&text, symbol, &state) {
                                    warn!("Failed to handle kline message: {}", e);
                                }
                            }
                            Ok(Message::Close(_)) => {
                                warn!("Kline WebSocket closed for {}", symbol);
                                break;
                            }
                            Err(e) => {
                                error!("Kline WebSocket error for {}: {}", symbol, e);
                                break;
                            }
                            _ => {}
                        }
                    }

                    warn!(
                        "Kline WebSocket stream ended for {}, reconnecting...",
                        symbol
                    );
                }
                Err(e) => {
                    error!("Failed to connect to Kline WebSocket: {}", e);
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    }

    fn handle_kline_message(
        text: &str,
        symbol: &str,
        state: &web::Data<AppState>,
    ) -> anyhow::Result<()> {
        let data: Value = serde_json::from_str(text)?;

        // Parse Binance kline data
        if let Some(k) = data["k"].as_object() {
            let kline = Kline {
                symbol: symbol.to_string(),
                open: k["o"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0),
                high: k["h"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0),
                low: k["l"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0),
                close: k["c"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0),
                volume: k["v"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0),
                timestamp: k["t"].as_i64().unwrap_or(0),
            };

            // Update shared state
            let mut klines = state.klines.lock().unwrap();
            let symbol_klines = klines.entry(symbol.to_string()).or_insert_with(Vec::new);

            // Keep only last 500 klines
            if symbol_klines.len() >= 500 {
                symbol_klines.remove(0);
            }

            symbol_klines.push(kline);

            info!("📈 Updated kline for {}: O:{:.2} H:{:.2} L:{:.2} C:{:.2}",
                symbol, kline.open, kline.high, kline.low, kline.close);
        }

        Ok(())
    }
}
