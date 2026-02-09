//
//  Models.swift
//  HEAN
//
//  Central export file for all data models
//  Created on 2026-01-31.
//

// Re-export all models for convenient importing
// Usage: import Models

// Market Models
// - Market: Trading pair data with price, volume, changes
// - Candle: OHLCV candlestick data for charts

// Trading Models
// - Position: Open position with PnL tracking
// - PositionSide: LONG or SHORT
// - Order: Order data with status and fills
// - OrderSide: BUY or SELL
// - OrderType: MARKET, LIMIT, STOP_MARKET, STOP_LIMIT
// - OrderStatus: NEW, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED, EXPIRED

// Portfolio Models
// - Portfolio: Account equity, balance, margin, PnL

// Event Models
// - TradingEvent: Trading activity events with timestamps
// - EventType: Signal, order, position, risk, system events

// WebSocket Models
// - WebSocketState: Connection state
// - WebSocketHealth: Connection health metrics

// All models are available by importing this file or individually
