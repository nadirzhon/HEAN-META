use pyo3::prelude::*;
use crate::types::OrderbookState;

/// Parse orderbook and compute key metrics
#[pyfunction]
pub fn parse_orderbook(
    bids: Vec<(f64, f64)>,
    asks: Vec<(f64, f64)>,
) -> PyResult<OrderbookState> {
    if bids.is_empty() || asks.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bids and asks cannot be empty",
        ));
    }

    let best_bid = bids[0].0;
    let best_ask = asks[0].0;
    let mid_price = (best_bid + best_ask) / 2.0;
    let spread = best_ask - best_bid;

    let bid_depth: f64 = bids.iter().map(|(_, vol)| vol).sum();
    let ask_depth: f64 = asks.iter().map(|(_, vol)| vol).sum();
    let total_volume = bid_depth + ask_depth;

    let imbalance = if total_volume > 0.0 {
        (bid_depth - ask_depth) / total_volume
    } else {
        0.0
    };

    let bid_weighted: f64 = bids.iter().map(|(price, vol)| price * vol).sum();
    let ask_weighted: f64 = asks.iter().map(|(price, vol)| price * vol).sum();
    let depth_weighted_price = if total_volume > 0.0 {
        (bid_weighted + ask_weighted) / total_volume
    } else {
        mid_price
    };

    Ok(OrderbookState::new(
        mid_price,
        spread,
        bid_depth,
        ask_depth,
        imbalance,
        depth_weighted_price,
        total_volume,
    ))
}

/// Calculate volume-weighted average price
#[pyfunction]
pub fn vwap(levels: Vec<(f64, f64)>) -> PyResult<f64> {
    if levels.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Levels cannot be empty",
        ));
    }

    let total_volume: f64 = levels.iter().map(|(_, vol)| vol).sum();
    if total_volume == 0.0 {
        return Ok(levels[0].0);
    }

    let weighted_sum: f64 = levels.iter().map(|(price, vol)| price * vol).sum();
    Ok(weighted_sum / total_volume)
}

/// Calculate orderbook imbalance at different depth levels
#[pyfunction]
pub fn depth_imbalance(
    bids: Vec<(f64, f64)>,
    asks: Vec<(f64, f64)>,
    depth_levels: Vec<usize>,
) -> PyResult<Vec<f64>> {
    let mut imbalances = Vec::new();

    for &level in &depth_levels {
        let bid_vol: f64 = bids.iter().take(level).map(|(_, vol)| vol).sum();
        let ask_vol: f64 = asks.iter().take(level).map(|(_, vol)| vol).sum();
        let total = bid_vol + ask_vol;

        let imbalance = if total > 0.0 {
            (bid_vol - ask_vol) / total
        } else {
            0.0
        };
        imbalances.push(imbalance);
    }

    Ok(imbalances)
}
