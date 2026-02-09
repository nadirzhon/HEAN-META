use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Orderbook state with key metrics
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderbookState {
    #[pyo3(get, set)]
    pub mid_price: f64,
    #[pyo3(get, set)]
    pub spread: f64,
    #[pyo3(get, set)]
    pub bid_depth: f64,
    #[pyo3(get, set)]
    pub ask_depth: f64,
    #[pyo3(get, set)]
    pub imbalance: f64,
    #[pyo3(get, set)]
    pub depth_weighted_price: f64,
    #[pyo3(get, set)]
    pub total_volume: f64,
}

#[pymethods]
impl OrderbookState {
    #[new]
    pub fn new(
        mid_price: f64,
        spread: f64,
        bid_depth: f64,
        ask_depth: f64,
        imbalance: f64,
        depth_weighted_price: f64,
        total_volume: f64,
    ) -> Self {
        Self {
            mid_price,
            spread,
            bid_depth,
            ask_depth,
            imbalance,
            depth_weighted_price,
            total_volume,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OrderbookState(mid={:.2}, spread={:.4}, imbalance={:.2})",
            self.mid_price, self.spread, self.imbalance
        )
    }
}

/// Participant breakdown by type
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParticipantBreakdown {
    #[pyo3(get, set)]
    pub mm_activity: f64,
    #[pyo3(get, set)]
    pub institutional_flow: f64,
    #[pyo3(get, set)]
    pub retail_ratio: f64,
    #[pyo3(get, set)]
    pub whale_presence: f64,
    #[pyo3(get, set)]
    pub arb_activity: f64,
}

#[pymethods]
impl ParticipantBreakdown {
    #[new]
    pub fn new(
        mm_activity: f64,
        institutional_flow: f64,
        retail_ratio: f64,
        whale_presence: f64,
        arb_activity: f64,
    ) -> Self {
        Self {
            mm_activity,
            institutional_flow,
            retail_ratio,
            whale_presence,
            arb_activity,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ParticipantBreakdown(MM={:.1}%, Inst={:.1}%, Retail={:.1}%, Whale={:.1}%, Arb={:.1}%)",
            self.mm_activity * 100.0,
            self.institutional_flow * 100.0,
            self.retail_ratio * 100.0,
            self.whale_presence * 100.0,
            self.arb_activity * 100.0
        )
    }
}

/// Market phase classification
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum MarketPhase {
    Ice,
    Water,
    Vapor,
    Transition,
}

#[pymethods]
impl MarketPhase {
    fn __repr__(&self) -> String {
        match self {
            MarketPhase::Ice => "ICE".to_string(),
            MarketPhase::Water => "WATER".to_string(),
            MarketPhase::Vapor => "VAPOR".to_string(),
            MarketPhase::Transition => "TRANSITION".to_string(),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}
