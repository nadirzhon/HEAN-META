use pyo3::prelude::*;

mod clustering;
mod orderbook;
mod thermodynamics;
mod types;

use clustering::*;
use orderbook::*;
use thermodynamics::*;
use types::*;

/// HEAN Core - High-performance Rust module for trading system
#[pymodule]
fn hean_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Orderbook functions
    m.add_function(wrap_pyfunction!(parse_orderbook, m)?)?;
    m.add_function(wrap_pyfunction!(vwap, m)?)?;
    m.add_function(wrap_pyfunction!(depth_imbalance, m)?)?;

    // Thermodynamics functions
    m.add_function(wrap_pyfunction!(market_temperature, m)?)?;
    m.add_function(wrap_pyfunction!(market_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(detect_phase, m)?)?;
    m.add_function(wrap_pyfunction!(szilard_profit, m)?)?;
    m.add_function(wrap_pyfunction!(information_bits, m)?)?;
    m.add_function(wrap_pyfunction!(thermal_efficiency, m)?)?;

    // Clustering functions
    m.add_function(wrap_pyfunction!(classify_participants, m)?)?;
    m.add_function(wrap_pyfunction!(market_concentration, m)?)?;
    m.add_function(wrap_pyfunction!(detect_coordination, m)?)?;

    // Types
    m.add_class::<OrderbookState>()?;
    m.add_class::<ParticipantBreakdown>()?;
    m.add_class::<MarketPhase>()?;

    Ok(())
}
