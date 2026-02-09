use pyo3::prelude::*;

/// Calculate market temperature: T = KE / N where KE = Sum (dP_i * V_i)^2
#[pyfunction]
pub fn market_temperature(prices: Vec<f64>, volumes: Vec<f64>) -> PyResult<f64> {
    if prices.len() < 2 {
        return Ok(0.0);
    }

    if prices.len() != volumes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Prices and volumes must have same length",
        ));
    }

    let n = prices.len() as f64;
    let mut kinetic_energy = 0.0;

    for i in 1..prices.len() {
        let price_change = prices[i] - prices[i - 1];
        let volume = volumes[i];
        kinetic_energy += (price_change * volume).powi(2);
    }

    Ok(kinetic_energy / n)
}

/// Calculate market entropy: S = -Sum p_i * log(p_i)
#[pyfunction]
pub fn market_entropy(volumes: Vec<f64>) -> PyResult<f64> {
    if volumes.is_empty() {
        return Ok(0.0);
    }

    let total_volume: f64 = volumes.iter().sum();
    if total_volume == 0.0 {
        return Ok(0.0);
    }

    let mut entropy = 0.0;
    for &vol in &volumes {
        if vol > 0.0 {
            let p = vol / total_volume;
            entropy -= p * p.ln();
        }
    }

    Ok(entropy)
}

/// Detect market phase based on temperature and entropy
#[pyfunction]
pub fn detect_phase(
    temperature: f64,
    entropy: f64,
    temp_history: Vec<f64>,
    entropy_history: Vec<f64>,
) -> PyResult<String> {
    let temp_threshold = if !temp_history.is_empty() {
        let avg: f64 = temp_history.iter().sum::<f64>() / temp_history.len() as f64;
        avg * 1.5
    } else {
        1.0
    };

    let entropy_threshold = if !entropy_history.is_empty() {
        let avg: f64 = entropy_history.iter().sum::<f64>() / entropy_history.len() as f64;
        avg * 1.2
    } else {
        2.0
    };

    let phase = if temperature < temp_threshold * 0.5 && entropy < entropy_threshold * 0.8 {
        "ICE"
    } else if temperature > temp_threshold && entropy > entropy_threshold {
        "VAPOR"
    } else if temperature > temp_threshold * 0.7 {
        "WATER"
    } else {
        "TRANSITION"
    };

    Ok(phase.to_string())
}

/// Calculate Szilard profit: Work = k * T * ln(2) * information_bits
#[pyfunction]
pub fn szilard_profit(temperature: f64, information_bits: f64) -> PyResult<f64> {
    const K_CRYPTO: f64 = 0.1;
    const LN_2: f64 = 0.693147180559945;
    Ok(K_CRYPTO * temperature * LN_2 * information_bits)
}

/// Calculate information content from prediction accuracy
#[pyfunction]
pub fn information_bits(prediction_accuracy: f64) -> PyResult<f64> {
    if prediction_accuracy <= 0.0 || prediction_accuracy >= 1.0 {
        return Ok(0.0);
    }
    Ok(-(1.0 - prediction_accuracy).log2())
}

/// Calculate thermal efficiency: eta = W / Q_in
#[pyfunction]
pub fn thermal_efficiency(profit: f64, risk_capital: f64) -> PyResult<f64> {
    if risk_capital <= 0.0 {
        return Ok(0.0);
    }
    Ok(profit / risk_capital)
}
