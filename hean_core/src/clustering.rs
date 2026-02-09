use pyo3::prelude::*;
use crate::types::ParticipantBreakdown;

/// Classify market participants based on trading behavior
#[pyfunction]
pub fn classify_participants(
    trade_sizes: Vec<f64>,
    trade_times: Vec<f64>,
    is_aggressive: Vec<bool>,
) -> PyResult<ParticipantBreakdown> {
    if trade_sizes.len() != trade_times.len() || trade_sizes.len() != is_aggressive.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All input vectors must have same length",
        ));
    }

    if trade_sizes.is_empty() {
        return Ok(ParticipantBreakdown::new(0.0, 0.0, 0.0, 0.0, 0.0));
    }

    let n = trade_sizes.len();
    let total_volume: f64 = trade_sizes.iter().sum();

    let mut sorted_sizes = trade_sizes.clone();
    sorted_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted_sizes[n / 2];
    let p90 = sorted_sizes[(n * 90) / 100];
    let p95 = sorted_sizes[(n * 95) / 100];

    let mut mm_volume = 0.0;
    let mut institutional_volume = 0.0;
    let mut retail_volume = 0.0;
    let mut whale_volume = 0.0;
    let mut arb_volume = 0.0;

    let mut time_diffs = Vec::new();
    for i in 1..trade_times.len() {
        time_diffs.push(trade_times[i] - trade_times[i - 1]);
    }
    let avg_time_diff = if !time_diffs.is_empty() {
        time_diffs.iter().sum::<f64>() / time_diffs.len() as f64
    } else {
        1.0
    };

    for i in 0..n {
        let size = trade_sizes[i];
        let aggressive = is_aggressive[i];
        let time_diff = if i > 0 {
            trade_times[i] - trade_times[i - 1]
        } else {
            avg_time_diff
        };

        if !aggressive && size < p90 && time_diff < avg_time_diff * 2.0 {
            mm_volume += size;
        }
        if size >= p95 {
            whale_volume += size;
        }
        if size >= p90 && size < p95 {
            institutional_volume += size;
        }
        if size < p50 {
            retail_volume += size;
        }
        if aggressive && time_diff < avg_time_diff * 0.5 {
            arb_volume += size;
        }
    }

    let safe_div = |a: f64, b: f64| -> f64 {
        if b > 0.0 { a / b } else { 0.0 }
    };

    Ok(ParticipantBreakdown::new(
        safe_div(mm_volume, total_volume),
        safe_div(institutional_volume, total_volume),
        safe_div(retail_volume, total_volume),
        safe_div(whale_volume, total_volume),
        safe_div(arb_volume, total_volume),
    ))
}

/// Calculate Herfindahl-Hirschman Index for market concentration
#[pyfunction]
pub fn market_concentration(trade_sizes: Vec<f64>) -> PyResult<f64> {
    if trade_sizes.is_empty() {
        return Ok(0.0);
    }

    let total_volume: f64 = trade_sizes.iter().sum();
    if total_volume == 0.0 {
        return Ok(0.0);
    }

    let mut hhi = 0.0;
    for &size in &trade_sizes {
        let share = size / total_volume;
        hhi += share * share;
    }

    Ok(hhi)
}

/// Detect coordinated trading patterns
#[pyfunction]
pub fn detect_coordination(
    trade_sizes: Vec<f64>,
    trade_times: Vec<f64>,
    similarity_threshold: f64,
    time_window: f64,
) -> PyResult<f64> {
    if trade_sizes.len() != trade_times.len() || trade_sizes.len() < 2 {
        return Ok(0.0);
    }

    let mut coordinated_volume = 0.0;
    let total_volume: f64 = trade_sizes.iter().sum();

    for i in 0..trade_sizes.len() {
        let mut cluster_volume = trade_sizes[i];
        let cluster_start = trade_times[i];

        for j in (i + 1)..trade_sizes.len() {
            let time_diff = trade_times[j] - cluster_start;
            if time_diff > time_window {
                break;
            }

            let size_ratio = trade_sizes[j] / trade_sizes[i];
            if size_ratio >= (1.0 - similarity_threshold)
                && size_ratio <= (1.0 + similarity_threshold)
            {
                cluster_volume += trade_sizes[j];
            }
        }

        if cluster_volume > trade_sizes[i] * 3.0 {
            coordinated_volume += cluster_volume;
        }
    }

    Ok(if total_volume > 0.0 {
        coordinated_volume / total_volume
    } else {
        0.0
    })
}
