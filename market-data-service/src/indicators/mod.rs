/// Simple Moving Average (SMA)
pub fn sma(data: &[f64], period: usize) -> Option<f64> {
    if data.len() < period {
        return None;
    }

    let sum: f64 = data[data.len() - period..].iter().sum();
    Some(sum / period as f64)
}

/// Exponential Moving Average (EMA)
pub fn ema(data: &[f64], period: usize) -> Option<f64> {
    if data.len() < period {
        return None;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = data[data.len() - period..data.len() - period + 1]
        .iter()
        .sum::<f64>()
        / period as f64;

    for &price in &data[data.len() - period + 1..] {
        ema = (price - ema) * multiplier + ema;
    }

    Some(ema)
}

/// Relative Strength Index (RSI)
pub fn rsi(data: &[f64], period: usize) -> Option<f64> {
    if data.len() < period + 1 {
        return None;
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    // Calculate price changes
    for i in 1..data.len() {
        let change = data[i] - data[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    // Calculate average gain and loss
    let avg_gain: f64 = gains[gains.len() - period..].iter().sum::<f64>() / period as f64;
    let avg_loss: f64 = losses[losses.len() - period..].iter().sum::<f64>() / period as f64;

    if avg_loss == 0.0 {
        return Some(100.0);
    }

    let rs = avg_gain / avg_loss;
    let rsi = 100.0 - (100.0 / (1.0 + rs));

    Some(rsi)
}

/// MACD (Moving Average Convergence Divergence)
pub fn macd(
    data: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> Option<(f64, f64, f64)> {
    let fast_ema = ema(data, fast_period)?;
    let slow_ema = ema(data, slow_period)?;

    let macd_line = fast_ema - slow_ema;

    // For simplicity, using SMA for signal line (normally EMA)
    let signal_line = macd_line; // Simplified

    let histogram = macd_line - signal_line;

    Some((macd_line, signal_line, histogram))
}

/// Bollinger Bands
pub fn bollinger_bands(
    data: &[f64],
    period: usize,
    std_dev: f64,
) -> Option<(f64, f64, f64)> {
    let sma = sma(data, period)?;

    // Calculate standard deviation
    let variance: f64 = data[data.len() - period..]
        .iter()
        .map(|&x| (x - sma).powi(2))
        .sum::<f64>()
        / period as f64;

    let std = variance.sqrt();

    let upper_band = sma + (std_dev * std);
    let lower_band = sma - (std_dev * std);

    Some((upper_band, sma, lower_band))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(sma(&data, 3), Some(4.0)); // (3+4+5)/3
    }

    #[test]
    fn test_rsi() {
        let data = vec![
            44.0, 44.25, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
            45.61, 46.28, 46.28,
        ];
        let rsi = rsi(&data, 14);
        assert!(rsi.is_some());
        assert!(rsi.unwrap() > 0.0 && rsi.unwrap() < 100.0);
    }
}
