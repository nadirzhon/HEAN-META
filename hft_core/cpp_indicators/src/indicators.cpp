#include <vector>
#include <cmath>
#include <immintrin.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// ============================================================================
// RSI - SIMD Optimized
// ============================================================================

std::vector<double> calculate_rsi(const std::vector<double>& prices, int period = 14) {
    const size_t n = prices.size();
    std::vector<double> rsi(n, 0.0);

    if (n < static_cast<size_t>(period + 1)) return rsi;

    // Calculate changes
    std::vector<double> changes(n - 1);
    for (size_t i = 0; i < n - 1; ++i) {
        changes[i] = prices[i + 1] - prices[i];
    }

    // Separate gains/losses with SIMD
    std::vector<double> gains(n - 1, 0.0);
    std::vector<double> losses(n - 1, 0.0);

    size_t i = 0;
    for (; i + 4 <= changes.size(); i += 4) {
        __m256d change = _mm256_loadu_pd(&changes[i]);
        __m256d zero = _mm256_setzero_pd();

        __m256d gain = _mm256_max_pd(change, zero);
        _mm256_storeu_pd(&gains[i], gain);

        __m256d loss = _mm256_sub_pd(zero, _mm256_min_pd(change, zero));
        _mm256_storeu_pd(&losses[i], loss);
    }

    for (; i < changes.size(); ++i) {
        gains[i] = std::max(changes[i], 0.0);
        losses[i] = std::max(-changes[i], 0.0);
    }

    // Calculate EMA
    double avg_gain = 0.0, avg_loss = 0.0;
    for (int i = 0; i < period; ++i) {
        avg_gain += gains[i];
        avg_loss += losses[i];
    }
    avg_gain /= period;
    avg_loss /= period;

    const double alpha = 1.0 / period;
    for (size_t i = period; i < n; ++i) {
        avg_gain = alpha * gains[i - 1] + (1 - alpha) * avg_gain;
        avg_loss = alpha * losses[i - 1] + (1 - alpha) * avg_loss;

        if (avg_loss == 0.0) {
            rsi[i] = 100.0;
        } else {
            double rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    return rsi;
}

// ============================================================================
// EMA - Fast Implementation
// ============================================================================

std::vector<double> calculate_ema(const std::vector<double>& data, int period) {
    std::vector<double> result(data.size());
    double multiplier = 2.0 / (period + 1.0);

    result[0] = data[0];
    for (size_t i = 1; i < data.size(); ++i) {
        result[i] = (data[i] - result[i-1]) * multiplier + result[i-1];
    }

    return result;
}

// ============================================================================
// MACD
// ============================================================================

struct MACDResult {
    std::vector<double> macd_line;
    std::vector<double> signal_line;
    std::vector<double> histogram;
};

MACDResult calculate_macd(
    const std::vector<double>& prices,
    int fast_period = 12,
    int slow_period = 26,
    int signal_period = 9
) {
    auto fast_ema = calculate_ema(prices, fast_period);
    auto slow_ema = calculate_ema(prices, slow_period);

    std::vector<double> macd_line(prices.size());
    for (size_t i = 0; i < prices.size(); ++i) {
        macd_line[i] = fast_ema[i] - slow_ema[i];
    }

    auto signal_line = calculate_ema(macd_line, signal_period);

    std::vector<double> histogram(prices.size());
    for (size_t i = 0; i < prices.size(); ++i) {
        histogram[i] = macd_line[i] - signal_line[i];
    }

    return {macd_line, signal_line, histogram};
}

// ============================================================================
// Bollinger Bands - SIMD Optimized
// ============================================================================

struct BBResult {
    std::vector<double> upper;
    std::vector<double> middle;
    std::vector<double> lower;
};

BBResult calculate_bollinger_bands(
    const std::vector<double>& prices,
    int period = 20,
    double num_std = 2.0
) {
    const size_t n = prices.size();
    BBResult result;
    result.upper.resize(n);
    result.middle.resize(n);
    result.lower.resize(n);

    for (size_t i = period - 1; i < n; ++i) {
        // Calculate SMA with SIMD
        double sum = 0.0;
        size_t j = i - period + 1;

        __m256d sum_vec = _mm256_setzero_pd();
        for (; j + 4 <= i + 1; j += 4) {
            __m256d prices_vec = _mm256_loadu_pd(&prices[j]);
            sum_vec = _mm256_add_pd(sum_vec, prices_vec);
        }

        double temp[4];
        _mm256_storeu_pd(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];

        for (; j <= i; ++j) {
            sum += prices[j];
        }

        double sma = sum / period;
        result.middle[i] = sma;

        // Calculate std dev with SIMD
        double variance = 0.0;
        j = i - period + 1;

        __m256d var_vec = _mm256_setzero_pd();
        __m256d sma_vec = _mm256_set1_pd(sma);

        for (; j + 4 <= i + 1; j += 4) {
            __m256d prices_vec = _mm256_loadu_pd(&prices[j]);
            __m256d diff = _mm256_sub_pd(prices_vec, sma_vec);
            __m256d sq = _mm256_mul_pd(diff, diff);
            var_vec = _mm256_add_pd(var_vec, sq);
        }

        _mm256_storeu_pd(temp, var_vec);
        variance = temp[0] + temp[1] + temp[2] + temp[3];

        for (; j <= i; ++j) {
            double diff = prices[j] - sma;
            variance += diff * diff;
        }

        double std_dev = std::sqrt(variance / period);

        result.upper[i] = sma + num_std * std_dev;
        result.lower[i] = sma - num_std * std_dev;
    }

    return result;
}

// ============================================================================
// Python Bindings
// ============================================================================

NB_MODULE(indicators_cpp, m) {
    m.doc() = "Ultra-fast technical indicators with SIMD optimization";

    m.def("rsi", &calculate_rsi,
          "prices"_a, "period"_a = 14,
          "Calculate RSI indicator (SIMD optimized)");

    m.def("ema", &calculate_ema,
          "data"_a, "period"_a,
          "Calculate EMA");

    m.def("macd", [](const std::vector<double>& prices, int fast, int slow, int signal) {
        auto result = calculate_macd(prices, fast, slow, signal);
        return std::make_tuple(result.macd_line, result.signal_line, result.histogram);
    }, "prices"_a, "fast_period"_a = 12, "slow_period"_a = 26, "signal_period"_a = 9,
       "Calculate MACD indicator");

    m.def("bollinger_bands", [](const std::vector<double>& prices, int period, double num_std) {
        auto result = calculate_bollinger_bands(prices, period, num_std);
        return std::make_tuple(result.upper, result.middle, result.lower);
    }, "prices"_a, "period"_a = 20, "num_std"_a = 2.0,
       "Calculate Bollinger Bands (SIMD optimized)");
}
