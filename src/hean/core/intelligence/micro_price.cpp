/**
 * Micro-Price Prediction System
 * VAMP (Volume Adjusted Mid Price) + 100ms Linear Regression Prediction
 * Target: >80% accuracy with nanosecond execution
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <array>

// Maximum orderbook depth to consider
#define MAX_DEPTH 50
#define PREDICTION_WINDOW_MS 100
#define MIN_VOLUME_THRESHOLD 0.0001

// Orderbook entry: [price, volume]
struct OrderbookEntry {
    double price;
    double volume;
};

// Pre-allocated orderbook snapshot (no dynamic allocation)
struct OrderbookSnapshot {
    OrderbookEntry bids[MAX_DEPTH];
    OrderbookEntry asks[MAX_DEPTH];
    int bid_count;
    int ask_count;
    int64_t timestamp_ns;
};

// Linear regression model for 100ms prediction
struct PredictionModel {
    double slope;      // Price change per ms
    double intercept;  // Current price estimate
    double r_squared;  // Model confidence
    int64_t last_update_ns;
};

// VAMP calculation result
struct VAMPResult {
    double vamp_price;      // Volume Adjusted Mid Price
    double imbalance_ratio; // Order flow imbalance (-1 to +1)
    double weighted_mid;    // Volume-weighted mid price
    double bid_volume;
    double ask_volume;
};

/**
 * Calculate VAMP (Volume Adjusted Mid Price)
 * Uses orderbook depth to find the "true" price accounting for volume
 */
VAMPResult calculate_vamp(const OrderbookSnapshot& snapshot) {
    VAMPResult result;
    result.bid_volume = 0.0;
    result.ask_volume = 0.0;
    result.vamp_price = 0.0;
    
    // Calculate total bid and ask volume within first N levels
    for (int i = 0; i < snapshot.bid_count && i < MAX_DEPTH; ++i) {
        result.bid_volume += snapshot.bids[i].volume;
    }
    
    for (int i = 0; i < snapshot.ask_count && i < MAX_DEPTH; ++i) {
        result.ask_volume += snapshot.asks[i].volume;
    }
    
    // Calculate volume-weighted mid price
    double total_volume = result.bid_volume + result.ask_volume;
    if (total_volume < MIN_VOLUME_THRESHOLD) {
        // Fallback to simple mid price
        if (snapshot.bid_count > 0 && snapshot.ask_count > 0) {
            result.vamp_price = (snapshot.bids[0].price + snapshot.asks[0].price) * 0.5;
            result.weighted_mid = result.vamp_price;
        }
        result.imbalance_ratio = 0.0;
        return result;
    }
    
    // Calculate volume-weighted bid and ask prices
    double weighted_bid = 0.0;
    double weighted_ask = 0.0;
    
    double bid_sum = 0.0;
    for (int i = 0; i < snapshot.bid_count && i < MAX_DEPTH; ++i) {
        double weight = snapshot.bids[i].volume / result.bid_volume;
        weighted_bid += snapshot.bids[i].price * weight;
        bid_sum += weight;
    }
    if (bid_sum > 0.0) {
        weighted_bid /= bid_sum;
    }
    
    double ask_sum = 0.0;
    for (int i = 0; i < snapshot.ask_count && i < MAX_DEPTH; ++i) {
        double weight = snapshot.asks[i].volume / result.ask_volume;
        weighted_ask += snapshot.asks[i].price * weight;
        ask_sum += weight;
    }
    if (ask_sum > 0.0) {
        weighted_ask /= ask_sum;
    }
    
    // Calculate order flow imbalance (-1 to +1)
    // Positive = more bid volume (bullish), Negative = more ask volume (bearish)
    result.imbalance_ratio = (result.bid_volume - result.ask_volume) / total_volume;
    
    // VAMP = weighted mid + imbalance adjustment
    result.weighted_mid = (weighted_bid + weighted_ask) * 0.5;
    double spread = weighted_ask - weighted_bid;
    if (spread > 0.0) {
        // Adjust price based on imbalance (move towards side with more volume)
        result.vamp_price = result.weighted_mid + (spread * 0.5 * result.imbalance_ratio);
    } else {
        result.vamp_price = result.weighted_mid;
    }
    
    return result;
}

/**
 * Update linear regression model for 100ms prediction
 * Uses exponential moving average for fast adaptation
 */
void update_prediction_model(
    PredictionModel& model,
    double current_price,
    double previous_price,
    int64_t time_delta_ns
) {
    if (time_delta_ns <= 0) {
        return;
    }
    
    // Calculate price change rate (per nanosecond)
    double price_delta = current_price - previous_price;
    double time_delta_ms = static_cast<double>(time_delta_ns) / 1e6; // Convert to ms
    
    if (time_delta_ms < 0.001) { // Less than 1 microsecond
        return; // Skip if time delta too small
    }
    
    // Calculate instantaneous slope (price change per ms)
    double instant_slope = price_delta / time_delta_ms;
    
    // Exponential moving average for slope (alpha = 0.3 for fast adaptation)
    const double alpha = 0.3;
    if (model.last_update_ns > 0) {
        model.slope = alpha * instant_slope + (1.0 - alpha) * model.slope;
    } else {
        model.slope = instant_slope;
    }
    
    model.intercept = current_price;
    model.last_update_ns = current_price; // Reuse field for tracking
    
    // Calculate R-squared approximation (simplified)
    // Higher confidence if slope is consistent
    double slope_variance = std::abs(instant_slope - model.slope);
    double max_slope = std::max(std::abs(instant_slope), std::abs(model.slope));
    if (max_slope > 0.0) {
        model.r_squared = 1.0 - (slope_variance / max_slope);
        model.r_squared = std::max(0.0, std::min(1.0, model.r_squared));
    } else {
        model.r_squared = 0.5; // Neutral confidence
    }
}

/**
 * Predict price 100ms into the future
 * @param model: Prediction model
 * @param current_vamp: Current VAMP price
 * @return Predicted price in 100ms
 */
double predict_price_100ms(const PredictionModel& model, double current_vamp) {
    if (model.last_update_ns == 0) {
        return current_vamp; // No prediction available
    }
    
    // Predict price = intercept + slope * time_horizon_ms
    double predicted = model.intercept + (model.slope * PREDICTION_WINDOW_MS);
    
    // Apply confidence weighting
    double confidence_weighted = current_vamp * (1.0 - model.r_squared) + predicted * model.r_squared;
    
    return confidence_weighted;
}

/**
 * Micro-price prediction engine (main interface)
 */
class MicroPricePredictor {
private:
    PredictionModel model_;
    std::array<double, 100> price_history_;  // Circular buffer
    std::array<int64_t, 100> time_history_;
    int history_index_;
    double last_price_;
    int64_t last_timestamp_ns_;
    
public:
    MicroPricePredictor() 
        : history_index_(0)
        , last_price_(0.0)
        , last_timestamp_ns_(0)
    {
        model_.slope = 0.0;
        model_.intercept = 0.0;
        model_.r_squared = 0.0;
        model_.last_update_ns = 0;
        price_history_.fill(0.0);
        time_history_.fill(0);
    }
    
    /**
     * Update with new orderbook snapshot
     * @param snapshot: Orderbook snapshot
     * @return VAMP result
     */
    VAMPResult update(const OrderbookSnapshot& snapshot) {
        VAMPResult vamp = calculate_vamp(snapshot);
        
        if (last_timestamp_ns_ > 0 && vamp.vamp_price > 0.0) {
            int64_t time_delta = snapshot.timestamp_ns - last_timestamp_ns_;
            update_prediction_model(model_, vamp.vamp_price, last_price_, time_delta);
        }
        
        // Update history
        price_history_[history_index_] = vamp.vamp_price;
        time_history_[history_index_] = snapshot.timestamp_ns;
        history_index_ = (history_index_ + 1) % 100;
        
        last_price_ = vamp.vamp_price;
        last_timestamp_ns_ = snapshot.timestamp_ns;
        
        return vamp;
    }
    
    /**
     * Predict price 100ms into the future
     */
    double predict_100ms() const {
        if (last_price_ <= 0.0) {
            return 0.0;
        }
        return predict_price_100ms(model_, last_price_);
    }
    
    /**
     * Get prediction confidence (0.0 to 1.0)
     */
    double get_confidence() const {
        return model_.r_squared;
    }
    
    /**
     * Get current VAMP price
     */
    double get_current_vamp() const {
        return last_price_;
    }
};

// C interface for Python bindings
extern "C" {
    static MicroPricePredictor* g_predictor = nullptr;
    
    void micro_price_init() {
        if (g_predictor == nullptr) {
            g_predictor = new MicroPricePredictor();
        }
    }
    
    double micro_price_update(
        const double* bids, const double* bid_volumes,
        const double* asks, const double* ask_volumes,
        int bid_count, int ask_count,
        int64_t timestamp_ns
    ) {
        if (g_predictor == nullptr) {
            micro_price_init();
        }
        
        OrderbookSnapshot snapshot;
        snapshot.bid_count = std::min(bid_count, MAX_DEPTH);
        snapshot.ask_count = std::min(ask_count, MAX_DEPTH);
        snapshot.timestamp_ns = timestamp_ns;
        
        for (int i = 0; i < snapshot.bid_count; ++i) {
            snapshot.bids[i].price = bids[i * 2];
            snapshot.bids[i].volume = bid_volumes[i];
        }
        
        for (int i = 0; i < snapshot.ask_count; ++i) {
            snapshot.asks[i].price = asks[i * 2];
            snapshot.asks[i].volume = ask_volumes[i];
        }
        
        VAMPResult vamp = g_predictor->update(snapshot);
        return vamp.vamp_price;
    }
    
    double micro_price_predict_100ms() {
        if (g_predictor == nullptr) {
            return 0.0;
        }
        return g_predictor->predict_100ms();
    }
    
    double micro_price_get_confidence() {
        if (g_predictor == nullptr) {
            return 0.0;
        }
        return g_predictor->get_confidence();
    }
    
    void micro_price_cleanup() {
        if (g_predictor != nullptr) {
            delete g_predictor;
            g_predictor = nullptr;
        }
    }
}
