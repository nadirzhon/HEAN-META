/**
 * Orderbook Toxicity Detector Implementation
 */

#include "ToxicityDetector.h"
#include "ELM_Regressor.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <deque>
#include <map>

ToxicityDetector::ToxicityDetector()
    : elm_(6, 100)  // 6 input features, 100 hidden neurons
{
}

ToxicityDetector::~ToxicityDetector() {
}

void ToxicityDetector::update_orderbook(const std::string& symbol,
                                       const std::vector<double>& bid_prices,
                                       const std::vector<double>& bid_sizes,
                                       const std::vector<double>& ask_prices,
                                       const std::vector<double>& ask_sizes,
                                       int64_t timestamp_ns) {
    OrderbookSnapshot snapshot;
    snapshot.timestamp_ns = timestamp_ns;
    
    // Store bid levels
    int num_bids = std::min(bid_prices.size(), bid_sizes.size());
    snapshot.bids.reserve(num_bids);
    for (int i = 0; i < num_bids; i++) {
        snapshot.bids.push_back(OrderbookLevel(bid_prices[i], bid_sizes[i], timestamp_ns));
    }
    
    // Store ask levels
    int num_asks = std::min(ask_prices.size(), ask_sizes.size());
    snapshot.asks.reserve(num_asks);
    for (int i = 0; i < num_asks; i++) {
        snapshot.asks.push_back(OrderbookLevel(ask_prices[i], ask_sizes[i], timestamp_ns));
    }
    
    // Calculate mid price
    if (!snapshot.bids.empty() && !snapshot.asks.empty()) {
        snapshot.mid_price = (snapshot.bids[0].price + snapshot.asks[0].price) / 2.0;
    }
    
    // Add to history
    auto& history = orderbook_history_[symbol];
    history.push_back(snapshot);
    
    // Keep only last N snapshots
    if (history.size() > HISTORY_SIZE) {
        history.pop_front();
    }
    
    // Extract features and predict using ELM
    if (history.size() >= 2) {
        // Get current and previous snapshots
        const OrderbookSnapshot& current = history.back();
        const OrderbookSnapshot& previous = history[history.size() - 2];
        
        // Extract features
        std::vector<double> bid_sizes_vec;
        std::vector<double> ask_sizes_vec;
        for (const auto& bid : current.bids) {
            bid_sizes_vec.push_back(bid.size);
        }
        for (const auto& ask : current.asks) {
            ask_sizes_vec.push_back(ask.size);
        }
        
        double ofi = ELM_Regressor::calculate_ofi(
            bid_sizes_vec.data(),
            ask_sizes_vec.data(),
            std::min(bid_sizes_vec.size(), ask_sizes_vec.size())
        );
        
        // Extract full feature vector
        std::vector<double> bid_prices_vec, bid_sizes_vec_full, ask_prices_vec, ask_sizes_vec_full;
        for (const auto& bid : current.bids) {
            bid_prices_vec.push_back(bid.price);
            bid_sizes_vec_full.push_back(bid.size);
        }
        for (const auto& ask : current.asks) {
            ask_prices_vec.push_back(ask.price);
            ask_sizes_vec_full.push_back(ask.size);
        }
        
        std::vector<double> features = ELM_Regressor::extract_features(
            bid_prices_vec.data(), bid_sizes_vec_full.data(), bid_prices_vec.size(),
            ask_prices_vec.data(), ask_sizes_vec_full.data(), ask_prices_vec.size(),
            previous.mid_price
        );
        
        // Predict price movement using ELM
        if (features.size() == 6) {
            double predicted_movement = elm_.predict(features);
            
            // Calculate actual price movement
            double actual_movement = 0.0;
            if (previous.mid_price > 0.0) {
                actual_movement = (current.mid_price - previous.mid_price) / previous.mid_price;
            }
        }
    }
}

ToxicitySignal ToxicityDetector::detect_toxicity(const std::string& symbol) const {
    ToxicitySignal signal;
    signal.symbol = symbol;
    signal.timestamp_ns = get_timestamp_ns();
    
    auto it = orderbook_history_.find(symbol);
    if (it == orderbook_history_.end() || it->second.size() < 2) {
        return signal;  // Not enough history
    }
    
    const auto& history = it->second;
    const OrderbookSnapshot& current = history.back();
    const OrderbookSnapshot& previous = history[history.size() - 2];
    
    // Extract features and predict
    std::vector<double> bid_prices, bid_sizes, ask_prices, ask_sizes;
    for (const auto& bid : current.bids) {
        bid_prices.push_back(bid.price);
        bid_sizes.push_back(bid.size);
    }
    for (const auto& ask : current.asks) {
        ask_prices.push_back(ask.price);
        ask_sizes.push_back(ask.size);
    }
    
    std::vector<double> features = ELM_Regressor::extract_features(
        bid_prices.data(), bid_sizes.data(), bid_prices.size(),
        ask_prices.data(), ask_sizes.data(), ask_prices.size(),
        previous.mid_price
    );
    
    if (features.size() != 6) {
        return signal;
    }
    
    // Predict price movement
    double predicted_movement = elm_.predict(features);
    
    // Calculate actual movement
    double actual_movement = 0.0;
    if (previous.mid_price > 0.0) {
        actual_movement = (current.mid_price - previous.mid_price) / previous.mid_price;
    }
    
    signal.predicted_price_movement = predicted_movement;
    signal.actual_price_movement = actual_movement;
    
    // Detect spoofing
    signal.spoofing_probability = calculate_spoofing_probability(
        symbol, predicted_movement, actual_movement
    );
    
    // Detect layering
    signal.layering_probability = calculate_layering_probability(symbol);
    
    // Determine if fake order
    signal.is_fake_order = (signal.spoofing_probability > SPOOFING_THRESHOLD) ||
                          (signal.layering_probability > LAYERING_THRESHOLD);
    
    return signal;
}

bool ToxicityDetector::is_fake_order(const std::string& symbol, double price, 
                                     double size, bool is_bid) const {
    // Check for suspicious order patterns
    if (is_suspicious_order_pattern(symbol, price, size, is_bid)) {
        return true;
    }
    
    // Check toxicity signal
    ToxicitySignal signal = detect_toxicity(symbol);
    return signal.is_fake_order;
}

double ToxicityDetector::calculate_spoofing_probability(const std::string& symbol,
                                                       double predicted_movement,
                                                       double actual_movement) const {
    // Spoofing: Large predicted movement but small actual movement
    return elm_.detect_spoofing(predicted_movement, actual_movement, 0.002);
}

double ToxicityDetector::calculate_layering_probability(const std::string& symbol) const {
    auto it = orderbook_history_.find(symbol);
    if (it == orderbook_history_.end() || it->second.size() < 10) {
        return 0.0;
    }
    
    const auto& history = it->second;
    
    // Layering: Multiple large orders at similar prices that disappear quickly
    double layering_score = 0.0;
    
    // Check last few snapshots for rapid order appearance/disappearance
    for (size_t i = history.size() - 1; i >= std::max(0UL, history.size() - 10); i--) {
        const OrderbookSnapshot& snapshot = history[i];
        
        // Check for large orders that appear and disappear quickly
        for (const auto& bid : snapshot.bids) {
            if (bid.size > 10.0) {  // Large order threshold
                // Check if it disappears in next snapshot
                if (i + 1 < history.size()) {
                    const OrderbookSnapshot& next = history[i + 1];
                    bool found = false;
                    for (const auto& next_bid : next.bids) {
                        if (std::abs(next_bid.price - bid.price) < 0.01 && 
                            std::abs(next_bid.size - bid.size) < 0.1) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        layering_score += 0.1;  // Order disappeared quickly
                    }
                }
            }
        }
    }
    
    return std::min(1.0, layering_score);
}

bool ToxicityDetector::is_suspicious_order_pattern(const std::string& symbol,
                                                   double price, double size,
                                                   bool is_bid) const {
    auto it = orderbook_history_.find(symbol);
    if (it == orderbook_history_.end() || it->second.empty()) {
        return false;
    }
    
    const OrderbookSnapshot& current = it->second.back();
    
    // Check for orders that are much larger than average
    double avg_size = 0.0;
    int count = 0;
    
    for (const auto& level : (is_bid ? current.bids : current.asks)) {
        avg_size += level.size;
        count++;
    }
    
    if (count > 0) {
        avg_size /= count;
        
        // If order is 5x larger than average, suspicious
        if (size > avg_size * 5.0) {
            return true;
        }
    }
    
    // Check for orders at round numbers (common in spoofing)
    double price_mod = std::fmod(price, 100.0);
    if (price_mod < 1.0 || price_mod > 99.0) {
        // Near round number
        return size > 50.0;  // Large order at round number
    }
    
    return false;
}

int64_t ToxicityDetector::get_timestamp_ns() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

// C interface for Python bindings
extern "C" {
    static ToxicityDetector* g_detector = nullptr;
    
    void toxicity_detector_init() {
        if (g_detector == nullptr) {
            g_detector = new ToxicityDetector();
        }
    }
    
    void toxicity_detector_update_orderbook(
        const char* symbol,
        const double* bid_prices,
        const double* bid_sizes,
        int num_bids,
        const double* ask_prices,
        const double* ask_sizes,
        int num_asks,
        int64_t timestamp_ns
    ) {
        if (g_detector) {
            std::vector<double> bid_p(bid_prices, bid_prices + num_bids);
            std::vector<double> bid_s(bid_sizes, bid_sizes + num_bids);
            std::vector<double> ask_p(ask_prices, ask_prices + num_asks);
            std::vector<double> ask_s(ask_sizes, ask_sizes + num_asks);
            
            g_detector->update_orderbook(
                std::string(symbol),
                bid_p, bid_s, ask_p, ask_s,
                timestamp_ns
            );
        }
    }
    
    int toxicity_detector_is_fake_order(
        const char* symbol,
        double price,
        double size,
        int is_bid
    ) {
        if (g_detector) {
            return g_detector->is_fake_order(
                std::string(symbol),
                price, size, is_bid != 0
            ) ? 1 : 0;
        }
        return 0;
    }
    
    void toxicity_detector_cleanup() {
        if (g_detector) {
            delete g_detector;
            g_detector = nullptr;
        }
    }
}
