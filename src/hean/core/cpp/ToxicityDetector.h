/**
 * Orderbook Toxicity Detector
 * 
 * Detects spoofing and layering patterns in orderbooks using:
 * - ELM-based prediction vs actual price movement
 * - Order size/time analysis
 * - Depth imbalance anomalies
 */

#ifndef TOXICITY_DETECTOR_H
#define TOXICITY_DETECTOR_H

#include "ELM_Regressor.h"
#include <deque>
#include <map>
#include <string>
#include <vector>
#include <chrono>

/**
 * Orderbook level
 */
struct OrderbookLevel {
    double price;
    double size;
    int64_t timestamp_ns;
    
    OrderbookLevel() : price(0.0), size(0.0), timestamp_ns(0) {}
    OrderbookLevel(double p, double s, int64_t ts) 
        : price(p), size(s), timestamp_ns(ts) {}
};

/**
 * Toxicity signal
 */
struct ToxicitySignal {
    std::string symbol;
    double spoofing_probability;  // 0-1, probability of spoofing
    double layering_probability;  // 0-1, probability of layering
    bool is_fake_order;  // True if detected as fake
    double predicted_price_movement;  // ELM prediction
    double actual_price_movement;  // Actual observed movement
    int64_t timestamp_ns;
    
    ToxicitySignal() : spoofing_probability(0.0), layering_probability(0.0),
                      is_fake_order(false), predicted_price_movement(0.0),
                      actual_price_movement(0.0), timestamp_ns(0) {}
};

/**
 * Orderbook Toxicity Detector
 */
class ToxicityDetector {
public:
    ToxicityDetector();
    ~ToxicityDetector();
    
    /**
     * Update orderbook snapshot
     */
    void update_orderbook(const std::string& symbol,
                         const std::vector<double>& bid_prices,
                         const std::vector<double>& bid_sizes,
                         const std::vector<double>& ask_prices,
                         const std::vector<double>& ask_sizes,
                         int64_t timestamp_ns);
    
    /**
     * Detect toxicity (spoofing/layering) in current orderbook
     */
    ToxicitySignal detect_toxicity(const std::string& symbol) const;
    
    /**
     * Check if a large order is fake (for immediate trade against it)
     */
    bool is_fake_order(const std::string& symbol, double price, double size, bool is_bid) const;
    
    /**
     * Get ELM regressor for OFI prediction
     */
    ELM_Regressor& get_elm() { return elm_; }
    
private:
    // Orderbook snapshots (circular buffer for history)
    struct OrderbookSnapshot {
        std::vector<OrderbookLevel> bids;
        std::vector<OrderbookLevel> asks;
        int64_t timestamp_ns;
        double mid_price;
        
        OrderbookSnapshot() : timestamp_ns(0), mid_price(0.0) {}
    };
    
    // Orderbook history per symbol (last N snapshots)
    std::map<std::string, std::deque<OrderbookSnapshot>> orderbook_history_;
    
    // ELM regressor for OFI prediction
    ELM_Regressor elm_;
    
    // Configuration
    static constexpr int HISTORY_SIZE = 100;  // Keep last 100 snapshots
    static constexpr double SPOOFING_THRESHOLD = 0.7;  // Probability threshold
    static constexpr double LAYERING_THRESHOLD = 0.6;
    
    // Internal methods
    double calculate_spoofing_probability(const std::string& symbol, 
                                         double predicted_movement,
                                         double actual_movement) const;
    double calculate_layering_probability(const std::string& symbol) const;
    bool is_suspicious_order_pattern(const std::string& symbol, 
                                    double price, double size, bool is_bid) const;
    int64_t get_timestamp_ns() const;
};

#endif // TOXICITY_DETECTOR_H
