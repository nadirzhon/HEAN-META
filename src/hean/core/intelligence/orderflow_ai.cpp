/**
 * AI Micro-Structure Layer: OrderFlow Intelligence
 * 
 * - VPIN (Volume-weighted Probability of Informed Trading) Analysis
 * - Spoofing Detection via Cancel-to-Fill Ratio Monitoring
 * - Real-time Order Flow Toxicity Assessment
 */

#include "../../cpp/SIMDJSONParser.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>

// Forward declarations
struct OrderFlowEvent {
    int64_t timestamp_ns;
    char symbol[16];
    double price;
    double size;
    bool is_buy;
    bool is_trade;
    bool is_cancel;
    bool is_new_order;
};

struct VPINBucket {
    double volume_buy = 0.0;
    double volume_sell = 0.0;
    int64_t bucket_start_ns = 0;
    int64_t bucket_end_ns = 0;
};

struct SpoofingPattern {
    std::string symbol;
    int64_t detection_time_ns;
    double cancel_to_fill_ratio;
    int consecutive_cancels;
    double suspected_order_size;
    bool is_spoofing;
};

class OrderFlowAI {
private:
    // VPIN parameters
    static constexpr int VPIN_BUCKETS = 50;  // Number of volume buckets
    static constexpr double VPIN_THRESHOLD = 0.7;  // High VPIN threshold
    
    // Spoofing detection parameters
    static constexpr double SPOOF_CANCEL_TO_FILL_THRESHOLD = 5.0;  // 5x cancels to fills = suspicious
    static constexpr int SPOOF_CONSECUTIVE_CANCELS_THRESHOLD = 3;  // 3+ consecutive cancels
    static constexpr double SPOOF_ORDER_SIZE_MULTIPLIER = 10.0;  // Orders 10x+ average = suspicious
    
    // Lock-free ring buffer for order flow events (single producer, single consumer)
    // Note: Using mutex for simplicity; can be replaced with lock-free structure
    std::mutex orderflow_mutex_;
    std::deque<OrderFlowEvent> orderflow_events_;
    size_t max_events_ = 10000;
    
    // VPIN state per symbol
    std::map<std::string, std::deque<VPINBucket>> vpin_buckets_;
    std::map<std::string, double> current_vpin_;
    std::map<std::string, int64_t> last_bucket_end_ns_;
    
    // Spoofing detection state
    std::map<std::string, int> cancel_count_;
    std::map<std::string, int> fill_count_;
    std::map<std::string, std::deque<int64_t>> cancel_timestamps_;  // Track cancel timestamps
    std::map<std::string, double> avg_order_size_;
    std::map<std::string, std::vector<SpoofingPattern>> detected_spoofing_;
    
    // Order book state for iceberg detection
    std::map<std::string, std::map<double, double>> bid_levels_;  // price -> size
    std::map<std::string, std::map<double, double>> ask_levels_;
    std::map<std::string, std::vector<std::pair<double, double>>> detected_icebergs_;  // (price, suspected_size)
    
    // Force inline for hot-path functions
    __attribute__((always_inline))
    inline double calculate_volume_imbalance(double v_buy, double v_sell) const noexcept {
        double total = v_buy + v_sell;
        if (total < 1e-10) return 0.0;
        return std::abs(v_buy - v_sell) / total;
    }
    
    __attribute__((always_inline))
    inline double calculate_vpin_bucket(const VPINBucket& bucket) const noexcept {
        double total_volume = bucket.volume_buy + bucket.volume_sell;
        if (total_volume < 1e-10) return 0.0;
        return calculate_volume_imbalance(bucket.volume_buy, bucket.volume_sell);
    }
    
    void update_vpin(const std::string& symbol, const OrderFlowEvent& event) {
        std::lock_guard<std::mutex> lock(orderflow_mutex_);
        
        if (vpin_buckets_.find(symbol) == vpin_buckets_.end()) {
            vpin_buckets_[symbol] = std::deque<VPINBucket>();
            current_vpin_[symbol] = 0.0;
            last_bucket_end_ns_[symbol] = 0;
        }
        
        auto& buckets = vpin_buckets_[symbol];
        int64_t bucket_duration_ns = 60'000'000'000LL / VPIN_BUCKETS;  // 60 seconds / 50 buckets = 1.2s per bucket
        
        // Find or create appropriate bucket
        int64_t target_bucket_start = (event.timestamp_ns / bucket_duration_ns) * bucket_duration_ns;
        
        if (buckets.empty() || buckets.back().bucket_start_ns < target_bucket_start) {
            // Create new bucket
            VPINBucket new_bucket;
            new_bucket.bucket_start_ns = target_bucket_start;
            new_bucket.bucket_end_ns = target_bucket_start + bucket_duration_ns;
            
            if (event.is_trade) {
                if (event.is_buy) {
                    new_bucket.volume_buy = event.size;
                } else {
                    new_bucket.volume_sell = event.size;
                }
            }
            
            buckets.push_back(new_bucket);
            
            // Maintain window size
            while (buckets.size() > static_cast<size_t>(VPIN_BUCKETS)) {
                buckets.pop_front();
            }
        } else {
            // Add to current bucket
            if (event.is_trade) {
                if (event.is_buy) {
                    buckets.back().volume_buy += event.size;
                } else {
                    buckets.back().volume_sell += event.size;
                }
            }
        }
        
        // Calculate VPIN as average volume imbalance over all buckets
        if (buckets.size() > 0) {
            double sum_vpin = 0.0;
            int count = 0;
            for (const auto& bucket : buckets) {
                double vpin_bucket = calculate_vpin_bucket(bucket);
                sum_vpin += vpin_bucket;
                count++;
            }
            current_vpin_[symbol] = count > 0 ? sum_vpin / count : 0.0;
        }
        
        last_bucket_end_ns_[symbol] = target_bucket_start + bucket_duration_ns;
    }
    
    void detect_spoofing(const std::string& symbol, const OrderFlowEvent& event) {
        std::lock_guard<std::mutex> lock(orderflow_mutex_);
        
        if (cancel_count_.find(symbol) == cancel_count_.end()) {
            cancel_count_[symbol] = 0;
            fill_count_[symbol] = 0;
            avg_order_size_[symbol] = 0.0;
            cancel_timestamps_[symbol] = std::deque<int64_t>();
        }
        
        auto& cancels = cancel_count_[symbol];
        auto& fills = fill_count_[symbol];
        auto& cancel_ts = cancel_timestamps_[symbol];
        auto& avg_size = avg_order_size_[symbol];
        
        // Update cancel/fill counts
        if (event.is_cancel) {
            cancels++;
            cancel_ts.push_back(event.timestamp_ns);
            
            // Keep only recent cancels (last 60 seconds)
            int64_t cutoff_ns = event.timestamp_ns - 60'000'000'000LL;
            while (!cancel_ts.empty() && cancel_ts.front() < cutoff_ns) {
                cancel_ts.pop_front();
            }
        } else if (event.is_trade) {
            fills++;
        }
        
        // Update average order size (exponential moving average)
        if (event.is_new_order && event.size > 0) {
            if (avg_size < 1e-10) {
                avg_size = event.size;
            } else {
                avg_size = 0.9 * avg_size + 0.1 * event.size;  // EMA with alpha=0.1
            }
        }
        
        // Check for spoofing pattern
        bool is_spoofing = false;
        double cancel_to_fill_ratio = 0.0;
        
        if (fills > 0) {
            cancel_to_fill_ratio = static_cast<double>(cancels) / static_cast<double>(fills);
        } else if (cancels >= SPOOF_CONSECUTIVE_CANCELS_THRESHOLD) {
            cancel_to_fill_ratio = static_cast<double>(cancels);  // High ratio when no fills
        }
        
        // Check consecutive cancels
        int consecutive_cancels = static_cast<int>(cancel_ts.size());
        
        // Check for suspiciously large orders
        bool suspicious_size = false;
        if (event.is_new_order && avg_size > 1e-10) {
            suspicious_size = (event.size / avg_size) >= SPOOF_ORDER_SIZE_MULTIPLIER;
        }
        
        // Detect spoofing if multiple conditions met
        if (cancel_to_fill_ratio >= SPOOF_CANCEL_TO_FILL_THRESHOLD ||
            (consecutive_cancels >= SPOOF_CONSECUTIVE_CANCELS_THRESHOLD && suspicious_size)) {
            is_spoofing = true;
            
            SpoofingPattern pattern;
            pattern.symbol = symbol;
            pattern.detection_time_ns = event.timestamp_ns;
            pattern.cancel_to_fill_ratio = cancel_to_fill_ratio;
            pattern.consecutive_cancels = consecutive_cancels;
            pattern.suspected_order_size = event.size;
            pattern.is_spoofing = true;
            
            if (detected_spoofing_.find(symbol) == detected_spoofing_.end()) {
                detected_spoofing_[symbol] = std::vector<SpoofingPattern>();
            }
            
            detected_spoofing_[symbol].push_back(pattern);
            
            // Keep only recent detections (last 100)
            if (detected_spoofing_[symbol].size() > 100) {
                detected_spoofing_[symbol].erase(detected_spoofing_[symbol].begin());
            }
        }
    }
    
    void detect_iceberg_orders(const std::string& symbol, const OrderFlowEvent& event) {
        std::lock_guard<std::mutex> lock(orderflow_mutex_);
        
        // Update order book levels
        if (event.is_new_order) {
            auto& levels = event.is_buy ? bid_levels_[symbol] : ask_levels_[symbol];
            
            if (levels.find(event.price) == levels.end()) {
                levels[event.price] = 0.0;
            }
            levels[event.price] += event.size;
        } else if (event.is_cancel) {
            auto& levels = event.is_buy ? bid_levels_[symbol] : ask_levels_[symbol];
            
            if (levels.find(event.price) != levels.end()) {
                levels[event.price] = std::max(0.0, levels[event.price] - event.size);
                if (levels[event.price] < 1e-10) {
                    levels.erase(event.price);
                }
            }
        } else if (event.is_trade) {
            auto& levels = event.is_buy ? bid_levels_[symbol] : ask_levels_[symbol];
            
            if (levels.find(event.price) != levels.end()) {
                levels[event.price] = std::max(0.0, levels[event.price] - event.size);
            }
        }
        
        // Detect iceberg: large hidden size at a price level that replenishes quickly
        auto& levels = event.is_buy ? bid_levels_[symbol] : ask_levels_[symbol];
        double avg_level_size = 0.0;
        int level_count = 0;
        
        for (const auto& [price, size] : levels) {
            avg_level_size += size;
            level_count++;
        }
        
        if (level_count > 0) {
            avg_level_size /= level_count;
        }
        
        // Check for levels significantly larger than average (potential iceberg)
        for (const auto& [price, size] : levels) {
            if (size > avg_level_size * 5.0 && size > 0.0) {  // 5x average and non-zero
                // Check if this price level has been detected before
                auto& icebergs = detected_icebergs_[symbol];
                bool already_detected = false;
                for (const auto& [iceberg_price, iceberg_size] : icebergs) {
                    if (std::abs(iceberg_price - price) < 0.01) {  // Same price level
                        already_detected = true;
                        break;
                    }
                }
                
                if (!already_detected) {
                    icebergs.push_back({price, size});
                    
                    // Keep only recent detections (last 50)
                    if (icebergs.size() > 50) {
                        icebergs.erase(icebergs.begin());
                    }
                }
            }
        }
    }
    
public:
    OrderFlowAI() {
        // Initialize
    }
    
    ~OrderFlowAI() = default;
    
    /**
     * Process an order flow event
     */
    void process_event(const OrderFlowEvent& event) {
        std::string symbol_str(event.symbol);
        
        // Update VPIN
        if (event.is_trade) {
            update_vpin(symbol_str, event);
        }
        
        // Detect spoofing
        if (event.is_cancel || event.is_trade || event.is_new_order) {
            detect_spoofing(symbol_str, event);
            detect_iceberg_orders(symbol_str, event);
        }
        
        // Store event
        {
            std::lock_guard<std::mutex> lock(orderflow_mutex_);
            orderflow_events_.push_back(event);
            if (orderflow_events_.size() > max_events_) {
                orderflow_events_.pop_front();
            }
        }
    }
    
    /**
     * Get current VPIN for a symbol
     */
    double get_vpin(const char* symbol) const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(orderflow_mutex_));
        std::string symbol_str(symbol);
        auto it = current_vpin_.find(symbol_str);
        return (it != current_vpin_.end()) ? it->second : 0.0;
    }
    
    /**
     * Check if VPIN is high (indicates informed trading)
     */
    bool is_high_vpin(const char* symbol) const {
        return get_vpin(symbol) >= VPIN_THRESHOLD;
    }
    
    /**
     * Get cancel-to-fill ratio for spoofing detection
     */
    double get_cancel_to_fill_ratio(const char* symbol) const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(orderflow_mutex_));
        std::string symbol_str(symbol);
        
        auto cancel_it = cancel_count_.find(symbol_str);
        auto fill_it = fill_count_.find(symbol_str);
        
        int cancels = (cancel_it != cancel_count_.end()) ? cancel_it->second : 0;
        int fills = (fill_it != fill_count_.end()) ? fill_it->second : 0;
        
        if (fills > 0) {
            return static_cast<double>(cancels) / static_cast<double>(fills);
        } else if (cancels > 0) {
            return static_cast<double>(cancels);  // High ratio when no fills
        }
        
        return 0.0;
    }
    
    /**
     * Get number of detected spoofing patterns for a symbol
     */
    int get_spoofing_count(const char* symbol) const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(orderflow_mutex_));
        std::string symbol_str(symbol);
        auto it = detected_spoofing_.find(symbol_str);
        return (it != detected_spoofing_.end()) ? static_cast<int>(it->second.size()) : 0;
    }
    
    /**
     * Get detected iceberg orders for a symbol
     */
    int get_iceberg_count(const char* symbol) const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(orderflow_mutex_));
        std::string symbol_str(symbol);
        auto it = detected_icebergs_.find(symbol_str);
        return (it != detected_icebergs_.end()) ? static_cast<int>(it->second.size()) : 0;
    }
    
    /**
     * Get iceberg order details (price and suspected size)
     */
    int get_iceberg_details(const char* symbol, double* prices, double* sizes, int max_count) const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(orderflow_mutex_));
        std::string symbol_str(symbol);
        auto it = detected_icebergs_.find(symbol_str);
        
        if (it == detected_icebergs_.end() || it->second.empty()) {
            return 0;
        }
        
        int count = 0;
        for (const auto& [price, size] : it->second) {
            if (count >= max_count) break;
            prices[count] = price;
            sizes[count] = size;
            count++;
        }
        
        return count;
    }
};

// Global instance
static OrderFlowAI* g_orderflow_ai = nullptr;
static std::mutex g_instance_mutex;

extern "C" {
    void orderflow_ai_init() {
        std::lock_guard<std::mutex> lock(g_instance_mutex);
        if (g_orderflow_ai == nullptr) {
            g_orderflow_ai = new OrderFlowAI();
        }
    }
    
    void orderflow_ai_process_event(
        int64_t timestamp_ns,
        const char* symbol,
        double price,
        double size,
        int is_buy,
        int is_trade,
        int is_cancel,
        int is_new_order
    ) {
        if (g_orderflow_ai == nullptr) {
            orderflow_ai_init();
        }
        
        OrderFlowEvent event;
        event.timestamp_ns = timestamp_ns;
        std::strncpy(event.symbol, symbol, sizeof(event.symbol) - 1);
        event.symbol[sizeof(event.symbol) - 1] = '\0';
        event.price = price;
        event.size = size;
        event.is_buy = (is_buy != 0);
        event.is_trade = (is_trade != 0);
        event.is_cancel = (is_cancel != 0);
        event.is_new_order = (is_new_order != 0);
        
        g_orderflow_ai->process_event(event);
    }
    
    double orderflow_ai_get_vpin(const char* symbol) {
        if (g_orderflow_ai == nullptr) {
            return 0.0;
        }
        return g_orderflow_ai->get_vpin(symbol);
    }
    
    int orderflow_ai_is_high_vpin(const char* symbol) {
        if (g_orderflow_ai == nullptr) {
            return 0;
        }
        return g_orderflow_ai->is_high_vpin(symbol) ? 1 : 0;
    }
    
    double orderflow_ai_get_cancel_to_fill_ratio(const char* symbol) {
        if (g_orderflow_ai == nullptr) {
            return 0.0;
        }
        return g_orderflow_ai->get_cancel_to_fill_ratio(symbol);
    }
    
    int orderflow_ai_get_spoofing_count(const char* symbol) {
        if (g_orderflow_ai == nullptr) {
            return 0;
        }
        return g_orderflow_ai->get_spoofing_count(symbol);
    }
    
    int orderflow_ai_get_iceberg_count(const char* symbol) {
        if (g_orderflow_ai == nullptr) {
            return 0;
        }
        return g_orderflow_ai->get_iceberg_count(symbol);
    }
    
    int orderflow_ai_get_iceberg_details(const char* symbol, double* prices, double* sizes, int max_count) {
        if (g_orderflow_ai == nullptr) {
            return 0;
        }
        return g_orderflow_ai->get_iceberg_details(symbol, prices, sizes, max_count);
    }
    
    void orderflow_ai_cleanup() {
        std::lock_guard<std::mutex> lock(g_instance_mutex);
        if (g_orderflow_ai != nullptr) {
            delete g_orderflow_ai;
            g_orderflow_ai = nullptr;
        }
    }
}