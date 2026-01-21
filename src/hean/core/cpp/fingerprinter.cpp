/**
 * Algorithmic Fingerprinting Engine
 * Tracks latency signatures of large limit orders to identify institutional bots
 * Detects spoofing and iceberg patterns
 * Provides Predictive Alpha signals to the Swarm
 */

#include "fingerprinter.h"
#include <deque>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace hean {

// Order fingerprint structure
struct OrderFingerprint {
    std::string order_id;
    std::string symbol;
    double price;
    double size;
    int64_t timestamp_ns;
    int64_t first_seen_ns;
    int64_t last_update_ns;
    int update_count;
    bool is_limit;
    bool is_spoof;
    bool is_iceberg;
    double volatility_score;
    double latency_signature;  // Key metric: latency pattern
    std::vector<int64_t> update_timestamps;  // Track order update latencies
    std::deque<double> size_history;  // Track size changes (iceberg detection)
    
    OrderFingerprint() : price(0.0), size(0.0), timestamp_ns(0), 
                        first_seen_ns(0), last_update_ns(0), update_count(0),
                        is_limit(false), is_spoof(false), is_iceberg(false),
                        volatility_score(0.0), latency_signature(0.0) {}
};

// Bot signature structure (identified institutional patterns)
struct BotSignature {
    std::string bot_id;  // Derived from fingerprint
    double avg_latency_ms;
    double latency_stddev;
    double spoof_ratio;
    double iceberg_ratio;
    int total_orders;
    double price_impact_score;
    std::vector<std::string> preferred_symbols;
    std::deque<double> recent_alpha_signals;  // Track predictive accuracy
    
    BotSignature() : avg_latency_ms(0.0), latency_stddev(0.0), 
                    spoof_ratio(0.0), iceberg_ratio(0.0), total_orders(0),
                    price_impact_score(0.0) {}
};

class AlgoFingerprinterImpl {
private:
    std::mutex data_mutex_;
    
    // Active order fingerprints
    std::unordered_map<std::string, OrderFingerprint> active_orders_;
    
    // Identified bot signatures
    std::unordered_map<std::string, BotSignature> bot_signatures_;
    
    // Configuration
    double min_order_size_usdt_;  // Minimum size to fingerprint (e.g., $10k)
    double spoof_threshold_;  // Size reduction ratio to detect spoofing
    double latency_window_ms_;  // Window for latency signature calculation
    int min_updates_for_fingerprint_;  // Minimum updates to create fingerprint
    
    // Calculate latency signature from order update timestamps
    double calculate_latency_signature(const OrderFingerprint& order) {
        if (order.update_timestamps.size() < 3) {
            return 0.0;
        }
        
        // Calculate inter-update latencies
        std::vector<double> latencies;
        for (size_t i = 1; i < order.update_timestamps.size(); i++) {
            int64_t delta_ns = order.update_timestamps[i] - order.update_timestamps[i-1];
            double delta_ms = delta_ns / 1e6;
            latencies.push_back(delta_ms);
        }
        
        // Calculate mean and stddev
        double mean = 0.0;
        for (double lat : latencies) {
            mean += lat;
        }
        mean /= latencies.size();
        
        double variance = 0.0;
        for (double lat : latencies) {
            variance += (lat - mean) * (lat - mean);
        }
        variance /= latencies.size();
        double stddev = std::sqrt(variance);
        
        // Latency signature = normalized pattern (low stddev = consistent bot)
        // Combine mean and consistency
        double consistency_score = 1.0 / (1.0 + stddev);
        return mean * 0.5 + (100.0 / (1.0 + mean)) * 0.5 * consistency_score;
    }
    
    // Detect spoofing: large order that disappears quickly
    bool detect_spoofing(const OrderFingerprint& order) {
        if (order.size_history.size() < 3) {
            return false;
        }
        
        double max_size = *std::max_element(order.size_history.begin(), order.size_history.end());
        double current_size = order.size_history.back();
        double reduction_ratio = current_size / max_size;
        
        // If size reduced by > 80% within short time, likely spoofing
        if (reduction_ratio < 0.2 && order.update_count < 5) {
            int64_t lifetime_ms = (order.last_update_ns - order.first_seen_ns) / 1e6;
            if (lifetime_ms < 2000) {  // Within 2 seconds
                return true;
            }
        }
        
        return false;
    }
    
    // Detect iceberg: order that refills after partial fill
    bool detect_iceberg(const OrderFingerprint& order) {
        if (order.size_history.size() < 5) {
            return false;
        }
        
        // Look for pattern: size decreases then increases (refill)
        int refill_count = 0;
        for (size_t i = 1; i < order.size_history.size() - 1; i++) {
            double prev = order.size_history[i-1];
            double curr = order.size_history[i];
            double next = order.size_history[i+1];
            
            // Decrease then increase (refill pattern)
            if (curr < prev * 0.8 && next > curr * 1.2) {
                refill_count++;
            }
        }
        
        // If we see multiple refill patterns, it's an iceberg
        return refill_count >= 2;
    }
    
    // Match order to existing bot signature or create new one
    std::string match_or_create_bot_signature(const OrderFingerprint& order) {
        // Simple matching: compare latency signature to known bots
        double order_signature = order.latency_signature;
        
        for (auto& [bot_id, bot] : bot_signatures_) {
            // If latency signature is similar (within 20%), match
            if (bot.avg_latency_ms > 0.0) {
                double diff = std::abs(order_signature - bot.avg_latency_ms);
                if (diff < bot.avg_latency_ms * 0.2 && bot.total_orders > 10) {
                    // Update bot signature with new data
                    bot.total_orders++;
                    bot.avg_latency_ms = (bot.avg_latency_ms * (bot.total_orders - 1) + order.latency_signature) / bot.total_orders;
                    if (order.is_spoof) bot.spoof_ratio = (bot.spoof_ratio * (bot.total_orders - 1) + 1.0) / bot.total_orders;
                    else bot.spoof_ratio = (bot.spoof_ratio * (bot.total_orders - 1)) / bot.total_orders;
                    if (order.is_iceberg) bot.iceberg_ratio = (bot.iceberg_ratio * (bot.total_orders - 1) + 1.0) / bot.total_orders;
                    else bot.iceberg_ratio = (bot.iceberg_ratio * (bot.total_orders - 1)) / bot.total_orders;
                    
                    // Add symbol to preferred symbols if not already there
                    bool symbol_exists = false;
                    for (const auto& sym : bot.preferred_symbols) {
                        if (sym == order.symbol) {
                            symbol_exists = true;
                            break;
                        }
                    }
                    if (!symbol_exists) {
                        bot.preferred_symbols.push_back(order.symbol);
                    }
                    
                    return bot_id;
                }
            }
        }
        
        // Create new bot signature
        std::string new_bot_id = "BOT_" + std::to_string(bot_signatures_.size() + 1);
        BotSignature new_bot;
        new_bot.bot_id = new_bot_id;
        new_bot.avg_latency_ms = order.latency_signature;
        new_bot.latency_stddev = 0.0;
        new_bot.total_orders = 1;
        new_bot.spoof_ratio = order.is_spoof ? 1.0 : 0.0;
        new_bot.iceberg_ratio = order.is_iceberg ? 1.0 : 0.0;
        new_bot.price_impact_score = 0.0;
        new_bot.preferred_symbols.push_back(order.symbol);
        
        bot_signatures_[new_bot_id] = new_bot;
        return new_bot_id;
    }
    
public:
    AlgoFingerprinterImpl() : min_order_size_usdt_(10000.0),  // $10k minimum
                             spoof_threshold_(0.2),
                             latency_window_ms_(5000.0),
                             min_updates_for_fingerprint_(3) {}
    
    void update_order(const char* order_id, const char* symbol, double price, 
                     double size, int64_t timestamp_ns, int is_limit) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // Only fingerprint large limit orders
        double order_value_usdt = price * size;
        if (!is_limit || order_value_usdt < min_order_size_usdt_) {
            return;
        }
        
        std::string oid(order_id);
        std::string sym(symbol);
        
        auto it = active_orders_.find(oid);
        if (it == active_orders_.end()) {
            // New order
            OrderFingerprint new_order;
            new_order.order_id = oid;
            new_order.symbol = sym;
            new_order.price = price;
            new_order.size = size;
            new_order.timestamp_ns = timestamp_ns;
            new_order.first_seen_ns = timestamp_ns;
            new_order.last_update_ns = timestamp_ns;
            new_order.update_count = 1;
            new_order.is_limit = true;
            new_order.update_timestamps.push_back(timestamp_ns);
            new_order.size_history.push_back(size);
            
            active_orders_[oid] = new_order;
        } else {
            // Update existing order
            OrderFingerprint& order = it->second;
            order.last_update_ns = timestamp_ns;
            order.update_count++;
            order.size = size;
            order.update_timestamps.push_back(timestamp_ns);
            order.size_history.push_back(size);
            
            // Maintain window size
            if (order.update_timestamps.size() > 20) {
                order.update_timestamps.erase(order.update_timestamps.begin());
            }
            if (order.size_history.size() > 20) {
                order.size_history.pop_front();
            }
            
            // Calculate latency signature when we have enough data
            if (order.update_count >= min_updates_for_fingerprint_) {
                order.latency_signature = calculate_latency_signature(order);
                order.is_spoof = detect_spoofing(order);
                order.is_iceberg = detect_iceberg(order);
            }
        }
    }
    
    void remove_order(const char* order_id) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        std::string oid(order_id);
        auto it = active_orders_.find(oid);
        if (it != active_orders_.end()) {
            OrderFingerprint& order = it->second;
            
            // Final fingerprint analysis before removing
            if (order.update_count >= min_updates_for_fingerprint_) {
                order.latency_signature = calculate_latency_signature(order);
                order.is_spoof = detect_spoofing(order);
                order.is_iceberg = detect_iceberg(order);
                
                // Match to bot or create new signature
                match_or_create_bot_signature(order);
            }
            
            active_orders_.erase(it);
        }
    }
    
    // Get predictive alpha signal: expected next move of identified bot
    int get_predictive_alpha(const char* symbol, double* alpha_signal, 
                            double* confidence, char* bot_id_out, int max_bot_id_len) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        std::string sym(symbol);
        
        // Find most active bot for this symbol
        std::string best_bot_id;
        double best_score = 0.0;
        
        for (auto& [bid, bot] : bot_signatures_) {
            // Check if this bot trades this symbol
            bool trades_symbol = false;
            for (const auto& pref_sym : bot.preferred_symbols) {
                if (pref_sym == sym) {
                    trades_symbol = true;
                    break;
                }
            }
            if (!trades_symbol) {
                // If bot has preferred symbols, check for partial match
                if (!bot.preferred_symbols.empty()) {
                    bool partial_match = false;
                    for (const auto& pref_sym : bot.preferred_symbols) {
                        if (sym.find(pref_sym) != std::string::npos || pref_sym.find(sym) != std::string::npos) {
                            partial_match = true;
                            break;
                        }
                    }
                    if (!partial_match) {
                        continue;
                    }
                } else {
                    continue;  // No preferred symbols, skip
                }
            }
            
            // Calculate bot activity score (higher = more predictive)
            double activity_score = bot.total_orders * (1.0 - bot.spoof_ratio) * bot.iceberg_ratio;
            if (activity_score > best_score) {
                best_score = activity_score;
                best_bot_id = bid;
            }
        }
        
        if (best_bot_id.empty() || best_score < 5.0) {
            *alpha_signal = 0.0;
            *confidence = 0.0;
            return 0;  // No signal
        }
        
        // Predict based on bot pattern
        BotSignature& bot = bot_signatures_[best_bot_id];
        
        // If bot has high iceberg ratio, expect continuation (icebergs refill)
        // If bot has high spoof ratio, expect reversal (spoofs disappear)
        if (bot.iceberg_ratio > 0.5) {
            *alpha_signal = 1.0;  // Bullish (iceberg buying pressure)
            *confidence = bot.iceberg_ratio;
        } else if (bot.spoof_ratio > 0.3) {
            *alpha_signal = -1.0;  // Bearish (spoof selling pressure)
            *confidence = bot.spoof_ratio;
        } else {
            *alpha_signal = 0.0;
            *confidence = 0.0;
        }
        
        // Copy bot ID
        strncpy(bot_id_out, best_bot_id.c_str(), max_bot_id_len - 1);
        bot_id_out[max_bot_id_len - 1] = '\0';
        
        return 1;  // Signal generated
    }
    
    int get_active_orders_count() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        return static_cast<int>(active_orders_.size());
    }
    
    int get_identified_bots_count() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        return static_cast<int>(bot_signatures_.size());
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        active_orders_.clear();
        bot_signatures_.clear();
    }
};

// Global instance
static AlgoFingerprinterImpl* g_fingerprinter = nullptr;

}  // namespace hean

// C interface
extern "C" {
    void algo_fingerprinter_init() {
        if (hean::g_fingerprinter == nullptr) {
            hean::g_fingerprinter = new hean::AlgoFingerprinterImpl();
        }
    }
    
    void algo_fingerprinter_update_order(const char* order_id, const char* symbol,
                                        double price, double size, int64_t timestamp_ns,
                                        int is_limit) {
        if (hean::g_fingerprinter != nullptr) {
            hean::g_fingerprinter->update_order(order_id, symbol, price, size, timestamp_ns, is_limit);
        }
    }
    
    void algo_fingerprinter_remove_order(const char* order_id) {
        if (hean::g_fingerprinter != nullptr) {
            hean::g_fingerprinter->remove_order(order_id);
        }
    }
    
    int algo_fingerprinter_get_predictive_alpha(const char* symbol, double* alpha_signal,
                                                double* confidence, char* bot_id_out, int max_bot_id_len) {
        if (hean::g_fingerprinter == nullptr) {
            return 0;
        }
        return hean::g_fingerprinter->get_predictive_alpha(symbol, alpha_signal, confidence, 
                                                           bot_id_out, max_bot_id_len);
    }
    
    int algo_fingerprinter_get_active_orders_count() {
        if (hean::g_fingerprinter == nullptr) {
            return 0;
        }
        return hean::g_fingerprinter->get_active_orders_count();
    }
    
    int algo_fingerprinter_get_identified_bots_count() {
        if (hean::g_fingerprinter == nullptr) {
            return 0;
        }
        return hean::g_fingerprinter->get_identified_bots_count();
    }
    
    void algo_fingerprinter_cleanup() {
        if (hean::g_fingerprinter != nullptr) {
            hean::g_fingerprinter->cleanup();
            delete hean::g_fingerprinter;
            hean::g_fingerprinter = nullptr;
        }
    }
}