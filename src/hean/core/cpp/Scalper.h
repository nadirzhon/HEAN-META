/**
 * Profit-Extraction Mode: Sub-second Scalping Engine
 * 
 * Strategy: Sub-second scalping with 0.02% profit target per trade.
 * Frequency: Unlimited - if gap exists, hit it.
 * Risk: Hard-stop calculated in ticks, not percentages.
 */

#ifndef SCALPER_H
#define SCALPER_H

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>

struct ScalpSignal {
    std::string symbol;
    bool is_long;  // true = buy, false = sell
    double entry_price;
    double target_price;  // 0.02% profit target
    double stop_loss_tick;  // Hard-stop in ticks (not percentage)
    int64_t signal_timestamp_ns;
    
    ScalpSignal() : is_long(true), entry_price(0.0), target_price(0.0),
                   stop_loss_tick(0.0), signal_timestamp_ns(0) {}
};

/**
 * Scalping engine for sub-second profit extraction
 */
class Scalper {
public:
    Scalper();
    ~Scalper();
    
    // Configuration
    void set_profit_target_pct(double target_pct);  // Default: 0.02% (0.0002)
    void set_max_position_size(double max_size);
    void set_tick_size(const std::string& symbol, double tick_size);
    void set_hard_stop_ticks(const std::string& symbol, double stop_ticks);  // Hard-stop in ticks
    
    // Price updates
    void update_price(const std::string& symbol, double price, double bid, double ask,
                     int64_t timestamp_ns);
    
    // Check for scalping opportunities
    ScalpSignal check_opportunity(const std::string& symbol) const;
    
    // Execute scalping trade
    bool execute_scalp(const ScalpSignal& signal);
    
    // Position management
    void close_position(const std::string& symbol);
    bool has_position(const std::string& symbol) const;
    
    // Statistics
    int64_t get_total_trades() const { return total_trades_.load(); }
    int64_t get_winning_trades() const { return winning_trades_.load(); }
    int64_t get_losing_trades() const { return losing_trades_.load(); }
    double get_total_profit() const { return total_profit_.load(); }
    double get_win_rate() const;
    int64_t get_avg_trade_duration_ns() const;
    
    // Callback for executed trades
    typedef void (*TradeCallback)(const ScalpSignal& signal, double profit, void* user_data);
    void set_trade_callback(TradeCallback callback, void* user_data);
    
private:
    // Configuration
    double profit_target_pct_;  // Default: 0.02% (0.0002)
    double max_position_size_;
    std::map<std::string, double> tick_sizes_;  // Per-symbol tick sizes
    std::map<std::string, double> hard_stop_ticks_;  // Per-symbol hard-stop in ticks
    std::mutex config_mutex_;
    
    // Price data
    struct PriceData {
        double price;
        double bid;
        double ask;
        int64_t timestamp_ns;
        
        PriceData() : price(0.0), bid(0.0), ask(0.0), timestamp_ns(0) {}
    };
    
    std::map<std::string, PriceData> prices_;
    std::mutex prices_mutex_;
    
    // Open positions
    struct Position {
        ScalpSignal signal;
        double entry_price;
        double target_price;
        double stop_loss_price;
        int64_t entry_timestamp_ns;
        
        Position() : entry_price(0.0), target_price(0.0), stop_loss_price(0.0),
                    entry_timestamp_ns(0) {}
    };
    
    std::map<std::string, Position> positions_;
    std::mutex positions_mutex_;
    
    // Statistics
    std::atomic<int64_t> total_trades_;
    std::atomic<int64_t> winning_trades_;
    std::atomic<int64_t> losing_trades_;
    std::atomic<double> total_profit_;
    std::atomic<int64_t> total_trade_duration_ns_;
    std::atomic<int64_t> trade_count_;
    
    // Callback
    TradeCallback trade_callback_;
    void* callback_user_data_;
    
    // Internal methods
    double calculate_target_price(double entry_price, bool is_long) const;
    double calculate_stop_loss_price(double entry_price, bool is_long, 
                                     const std::string& symbol) const;
    void update_positions();
    int64_t get_timestamp_ns() const;
};

#endif // SCALPER_H
