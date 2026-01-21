/**
 * "The Sniper" - Ultra-Low Latency Lead-Lag Arbitrage Engine
 * 
 * Direct simultaneous connections to Binance and Bybit streams.
 * Detects micro-price deltas and executes market orders in <2ms.
 */

#ifndef SNIPER_H
#define SNIPER_H

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Forward declarations
struct PriceUpdate;
struct ArbitrageSignal;

/**
 * Exchange identifiers
 */
enum class Exchange {
    BINANCE = 0,
    BYBIT = 1
};

/**
 * Price update structure with nanosecond timestamp
 */
struct PriceUpdate {
    std::string symbol;
    double price;
    double bid_price;
    double ask_price;
    int64_t timestamp_ns;  // Nanoseconds since epoch
    Exchange exchange;
    
    PriceUpdate() : price(0.0), bid_price(0.0), ask_price(0.0), 
                    timestamp_ns(0), exchange(Exchange::BINANCE) {}
};

/**
 * Arbitrage signal structure
 */
struct ArbitrageSignal {
    std::string symbol;
    double binance_price;
    double bybit_price;
    double price_delta_pct;  // Percentage difference
    double profit_estimate_pct;  // Estimated profit after fees
    int64_t signal_timestamp_ns;
    bool is_long_opportunity;  // true = buy Bybit, false = sell Bybit
    
    ArbitrageSignal() : binance_price(0.0), bybit_price(0.0),
                       price_delta_pct(0.0), profit_estimate_pct(0.0),
                       signal_timestamp_ns(0), is_long_opportunity(true) {}
};

/**
 * The Sniper - Ultra-fast arbitrage execution engine
 */
class Sniper {
public:
    Sniper();
    ~Sniper();
    
    // Configuration
    void set_delta_threshold(double threshold_pct);  // Default: 0.05% (0.0005)
    void set_execution_delay_limit_ns(int64_t max_delay_ns);  // Default: 2000000 (2ms)
    void set_max_position_size(double max_size);
    void set_maker_fee(double fee);  // Trading fee percentage
    void set_taker_fee(double fee);
    
    // Start/Stop
    bool start();
    void stop();
    bool is_running() const { return running_.load(); }
    
    // Price feed (called by WebSocket handlers)
    void update_binance_price(const std::string& symbol, double price, 
                             double bid, double ask, int64_t timestamp_ns);
    void update_bybit_price(const std::string& symbol, double price,
                           double bid, double ask, int64_t timestamp_ns);
    
    // Subscription
    void subscribe_symbol(const std::string& symbol);
    void unsubscribe_symbol(const std::string& symbol);
    
    // Statistics
    int64_t get_total_signals() const { return total_signals_.load(); }
    int64_t get_executed_trades() const { return executed_trades_.load(); }
    double get_total_profit() const { return total_profit_.load(); }
    int64_t get_avg_execution_time_ns() const;
    int64_t get_max_execution_time_ns() const { return max_execution_time_ns_.load(); }
    
    // Signal callback (register from Python)
    typedef void (*SignalCallback)(const ArbitrageSignal& signal, void* user_data);
    void set_signal_callback(SignalCallback callback, void* user_data);
    
private:
    // Configuration
    double delta_threshold_;  // Minimum price delta to trigger (default: 0.05%)
    int64_t execution_delay_limit_ns_;  // Maximum acceptable delay (2ms)
    double max_position_size_;
    double maker_fee_;
    double taker_fee_;
    
    // State
    std::atomic<bool> running_;
    std::map<std::string, PriceUpdate> binance_prices_;
    std::map<std::string, PriceUpdate> bybit_prices_;
    std::vector<std::string> subscribed_symbols_;
    std::mutex prices_mutex_;
    std::mutex symbols_mutex_;
    
    // Execution thread
    std::thread execution_thread_;
    std::atomic<bool> execution_thread_running_;
    
    // Callback
    SignalCallback signal_callback_;
    void* callback_user_data_;
    
    // Statistics
    std::atomic<int64_t> total_signals_;
    std::atomic<int64_t> executed_trades_;
    std::atomic<double> total_profit_;
    std::atomic<int64_t> max_execution_time_ns_;
    std::atomic<int64_t> total_execution_time_ns_;
    std::atomic<int64_t> execution_count_;
    
    // Internal methods
    void execution_loop();
    void check_arbitrage_opportunity(const std::string& symbol);
    double calculate_profit_estimate(double binance_price, double bybit_price, bool buy_bybit) const;
    int64_t get_timestamp_ns() const;
};

#endif // SNIPER_H
