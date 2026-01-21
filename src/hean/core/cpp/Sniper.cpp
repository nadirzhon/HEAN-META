/**
 * "The Sniper" - Ultra-Low Latency Lead-Lag Arbitrage Engine Implementation
 */

#include "Sniper.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>

Sniper::Sniper() 
    : delta_threshold_(0.0005)  // 0.05%
    , execution_delay_limit_ns_(2000000)  // 2ms in nanoseconds
    , max_position_size_(1.0)
    , maker_fee_(0.0001)  // 0.01% maker fee
    , taker_fee_(0.0006)  // 0.06% taker fee
    , running_(false)
    , execution_thread_running_(false)
    , signal_callback_(nullptr)
    , callback_user_data_(nullptr)
    , total_signals_(0)
    , executed_trades_(0)
    , total_profit_(0.0)
    , max_execution_time_ns_(0)
    , total_execution_time_ns_(0)
    , execution_count_(0)
{
}

Sniper::~Sniper() {
    stop();
}

void Sniper::set_delta_threshold(double threshold_pct) {
    delta_threshold_ = threshold_pct;
}

void Sniper::set_execution_delay_limit_ns(int64_t max_delay_ns) {
    execution_delay_limit_ns_ = max_delay_ns;
}

void Sniper::set_max_position_size(double max_size) {
    max_position_size_ = max_size;
}

void Sniper::set_maker_fee(double fee) {
    maker_fee_ = fee;
}

void Sniper::set_taker_fee(double fee) {
    taker_fee_ = fee;
}

bool Sniper::start() {
    if (running_.load()) {
        return false;
    }
    
    running_.store(true);
    execution_thread_running_.store(true);
    execution_thread_ = std::thread(&Sniper::execution_loop, this);
    
    return true;
}

void Sniper::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    execution_thread_running_.store(false);
    
    if (execution_thread_.joinable()) {
        execution_thread_.join();
    }
}

void Sniper::subscribe_symbol(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(symbols_mutex_);
    if (std::find(subscribed_symbols_.begin(), subscribed_symbols_.end(), symbol) 
        == subscribed_symbols_.end()) {
        subscribed_symbols_.push_back(symbol);
    }
}

void Sniper::unsubscribe_symbol(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(symbols_mutex_);
    subscribed_symbols_.erase(
        std::remove(subscribed_symbols_.begin(), subscribed_symbols_.end(), symbol),
        subscribed_symbols_.end()
    );
}

void Sniper::update_binance_price(const std::string& symbol, double price,
                                  double bid, double ask, int64_t timestamp_ns) {
    std::lock_guard<std::mutex> lock(prices_mutex_);
    
    PriceUpdate update;
    update.symbol = symbol;
    update.price = price;
    update.bid_price = bid;
    update.ask_price = ask;
    update.timestamp_ns = timestamp_ns;
    update.exchange = Exchange::BINANCE;
    
    binance_prices_[symbol] = update;
    
    // Immediately check for arbitrage opportunity
    check_arbitrage_opportunity(symbol);
}

void Sniper::update_bybit_price(const std::string& symbol, double price,
                                double bid, double ask, int64_t timestamp_ns) {
    std::lock_guard<std::mutex> lock(prices_mutex_);
    
    PriceUpdate update;
    update.symbol = symbol;
    update.price = price;
    update.bid_price = bid;
    update.ask_price = ask;
    update.timestamp_ns = timestamp_ns;
    update.exchange = Exchange::BYBIT;
    
    bybit_prices_[symbol] = update;
    
    // Immediately check for arbitrage opportunity
    check_arbitrage_opportunity(symbol);
}

void Sniper::check_arbitrage_opportunity(const std::string& symbol) {
    // Check if we have prices from both exchanges
    auto binance_it = binance_prices_.find(symbol);
    auto bybit_it = bybit_prices_.find(symbol);
    
    if (binance_it == binance_prices_.end() || bybit_it == bybit_prices_.end()) {
        return;
    }
    
    const PriceUpdate& binance = binance_it->second;
    const PriceUpdate& bybit = bybit_it->second;
    
    // Calculate price delta (Binance moving while Bybit stationary)
    double binance_price = binance.price;
    double bybit_price = bybit.price;
    
    if (binance_price <= 0.0 || bybit_price <= 0.0) {
        return;
    }
    
    // Calculate percentage delta
    double price_delta = std::abs(binance_price - bybit_price) / std::min(binance_price, bybit_price);
    
    // Check if delta exceeds threshold
    if (price_delta < delta_threshold_) {
        return;
    }
    
    // Determine direction: if Binance > Bybit, buy Bybit (long)
    bool is_long = binance_price > bybit_price;
    
    // Calculate profit estimate (accounting for fees)
    double profit_pct = calculate_profit_estimate(binance_price, bybit_price, is_long);
    
    // Only proceed if profit after fees is positive
    if (profit_pct <= 0.0) {
        return;
    }
    
    // Create arbitrage signal
    ArbitrageSignal signal;
    signal.symbol = symbol;
    signal.binance_price = binance_price;
    signal.bybit_price = bybit_price;
    signal.price_delta_pct = price_delta;
    signal.profit_estimate_pct = profit_pct;
    signal.signal_timestamp_ns = get_timestamp_ns();
    signal.is_long_opportunity = is_long;
    
    // Record signal
    total_signals_.fetch_add(1);
    
    // Execute immediately if within delay limit
    int64_t signal_age_ns = get_timestamp_ns() - signal.signal_timestamp_ns;
    if (signal_age_ns <= execution_delay_limit_ns_) {
        // Execute trade
        int64_t execution_start = get_timestamp_ns();
        
        // Call callback if registered
        if (signal_callback_) {
            signal_callback_(signal, callback_user_data_);
        }
        
        int64_t execution_time_ns = get_timestamp_ns() - execution_start;
        
        // Update statistics
        executed_trades_.fetch_add(1);
        total_profit_.fetch_add(profit_pct * max_position_size_);
        
        // Track execution time
        int64_t current_max = max_execution_time_ns_.load();
        while (execution_time_ns > current_max && 
               !max_execution_time_ns_.compare_exchange_weak(current_max, execution_time_ns)) {
            current_max = max_execution_time_ns_.load();
        }
        
        total_execution_time_ns_.fetch_add(execution_time_ns);
        execution_count_.fetch_add(1);
    }
}

double Sniper::calculate_profit_estimate(double binance_price, double bybit_price, 
                                         bool buy_bybit) const {
    if (buy_bybit) {
        // Buy on Bybit, sell on Binance
        // Cost: bybit_price * (1 + taker_fee_)
        // Revenue: binance_price * (1 - taker_fee_)
        double cost = bybit_price * (1.0 + taker_fee_);
        double revenue = binance_price * (1.0 - taker_fee_);
        double profit = revenue - cost;
        return profit / cost;  // Return as percentage
    } else {
        // Sell on Bybit, buy on Binance
        // Revenue: bybit_price * (1 - taker_fee_)
        // Cost: binance_price * (1 + taker_fee_)
        double revenue = bybit_price * (1.0 - taker_fee_);
        double cost = binance_price * (1.0 + taker_fee_);
        double profit = revenue - cost;
        return profit / cost;  // Return as percentage
    }
}

void Sniper::execution_loop() {
    // High-frequency polling loop
    while (execution_thread_running_.load()) {
        // Check all subscribed symbols continuously
        std::lock_guard<std::mutex> symbols_lock(symbols_mutex_);
        for (const auto& symbol : subscribed_symbols_) {
            // Check is done in update_*_price methods, but we can also poll here
            // for maximum responsiveness
            std::lock_guard<std::mutex> prices_lock(prices_mutex_);
            check_arbitrage_opportunity(symbol);
        }
        
        // Spin-wait with minimal sleep for ultra-low latency
        std::this_thread::sleep_for(std::chrono::microseconds(10));  // 10us = 0.01ms
    }
}

int64_t Sniper::get_timestamp_ns() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

int64_t Sniper::get_avg_execution_time_ns() const {
    int64_t count = execution_count_.load();
    if (count == 0) {
        return 0;
    }
    return total_execution_time_ns_.load() / count;
}

void Sniper::set_signal_callback(SignalCallback callback, void* user_data) {
    signal_callback_ = callback;
    callback_user_data_ = user_data;
}

// C interface for Python bindings
extern "C" {
    static Sniper* g_sniper = nullptr;
    
    void sniper_init() {
        if (g_sniper == nullptr) {
            g_sniper = new Sniper();
        }
    }
    
    void sniper_set_delta_threshold(double threshold) {
        if (g_sniper) {
            g_sniper->set_delta_threshold(threshold);
        }
    }
    
    int sniper_start() {
        if (g_sniper) {
            return g_sniper->start() ? 1 : 0;
        }
        return 0;
    }
    
    void sniper_stop() {
        if (g_sniper) {
            g_sniper->stop();
        }
    }
    
    void sniper_update_binance_price(const char* symbol, double price, 
                                     double bid, double ask, int64_t timestamp_ns) {
        if (g_sniper) {
            g_sniper->update_binance_price(std::string(symbol), price, bid, ask, timestamp_ns);
        }
    }
    
    void sniper_update_bybit_price(const char* symbol, double price,
                                   double bid, double ask, int64_t timestamp_ns) {
        if (g_sniper) {
            g_sniper->update_bybit_price(std::string(symbol), price, bid, ask, timestamp_ns);
        }
    }
    
    void sniper_subscribe_symbol(const char* symbol) {
        if (g_sniper) {
            g_sniper->subscribe_symbol(std::string(symbol));
        }
    }
    
    int64_t sniper_get_total_signals() {
        return g_sniper ? g_sniper->get_total_signals() : 0;
    }
    
    int64_t sniper_get_executed_trades() {
        return g_sniper ? g_sniper->get_executed_trades() : 0;
    }
    
    double sniper_get_total_profit() {
        return g_sniper ? g_sniper->get_total_profit() : 0.0;
    }
    
    int64_t sniper_get_avg_execution_time_ns() {
        return g_sniper ? g_sniper->get_avg_execution_time_ns() : 0;
    }
    
    void sniper_cleanup() {
        if (g_sniper) {
            delete g_sniper;
            g_sniper = nullptr;
        }
    }
}
