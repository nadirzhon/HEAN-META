/**
 * Profit-Extraction Mode: Sub-second Scalping Engine Implementation
 */

#include "Scalper.h"
#include <algorithm>
#include <cmath>
#include <iostream>

Scalper::Scalper()
    : profit_target_pct_(0.0002)  // 0.02%
    , max_position_size_(1.0)
    , total_trades_(0)
    , winning_trades_(0)
    , losing_trades_(0)
    , total_profit_(0.0)
    , total_trade_duration_ns_(0)
    , trade_count_(0)
    , trade_callback_(nullptr)
    , callback_user_data_(nullptr)
{
    // Default tick sizes (in USD)
    tick_sizes_["BTCUSDT"] = 0.01;
    tick_sizes_["ETHUSDT"] = 0.01;
    
    // Default hard-stop: 5 ticks
    hard_stop_ticks_["BTCUSDT"] = 5.0;
    hard_stop_ticks_["ETHUSDT"] = 5.0;
}

Scalper::~Scalper() {
}

void Scalper::set_profit_target_pct(double target_pct) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    profit_target_pct_ = target_pct;
}

void Scalper::set_max_position_size(double max_size) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    max_position_size_ = max_size;
}

void Scalper::set_tick_size(const std::string& symbol, double tick_size) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    tick_sizes_[symbol] = tick_size;
}

void Scalper::set_hard_stop_ticks(const std::string& symbol, double stop_ticks) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    hard_stop_ticks_[symbol] = stop_ticks;
}

void Scalper::update_price(const std::string& symbol, double price, double bid, double ask,
                           int64_t timestamp_ns) {
    std::lock_guard<std::mutex> lock(prices_mutex_);
    
    PriceData data;
    data.price = price;
    data.bid = bid;
    data.ask = ask;
    data.timestamp_ns = timestamp_ns;
    
    prices_[symbol] = data;
    
    // Update positions (check stop-loss/take-profit)
    update_positions();
}

ScalpSignal Scalper::check_opportunity(const std::string& symbol) const {
    ScalpSignal signal;
    signal.symbol = symbol;
    
    std::lock_guard<std::mutex> prices_lock(prices_mutex_);
    auto it = prices_.find(symbol);
    if (it == prices_.end()) {
        return signal;  // No price data
    }
    
    const PriceData& data = it->second;
    
    // Check if we already have a position
    std::lock_guard<std::mutex> positions_lock(positions_mutex_);
    if (positions_.find(symbol) != positions_.end()) {
        return signal;  // Already have position
    }
    
    // Calculate spread
    double spread = data.ask - data.bid;
    double mid_price = (data.bid + data.ask) / 2.0;
    double spread_pct = spread / mid_price;
    
    // Only trade if spread is tight (<0.01%)
    if (spread_pct > 0.0001) {
        return signal;  // Spread too wide
    }
    
    // Look for micro-movements (0.01% or more)
    // For scalping, we want to catch small movements quickly
    
    // Calculate entry based on current price
    double entry_price = mid_price;
    
    // Target: 0.02% profit
    double target_price = calculate_target_price(entry_price, true);  // Try long first
    
    // Check if target is reasonable (not too far from current ask)
    double ask_distance = (target_price - data.ask) / data.ask;
    if (ask_distance < profit_target_pct_ * 0.5) {  // Target is close to ask
        signal.is_long = true;
        signal.entry_price = data.ask;  // Market buy at ask
        signal.target_price = target_price;
        
        // Calculate stop-loss in ticks
        std::lock_guard<std::mutex> config_lock(config_mutex_);
        double stop_ticks = hard_stop_ticks_[symbol];
        if (stop_ticks == 0.0) {
            stop_ticks = 5.0;  // Default: 5 ticks
        }
        
        double tick_size = tick_sizes_[symbol];
        if (tick_size == 0.0) {
            tick_size = 0.01;  // Default tick size
        }
        
        signal.stop_loss_tick = stop_ticks * tick_size;
        signal.signal_timestamp_ns = get_timestamp_ns();
    } else {
        // Try short instead
        target_price = calculate_target_price(entry_price, false);
        double bid_distance = (data.bid - target_price) / data.bid;
        
        if (bid_distance < profit_target_pct_ * 0.5) {
            signal.is_long = false;
            signal.entry_price = data.bid;  // Market sell at bid
            signal.target_price = target_price;
            
            std::lock_guard<std::mutex> config_lock(config_mutex_);
            double stop_ticks = hard_stop_ticks_[symbol];
            if (stop_ticks == 0.0) {
                stop_ticks = 5.0;
            }
            
            double tick_size = tick_sizes_[symbol];
            if (tick_size == 0.0) {
                tick_size = 0.01;
            }
            
            signal.stop_loss_tick = stop_ticks * tick_size;
            signal.signal_timestamp_ns = get_timestamp_ns();
        }
    }
    
    return signal;
}

bool Scalper::execute_scalp(const ScalpSignal& signal) {
    // Check if we already have a position
    std::lock_guard<std::mutex> lock(positions_mutex_);
    if (positions_.find(signal.symbol) != positions_.end()) {
        return false;  // Already have position
    }
    
    // Create position
    Position position;
    position.signal = signal;
    position.entry_price = signal.entry_price;
    position.target_price = signal.target_price;
    position.stop_loss_price = calculate_stop_loss_price(
        signal.entry_price, signal.is_long, signal.symbol
    );
    position.entry_timestamp_ns = get_timestamp_ns();
    
    positions_[signal.symbol] = position;
    
    total_trades_.fetch_add(1);
    
    return true;
}

void Scalper::close_position(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    auto it = positions_.find(symbol);
    if (it == positions_.end()) {
        return;  // No position
    }
    
    // Calculate profit/loss
    const Position& position = it->second;
    
    // Get current price
    std::lock_guard<std::mutex> prices_lock(prices_mutex_);
    auto price_it = prices_.find(symbol);
    if (price_it == prices_.end()) {
        positions_.erase(it);
        return;
    }
    
    const PriceData& data = price_it->second;
    double exit_price = position.is_long ? data.bid : data.ask;  // Exit at opposite side
    
    // Calculate PnL
    double pnl = 0.0;
    if (position.is_long) {
        pnl = (exit_price - position.entry_price) / position.entry_price;
    } else {
        pnl = (position.entry_price - exit_price) / position.entry_price;
    }
    
    // Convert to USD
    double profit_usd = pnl * max_position_size_;
    
    // Update statistics
    if (pnl > 0.0) {
        winning_trades_.fetch_add(1);
    } else {
        losing_trades_.fetch_add(1);
    }
    
    total_profit_.fetch_add(profit_usd);
    
    // Track trade duration
    int64_t duration_ns = get_timestamp_ns() - position.entry_timestamp_ns;
    total_trade_duration_ns_.fetch_add(duration_ns);
    trade_count_.fetch_add(1);
    
    // Call callback
    if (trade_callback_) {
        trade_callback_(position.signal, profit_usd, callback_user_data_);
    }
    
    // Remove position
    positions_.erase(it);
}

bool Scalper::has_position(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    return positions_.find(symbol) != positions_.end();
}

double Scalper::calculate_target_price(double entry_price, bool is_long) const {
    if (is_long) {
        // Long: target = entry * (1 + profit_target_pct)
        return entry_price * (1.0 + profit_target_pct_);
    } else {
        // Short: target = entry * (1 - profit_target_pct)
        return entry_price * (1.0 - profit_target_pct_);
    }
}

double Scalper::calculate_stop_loss_price(double entry_price, bool is_long,
                                         const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    double stop_ticks = hard_stop_ticks_[symbol];
    if (stop_ticks == 0.0) {
        stop_ticks = 5.0;  // Default: 5 ticks
    }
    
    double tick_size = tick_sizes_[symbol];
    if (tick_size == 0.0) {
        tick_size = 0.01;  // Default tick size
    }
    
    double stop_distance = stop_ticks * tick_size;
    
    if (is_long) {
        // Long: stop = entry - stop_distance
        return entry_price - stop_distance;
    } else {
        // Short: stop = entry + stop_distance
        return entry_price + stop_distance;
    }
}

void Scalper::update_positions() {
    std::lock_guard<std::mutex> positions_lock(positions_mutex_);
    std::lock_guard<std::mutex> prices_lock(prices_mutex_);
    
    std::vector<std::string> to_close;
    
    for (auto& [symbol, position] : positions_) {
        auto price_it = prices_.find(symbol);
        if (price_it == prices_.end()) {
            continue;
        }
        
        const PriceData& data = price_it->second;
        double current_price = data.price;
        
        // Check stop-loss
        if (position.is_long && current_price <= position.stop_loss_price) {
            to_close.push_back(symbol);
            continue;
        }
        
        if (!position.is_long && current_price >= position.stop_loss_price) {
            to_close.push_back(symbol);
            continue;
        }
        
        // Check take-profit
        if (position.is_long && current_price >= position.target_price) {
            to_close.push_back(symbol);
            continue;
        }
        
        if (!position.is_long && current_price <= position.target_price) {
            to_close.push_back(symbol);
            continue;
        }
    }
    
    // Close positions (will be handled by close_position)
    positions_lock.unlock();
    for (const auto& symbol : to_close) {
        close_position(symbol);
    }
}

int64_t Scalper::get_timestamp_ns() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

double Scalper::get_win_rate() const {
    int64_t total = winning_trades_.load() + losing_trades_.load();
    if (total == 0) {
        return 0.0;
    }
    return static_cast<double>(winning_trades_.load()) / total;
}

int64_t Scalper::get_avg_trade_duration_ns() const {
    int64_t count = trade_count_.load();
    if (count == 0) {
        return 0;
    }
    return total_trade_duration_ns_.load() / count;
}

void Scalper::set_trade_callback(TradeCallback callback, void* user_data) {
    trade_callback_ = callback;
    callback_user_data_ = user_data;
}

// C interface for Python bindings
extern "C" {
    static Scalper* g_scalper = nullptr;
    
    void scalper_init() {
        if (g_scalper == nullptr) {
            g_scalper = new Scalper();
        }
    }
    
    void scalper_set_profit_target_pct(double target) {
        if (g_scalper) {
            g_scalper->set_profit_target_pct(target);
        }
    }
    
    void scalper_update_price(const char* symbol, double price, double bid, double ask,
                              int64_t timestamp_ns) {
        if (g_scalper) {
            g_scalper->update_price(std::string(symbol), price, bid, ask, timestamp_ns);
        }
    }
    
    void scalper_set_hard_stop_ticks(const char* symbol, double stop_ticks) {
        if (g_scalper) {
            g_scalper->set_hard_stop_ticks(std::string(symbol), stop_ticks);
        }
    }
    
    int scalper_execute_scalp(const char* symbol, int is_long, double entry_price,
                              double target_price, double stop_loss_tick) {
        if (g_scalper) {
            ScalpSignal signal;
            signal.symbol = std::string(symbol);
            signal.is_long = is_long != 0;
            signal.entry_price = entry_price;
            signal.target_price = target_price;
            signal.stop_loss_tick = stop_loss_tick;
            signal.signal_timestamp_ns = 0;
            
            return g_scalper->execute_scalp(signal) ? 1 : 0;
        }
        return 0;
    }
    
    int64_t scalper_get_total_trades() {
        return g_scalper ? g_scalper->get_total_trades() : 0;
    }
    
    double scalper_get_total_profit() {
        return g_scalper ? g_scalper->get_total_profit() : 0.0;
    }
    
    double scalper_get_win_rate() {
        return g_scalper ? g_scalper->get_win_rate() : 0.0;
    }
    
    void scalper_cleanup() {
        if (g_scalper) {
            delete g_scalper;
            g_scalper = nullptr;
        }
    }
}
