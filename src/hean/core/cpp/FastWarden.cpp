/**
 * FastWarden: High-frequency orderbook processing with TDA integration
 * Continuously updates Persistent Homology map of L2 Orderbook
 * 
 * Phase 19: Global Heartbeat Listener
 * Monitors master node health and triggers <10ms failover if master goes offline.
 */

#include "TDA_Engine.h"
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <functional>
#include <memory>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <tuple>

// Forward declaration for heartbeat callback (C-compatible)
extern "C" {
    typedef void (*HeartbeatCallbackC)(int is_master_online, int64_t last_heartbeat_ns);
}

// C++ wrapper for callback
typedef void (*HeartbeatCallback)(bool is_master_online, int64_t last_heartbeat_ns);

// Fractal Risk Management: Position structure
struct PositionRisk {
    std::string symbol;
    double size;  // Position size (notional)
    double entry_price;
    double current_price;
    double notional_value;  // size * current_price
    double unrealized_pnl;
    double risk_amount;  // Risk amount (e.g., distance to stop loss)
    int side;  // 0 = long, 1 = short
};

class FastWarden {
private:
    TDA_Engine _tda_engine;
    std::mutex _orderbook_mutex;
    std::map<std::string, std::vector<OrderbookLevel>> _l2_orderbooks;
    std::atomic<bool> _running;
    std::thread _update_thread;
    
    // Phase 19: Global Heartbeat
    std::mutex _heartbeat_mutex;
    std::atomic<bool> _is_master_online;
    std::atomic<int64_t> _last_master_heartbeat_ns;
    std::atomic<int64_t> _last_heartbeat_check_ns;
    std::thread _heartbeat_thread;
    HeartbeatCallback _heartbeat_callback;
    HeartbeatCallbackC _heartbeat_callback_c;  // C-compatible callback
    int64_t _heartbeat_timeout_ns;  // Timeout in nanoseconds (default: 10ms = 10,000,000 ns)
    bool _is_master_node;
    
    // Orderbook update processing
    void _process_orderbook_updates();
    
    // Phase 19: Global Heartbeat processing
    void _process_heartbeat();
    
    // Fractal Risk Management: Portfolio-wide risk calculation
    std::mutex _risk_mutex;
    std::map<std::string, double> _symbol_correlations;  // Correlation cache (symbol1_symbol2 -> correlation)
    double _calculate_position_correlation(const std::string& symbol1, const std::string& symbol2);
    double _calculate_portfolio_var(const std::vector<PositionRisk>& positions, double total_capital);
    double _calculate_systemic_risk(const std::vector<PositionRisk>& positions, double total_capital);
    
public:
    FastWarden() 
        : _tda_engine(50, 1, 0.1, 100),  // max_levels=50, max_dim=1, max_dist=0.1, update_ms=100
          _running(false),
          _is_master_online(true),
          _last_master_heartbeat_ns(0),
          _last_heartbeat_check_ns(0),
          _heartbeat_callback(nullptr),
          _heartbeat_callback_c(nullptr),
          _heartbeat_timeout_ns(10'000'000),  // 10ms default timeout
          _is_master_node(false) {
    }
    
    ~FastWarden() {
        stop();
    }
    
    // Update L2 orderbook snapshot
    void update_l2_orderbook(
        const std::string& symbol,
        const std::vector<std::pair<double, double>>& bids,  // (price, size)
        const std::vector<std::pair<double, double>>& asks
    );
    
    // Get market topology score (Python-exposed)
    double get_market_topology_score();
    
    // Get topology score for a specific symbol
    TopologyScore get_symbol_topology_score(const std::string& symbol);
    
    // Predict slippage using Riemannian curvature
    double predict_slippage(const std::string& symbol, double order_size, bool is_buy);
    
    // Check if market manifold is disconnected (watchdog)
    bool is_market_disconnected();
    
    // Phase 19: Global Heartbeat methods
    void update_master_heartbeat(int64_t timestamp_ns);
    bool is_master_online() const;
    int64_t get_last_master_heartbeat_ns() const;
    void set_heartbeat_callback(HeartbeatCallback callback);
    void set_heartbeat_timeout_ns(int64_t timeout_ns);
    void set_is_master_node(bool is_master);
    bool should_takeover_master() const;
    
    // Start background processing
    void start();
    
    // Stop background processing
    void stop();
    
    // Fractal Risk Management: Portfolio-wide risk assessment
    // Calculate risk for entire web of correlated positions
    // Returns: (portfolio_risk_pct, systemic_risk_pct, allowed_new_position_size)
    std::tuple<double, double, double> calculate_fractal_risk(
        const std::vector<PositionRisk>& positions,
        double total_capital,
        const std::string& new_symbol = "",
        double new_position_size = 0.0
    );
    
    // Check if adding a new position would exceed risk limits
    // Returns: (allowed, max_allowed_size, reason)
    std::tuple<bool, double, std::string> check_portfolio_risk_limits(
        const std::vector<PositionRisk>& positions,
        double total_capital,
        const std::string& new_symbol,
        double new_position_size,
        double max_portfolio_risk_pct = 0.05  // 5% max portfolio risk (protect $300 seed)
    );
    
    // Get correlation between two symbols (for risk calculation)
    double get_symbol_correlation(const std::string& symbol1, const std::string& symbol2);
};

void FastWarden::update_l2_orderbook(
    const std::string& symbol,
    const std::vector<std::pair<double, double>>& bids,
    const std::vector<std::pair<double, double>>& asks
) {
    std::lock_guard<std::mutex> lock(_orderbook_mutex);
    
    std::vector<OrderbookLevel> levels;
    auto now = std::chrono::system_clock::now();
    int64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
    
    // Combine bids (ascending price) and asks (ascending price)
    // For TDA, we want a unified price-size curve
    for (auto it = bids.rbegin(); it != bids.rend(); ++it) {
        OrderbookLevel level;
        level.price = it->first;
        level.size = it->second;
        level.timestamp_ns = timestamp_ns;
        levels.push_back(level);
    }
    
    for (const auto& ask : asks) {
        OrderbookLevel level;
        level.price = ask.first;
        level.size = ask.second;
        level.timestamp_ns = timestamp_ns;
        levels.push_back(level);
    }
    
    _l2_orderbooks[symbol] = levels;
    
    // Update TDA engine immediately
    _tda_engine.update_orderbook(symbol, levels);
}

double FastWarden::get_market_topology_score() {
    return _tda_engine.get_market_topology_score();
}

TopologyScore FastWarden::get_symbol_topology_score(const std::string& symbol) {
    return _tda_engine.get_topology_score(symbol);
}

double FastWarden::predict_slippage(const std::string& symbol, double order_size, bool is_buy) {
    return _tda_engine.predict_slippage(symbol, order_size, is_buy);
}

bool FastWarden::is_market_disconnected() {
    return _tda_engine.is_market_disconnected();
}

void FastWarden::_process_orderbook_updates() {
    while (_running) {
        {
            std::lock_guard<std::mutex> lock(_orderbook_mutex);
            
            // Periodically force update for all symbols
            for (const auto& [symbol, levels] : _l2_orderbooks) {
                if (!levels.empty()) {
                    _tda_engine.force_update(symbol);
                }
            }
        }
        
        // Sleep briefly
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// Phase 19: Global Heartbeat processing
void FastWarden::_process_heartbeat() {
    while (_running) {
        auto now = std::chrono::system_clock::now();
        int64_t current_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count();
        
        {
            std::lock_guard<std::mutex> lock(_heartbeat_mutex);
            _last_heartbeat_check_ns = current_ns;
            
            // If we're the master node, we don't need to check for master offline
            if (_is_master_node) {
                _is_master_online = true;
            } else {
                // Check if master heartbeat is stale
                int64_t time_since_heartbeat = current_ns - _last_master_heartbeat_ns;
                bool master_was_online = _is_master_online.load();
                bool master_is_online = (time_since_heartbeat < _heartbeat_timeout_ns);
                
                _is_master_online = master_is_online;
                
                // If master just went offline, trigger callback immediately
                if (master_was_online && !master_is_online) {
                    if (_heartbeat_callback) {
                        _heartbeat_callback(false, _last_master_heartbeat_ns);
                    }
                    // Note: C callback is handled separately via extern "C" wrapper
                }
            }
        }
        
        // Check at high frequency (every 1ms for <10ms failover)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void FastWarden::update_master_heartbeat(int64_t timestamp_ns) {
    std::lock_guard<std::mutex> lock(_heartbeat_mutex);
    _last_master_heartbeat_ns = timestamp_ns;
    _is_master_online = true;
}

bool FastWarden::is_master_online() const {
    return _is_master_online.load();
}

int64_t FastWarden::get_last_master_heartbeat_ns() const {
    return _last_master_heartbeat_ns.load();
}

void FastWarden::set_heartbeat_callback(HeartbeatCallback callback) {
    std::lock_guard<std::mutex> lock(_heartbeat_mutex);
    _heartbeat_callback = callback;
    // Note: _heartbeat_callback_c is only set from C API, not here
    // This maintains separation between C and C++ callbacks
}

void FastWarden::set_heartbeat_timeout_ns(int64_t timeout_ns) {
    std::lock_guard<std::mutex> lock(_heartbeat_mutex);
    _heartbeat_timeout_ns = timeout_ns;
}

void FastWarden::set_is_master_node(bool is_master) {
    std::lock_guard<std::mutex> lock(_heartbeat_mutex);
    _is_master_node = is_master;
    if (is_master) {
        _is_master_online = true;
    }
}

bool FastWarden::should_takeover_master() const {
    if (_is_master_node) {
        return false;  // Already master
    }
    
    // Should takeover if master is offline
    auto now = std::chrono::system_clock::now();
    int64_t current_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
    
    int64_t time_since_heartbeat = current_ns - _last_master_heartbeat_ns.load();
    return (time_since_heartbeat >= _heartbeat_timeout_ns);
}

void FastWarden::start() {
    if (_running) {
        return;
    }
    
    _running = true;
    _tda_engine.start();
    _update_thread = std::thread(&FastWarden::_process_orderbook_updates, this);
    
    // Phase 19: Start global heartbeat thread
    _heartbeat_thread = std::thread(&FastWarden::_process_heartbeat, this);
}

void FastWarden::stop() {
    if (!_running) {
        return;
    }
    
    _running = false;
    _tda_engine.stop();
    
    if (_update_thread.joinable()) {
        _update_thread.join();
    }
    
    // Phase 19: Stop global heartbeat thread
    if (_heartbeat_thread.joinable()) {
        _heartbeat_thread.join();
    }
}

// Fractal Risk Management: Calculate correlation between two symbols
double FastWarden::_calculate_position_correlation(const std::string& symbol1, const std::string& symbol2) {
    // Use cached correlation if available
    std::string cache_key = symbol1 < symbol2 ? symbol1 + "_" + symbol2 : symbol2 + "_" + symbol1;
    
    {
        std::lock_guard<std::mutex> lock(_risk_mutex);
        auto it = _symbol_correlations.find(cache_key);
        if (it != _symbol_correlations.end()) {
            return it->second;
        }
    }
    
    // Calculate correlation from orderbook topology
    // For now, use a simplified correlation estimate based on symbol similarity
    // In production, this would use historical price correlation
    
    // BTC/ETH pairs tend to be highly correlated
    bool is_btc_pair1 = symbol1.find("BTC") != std::string::npos;
    bool is_btc_pair2 = symbol2.find("BTC") != std::string::npos;
    bool is_eth_pair1 = symbol1.find("ETH") != std::string::npos;
    bool is_eth_pair2 = symbol2.find("ETH") != std::string::npos;
    
    double correlation = 0.3;  // Base correlation
    
    // High correlation if both are BTC pairs or both are ETH pairs
    if ((is_btc_pair1 && is_btc_pair2) || (is_eth_pair1 && is_eth_pair2)) {
        correlation = 0.7;
    } else if ((is_btc_pair1 && is_eth_pair2) || (is_eth_pair1 && is_btc_pair2)) {
        correlation = 0.6;  // BTC and ETH pairs are also correlated
    }
    
    // Cache the correlation
    {
        std::lock_guard<std::mutex> lock(_risk_mutex);
        _symbol_correlations[cache_key] = correlation;
    }
    
    return correlation;
}

// Fractal Risk Management: Calculate portfolio Value at Risk (VaR)
double FastWarden::_calculate_portfolio_var(const std::vector<PositionRisk>& positions, double total_capital) {
    if (positions.empty() || total_capital <= 0) {
        return 0.0;
    }
    
    // Calculate portfolio variance using correlation matrix
    double portfolio_variance = 0.0;
    
    for (size_t i = 0; i < positions.size(); i++) {
        double risk_i = positions[i].risk_amount;
        
        // Add individual position variance
        portfolio_variance += risk_i * risk_i;
        
        // Add correlation terms
        for (size_t j = i + 1; j < positions.size(); j++) {
            double risk_j = positions[j].risk_amount;
            double correlation = _calculate_position_correlation(positions[i].symbol, positions[j].symbol);
            portfolio_variance += 2.0 * risk_i * risk_j * correlation;
        }
    }
    
    // Portfolio risk = sqrt(portfolio_variance)
    double portfolio_std = std::sqrt(portfolio_variance);
    
    // VaR at 95% confidence (1.65 standard deviations)
    double var_95 = 1.65 * portfolio_std;
    
    // Return as percentage of capital
    return (var_95 / total_capital) * 100.0;
}

// Fractal Risk Management: Calculate systemic risk (risk of all positions moving together)
double FastWarden::_calculate_systemic_risk(const std::vector<PositionRisk>& positions, double total_capital) {
    if (positions.empty() || total_capital <= 0) {
        return 0.0;
    }
    
    // Systemic risk assumes all positions are perfectly correlated (worst case)
    double total_risk_amount = 0.0;
    
    for (const auto& pos : positions) {
        total_risk_amount += pos.risk_amount;
    }
    
    // Return as percentage of capital
    return (total_risk_amount / total_capital) * 100.0;
}

// Fractal Risk Management: Calculate portfolio-wide risk
std::tuple<double, double, double> FastWarden::calculate_fractal_risk(
    const std::vector<PositionRisk>& positions,
    double total_capital,
    const std::string& new_symbol,
    double new_position_size
) {
    std::lock_guard<std::mutex> lock(_risk_mutex);
    
    std::vector<PositionRisk> all_positions = positions;
    
    // Add new position if provided
    if (!new_symbol.empty() && new_position_size > 0.0) {
        PositionRisk new_pos;
        new_pos.symbol = new_symbol;
        new_pos.size = new_position_size;
        new_pos.risk_amount = new_position_size * 0.02;  // Assume 2% risk per position (stop loss)
        all_positions.push_back(new_pos);
    }
    
    // Calculate portfolio VaR (correlated risk)
    double portfolio_risk_pct = _calculate_portfolio_var(all_positions, total_capital);
    
    // Calculate systemic risk (worst case: all positions move together)
    double systemic_risk_pct = _calculate_systemic_risk(all_positions, total_capital);
    
    // Calculate maximum allowed new position size
    // Ensure portfolio risk stays below 5% of capital (protect $300 seed)
    double max_risk_amount = total_capital * 0.05;  // 5% max risk
    double current_risk = _calculate_portfolio_var(positions, total_capital) * total_capital / 100.0;
    double remaining_risk = max_risk_amount - current_risk;
    
    double max_allowed_size = 0.0;
    if (remaining_risk > 0) {
        max_allowed_size = remaining_risk / 0.02;  // Divide by assumed 2% risk per position
    }
    
    return std::make_tuple(portfolio_risk_pct, systemic_risk_pct, max_allowed_size);
}

// Fractal Risk Management: Check if adding new position exceeds risk limits
std::tuple<bool, double, std::string> FastWarden::check_portfolio_risk_limits(
    const std::vector<PositionRisk>& positions,
    double total_capital,
    const std::string& new_symbol,
    double new_position_size,
    double max_portfolio_risk_pct
) {
    if (total_capital <= 0) {
        return std::make_tuple(false, 0.0, "Invalid capital");
    }
    
    // Calculate risk with new position
    auto [portfolio_risk, systemic_risk, max_allowed] = calculate_fractal_risk(
        positions, total_capital, new_symbol, new_position_size
    );
    
    // Check if portfolio risk exceeds limit
    if (portfolio_risk > max_portfolio_risk_pct * 100.0) {
        return std::make_tuple(
            false,
            max_allowed,
            "Portfolio risk exceeds limit: " + std::to_string(portfolio_risk) + "% > " + 
            std::to_string(max_portfolio_risk_pct * 100.0) + "%"
        );
    }
    
    // Check if systemic risk is too high
    if (systemic_risk > max_portfolio_risk_pct * 150.0) {  // Allow higher systemic risk threshold
        return std::make_tuple(
            false,
            max_allowed,
            "Systemic risk too high: " + std::to_string(systemic_risk) + "%"
        );
    }
    
    return std::make_tuple(true, max_allowed, "OK");
}

// Fractal Risk Management: Get correlation between two symbols
double FastWarden::get_symbol_correlation(const std::string& symbol1, const std::string& symbol2) {
    return _calculate_position_correlation(symbol1, symbol2);
}

// Global instance (singleton pattern)
static FastWarden* g_fast_warden = nullptr;
static std::mutex g_instance_mutex;

extern "C" {
    void fast_warden_init() {
        std::lock_guard<std::mutex> lock(g_instance_mutex);
        if (g_fast_warden == nullptr) {
            g_fast_warden = new FastWarden();
            g_fast_warden->start();
        }
    }
    
    void fast_warden_update_orderbook(
        const char* symbol,
        const double* bid_prices,
        const double* bid_sizes,
        int num_bids,
        const double* ask_prices,
        const double* ask_sizes,
        int num_asks
    ) {
        if (g_fast_warden == nullptr) {
            fast_warden_init();
        }
        
        std::vector<std::pair<double, double>> bids;
        for (int i = 0; i < num_bids; i++) {
            bids.push_back({bid_prices[i], bid_sizes[i]});
        }
        
        std::vector<std::pair<double, double>> asks;
        for (int i = 0; i < num_asks; i++) {
            asks.push_back({ask_prices[i], ask_sizes[i]});
        }
        
        g_fast_warden->update_l2_orderbook(std::string(symbol), bids, asks);
    }
    
    double fast_warden_get_market_topology_score() {
        if (g_fast_warden == nullptr) {
            return 1.0;  // Default: stable market
        }
        return g_fast_warden->get_market_topology_score();
    }
    
    double fast_warden_predict_slippage(const char* symbol, double order_size, int is_buy) {
        if (g_fast_warden == nullptr) {
            return 0.01;  // Default: 1% slippage
        }
        return g_fast_warden->predict_slippage(std::string(symbol), order_size, is_buy != 0);
    }
    
    int fast_warden_is_market_disconnected() {
        if (g_fast_warden == nullptr) {
            return 0;  // Default: connected
        }
        return g_fast_warden->is_market_disconnected() ? 1 : 0;
    }
    
    // Phase 19: Global Heartbeat C API
    void fast_warden_update_master_heartbeat(int64_t timestamp_ns) {
        if (g_fast_warden == nullptr) {
            fast_warden_init();
        }
        g_fast_warden->update_master_heartbeat(timestamp_ns);
    }
    
    int fast_warden_is_master_online() {
        if (g_fast_warden == nullptr) {
            return 1;  // Default: assume online
        }
        return g_fast_warden->is_master_online() ? 1 : 0;
    }
    
    int64_t fast_warden_get_last_master_heartbeat_ns() {
        if (g_fast_warden == nullptr) {
            return 0;
        }
        return g_fast_warden->get_last_master_heartbeat_ns();
    }
    
    void fast_warden_set_heartbeat_callback(HeartbeatCallbackC callback) {
        if (g_fast_warden == nullptr) {
            fast_warden_init();
        }
        // Store C callback directly in the instance
        // We'll need to access private member, so use a friend or setter
        // For now, store via a static wrapper
        std::lock_guard<std::mutex> lock(g_instance_mutex);
        if (g_fast_warden) {
            // Store the C callback and create a C++ wrapper
            static HeartbeatCallbackC stored_c_callback = nullptr;
            stored_c_callback = callback;
            
            if (callback) {
                // Create a C++ callback wrapper that calls the C callback
                HeartbeatCallback cpp_wrapper = [](bool is_online, int64_t ts) {
                    if (stored_c_callback) {
                        stored_c_callback(is_online ? 1 : 0, ts);
                    }
                };
                g_fast_warden->set_heartbeat_callback(cpp_wrapper);
            } else {
                g_fast_warden->set_heartbeat_callback(nullptr);
            }
        }
    }
    
    void fast_warden_set_heartbeat_timeout_ns(int64_t timeout_ns) {
        if (g_fast_warden == nullptr) {
            fast_warden_init();
        }
        g_fast_warden->set_heartbeat_timeout_ns(timeout_ns);
    }
    
    void fast_warden_set_is_master_node(int is_master) {
        if (g_fast_warden == nullptr) {
            fast_warden_init();
        }
        g_fast_warden->set_is_master_node(is_master != 0);
    }
    
    int fast_warden_should_takeover_master() {
        if (g_fast_warden == nullptr) {
            return 0;
        }
        return g_fast_warden->should_takeover_master() ? 1 : 0;
    }
    
    void fast_warden_cleanup() {
        std::lock_guard<std::mutex> lock(g_instance_mutex);
        if (g_fast_warden != nullptr) {
            g_fast_warden->stop();
            delete g_fast_warden;
            g_fast_warden = nullptr;
        }
    }
}