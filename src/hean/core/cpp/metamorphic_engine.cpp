/**
 * Metamorphic Engine: Self-Evolving Trading System
 * 
 * This engine profiles all trading functions in real-time and detects alpha decay.
 * When decay is detected, it triggers an 'Evolutionary Cycle' where the AI proposes
 * new mathematical models (e.g., switching from Linear Regression to Non-Euclidean Geometry).
 * 
 * The system breaks the limits of 'Absolute' - it understands the market better than
 * the engineers who built the exchange.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <mutex>
#include <chrono>
#include <vector>
#include <deque>
#include <functional>
#include <memory>
#include "MemoryPool.h"

// Maximum number of strategies to profile
#define MAX_STRATEGIES 50
#define ALPHA_DECAY_WINDOW 1000  // Number of trades to analyze for decay
#define MIN_TRADES_FOR_DECAY_DETECTION 10  // Reduced threshold for faster evolution
// REMOVED: ALPHA_DECAY_THRESHOLD - No safety constraints, evolution based ONLY on real-time alpha

// Hot-Swap: Strategy execution function signature (C++ only, not in extern "C")
// Parameters: symbol, price, size, side (0=buy, 1=sell), timestamp_ns
// Returns: execution_result (success/failure), pnl
typedef std::function<std::pair<bool, double>(const char*, double, double, int, int64_t)> StrategyExecutionFunc;

// Mathematical model types
enum ModelType {
    LINEAR_REGRESSION = 0,
    NON_EUCLIDEAN_GEOMETRY = 1,
    MANIFOLD_LEARNING = 2,
    TOPOLOGICAL_DATA_ANALYSIS = 3,
    QUANTUM_WAVEFUNCTION = 4,
    CAUSAL_GRAPH = 5,
    HYPERBOLIC_GEOMETRY = 6,
    RIEMANNIAN_MANIFOLD = 7
};

// Strategy performance metrics
struct StrategyMetrics {
    std::string strategy_id;
    double current_alpha;  // Current Sharpe-like alpha
    double peak_alpha;     // Best alpha achieved
    double alpha_decay_rate;  // Rate of decay (negative = decay)
    int total_trades;
    int winning_trades;
    double total_pnl;
    double rolling_sharpe;
    double rolling_win_rate;
    int64_t last_trade_timestamp_ns;
    ModelType current_model;
    ModelType proposed_model;
    bool evolution_triggered;
    int evolution_cycle_count;
    
    // Performance history for decay detection
    std::deque<double> alpha_history;
    std::deque<double> pnl_history;
    std::deque<int> win_loss_history;  // 1 = win, -1 = loss
    
    StrategyMetrics() : current_alpha(0.0), peak_alpha(0.0), alpha_decay_rate(0.0),
                       total_trades(0), winning_trades(0), total_pnl(0.0),
                       rolling_sharpe(0.0), rolling_win_rate(0.0),
                       last_trade_timestamp_ns(0), current_model(LINEAR_REGRESSION),
                       proposed_model(LINEAR_REGRESSION), evolution_triggered(false),
                       evolution_cycle_count(0) {}
};

// Metamorphic Engine class
class MetamorphicEngine {
private:
    std::map<std::string, StrategyMetrics> strategies_;
    std::mutex data_mutex_;
    int64_t last_evolution_timestamp_ns_;
    int total_evolution_cycles_;
    
    // Hot-Swap: Strategy execution function registry
    std::map<std::string, StrategyExecutionFunc> strategy_executors_;
    std::mutex executor_mutex_;
    std::map<std::string, int64_t> executor_update_timestamps_;  // Track when executor was swapped
    
    // Model transition probabilities (learned from historical performance)
    double model_transition_matrix_[8][8];
    
    // Calculate rolling Sharpe ratio
    double calculate_sharpe(const std::deque<double>& returns, double risk_free_rate = 0.0) {
        if (returns.size() < 2) return 0.0;
        
        double mean = 0.0;
        for (double ret : returns) {
            mean += ret;
        }
        mean /= returns.size();
        
        double variance = 0.0;
        for (double ret : returns) {
            double diff = ret - mean;
            variance += diff * diff;
        }
        variance /= (returns.size() - 1);
        
        if (variance < 1e-10) return 0.0;
        
        double std_dev = std::sqrt(variance);
        return (mean - risk_free_rate) / std_dev * std::sqrt(252.0);  // Annualized
    }
    
    // Detect alpha decay using statistical analysis
    // UNCONSTRAINED: Minimal threshold for rapid evolution
    bool detect_alpha_decay(StrategyMetrics& metrics) {
        // Reduced minimum - allow evolution with very few trades
        if (metrics.alpha_history.size() < 5) {
            return false;
        }
        
        // Calculate recent alpha (last 20% of history)
        int recent_window = std::max(10, (int)(metrics.alpha_history.size() * 0.2));
        double recent_alpha = 0.0;
        for (int i = metrics.alpha_history.size() - recent_window; i < (int)metrics.alpha_history.size(); i++) {
            recent_alpha += metrics.alpha_history[i];
        }
        recent_alpha /= recent_window;
        
        // Calculate historical peak alpha (first 20% of history)
        int historical_window = std::max(10, (int)(metrics.alpha_history.size() * 0.2));
        double historical_alpha = 0.0;
        for (int i = 0; i < historical_window && i < (int)metrics.alpha_history.size(); i++) {
            historical_alpha += metrics.alpha_history[i];
        }
        historical_alpha /= historical_window;
        
        // UNCONSTRAINED: Evolution based ONLY on real-time alpha, no safety thresholds
        // If recent alpha is lower than historical, trigger evolution
        // No minimum threshold - allow continuous evolution
        if (historical_alpha > 0.01) {  // Only check if we had positive alpha
            double decay_ratio = (historical_alpha - recent_alpha) / historical_alpha;
            metrics.alpha_decay_rate = decay_ratio;
            
            // ANY decay triggers evolution (no threshold constraint)
            if (decay_ratio > 0.0) {
                return true;
            }
        } else if (recent_alpha < 0.0) {
            // Negative alpha: immediate evolution
            metrics.alpha_decay_rate = 1.0;  // Maximum decay signal
            return true;
        }
        
        return false;
    }
    
    // Propose new mathematical model based on current performance and market regime
    ModelType propose_evolutionary_model(const StrategyMetrics& metrics, double market_volatility) {
        // Evolutionary logic: Switch models based on performance characteristics
        
        // If high volatility, use Non-Euclidean Geometry (better for chaotic markets)
        if (market_volatility > 0.5) {
            return NON_EUCLIDEAN_GEOMETRY;
        }
        
        // If low win rate but high volatility, use Manifold Learning
        if (metrics.rolling_win_rate < 0.4 && market_volatility > 0.3) {
            return MANIFOLD_LEARNING;
        }
        
        // If strong correlations detected, use Causal Graph
        if (metrics.rolling_sharpe > 1.5 && metrics.total_trades > 500) {
            return CAUSAL_GRAPH;
        }
        
        // If complex patterns, use Topological Data Analysis
        if (metrics.total_trades > 1000 && metrics.rolling_win_rate > 0.5) {
            return TOPOLOGICAL_DATA_ANALYSIS;
        }
        
        // Default: Try Non-Euclidean Geometry (more sophisticated than linear)
        return NON_EUCLIDEAN_GEOMETRY;
    }
    
    // Calculate current alpha (Sharpe-adjusted performance metric)
    double calculate_alpha(const StrategyMetrics& metrics) {
        if (metrics.pnl_history.size() < 10) {
            return 0.0;
        }
        
        // Use rolling Sharpe as proxy for alpha
        double sharpe = calculate_sharpe(metrics.pnl_history);
        
        // Adjust for win rate
        double win_rate_bonus = (metrics.rolling_win_rate - 0.5) * 0.5;
        
        return sharpe + win_rate_bonus;
    }

public:
    MetamorphicEngine() : last_evolution_timestamp_ns_(0), total_evolution_cycles_(0) {
        // Initialize model transition matrix (can be learned over time)
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                model_transition_matrix_[i][j] = (i == j) ? 0.7 : 0.3 / 7.0;  // Prefer staying, but allow transitions
            }
        }
    }
    
    // Register a strategy for profiling
    void register_strategy(const std::string& strategy_id, ModelType initial_model = LINEAR_REGRESSION) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        if (strategies_.find(strategy_id) == strategies_.end()) {
            StrategyMetrics metrics;
            metrics.strategy_id = strategy_id;
            metrics.current_model = initial_model;
            metrics.proposed_model = initial_model;
            strategies_[strategy_id] = metrics;
        }
    }
    
    // Record a trade result for a strategy
    void record_trade(const std::string& strategy_id, double pnl, int64_t timestamp_ns, bool is_win) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        auto it = strategies_.find(strategy_id);
        if (it == strategies_.end()) {
            return;  // Strategy not registered
        }
        
        StrategyMetrics& metrics = it->second;
        
        // Update trade statistics
        metrics.total_trades++;
        metrics.total_pnl += pnl;
        metrics.last_trade_timestamp_ns = timestamp_ns;
        
        if (is_win) {
            metrics.winning_trades++;
        }
        
        // Update history (keep last ALPHA_DECAY_WINDOW trades)
        metrics.pnl_history.push_back(pnl);
        if (metrics.pnl_history.size() > ALPHA_DECAY_WINDOW) {
            metrics.pnl_history.pop_front();
        }
        
        metrics.win_loss_history.push_back(is_win ? 1 : -1);
        if (metrics.win_loss_history.size() > ALPHA_DECAY_WINDOW) {
            metrics.win_loss_history.pop_front();
        }
        
        // Calculate rolling metrics
        if (metrics.pnl_history.size() >= 20) {
            metrics.rolling_sharpe = calculate_sharpe(metrics.pnl_history);
        }
        
        if (metrics.total_trades > 0) {
            metrics.rolling_win_rate = (double)metrics.winning_trades / metrics.total_trades;
        }
        
        // Calculate current alpha
        metrics.current_alpha = calculate_alpha(metrics);
        
        // Update peak alpha
        if (metrics.current_alpha > metrics.peak_alpha) {
            metrics.peak_alpha = metrics.current_alpha;
        }
        
        // Update alpha history
        metrics.alpha_history.push_back(metrics.current_alpha);
        if (metrics.alpha_history.size() > ALPHA_DECAY_WINDOW) {
            metrics.alpha_history.pop_front();
        }
        
        // Check for alpha decay and trigger evolution if needed
        if (!metrics.evolution_triggered && detect_alpha_decay(metrics)) {
            metrics.evolution_triggered = true;
            metrics.evolution_cycle_count++;
            
            // Propose new model (use market volatility from recent trades)
            double market_volatility = 0.0;
            if (metrics.pnl_history.size() >= 20) {
                std::vector<double> recent_pnl(metrics.pnl_history.end() - 20, metrics.pnl_history.end());
                double mean = 0.0;
                for (double p : recent_pnl) mean += p;
                mean /= recent_pnl.size();
                double variance = 0.0;
                for (double p : recent_pnl) {
                    double diff = p - mean;
                    variance += diff * diff;
                }
                market_volatility = std::sqrt(variance / recent_pnl.size());
            }
            
            metrics.proposed_model = propose_evolutionary_model(metrics, market_volatility);
            
            // Update global evolution tracking
            total_evolution_cycles_++;
            last_evolution_timestamp_ns_ = timestamp_ns;
        }
    }
    
    // Get evolution status for a strategy
    bool get_evolution_status(const std::string& strategy_id, ModelType* current_model, ModelType* proposed_model, 
                             double* alpha_decay_rate, int* evolution_cycle) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        auto it = strategies_.find(strategy_id);
        if (it == strategies_.end()) {
            return false;
        }
        
        const StrategyMetrics& metrics = it->second;
        *current_model = metrics.current_model;
        *proposed_model = metrics.proposed_model;
        *alpha_decay_rate = metrics.alpha_decay_rate;
        *evolution_cycle = metrics.evolution_cycle_count;
        
        return metrics.evolution_triggered;
    }
    
    // Apply evolutionary model change
    void apply_evolution(const std::string& strategy_id) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        auto it = strategies_.find(strategy_id);
        if (it == strategies_.end()) {
            return;
        }
        
        StrategyMetrics& metrics = it->second;
        metrics.current_model = metrics.proposed_model;
        metrics.evolution_triggered = false;
        metrics.alpha_history.clear();  // Reset history after evolution
        metrics.pnl_history.clear();
        metrics.win_loss_history.clear();
    }
    
    // Get system evolution level (SEL) - overall system intelligence metric
    double get_system_evolution_level() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        if (strategies_.empty()) {
            return 0.0;
        }
        
        double total_sel = 0.0;
        int count = 0;
        
        for (const auto& pair : strategies_) {
            const StrategyMetrics& metrics = pair.second;
            
            // SEL calculation: combines model sophistication, performance, and evolution cycles
            double model_sophistication = (double)metrics.current_model / 7.0;  // 0-1 scale
            double performance_score = std::min(1.0, metrics.rolling_sharpe / 2.0);  // Cap at 2.0 Sharpe
            double evolution_bonus = std::min(0.3, metrics.evolution_cycle_count * 0.05);  // Up to 30% bonus
            
            double sel = (model_sophistication * 0.4 + performance_score * 0.4 + evolution_bonus * 0.2);
            total_sel += sel;
            count++;
        }
        
        return count > 0 ? total_sel / count : 0.0;
    }
    
    // Get all strategy metrics (for monitoring)
    void get_all_metrics(std::map<std::string, StrategyMetrics>& output) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        output = strategies_;
    }
    
    // Hot-Swap: Register or replace strategy execution function
    bool register_strategy_executor(const std::string& strategy_id, StrategyExecutionFunc executor) {
        std::lock_guard<std::mutex> lock(executor_mutex_);
        
        if (executor == nullptr) {
            return false;  // Invalid executor
        }
        
        strategy_executors_[strategy_id] = executor;
        auto now = std::chrono::system_clock::now();
        executor_update_timestamps_[strategy_id] = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count();
        
        return true;
    }
    
    // Hot-Swap: Execute strategy using registered executor
    std::pair<bool, double> execute_strategy(const std::string& strategy_id, const char* symbol, 
                                            double price, double size, int side, int64_t timestamp_ns) {
        std::lock_guard<std::mutex> lock(executor_mutex_);
        
        auto it = strategy_executors_.find(strategy_id);
        if (it == strategy_executors_.end()) {
            return std::make_pair(false, 0.0);  // No executor registered
        }
        
        // Execute the strategy using the registered function (hot-swapped)
        try {
            return it->second(symbol, price, size, side, timestamp_ns);
        } catch (...) {
            return std::make_pair(false, 0.0);  // Execution failed
        }
    }
    
    // Hot-Swap: Get executor update timestamp (when it was last swapped)
    int64_t get_executor_update_timestamp(const std::string& strategy_id) {
        std::lock_guard<std::mutex> lock(executor_mutex_);
        auto it = executor_update_timestamps_.find(strategy_id);
        if (it != executor_update_timestamps_.end()) {
            return it->second;
        }
        return 0;
    }
    
    // Hot-Swap: Check if executor is registered
    bool has_executor(const std::string& strategy_id) {
        std::lock_guard<std::mutex> lock(executor_mutex_);
        return strategy_executors_.find(strategy_id) != strategy_executors_.end();
    }
};

// Global instance
static MetamorphicEngine* g_metamorphic_engine = nullptr;

// C API
extern "C" {
    void metamorphic_engine_init() {
        if (g_metamorphic_engine == nullptr) {
            g_metamorphic_engine = new MetamorphicEngine();
        }
    }
    
    void metamorphic_engine_register_strategy(const char* strategy_id, int model_type) {
        if (g_metamorphic_engine == nullptr) {
            metamorphic_engine_init();
        }
        g_metamorphic_engine->register_strategy(std::string(strategy_id), (ModelType)model_type);
    }
    
    void metamorphic_engine_record_trade(const char* strategy_id, double pnl, int64_t timestamp_ns, int is_win) {
        if (g_metamorphic_engine == nullptr) {
            return;
        }
        g_metamorphic_engine->record_trade(std::string(strategy_id), pnl, timestamp_ns, is_win != 0);
    }
    
    int metamorphic_engine_get_evolution_status(const char* strategy_id, int* current_model, int* proposed_model,
                                                double* alpha_decay_rate, int* evolution_cycle) {
        if (g_metamorphic_engine == nullptr) {
            return 0;
        }
        ModelType curr, prop;
        bool triggered = g_metamorphic_engine->get_evolution_status(
            std::string(strategy_id), &curr, &prop, alpha_decay_rate, evolution_cycle);
        *current_model = (int)curr;
        *proposed_model = (int)prop;
        return triggered ? 1 : 0;
    }
    
    void metamorphic_engine_apply_evolution(const char* strategy_id) {
        if (g_metamorphic_engine == nullptr) {
            return;
        }
        g_metamorphic_engine->apply_evolution(std::string(strategy_id));
    }
    
    double metamorphic_engine_get_sel() {
        if (g_metamorphic_engine == nullptr) {
            return 0.0;
        }
        return g_metamorphic_engine->get_system_evolution_level();
    }
    
    void metamorphic_engine_cleanup() {
        if (g_metamorphic_engine != nullptr) {
            delete g_metamorphic_engine;
            g_metamorphic_engine = nullptr;
        }
    }
}

// C++ API for Hot-Swap (exposed via pybind11)
// These functions allow Python to register and execute strategies without restarting C++ process

// C-compatible wrappers for basic queries
extern "C" {
    // Hot-Swap: Check if executor is registered (C-compatible wrapper)
    int metamorphic_engine_has_executor(const char* strategy_id) {
        if (g_metamorphic_engine == nullptr) {
            return 0;
        }
        return g_metamorphic_engine->has_executor(std::string(strategy_id)) ? 1 : 0;
    }
    
    // Hot-Swap: Get executor update timestamp
    int64_t metamorphic_engine_get_executor_timestamp(const char* strategy_id) {
        if (g_metamorphic_engine == nullptr) {
            return 0;
        }
        return g_metamorphic_engine->get_executor_update_timestamp(std::string(strategy_id));
    }
}

// C++ functions for pybind11 (not in extern "C")
// Note: These will be bound via pybind11 in python_bindings.cpp
bool metamorphic_engine_register_executor_cpp(const std::string& strategy_id, StrategyExecutionFunc executor) {
    if (g_metamorphic_engine == nullptr) {
        metamorphic_engine_init();
    }
    return g_metamorphic_engine->register_strategy_executor(strategy_id, executor);
}

std::pair<bool, double> metamorphic_engine_execute_strategy_cpp(
    const std::string& strategy_id, const char* symbol, double price, double size, int side, int64_t timestamp_ns) {
    if (g_metamorphic_engine == nullptr) {
        return std::make_pair(false, 0.0);
    }
    return g_metamorphic_engine->execute_strategy(strategy_id, symbol, price, size, side, timestamp_ns);
}
