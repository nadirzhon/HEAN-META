/**
 * High-Frequency Triangular Arbitrage Scanner
 * Ultra-low latency cycle detection for 50+ trading pairs
 * Target: < 500 microseconds from detection to execution signal
 * 
 * Algorithm: Depth-First Search with pruning for triangular cycles
 * Formula: (Price_AB * Price_BC * Price_CA) > 1 + fee_buffer
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <chrono>
#include <thread>
#include <queue>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of trading pairs
#define MAX_PAIRS 100
#define MIN_PROFIT_BPS 5  // Minimum profit in basis points (0.05%)
#define FEE_BUFFER_BPS 10  // Fee buffer in basis points (0.10%)

// Trading pair structure
struct TradingPair {
    std::string symbol;      // e.g., "BTCUSDT"
    std::string base;        // e.g., "BTC"
    std::string quote;       // e.g., "USDT"
    double bid_price;        // Best bid price
    double ask_price;        // Best ask price
    double bid_size;         // Best bid size
    double ask_size;         // Best ask size
    int64_t timestamp_ns;    // Nanosecond timestamp
    bool is_active;          // Whether pair is active
    
    TradingPair() : bid_price(0.0), ask_price(0.0), bid_size(0.0), 
                    ask_size(0.0), timestamp_ns(0), is_active(false) {}
};

// Triangular cycle structure
struct TriangularCycle {
    std::string pair_a;      // First pair (A->B)
    std::string pair_b;      // Second pair (B->C)
    std::string pair_c;      // Third pair (C->A)
    std::string asset_a;     // Starting asset
    std::string asset_b;     // Intermediate asset
    std::string asset_c;     // Final asset
    double profit_ratio;     // (Price_AB * Price_BC * Price_CA) - 1
    double profit_bps;       // Profit in basis points
    double max_size;         // Maximum size we can trade (limited by liquidity)
    int64_t detection_time_ns;  // When cycle was detected
    
    TriangularCycle() : profit_ratio(0.0), profit_bps(0.0), max_size(0.0),
                        detection_time_ns(0) {}
};

// Asset graph node (for DFS traversal)
struct AssetNode {
    std::string asset;
    std::vector<std::pair<std::string, double>> edges;  // (pair_symbol, price)
    std::vector<std::pair<std::string, double>> reverse_edges;  // Reverse direction
    
    AssetNode() {}
};

// Triangular Scanner class
class TriangularScanner {
private:
    // Pair storage: symbol -> TradingPair
    std::unordered_map<std::string, TradingPair> pairs_;
    std::mutex pairs_mutex_;
    
    // Asset graph: asset -> AssetNode
    std::unordered_map<std::string, AssetNode> asset_graph_;
    std::mutex graph_mutex_;
    
    // Detected cycles (priority queue by profit)
    std::priority_queue<TriangularCycle, std::vector<TriangularCycle>,
                       std::function<bool(const TriangularCycle&, const TriangularCycle&)>> cycle_queue_;
    std::mutex queue_mutex_;
    
    // Configuration
    double fee_buffer_;           // Fee buffer (as ratio, e.g., 0.001 = 0.1%)
    double min_profit_bps_;       // Minimum profit in basis points
    int max_depth_;               // Maximum cycle depth (always 3 for triangular)
    int64_t max_age_ns_;          // Maximum age for price data (1ms = 1e6 ns)
    
    // Performance metrics
    int64_t total_scans_;
    int64_t cycles_found_;
    int64_t total_latency_ns_;
    
    // Extract base and quote from symbol (e.g., "BTCUSDT" -> "BTC", "USDT")
    void parse_symbol(const std::string& symbol, std::string& base, std::string& quote) {
        // Common quote currencies to try
        std::vector<std::string> quotes = {"USDT", "USDC", "BTC", "ETH", "BNB", "BUSD"};
        
        base = "";
        quote = "";
        
        for (const auto& q : quotes) {
            if (symbol.size() > q.size() && 
                symbol.substr(symbol.size() - q.size()) == q) {
                base = symbol.substr(0, symbol.size() - q.size());
                quote = q;
                return;
            }
        }
        
        // Fallback: assume last 4 chars are quote
        if (symbol.size() > 4) {
            base = symbol.substr(0, symbol.size() - 4);
            quote = symbol.substr(symbol.size() - 4);
        } else {
            base = symbol;
            quote = "";
        }
    }
    
    // Build asset graph from pairs (for fast cycle detection)
    void rebuild_graph() {
        std::lock_guard<std::mutex> lock(pairs_mutex_);
        std::lock_guard<std::mutex> graph_lock(graph_mutex_);
        
        asset_graph_.clear();
        
        for (const auto& pair_it : pairs_) {
            const TradingPair& pair = pair_it.second;
            if (!pair.is_active || pair.bid_price <= 0 || pair.ask_price <= 0) {
                continue;
            }
            
            std::string base, quote;
            parse_symbol(pair.symbol, base, quote);
            
            if (base.empty() || quote.empty()) {
                continue;
            }
            
            // Add forward edge: base -> quote (sell base, buy quote)
            asset_graph_[base].asset = base;
            asset_graph_[base].edges.push_back({pair.symbol, pair.bid_price});
            asset_graph_[quote].reverse_edges.push_back({pair.symbol, pair.bid_price});
            
            // Add reverse edge: quote -> base (sell quote, buy base)
            asset_graph_[quote].asset = quote;
            asset_graph_[quote].edges.push_back({pair.symbol, 1.0 / pair.ask_price});
            asset_graph_[base].reverse_edges.push_back({pair.symbol, 1.0 / pair.ask_price});
        }
    }
    
    // Depth-First Search for triangular cycles (A->B->C->A)
    void find_cycles_dfs(const std::string& start_asset, 
                        const std::string& current_asset,
                        std::vector<std::string>& path,
                        std::unordered_set<std::string>& visited,
                        std::vector<TriangularCycle>& cycles,
                        double cumulative_ratio,
                        int depth) {
        // Triangular arbitrage: exactly 3 steps (depth 0, 1, 2 -> return to start at depth 3)
        if (depth >= 3) {
            // Check if we've returned to start (triangular cycle)
            if (depth == 3 && current_asset == start_asset) {
                // Calculate profit ratio
                double profit_ratio = cumulative_ratio - 1.0;
                double profit_bps = profit_ratio * 10000.0;  // Convert to basis points
                
                // Check if profit exceeds minimum threshold (including fees)
                double required_profit_bps = min_profit_bps_ + FEE_BUFFER_BPS;
                
                if (profit_bps >= required_profit_bps && path.size() == 3) {
                    TriangularCycle cycle;
                    cycle.pair_a = path[0];
                    cycle.pair_b = path[1];
                    cycle.pair_c = path[2];
                    cycle.asset_a = start_asset;
                    cycle.asset_b = "";  // Will be determined from pairs
                    cycle.asset_c = "";
                    cycle.profit_ratio = profit_ratio;
                    cycle.profit_bps = profit_bps;
                    cycle.detection_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()
                    ).count();
                    
                    // Calculate maximum size (limited by smallest liquidity)
                    cycle.max_size = calculate_max_size(cycle);
                    
                    cycles.push_back(cycle);
                }
            }
            return;
        }
        
        // Prune: if cumulative ratio is already too low, skip
        double required_profit_bps = min_profit_bps_ + FEE_BUFFER_BPS;
        double min_required_ratio = 1.0 + (required_profit_bps / 10000.0);
        if (cumulative_ratio < min_required_ratio * 0.8) {  // Early pruning
            return;
        }
        
        // Get current node
        auto node_it = asset_graph_.find(current_asset);
        if (node_it == asset_graph_.end()) {
            return;
        }
        
        const AssetNode& node = node_it->second;
        
        // Try all edges from current asset
        for (const auto& edge : node.edges) {
            const std::string& pair_symbol = edge.first;
            double price = edge.second;
            
            // Skip if pair already in path (avoid revisiting same pair)
            if (std::find(path.begin(), path.end(), pair_symbol) != path.end()) {
                continue;
            }
            
            // Get destination asset from pair
            TradingPair pair;
            {
                std::lock_guard<std::mutex> lock(pairs_mutex_);
                auto pair_it = pairs_.find(pair_symbol);
                if (pair_it == pairs_.end() || !pair_it->second.is_active) {
                    continue;
                }
                pair = pair_it->second;
            }
            
            std::string base, quote;
            parse_symbol(pair_symbol, base, quote);
            
            // Determine destination asset
            std::string next_asset;
            if (current_asset == base) {
                next_asset = quote;
            } else if (current_asset == quote) {
                next_asset = base;
            } else {
                continue;  // Invalid edge
            }
            
            // Skip if already visited in this path (except start at end)
            if (depth < 2 && visited.find(next_asset) != visited.end()) {
                continue;
            }
            
            // Recursive search
            path.push_back(pair_symbol);
            if (depth < 2) {
                visited.insert(next_asset);
            }
            
            find_cycles_dfs(start_asset, next_asset, path, visited, cycles,
                          cumulative_ratio * price, depth + 1);
            
            // Backtrack
            path.pop_back();
            if (depth < 2) {
                visited.erase(next_asset);
            }
        }
    }
    
    // Calculate maximum tradeable size (limited by smallest liquidity)
    double calculate_max_size(const TriangularCycle& cycle) {
        std::lock_guard<std::mutex> lock(pairs_mutex_);
        
        double max_size = 1e10;  // Start with large value
        
        // Check liquidity for each leg
        auto pair_a_it = pairs_.find(cycle.pair_a);
        auto pair_b_it = pairs_.find(cycle.pair_b);
        auto pair_c_it = pairs_.find(cycle.pair_c);
        
        if (pair_a_it == pairs_.end() || pair_b_it == pairs_.end() || pair_c_it == pairs_.end()) {
            return 0.0;
        }
        
        const TradingPair& pair_a = pair_a_it->second;
        const TradingPair& pair_b = pair_b_it->second;
        const TradingPair& pair_c = pair_c_it->second;
        
        // For each pair, take minimum of bid/ask size (conservative)
        double size_a = std::min(pair_a.bid_size, pair_a.ask_size);
        double size_b = std::min(pair_b.bid_size, pair_b.ask_size);
        double size_c = std::min(pair_c.bid_size, pair_c.ask_size);
        
        // Maximum size is limited by smallest liquidity
        max_size = std::min({size_a, size_b, size_c, max_size});
        
        return max_size;
    }
    
public:
    TriangularScanner(double fee_buffer = 0.001, double min_profit_bps = 5.0) 
        : fee_buffer_(fee_buffer), min_profit_bps_(min_profit_bps), max_depth_(3),
          max_age_ns_(1000000),  // 1ms max age
          total_scans_(0), cycles_found_(0), total_latency_ns_(0),
          cycle_queue_([](const TriangularCycle& a, const TriangularCycle& b) {
              return a.profit_bps < b.profit_bps;  // Max heap by profit
          }) {
    }
    
    ~TriangularScanner() {}
    
    // Add or update trading pair
    void update_pair(const std::string& symbol, double bid_price, double ask_price,
                    double bid_size, double ask_size, int64_t timestamp_ns) {
        std::lock_guard<std::mutex> lock(pairs_mutex_);
        
        TradingPair& pair = pairs_[symbol];
        pair.symbol = symbol;
        pair.bid_price = bid_price;
        pair.ask_price = ask_price;
        pair.bid_size = bid_size;
        pair.ask_size = ask_size;
        pair.timestamp_ns = timestamp_ns;
        pair.is_active = (bid_price > 0 && ask_price > 0 && 
                         bid_size > 0 && ask_size > 0);
        
        // Update asset graph if pair became active
        if (pair.is_active) {
            std::lock_guard<std::mutex> graph_lock(graph_mutex_);
            rebuild_graph();
        }
    }
    
    // Scan for triangular arbitrage opportunities
    std::vector<TriangularCycle> scan_cycles() {
        auto start_time = std::chrono::steady_clock::now();
        
        std::vector<TriangularCycle> cycles;
        
        // Rebuild graph if needed (should be done incrementally, but this is safe)
        rebuild_graph();
        
        // Get all assets
        std::vector<std::string> assets;
        {
            std::lock_guard<std::mutex> graph_lock(graph_mutex_);
            for (const auto& node_it : asset_graph_) {
                assets.push_back(node_it.first);
            }
        }
        
        // Try each asset as starting point
        for (const std::string& start_asset : assets) {
            std::vector<std::string> path;
            std::unordered_set<std::string> visited;
            visited.insert(start_asset);
            
            find_cycles_dfs(start_asset, start_asset, path, visited, cycles, 1.0, 0);
        }
        
        // Sort by profit (highest first)
        std::sort(cycles.begin(), cycles.end(),
                 [](const TriangularCycle& a, const TriangularCycle& b) {
                     return a.profit_bps > b.profit_bps;
                 });
        
        // Keep top 10 cycles
        if (cycles.size() > 10) {
            cycles.resize(10);
        }
        
        // Update metrics
        auto end_time = std::chrono::steady_clock::now();
        int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time
        ).count();
        
        total_scans_++;
        cycles_found_ += cycles.size();
        total_latency_ns_ += latency_ns;
        
        return cycles;
    }
    
    // Get average latency in microseconds
    double get_avg_latency_us() const {
        if (total_scans_ == 0) return 0.0;
        return (total_latency_ns_ / static_cast<double>(total_scans_)) / 1000.0;
    }
    
    // Get statistics
    void get_stats(int64_t& total_scans, int64_t& cycles_found, double& avg_latency_us) const {
        total_scans = total_scans_;
        cycles_found = cycles_found_;
        avg_latency_us = get_avg_latency_us();
    }
    
    // Get number of active pairs
    int get_active_pair_count() const {
        std::lock_guard<std::mutex> lock(pairs_mutex_);
        int count = 0;
        for (const auto& pair_it : pairs_) {
            if (pair_it.second.is_active) {
                count++;
            }
        }
        return count;
    }
};

// C interface for Python bindings
static TriangularScanner* g_scanner = nullptr;

extern "C" {
    void triangular_scanner_init(double fee_buffer, double min_profit_bps) {
        if (g_scanner == nullptr) {
            g_scanner = new TriangularScanner(fee_buffer, min_profit_bps);
        }
    }
    
    void triangular_scanner_update_pair(const char* symbol, double bid_price, double ask_price,
                                       double bid_size, double ask_size, int64_t timestamp_ns) {
        if (g_scanner != nullptr) {
            g_scanner->update_pair(std::string(symbol), bid_price, ask_price,
                                  bid_size, ask_size, timestamp_ns);
        }
    }
    
    // Scan and return top cycle (simplified for C interface)
    // Full implementation would return array of cycles
    int triangular_scanner_scan_top_cycle(char* pair_a, char* pair_b, char* pair_c,
                                          double* profit_bps, double* max_size) {
        if (g_scanner == nullptr) {
            return 0;
        }
        
        std::vector<TriangularCycle> cycles = g_scanner->scan_cycles();
        
        if (cycles.empty()) {
            return 0;
        }
        
        const TriangularCycle& top = cycles[0];
        strncpy(pair_a, top.pair_a.c_str(), 32);
        strncpy(pair_b, top.pair_b.c_str(), 32);
        strncpy(pair_c, top.pair_c.c_str(), 32);
        *profit_bps = top.profit_bps;
        *max_size = top.max_size;
        
        return 1;
    }
    
    void triangular_scanner_get_stats(int64_t* total_scans, int64_t* cycles_found,
                                     double* avg_latency_us) {
        if (g_scanner != nullptr) {
            g_scanner->get_stats(*total_scans, *cycles_found, *avg_latency_us);
        } else {
            *total_scans = 0;
            *cycles_found = 0;
            *avg_latency_us = 0.0;
        }
    }
    
    int triangular_scanner_get_active_pair_count() {
        if (g_scanner == nullptr) return 0;
        return g_scanner->get_active_pair_count();
    }
    
    void triangular_scanner_cleanup() {
        if (g_scanner != nullptr) {
            delete g_scanner;
            g_scanner = nullptr;
        }
    }
}

#ifdef __cplusplus
}
#endif
