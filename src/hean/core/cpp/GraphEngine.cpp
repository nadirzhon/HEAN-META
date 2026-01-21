/**
 * Quantum Computing & AI Research: Predictive Graph-Based Engine
 * Real-time Adjacency Matrix for 50+ crypto assets with Lead-Lag Detection
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <mutex>
#include <chrono>
#include <thread>
#include "MemoryPool.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of assets (expandable)
#define MAX_ASSETS 100
#define MIN_CORRELATION_WINDOW 50
#define LEAD_LAG_DETECTION_WINDOW 100

// Asset data structure - using std::array instead of std::deque for zero-allocation
struct AssetData {
    std::string symbol;
    std::array<double, 100> prices;  // Fixed-size circular buffer
    std::array<double, 100> returns;
    int price_index;
    int return_index;
    int price_count;
    int return_count;
    double current_price;
    double volatility;
    int64_t last_update_timestamp_ns;
    bool is_leader;
    double leader_score;
    
    AssetData() : price_index(0), return_index(0), price_count(0), return_count(0),
                  current_price(0.0), volatility(0.0), last_update_timestamp_ns(0),
                  is_leader(false), leader_score(0.0) {
        prices.fill(0.0);
        returns.fill(0.0);
    }
    
    void add_price(double price) {
        prices[price_index] = price;
        price_index = (price_index + 1) % 100;
        if (price_count < 100) price_count++;
    }
    
    void add_return(double ret) {
        returns[return_index] = ret;
        return_index = (return_index + 1) % 100;
        if (return_count < 100) return_count++;
    }
};

// Correlation pair structure
struct CorrelationPair {
    int asset_a_idx;
    int asset_b_idx;
    double correlation;
    double lead_lag_score;  // Positive = A leads B, Negative = B leads A
    double p_value;
};

// Graph Engine class
class GraphEngine {
private:
    std::map<std::string, int> symbol_to_index_;
    std::array<AssetData, MAX_ASSETS> assets_;  // Fixed-size array
    double correlation_matrix_[MAX_ASSETS][MAX_ASSETS];
    double lead_lag_matrix_[MAX_ASSETS][MAX_ASSETS];
    std::mutex data_mutex_;
    int asset_count_;
    int window_size_;
    
    // Lead-lag detection using cross-correlation (optimized for arrays)
    double calculate_cross_correlation(const AssetData& asset_a, const AssetData& asset_b,
                                       int max_lag = 10) {
        if (asset_a.return_count < MIN_CORRELATION_WINDOW || 
            asset_b.return_count < MIN_CORRELATION_WINDOW) {
            return 0.0;
        }
        
        int min_size = std::min(asset_a.return_count, asset_b.return_count);
        if (min_size < MIN_CORRELATION_WINDOW) return 0.0;
        
        double best_correlation = 0.0;
        int best_lag = 0;
        
        // Try different lags to find maximum correlation
        for (int lag = -max_lag; lag <= max_lag; lag++) {
            double correlation = 0.0;
            int valid_pairs = 0;
            
            // Get aligned arrays from circular buffers
            int start_a = (asset_a.return_index - min_size + 100) % 100;
            int start_b = (asset_b.return_index - min_size + 100) % 100;
            
            for (int i = std::max(0, lag); i < min_size + lag && i < min_size; i++) {
                int j = i - lag;
                if (j >= 0 && j < min_size) {
                    int idx_a = (start_a + i) % 100;
                    int idx_b = (start_b + j) % 100;
                    correlation += asset_a.returns[idx_a] * asset_b.returns[idx_b];
                    valid_pairs++;
                }
            }
            
            if (valid_pairs > 0) {
                correlation /= valid_pairs;
                if (std::abs(correlation) > std::abs(best_correlation)) {
                    best_correlation = correlation;
                    best_lag = lag;
                }
            }
        }
        
        // Normalize and return lead-lag score (positive = A leads, negative = B leads)
        return best_lag * (best_correlation / std::abs(best_correlation));
    }
    
    // Calculate Pearson correlation (optimized for arrays)
    double calculate_pearson_correlation(const AssetData& asset_a, const AssetData& asset_b) {
        int count_a = asset_a.return_count;
        int count_b = asset_b.return_count;
        if (count_a != count_b || count_a < 2) {
            return 0.0;
        }
        
        int n = count_a;
        double mean_a = 0.0, mean_b = 0.0;
        
        int start_a = (asset_a.return_index - n + 100) % 100;
        int start_b = (asset_b.return_index - n + 100) % 100;
        
        for (int i = 0; i < n; i++) {
            mean_a += asset_a.returns[(start_a + i) % 100];
            mean_b += asset_b.returns[(start_b + i) % 100];
        }
        mean_a /= n;
        mean_b /= n;
        
        double covariance = 0.0, variance_a = 0.0, variance_b = 0.0;
        for (int i = 0; i < n; i++) {
            double val_a = asset_a.returns[(start_a + i) % 100] - mean_a;
            double val_b = asset_b.returns[(start_b + i) % 100] - mean_b;
            covariance += val_a * val_b;
            variance_a += val_a * val_a;
            variance_b += val_b * val_b;
        }
        
        double denominator = std::sqrt(variance_a * variance_b);
        if (denominator < 1e-10) return 0.0;
        
        return covariance / denominator;
    }
    
    // Identify current leader (highest leader score)
    void update_leader_scores() {
        for (int i = 0; i < asset_count_; i++) {
            double total_lead_score = 0.0;
            int count = 0;
            
            for (int j = 0; j < asset_count_; j++) {
                if (i != j && std::abs(lead_lag_matrix_[i][j]) > 0.1) {
                    // If lead_lag_matrix_[i][j] > 0, asset i leads asset j
                    total_lead_score += lead_lag_matrix_[i][j];
                    count++;
                }
            }
            
            assets_[i].leader_score = count > 0 ? total_lead_score / count : 0.0;
        }
        
        // Find asset with highest leader score
        int leader_idx = 0;
        double max_score = assets_[0].leader_score;
        for (int i = 1; i < asset_count_; i++) {
            if (assets_[i].leader_score > max_score) {
                max_score = assets_[i].leader_score;
                leader_idx = i;
            }
        }
        
        // Mark leader (only if score is significantly positive)
        for (int i = 0; i < asset_count_; i++) {
            assets_[i].is_leader = (i == leader_idx && max_score > 0.3);
        }
    }
    
public:
    GraphEngine(int window_size = 100) : asset_count_(0), window_size_(window_size) {
        // Initialize matrices
        for (int i = 0; i < MAX_ASSETS; i++) {
            for (int j = 0; j < MAX_ASSETS; j++) {
                correlation_matrix_[i][j] = 0.0;
                lead_lag_matrix_[i][j] = 0.0;
            }
        }
    }
    
    // Add or update asset
    int add_asset(const std::string& symbol) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        auto it = symbol_to_index_.find(symbol);
        if (it != symbol_to_index_.end()) {
            return it->second;
        }
        
        if (asset_count_ >= MAX_ASSETS) {
            return -1;  // Max assets reached
        }
        
        int idx = asset_count_++;
        symbol_to_index_[symbol] = idx;
        
        AssetData asset;
        asset.symbol = symbol;
        assets_[idx] = asset;
        return idx;
    }
    
    // Update price for an asset
    bool update_price(const std::string& symbol, double price, int64_t timestamp_ns) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        auto it = symbol_to_index_.find(symbol);
        if (it == symbol_to_index_.end()) {
            return false;
        }
        
        int idx = it->second;
        AssetData& asset = assets_[idx];
        
        // Calculate return if we have previous price
        if (asset.price_count > 0 && asset.current_price > 0.0) {
            double prev_price = asset.current_price;
            double ret = (price - prev_price) / prev_price;
            asset.add_return(ret);
            
            // Calculate rolling volatility
            if (asset.return_count >= 20) {
                int n = asset.return_count;
                int start_idx = (asset.return_index - n + 100) % 100;
                
                double mean = 0.0;
                for (int i = 0; i < n; i++) {
                    mean += asset.returns[(start_idx + i) % 100];
                }
                mean /= n;
                
                double variance = 0.0;
                for (int i = 0; i < n; i++) {
                    double diff = asset.returns[(start_idx + i) % 100] - mean;
                    variance += diff * diff;
                }
                variance /= n;
                asset.volatility = std::sqrt(variance * 252.0);  // Annualized
            }
        }
        
        asset.add_price(price);
        asset.current_price = price;
        asset.last_update_timestamp_ns = timestamp_ns;
        
        return true;
    }
    
    // Recalculate adjacency matrix and lead-lag relationships
    void recalculate_matrix() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // Calculate correlation matrix
        for (int i = 0; i < asset_count_; i++) {
            for (int j = 0; j < asset_count_; j++) {
                if (i == j) {
                    correlation_matrix_[i][j] = 1.0;
                    lead_lag_matrix_[i][j] = 0.0;
                } else if (assets_[i].return_count >= MIN_CORRELATION_WINDOW &&
                          assets_[j].return_count >= MIN_CORRELATION_WINDOW) {
                    // Align series lengths
                    int min_len = std::min(assets_[i].returns.size(), assets_[j].returns.size());
                    
                    correlation_matrix_[i][j] = calculate_pearson_correlation(assets_[i], assets_[j]);
                    lead_lag_matrix_[i][j] = calculate_cross_correlation(assets_[i], assets_[j]);
                } else {
                    correlation_matrix_[i][j] = 0.0;
                    lead_lag_matrix_[i][j] = 0.0;
                }
            }
        }
        
        // Update leader scores
        update_leader_scores();
    }
    
    // Get high-dimensional feature vector (flattened adjacency matrix + metadata)
    void get_feature_vector(double* output, int max_size) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        int idx = 0;
        
        // Flatten correlation matrix (upper triangle only, symmetric)
        for (int i = 0; i < asset_count_ && idx < max_size; i++) {
            for (int j = i; j < asset_count_ && idx < max_size; j++) {
                output[idx++] = correlation_matrix_[i][j];
            }
        }
        
        // Flatten lead-lag matrix (upper triangle)
        for (int i = 0; i < asset_count_ && idx < max_size; i++) {
            for (int j = i; j < asset_count_ && idx < max_size; j++) {
                output[idx++] = lead_lag_matrix_[i][j];
            }
        }
        
        // Add asset metadata (volatility, leader scores)
        for (int i = 0; i < asset_count_ && idx < max_size; i++) {
            output[idx++] = assets_[i].volatility;
        }
        
        for (int i = 0; i < asset_count_ && idx < max_size; i++) {
            output[idx++] = assets_[i].leader_score;
        }
        
        // Pad with zeros if needed
        while (idx < max_size) {
            output[idx++] = 0.0;
        }
    }
    
    // Get current leader
    const char* get_current_leader() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        for (int i = 0; i < asset_count_; i++) {
            if (assets_[i].is_leader) {
                return assets_[i].symbol.c_str();
            }
        }
        return nullptr;
    }
    
    // Get laggards (assets with negative lead scores relative to leader)
    int get_laggards(const char** output, int max_count) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        int leader_idx = -1;
        for (int i = 0; i < asset_count_; i++) {
            if (assets_[i].is_leader) {
                leader_idx = i;
                break;
            }
        }
        
        if (leader_idx == -1) {
            return 0;
        }
        
        int count = 0;
        for (int i = 0; i < asset_count_ && count < max_count; i++) {
            if (i != leader_idx && lead_lag_matrix_[leader_idx][i] > 0.2) {
                // Asset i is lagging behind leader
                static std::array<std::string, MAX_ASSETS> laggard_symbols;
                static int laggard_idx = 0;
                if (laggard_idx < MAX_ASSETS) {
                    laggard_symbols[laggard_idx] = assets_[i].symbol;
                    output[count++] = laggard_symbols[laggard_idx].c_str();
                    laggard_idx++;
                }
            }
        }
        
        return count;
    }
    
    // Get correlation between two assets
    double get_correlation(const std::string& symbol_a, const std::string& symbol_b) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        auto it_a = symbol_to_index_.find(symbol_a);
        auto it_b = symbol_to_index_.find(symbol_b);
        
        if (it_a == symbol_to_index_.end() || it_b == symbol_to_index_.end()) {
            return 0.0;
        }
        
        return correlation_matrix_[it_a->second][it_b->second];
    }
    
    // Get lead-lag relationship (positive = a leads b, negative = b leads a)
    double get_lead_lag(const std::string& symbol_a, const std::string& symbol_b) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        auto it_a = symbol_to_index_.find(symbol_a);
        auto it_b = symbol_to_index_.find(symbol_b);
        
        if (it_a == symbol_to_index_.end() || it_b == symbol_to_index_.end()) {
            return 0.0;
        }
        
        return lead_lag_matrix_[it_a->second][it_b->second];
    }
    
    int get_asset_count() const {
        return asset_count_;
    }
};

// C interface for Python bindings
static GraphEngine* g_engine = nullptr;

extern "C" {
    void graph_engine_init(int window_size) {
        if (g_engine == nullptr) {
            g_engine = new GraphEngine(window_size);
        }
    }
    
    int graph_engine_add_asset(const char* symbol) {
        if (g_engine == nullptr) return -1;
        return g_engine->add_asset(std::string(symbol));
    }
    
    int graph_engine_update_price(const char* symbol, double price, int64_t timestamp_ns) {
        if (g_engine == nullptr) return 0;
        return g_engine->update_price(std::string(symbol), price, timestamp_ns) ? 1 : 0;
    }
    
    void graph_engine_recalculate() {
        if (g_engine != nullptr) {
            g_engine->recalculate_matrix();
        }
    }
    
    void graph_engine_get_feature_vector(double* output, int size) {
        if (g_engine != nullptr) {
            g_engine->get_feature_vector(output, size);
        }
    }
    
    const char* graph_engine_get_leader() {
        if (g_engine == nullptr) return nullptr;
        return g_engine->get_current_leader();
    }
    
    double graph_engine_get_correlation(const char* symbol_a, const char* symbol_b) {
        if (g_engine == nullptr) return 0.0;
        return g_engine->get_correlation(std::string(symbol_a), std::string(symbol_b));
    }
    
    double graph_engine_get_lead_lag(const char* symbol_a, const char* symbol_b) {
        if (g_engine == nullptr) return 0.0;
        return g_engine->get_lead_lag(std::string(symbol_a), std::string(symbol_b));
    }
    
    int graph_engine_get_asset_count() {
        if (g_engine == nullptr) return 0;
        return g_engine->get_asset_count();
    }
    
    void graph_engine_cleanup() {
        if (g_engine != nullptr) {
            delete g_engine;
            g_engine = nullptr;
        }
    }
}

#ifdef __cplusplus
}
#endif
