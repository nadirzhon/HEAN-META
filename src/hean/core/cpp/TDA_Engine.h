/**
 * Topological Data Analysis (TDA) Engine
 * Persistent Homology computation for L2 Orderbook structure analysis
 */

#ifndef TDA_ENGINE_H
#define TDA_ENGINE_H

#include <deque>
#include <map>
#include <mutex>
#include <vector>
#include <thread>
#include <atomic>
#include <string>
#include <chrono>

// Orderbook entry structure
struct OrderbookLevel {
    double price;
    double size;
    int64_t timestamp_ns;
};

// Persistent Homology Barcode (birth-death pairs)
struct PersistencePair {
    double birth;
    double death;
    int dimension;  // 0 = connected components, 1 = loops, 2 = voids
};

// Topology score metrics
struct TopologyScore {
    double stability_score;      // 0-1: 1 = perfectly stable, 0 = collapsing
    double connectivity_score;   // 0-1: 1 = well-connected, 0 = disconnected
    double curvature_score;      // Riemannian curvature proxy (-inf to +inf)
    bool is_disconnected;        // True if manifold has disconnected components
    int num_holes;               // Number of topological holes (1D homology)
    double avg_persistence;      // Average persistence lifetime
    int64_t last_update_ns;      // Last update timestamp
};

class TDA_Engine {
private:
    // Orderbook data per symbol
    std::map<std::string, std::vector<OrderbookLevel>> _orderbooks;
    std::mutex _data_mutex;
    
    // Persistent Homology computation cache
    std::map<std::string, std::vector<PersistencePair>> _persistence_barcodes;
    std::map<std::string, TopologyScore> _topology_scores;
    
    // Background update thread
    std::atomic<bool> _running;
    std::thread _update_thread;
    
    // Configuration
    int _max_levels;              // Maximum orderbook levels to analyze
    int _max_dimension;           // Maximum homology dimension (0, 1, 2)
    double _max_distance;         // Maximum filtration distance
    int64_t _update_interval_ns;  // Update interval in nanoseconds
    
    // Point cloud construction from orderbook
    std::vector<std::vector<double>> _build_point_cloud(
        const std::vector<OrderbookLevel>& orderbook,
        int max_points = 50
    );
    
    // Compute Vietoris-Rips complex persistence
    std::vector<PersistencePair> _compute_persistence(
        const std::vector<std::vector<double>>& point_cloud,
        double max_filtration
    );
    
    // Compute Riemannian curvature proxy from orderbook geometry
    double _compute_riemannian_curvature(
        const std::vector<OrderbookLevel>& orderbook
    );
    
    // Check if manifold is disconnected (critical for flash-crash detection)
    bool _check_disconnected(
        const std::vector<PersistencePair>& persistence_pairs
    );
    
    // Background update loop
    void _update_loop();
    
public:
    TDA_Engine(
        int max_levels = 50,
        int max_dimension = 1,
        double max_distance = 0.1,
        int64_t update_interval_ms = 100
    );
    
    ~TDA_Engine();
    
    // Update orderbook snapshot for a symbol
    void update_orderbook(
        const std::string& symbol,
        const std::vector<OrderbookLevel>& levels
    );
    
    // Get topology score (thread-safe, cached)
    TopologyScore get_topology_score(const std::string& symbol);
    
    // Get market topology score (aggregate across all symbols)
    double get_market_topology_score();
    
    // Get predicted slippage based on Riemannian curvature
    double predict_slippage(
        const std::string& symbol,
        double order_size,
        bool is_buy
    );
    
    // Check if market manifold is disconnected (watchdog)
    bool is_market_disconnected();
    
    // Start background update thread
    void start();
    
    // Stop background update thread
    void stop();
    
    // Force immediate update for a symbol
    void force_update(const std::string& symbol);
};

#endif // TDA_ENGINE_H