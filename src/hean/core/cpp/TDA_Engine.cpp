/**
 * Topological Data Analysis Engine Implementation
 * Computes Persistent Homology of L2 Orderbook for market structure analysis
 */

#include "TDA_Engine.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

TDA_Engine::TDA_Engine(
    int max_levels,
    int max_dimension,
    double max_distance,
    int64_t update_interval_ms
) : _max_levels(max_levels),
    _max_dimension(max_dimension),
    _max_distance(max_distance),
    _update_interval_ns(update_interval_ms * 1000000),
    _running(false)
{
}

TDA_Engine::~TDA_Engine() {
    stop();
}

std::vector<std::vector<double>> TDA_Engine::_build_point_cloud(
    const std::vector<OrderbookLevel>& orderbook,
    int max_points
) {
    std::vector<std::vector<double>> point_cloud;
    
    if (orderbook.empty()) {
        return point_cloud;
    }
    
    // Extract prices and sizes, normalize to [0, 1] range
    std::vector<double> prices;
    std::vector<double> sizes;
    
    int num_levels = std::min(static_cast<int>(orderbook.size()), max_points);
    for (int i = 0; i < num_levels; i++) {
        prices.push_back(orderbook[i].price);
        sizes.push_back(orderbook[i].size);
    }
    
    if (prices.empty()) {
        return point_cloud;
    }
    
    // Normalize prices (relative to mid-price)
    double mid_price = (prices.front() + prices.back()) / 2.0;
    double price_range = prices.back() - prices.front();
    if (price_range < 1e-10) price_range = 1.0;
    
    // Normalize sizes
    double max_size = *std::max_element(sizes.begin(), sizes.end());
    if (max_size < 1e-10) max_size = 1.0;
    
    // Build 2D point cloud: [normalized_price_offset, normalized_size]
    for (size_t i = 0; i < prices.size(); i++) {
        double norm_price = (prices[i] - mid_price) / price_range;
        double norm_size = sizes[i] / max_size;
        point_cloud.push_back({norm_price, norm_size});
    }
    
    return point_cloud;
}

std::vector<PersistencePair> TDA_Engine::_compute_persistence(
    const std::vector<std::vector<double>>& point_cloud,
    double max_filtration
) {
    std::vector<PersistencePair> persistence_pairs;
    
    if (point_cloud.size() < 2) {
        return persistence_pairs;
    }
    
    // Simplified Vietoris-Rips persistence computation
    // In production, use a library like GUDHI or Ripser
    
    // Compute distance matrix
    size_t n = point_cloud.size();
    std::vector<std::vector<double>> distances(n, std::vector<double>(n, 0.0));
    
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            double dist = 0.0;
            for (size_t k = 0; k < point_cloud[i].size(); k++) {
                double diff = point_cloud[i][k] - point_cloud[j][k];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }
    
    // Find connected components (0D homology) at different filtration values
    // Birth = distance when component appears, death = when it merges
    std::vector<bool> visited(n, false);
    std::vector<double> birth_times(n, 0.0);
    
    // Each point is born at filtration = 0
    for (size_t i = 0; i < n; i++) {
        birth_times[i] = 0.0;
        persistence_pairs.push_back({0.0, std::numeric_limits<double>::infinity(), 0});
    }
    
    // Find merge events (deaths of components)
    std::vector<std::pair<double, std::pair<int, int>>> edges;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (distances[i][j] <= max_filtration) {
                edges.push_back({distances[i][j], {i, j}});
            }
        }
    }
    std::sort(edges.begin(), edges.end());
    
    // Union-Find for connected components
    std::vector<int> parent(n);
    for (size_t i = 0; i < n; i++) {
        parent[i] = static_cast<int>(i);
    }
    
    auto find = [&](int x) -> int {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    
    // Process edges and record deaths
    for (const auto& edge : edges) {
        double dist = edge.first;
        int u = edge.second.first;
        int v = edge.second.second;
        
        int root_u = find(u);
        int root_v = find(v);
        
        if (root_u != root_v) {
            // Components merge at this distance
            // The older component dies, the younger survives
            if (birth_times[root_u] < birth_times[root_v]) {
                // root_u dies
                persistence_pairs[root_u].death = dist;
                parent[root_u] = root_v;
            } else {
                // root_v dies
                persistence_pairs[root_v].death = dist;
                parent[root_v] = root_u;
            }
        }
    }
    
    // Compute 1D homology (loops) - simplified
    // In production, use proper simplicial complex computation
    // For now, estimate based on gaps in point cloud
    int num_holes = 0;
    for (size_t i = 1; i < point_cloud.size(); i++) {
        double gap = std::abs(point_cloud[i][0] - point_cloud[i-1][0]);
        if (gap > max_filtration * 0.5) {
            num_holes++;
            persistence_pairs.push_back({gap * 0.3, gap, 1});
        }
    }
    
    return persistence_pairs;
}

double TDA_Engine::_compute_riemannian_curvature(
    const std::vector<OrderbookLevel>& orderbook
) {
    if (orderbook.size() < 3) {
        return 0.0;
    }
    
    // Compute curvature proxy from orderbook geometry
    // Using discrete approximation: K ≈ (d²y/dx²) / (1 + (dy/dx)²)^(3/2)
    
    std::vector<double> prices;
    std::vector<double> sizes;
    
    for (const auto& level : orderbook) {
        prices.push_back(level.price);
        sizes.push_back(level.size);
    }
    
    // Compute second derivative of size w.r.t. price (curvature indicator)
    double total_curvature = 0.0;
    int count = 0;
    
    for (size_t i = 1; i < prices.size() - 1; i++) {
        double dx1 = prices[i] - prices[i-1];
        double dx2 = prices[i+1] - prices[i];
        double dy1 = sizes[i] - sizes[i-1];
        double dy2 = sizes[i+1] - sizes[i];
        
        if (std::abs(dx1) > 1e-10 && std::abs(dx2) > 1e-10) {
            double dydx1 = dy1 / dx1;
            double dydx2 = dy2 / dx2;
            double d2ydx2 = (dydx2 - dydx1) / ((dx1 + dx2) / 2.0);
            
            // Riemannian curvature approximation
            double curvature = d2ydx2 / std::pow(1.0 + dydx1 * dydx1, 1.5);
            total_curvature += curvature;
            count++;
        }
    }
    
    return count > 0 ? total_curvature / count : 0.0;
}

bool TDA_Engine::_check_disconnected(
    const std::vector<PersistencePair>& persistence_pairs
) {
    // Check if there are long-lived 0D homology features (disconnected components)
    for (const auto& pair : persistence_pairs) {
        if (pair.dimension == 0) {
            double persistence = pair.death - pair.birth;
            // If a component persists for a long time, manifold is disconnected
            if (persistence > _max_distance * 0.5 && !std::isinf(pair.death)) {
                return true;
            }
        }
    }
    return false;
}

void TDA_Engine::_update_loop() {
    while (_running) {
        auto start = std::chrono::steady_clock::now();
        
        // Update topology for all symbols
        {
            std::lock_guard<std::mutex> lock(_data_mutex);
            
            for (auto& [symbol, orderbook] : _orderbooks) {
                if (orderbook.empty()) continue;
                
                // Build point cloud
                auto point_cloud = _build_point_cloud(orderbook, _max_levels);
                if (point_cloud.size() < 2) continue;
                
                // Compute persistence
                auto persistence = _compute_persistence(point_cloud, _max_distance);
                _persistence_barcodes[symbol] = persistence;
                
                // Compute topology score
                TopologyScore score;
                
                // Stability score: based on average persistence
                double total_persistence = 0.0;
                int num_pairs = 0;
                for (const auto& pair : persistence) {
                    if (!std::isinf(pair.death)) {
                        total_persistence += (pair.death - pair.birth);
                        num_pairs++;
                    }
                }
                score.avg_persistence = num_pairs > 0 ? total_persistence / num_pairs : 0.0;
                score.stability_score = std::min(1.0, score.avg_persistence / _max_distance);
                
                // Connectivity score: inverse of disconnectedness
                score.is_disconnected = _check_disconnected(persistence);
                score.connectivity_score = score.is_disconnected ? 0.0 : 1.0;
                
                // Count holes (1D homology features)
                score.num_holes = 0;
                for (const auto& pair : persistence) {
                    if (pair.dimension == 1 && !std::isinf(pair.death)) {
                        score.num_holes++;
                    }
                }
                
                // Riemannian curvature
                score.curvature_score = _compute_riemannian_curvature(orderbook);
                
                // Timestamp
                auto now = std::chrono::system_clock::now();
                score.last_update_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now.time_since_epoch()
                ).count();
                
                _topology_scores[symbol] = score;
            }
        }
        
        // Sleep until next update interval
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto sleep_time = std::chrono::nanoseconds(_update_interval_ns) - elapsed;
        if (sleep_time.count() > 0) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
}

void TDA_Engine::update_orderbook(
    const std::string& symbol,
    const std::vector<OrderbookLevel>& levels
) {
    std::lock_guard<std::mutex> lock(_data_mutex);
    _orderbooks[symbol] = levels;
}

TopologyScore TDA_Engine::get_topology_score(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(_data_mutex);
    auto it = _topology_scores.find(symbol);
    if (it != _topology_scores.end()) {
        return it->second;
    }
    
    // Return default score if not found
    TopologyScore default_score;
    default_score.stability_score = 1.0;
    default_score.connectivity_score = 1.0;
    default_score.curvature_score = 0.0;
    default_score.is_disconnected = false;
    default_score.num_holes = 0;
    default_score.avg_persistence = 0.0;
    default_score.last_update_ns = 0;
    return default_score;
}

double TDA_Engine::get_market_topology_score() {
    std::lock_guard<std::mutex> lock(_data_mutex);
    
    if (_topology_scores.empty()) {
        return 1.0;  // Default: stable market
    }
    
    // Aggregate scores across all symbols
    double total_stability = 0.0;
    double total_connectivity = 0.0;
    int count = 0;
    
    for (const auto& [symbol, score] : _topology_scores) {
        total_stability += score.stability_score;
        total_connectivity += score.connectivity_score;
        count++;
    }
    
    if (count == 0) {
        return 1.0;
    }
    
    // Weighted average: connectivity is critical
    double avg_stability = total_stability / count;
    double avg_connectivity = total_connectivity / count;
    
    // Market score: penalize heavily for disconnection
    return avg_connectivity * 0.7 + avg_stability * 0.3;
}

double TDA_Engine::predict_slippage(
    const std::string& symbol,
    double order_size,
    bool is_buy
) {
    std::lock_guard<std::mutex> lock(_data_mutex);
    
    auto it = _topology_scores.find(symbol);
    if (it == _topology_scores.end()) {
        return 0.01;  // Default: 1% slippage
    }
    
    const TopologyScore& score = it->second;
    auto orderbook_it = _orderbooks.find(symbol);
    if (orderbook_it == _orderbooks.end() || orderbook_it->second.empty()) {
        return 0.01;
    }
    
    // Predict slippage based on Riemannian curvature
    // High positive curvature = convex orderbook = lower slippage
    // High negative curvature = concave orderbook = higher slippage
    double base_slippage = 0.005;  // 0.5% base
    double curvature_factor = std::abs(score.curvature_score) * 10.0;  // Scale curvature
    
    // If curvature is positive (convex), reduce slippage
    // If curvature is negative (concave), increase slippage
    double predicted_slippage = base_slippage;
    if (score.curvature_score < 0) {
        predicted_slippage += curvature_factor;  // Increase for concave
    } else {
        predicted_slippage -= std::min(curvature_factor, base_slippage * 0.5);  // Decrease for convex
    }
    
    // Adjust for order size (larger orders = more slippage)
    double size_factor = std::min(1.0, order_size / 1000.0);  // Normalize to 1000 units
    predicted_slippage *= (1.0 + size_factor);
    
    // Ensure reasonable bounds
    predicted_slippage = std::max(0.001, std::min(0.1, predicted_slippage));  // 0.1% to 10%
    
    return predicted_slippage;
}

bool TDA_Engine::is_market_disconnected() {
    std::lock_guard<std::mutex> lock(_data_mutex);
    
    // Check if any symbol has disconnected manifold
    for (const auto& [symbol, score] : _topology_scores) {
        if (score.is_disconnected) {
            return true;
        }
    }
    
    return false;
}

void TDA_Engine::start() {
    if (_running) {
        return;
    }
    
    _running = true;
    _update_thread = std::thread(&TDA_Engine::_update_loop, this);
}

void TDA_Engine::stop() {
    if (!_running) {
        return;
    }
    
    _running = false;
    if (_update_thread.joinable()) {
        _update_thread.join();
    }
}

void TDA_Engine::force_update(const std::string& symbol) {
    // Trigger immediate update for a symbol
    // In a full implementation, this would wake the update thread
    // For now, just ensure data is ready
    std::lock_guard<std::mutex> lock(_data_mutex);
    if (_orderbooks.find(symbol) != _orderbooks.end()) {
        // Force recomputation on next cycle
    }
}