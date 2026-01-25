#pragma once

#include "common.h"
#include <unordered_map>
#include <vector>
#include <mutex>
#include <atomic>
#include <optional>

namespace hean {

/**
 * High-performance order execution engine.
 *
 * Thread-safe, lock-free where possible, optimized for low latency.
 * Target: <100 microseconds per operation.
 */
class OrderEngine {
public:
    OrderEngine();
    ~OrderEngine();

    // Order placement (thread-safe)
    OrderResult place_market_order(
        const std::string& symbol,
        Side side,
        double quantity
    );

    OrderResult place_limit_order(
        const std::string& symbol,
        Side side,
        double quantity,
        double price
    );

    // Order management
    bool cancel_order(const std::string& order_id);
    bool update_order(const std::string& order_id, double new_price);

    // Order status
    std::optional<Order> get_order(const std::string& order_id) const;
    std::vector<Order> get_active_orders() const;
    std::vector<Order> get_orders_by_symbol(const std::string& symbol) const;

    // Statistics
    size_t get_total_orders() const { return order_count_.load(); }
    size_t get_active_order_count() const;
    int64_t get_avg_latency_us() const { return avg_latency_us_.load(); }

    // Simulated fill (for testing/paper trading)
    void simulate_fill(const std::string& order_id, double fill_price);

private:
    // Order storage (thread-safe access via mutex)
    std::unordered_map<std::string, Order> orders_;
    mutable std::mutex orders_mutex_;

    // Atomic counters (lock-free)
    std::atomic<size_t> order_count_{0};
    std::atomic<int64_t> total_latency_us_{0};
    std::atomic<int64_t> avg_latency_us_{0};

    // Internal helpers
    std::string generate_order_id();
    void update_latency_stats(int64_t latency_us);
    OrderResult place_order_internal(Order&& order);
};

} // namespace hean
