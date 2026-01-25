#include "order_engine.h"
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace hean {

OrderEngine::OrderEngine() {
    // Reserve space to avoid reallocations
    orders_.reserve(10000);
}

OrderEngine::~OrderEngine() = default;

std::string OrderEngine::generate_order_id() {
    // Generate unique order ID: timestamp + counter
    static std::atomic<uint64_t> counter{0};

    auto now_us = timestamp_us();
    auto count = counter.fetch_add(1, std::memory_order_relaxed);

    std::ostringstream oss;
    oss << "ORD_" << std::setfill('0') << std::setw(16) << now_us
        << "_" << std::setw(6) << count;

    return oss.str();
}

void OrderEngine::update_latency_stats(int64_t latency_us) {
    // Update moving average latency
    auto count = order_count_.load(std::memory_order_relaxed);
    auto total = total_latency_us_.fetch_add(latency_us, std::memory_order_relaxed);

    if (count > 0) {
        avg_latency_us_.store((total + latency_us) / (count + 1), std::memory_order_relaxed);
    }
}

OrderResult OrderEngine::place_order_internal(Order&& order) {
    auto start = now();

    // Generate order ID
    order.order_id = generate_order_id();
    order.created_at_us = timestamp_us();
    order.updated_at_us = order.created_at_us;
    order.status = OrderStatus::SUBMITTED;

    // Store order (thread-safe)
    {
        std::lock_guard<std::mutex> lock(orders_mutex_);
        orders_[order.order_id] = std::move(order);
    }

    // Update counters
    order_count_.fetch_add(1, std::memory_order_relaxed);

    // Calculate latency
    auto end = now();
    auto latency_us = std::chrono::duration_cast<Microseconds>(end - start).count();
    update_latency_stats(latency_us);

    return OrderResult(true, order.order_id, latency_us);
}

OrderResult OrderEngine::place_market_order(
    const std::string& symbol,
    Side side,
    double quantity
) {
    if (quantity <= 0) {
        return OrderResult(false, "", 0);
    }

    Order order;
    order.symbol = symbol;
    order.side = side;
    order.type = OrderType::MARKET;
    order.quantity = quantity;
    order.price = 0.0;  // Market orders don't have price

    return place_order_internal(std::move(order));
}

OrderResult OrderEngine::place_limit_order(
    const std::string& symbol,
    Side side,
    double quantity,
    double price
) {
    if (quantity <= 0 || price <= 0) {
        return OrderResult(false, "", 0);
    }

    Order order;
    order.symbol = symbol;
    order.side = side;
    order.type = OrderType::LIMIT;
    order.quantity = quantity;
    order.price = price;

    return place_order_internal(std::move(order));
}

bool OrderEngine::cancel_order(const std::string& order_id) {
    std::lock_guard<std::mutex> lock(orders_mutex_);

    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }

    auto& order = it->second;

    // Can only cancel active orders
    if (!order.is_active()) {
        return false;
    }

    order.status = OrderStatus::CANCELLED;
    order.updated_at_us = timestamp_us();

    return true;
}

bool OrderEngine::update_order(const std::string& order_id, double new_price) {
    if (new_price <= 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(orders_mutex_);

    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }

    auto& order = it->second;

    // Can only update limit orders
    if (order.type != OrderType::LIMIT || !order.is_active()) {
        return false;
    }

    order.price = new_price;
    order.updated_at_us = timestamp_us();

    return true;
}

std::optional<Order> OrderEngine::get_order(const std::string& order_id) const {
    std::lock_guard<std::mutex> lock(orders_mutex_);

    auto it = orders_.find(order_id);
    if (it != orders_.end()) {
        return it->second;
    }

    return std::nullopt;
}

std::vector<Order> OrderEngine::get_active_orders() const {
    std::lock_guard<std::mutex> lock(orders_mutex_);

    std::vector<Order> active_orders;
    active_orders.reserve(orders_.size());

    for (const auto& [id, order] : orders_) {
        if (order.is_active()) {
            active_orders.push_back(order);
        }
    }

    return active_orders;
}

std::vector<Order> OrderEngine::get_orders_by_symbol(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(orders_mutex_);

    std::vector<Order> symbol_orders;
    symbol_orders.reserve(orders_.size() / 10);  // Estimate

    for (const auto& [id, order] : orders_) {
        if (order.symbol == symbol) {
            symbol_orders.push_back(order);
        }
    }

    return symbol_orders;
}

size_t OrderEngine::get_active_order_count() const {
    std::lock_guard<std::mutex> lock(orders_mutex_);

    return std::count_if(orders_.begin(), orders_.end(),
        [](const auto& pair) { return pair.second.is_active(); });
}

void OrderEngine::simulate_fill(const std::string& order_id, double fill_price) {
    std::lock_guard<std::mutex> lock(orders_mutex_);

    auto it = orders_.find(order_id);
    if (it == orders_.end() || !it->second.is_active()) {
        return;
    }

    auto& order = it->second;

    // Simulate complete fill
    order.filled_quantity = order.quantity;
    order.avg_fill_price = fill_price;
    order.status = OrderStatus::FILLED;
    order.updated_at_us = timestamp_us();
}

} // namespace hean
