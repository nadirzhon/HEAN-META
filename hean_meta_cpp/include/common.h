#pragma once

#include <string>
#include <chrono>
#include <atomic>
#include <memory>

namespace hean {

// High-resolution timestamp
using Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using Microseconds = std::chrono::microseconds;

inline Timestamp now() {
    return std::chrono::high_resolution_clock::now();
}

inline int64_t timestamp_us() {
    return std::chrono::duration_cast<Microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

// Order side
enum class Side {
    BUY,
    SELL
};

inline std::string to_string(Side side) {
    return side == Side::BUY ? "BUY" : "SELL";
}

inline Side side_from_string(const std::string& s) {
    return (s == "BUY" || s == "buy") ? Side::BUY : Side::SELL;
}

// Order type
enum class OrderType {
    MARKET,
    LIMIT,
    STOP_MARKET,
    STOP_LIMIT
};

inline std::string to_string(OrderType type) {
    switch (type) {
        case OrderType::MARKET: return "MARKET";
        case OrderType::LIMIT: return "LIMIT";
        case OrderType::STOP_MARKET: return "STOP_MARKET";
        case OrderType::STOP_LIMIT: return "STOP_LIMIT";
        default: return "UNKNOWN";
    }
}

// Order status
enum class OrderStatus {
    PENDING,
    SUBMITTED,
    PARTIALLY_FILLED,
    FILLED,
    CANCELLED,
    REJECTED,
    EXPIRED
};

inline std::string to_string(OrderStatus status) {
    switch (status) {
        case OrderStatus::PENDING: return "PENDING";
        case OrderStatus::SUBMITTED: return "SUBMITTED";
        case OrderStatus::PARTIALLY_FILLED: return "PARTIALLY_FILLED";
        case OrderStatus::FILLED: return "FILLED";
        case OrderStatus::CANCELLED: return "CANCELLED";
        case OrderStatus::REJECTED: return "REJECTED";
        case OrderStatus::EXPIRED: return "EXPIRED";
        default: return "UNKNOWN";
    }
}

// Order structure
struct Order {
    std::string order_id;
    std::string symbol;
    Side side;
    OrderType type;
    double quantity;
    double price;
    double filled_quantity;
    double avg_fill_price;
    OrderStatus status;
    int64_t created_at_us;
    int64_t updated_at_us;
    std::string exchange_order_id;
    std::string error_message;

    Order()
        : side(Side::BUY)
        , type(OrderType::MARKET)
        , quantity(0.0)
        , price(0.0)
        , filled_quantity(0.0)
        , avg_fill_price(0.0)
        , status(OrderStatus::PENDING)
        , created_at_us(timestamp_us())
        , updated_at_us(timestamp_us())
    {}

    bool is_complete() const {
        return status == OrderStatus::FILLED
            || status == OrderStatus::CANCELLED
            || status == OrderStatus::REJECTED
            || status == OrderStatus::EXPIRED;
    }

    bool is_active() const {
        return status == OrderStatus::PENDING
            || status == OrderStatus::SUBMITTED
            || status == OrderStatus::PARTIALLY_FILLED;
    }

    double remaining_quantity() const {
        return quantity - filled_quantity;
    }
};

// Position structure
struct Position {
    std::string symbol;
    double quantity;      // Positive = long, negative = short
    double entry_price;
    double unrealized_pnl;
    double realized_pnl;
    int64_t updated_at_us;

    Position()
        : quantity(0.0)
        , entry_price(0.0)
        , unrealized_pnl(0.0)
        , realized_pnl(0.0)
        , updated_at_us(timestamp_us())
    {}

    bool is_long() const { return quantity > 0; }
    bool is_short() const { return quantity < 0; }
    bool is_flat() const { return quantity == 0; }

    Side get_side() const {
        return quantity >= 0 ? Side::BUY : Side::SELL;
    }
};

// Order result
struct OrderResult {
    bool success;
    std::string order_id;
    std::string error_message;
    int64_t latency_us;

    OrderResult()
        : success(false)
        , latency_us(0)
    {}

    OrderResult(bool success_, const std::string& order_id_, int64_t latency_)
        : success(success_)
        , order_id(order_id_)
        , latency_us(latency_)
    {}
};

} // namespace hean
