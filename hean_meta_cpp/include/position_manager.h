#pragma once

#include "common.h"
#include <unordered_map>
#include <mutex>
#include <optional>

namespace hean {

/**
 * Thread-safe position manager.
 *
 * Tracks positions across multiple symbols with PnL calculation.
 */
class PositionManager {
public:
    PositionManager();
    ~PositionManager();

    // Position updates
    void update_position(
        const std::string& symbol,
        double quantity_delta,
        double price
    );

    void close_position(const std::string& symbol, double price);

    // Position queries
    std::optional<Position> get_position(const std::string& symbol) const;
    std::vector<Position> get_all_positions() const;
    std::vector<Position> get_open_positions() const;

    // PnL calculations
    double get_total_unrealized_pnl(
        const std::unordered_map<std::string, double>& current_prices
    ) const;

    double get_total_realized_pnl() const;

    void update_unrealized_pnl(
        const std::string& symbol,
        double current_price
    );

    // Statistics
    size_t get_position_count() const;
    size_t get_open_position_count() const;

private:
    std::unordered_map<std::string, Position> positions_;
    mutable std::mutex positions_mutex_;

    // Helper to calculate PnL
    double calculate_pnl(
        const Position& pos,
        double current_price
    ) const;
};

} // namespace hean
