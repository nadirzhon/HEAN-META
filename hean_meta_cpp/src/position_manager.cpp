#include "position_manager.h"
#include <algorithm>
#include <cmath>

namespace hean {

PositionManager::PositionManager() {
    positions_.reserve(100);
}

PositionManager::~PositionManager() = default;

void PositionManager::update_position(
    const std::string& symbol,
    double quantity_delta,
    double price
) {
    std::lock_guard<std::mutex> lock(positions_mutex_);

    auto& pos = positions_[symbol];

    if (pos.symbol.empty()) {
        // New position
        pos.symbol = symbol;
        pos.quantity = quantity_delta;
        pos.entry_price = price;
        pos.realized_pnl = 0.0;
        pos.unrealized_pnl = 0.0;
    } else {
        // Existing position
        bool closing = (pos.quantity > 0 && quantity_delta < 0) ||
                       (pos.quantity < 0 && quantity_delta > 0);

        if (closing) {
            // Calculate realized PnL on position reduction/close
            double closed_qty = std::min(std::abs(quantity_delta), std::abs(pos.quantity));
            double pnl_per_unit = (price - pos.entry_price) * (pos.quantity > 0 ? 1.0 : -1.0);
            pos.realized_pnl += pnl_per_unit * closed_qty;
        }

        // Update position size
        double old_qty = pos.quantity;
        pos.quantity += quantity_delta;

        // Update entry price if adding to position (same direction)
        if (!closing && pos.quantity != 0) {
            double old_value = old_qty * pos.entry_price;
            double new_value = quantity_delta * price;
            pos.entry_price = (old_value + new_value) / pos.quantity;
        }

        // If position is now flat, reset entry price
        if (std::abs(pos.quantity) < 1e-8) {
            pos.quantity = 0.0;
            pos.entry_price = 0.0;
            pos.unrealized_pnl = 0.0;
        }
    }

    pos.updated_at_us = timestamp_us();
}

void PositionManager::close_position(const std::string& symbol, double price) {
    std::lock_guard<std::mutex> lock(positions_mutex_);

    auto it = positions_.find(symbol);
    if (it == positions_.end() || it->second.is_flat()) {
        return;
    }

    auto& pos = it->second;

    // Calculate final realized PnL
    double pnl_per_unit = (price - pos.entry_price) * (pos.quantity > 0 ? 1.0 : -1.0);
    pos.realized_pnl += pnl_per_unit * std::abs(pos.quantity);

    // Clear position
    pos.quantity = 0.0;
    pos.entry_price = 0.0;
    pos.unrealized_pnl = 0.0;
    pos.updated_at_us = timestamp_us();
}

std::optional<Position> PositionManager::get_position(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(positions_mutex_);

    auto it = positions_.find(symbol);
    if (it != positions_.end()) {
        return it->second;
    }

    return std::nullopt;
}

std::vector<Position> PositionManager::get_all_positions() const {
    std::lock_guard<std::mutex> lock(positions_mutex_);

    std::vector<Position> result;
    result.reserve(positions_.size());

    for (const auto& [symbol, pos] : positions_) {
        result.push_back(pos);
    }

    return result;
}

std::vector<Position> PositionManager::get_open_positions() const {
    std::lock_guard<std::mutex> lock(positions_mutex_);

    std::vector<Position> result;
    result.reserve(positions_.size());

    for (const auto& [symbol, pos] : positions_) {
        if (!pos.is_flat()) {
            result.push_back(pos);
        }
    }

    return result;
}

double PositionManager::calculate_pnl(
    const Position& pos,
    double current_price
) const {
    if (pos.is_flat()) {
        return 0.0;
    }

    double pnl_per_unit = (current_price - pos.entry_price) * (pos.quantity > 0 ? 1.0 : -1.0);
    return pnl_per_unit * std::abs(pos.quantity);
}

double PositionManager::get_total_unrealized_pnl(
    const std::unordered_map<std::string, double>& current_prices
) const {
    std::lock_guard<std::mutex> lock(positions_mutex_);

    double total_pnl = 0.0;

    for (const auto& [symbol, pos] : positions_) {
        auto price_it = current_prices.find(symbol);
        if (price_it != current_prices.end()) {
            total_pnl += calculate_pnl(pos, price_it->second);
        }
    }

    return total_pnl;
}

double PositionManager::get_total_realized_pnl() const {
    std::lock_guard<std::mutex> lock(positions_mutex_);

    double total_pnl = 0.0;

    for (const auto& [symbol, pos] : positions_) {
        total_pnl += pos.realized_pnl;
    }

    return total_pnl;
}

void PositionManager::update_unrealized_pnl(
    const std::string& symbol,
    double current_price
) {
    std::lock_guard<std::mutex> lock(positions_mutex_);

    auto it = positions_.find(symbol);
    if (it != positions_.end()) {
        it->second.unrealized_pnl = calculate_pnl(it->second, current_price);
        it->second.updated_at_us = timestamp_us();
    }
}

size_t PositionManager::get_position_count() const {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    return positions_.size();
}

size_t PositionManager::get_open_position_count() const {
    std::lock_guard<std::mutex> lock(positions_mutex_);

    return std::count_if(positions_.begin(), positions_.end(),
        [](const auto& pair) { return !pair.second.is_flat(); });
}

} // namespace hean
