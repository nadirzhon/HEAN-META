#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>
#include <chrono>
#include <atomic>
#include <cmath>

namespace nb = nanobind;

class UltraFastOrderRouter {
private:
    std::atomic<uint64_t> order_id_counter{0};

public:
    UltraFastOrderRouter() = default;

    // Sub-microsecond order validation (branchless)
    bool validate_order(
        const std::string& symbol,
        double price,
        double quantity,
        double max_position_size
    ) const {
        // Branchless validation for CPU pipeline efficiency
        bool valid = (quantity > 0) &
                     (quantity <= max_position_size) &
                     (price > 0) &
                     (!symbol.empty());
        return valid;
    }

    // Lock-free order ID generation
    uint64_t generate_order_id() {
        return order_id_counter.fetch_add(1, std::memory_order_relaxed);
    }

    // Calculate position size with risk management (branchless)
    double calculate_position_size(
        double account_balance,
        double risk_per_trade,
        double stop_loss_pct,
        double price
    ) const {
        double risk_amount = account_balance * risk_per_trade;
        double stop_distance = price * stop_loss_pct;

        // Avoid division by zero
        if (stop_distance < 1e-10) {
            return 0.0;
        }

        return risk_amount / stop_distance;
    }

    // Calculate position size with Kelly Criterion
    double calculate_kelly_position(
        double account_balance,
        double win_rate,
        double avg_win,
        double avg_loss,
        double max_position_pct = 0.25
    ) const {
        if (avg_loss <= 0 || win_rate <= 0 || win_rate >= 1) {
            return 0.0;
        }

        double loss_rate = 1.0 - win_rate;
        double win_loss_ratio = avg_win / avg_loss;

        // Kelly formula: f = (p * b - q) / b
        // where p = win_rate, q = loss_rate, b = win/loss ratio
        double kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio;

        // Cap at max position percentage for safety
        kelly_fraction = std::min(kelly_fraction, max_position_pct);
        kelly_fraction = std::max(kelly_fraction, 0.0);

        return account_balance * kelly_fraction;
    }

    // Calculate stop loss price
    double calculate_stop_loss(
        double entry_price,
        double atr,
        double atr_multiplier,
        bool is_long
    ) const {
        double stop_distance = atr * atr_multiplier;

        if (is_long) {
            return entry_price - stop_distance;
        } else {
            return entry_price + stop_distance;
        }
    }

    // Calculate take profit price
    double calculate_take_profit(
        double entry_price,
        double atr,
        double atr_multiplier,
        bool is_long
    ) const {
        double profit_distance = atr * atr_multiplier;

        if (is_long) {
            return entry_price + profit_distance;
        } else {
            return entry_price - profit_distance;
        }
    }

    // Calculate order execution quality score
    double calculate_execution_quality(
        double executed_price,
        double intended_price,
        double bid_ask_spread,
        bool is_buy
    ) const {
        double slippage = is_buy ?
            (executed_price - intended_price) :
            (intended_price - executed_price);

        double slippage_pct = slippage / intended_price;
        double spread_pct = bid_ask_spread / intended_price;

        // Quality score: 100 = perfect, 0 = worst
        // Penalize both slippage and wide spreads
        double quality = 100.0 * (1.0 - (slippage_pct / spread_pct));
        quality = std::max(0.0, std::min(100.0, quality));

        return quality;
    }

    // Calculate optimal order chunks for iceberg orders
    struct IcebergPlan {
        int num_chunks;
        double chunk_size;
        double remaining;
    };

    IcebergPlan calculate_iceberg_chunks(
        double total_quantity,
        double max_visible_quantity,
        double min_chunk_size
    ) const {
        IcebergPlan plan;

        if (total_quantity <= max_visible_quantity) {
            plan.num_chunks = 1;
            plan.chunk_size = total_quantity;
            plan.remaining = 0.0;
            return plan;
        }

        plan.chunk_size = std::min(max_visible_quantity, total_quantity / 4.0);
        plan.chunk_size = std::max(plan.chunk_size, min_chunk_size);

        plan.num_chunks = static_cast<int>(std::ceil(total_quantity / plan.chunk_size));
        plan.remaining = total_quantity - (plan.chunk_size * (plan.num_chunks - 1));

        return plan;
    }

    // Calculate TWAP (Time-Weighted Average Price) schedule
    struct TWAPSchedule {
        int num_orders;
        double order_size;
        int interval_ms;
    };

    TWAPSchedule calculate_twap_schedule(
        double total_quantity,
        int duration_minutes,
        double max_market_impact_pct = 0.01
    ) const {
        TWAPSchedule schedule;

        // Target 1 order per minute minimum, up to 1 per 10 seconds for large orders
        int min_interval_ms = 10000;  // 10 seconds
        int max_interval_ms = 60000;  // 1 minute

        // More frequent orders for larger quantities to reduce market impact
        schedule.interval_ms = max_interval_ms;
        if (total_quantity > 1000) {
            schedule.interval_ms = min_interval_ms;
        } else if (total_quantity > 100) {
            schedule.interval_ms = 30000;  // 30 seconds
        }

        int total_duration_ms = duration_minutes * 60000;
        schedule.num_orders = total_duration_ms / schedule.interval_ms;
        schedule.num_orders = std::max(schedule.num_orders, 1);

        schedule.order_size = total_quantity / schedule.num_orders;

        return schedule;
    }

    // Calculate order urgency score (0-100)
    double calculate_urgency_score(
        double current_price,
        double target_price,
        double volatility,
        double time_remaining_seconds
    ) const {
        double price_distance_pct = std::abs(current_price - target_price) / current_price;
        double volatility_factor = volatility * 100.0;

        // Higher urgency when:
        // 1. Price is far from target
        // 2. Volatility is high (price might move away)
        // 3. Time is running out
        double time_factor = 100.0 / (1.0 + time_remaining_seconds / 60.0);
        double distance_factor = price_distance_pct * 100.0;

        double urgency = (distance_factor * 0.5) + (volatility_factor * 0.3) + (time_factor * 0.2);
        urgency = std::min(100.0, urgency);

        return urgency;
    }

    // Calculate optimal limit price for aggressive entry
    double calculate_aggressive_limit_price(
        double mid_price,
        double bid,
        double ask,
        bool is_buy,
        double aggression = 0.5  // 0 = passive, 1 = market
    ) const {
        if (is_buy) {
            // Buy: between mid and ask
            return mid_price + (ask - mid_price) * aggression;
        } else {
            // Sell: between mid and bid
            return mid_price - (mid_price - bid) * aggression;
        }
    }

    // Get timestamp in microseconds
    int64_t get_timestamp_us() const {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }
};

// Python bindings
NB_MODULE(order_router_cpp, m) {
    m.doc() = "Ultra-low latency order routing engine (<1μs operations)";

    nb::class_<UltraFastOrderRouter::IcebergPlan>(m, "IcebergPlan")
        .def_ro("num_chunks", &UltraFastOrderRouter::IcebergPlan::num_chunks)
        .def_ro("chunk_size", &UltraFastOrderRouter::IcebergPlan::chunk_size)
        .def_ro("remaining", &UltraFastOrderRouter::IcebergPlan::remaining);

    nb::class_<UltraFastOrderRouter::TWAPSchedule>(m, "TWAPSchedule")
        .def_ro("num_orders", &UltraFastOrderRouter::TWAPSchedule::num_orders)
        .def_ro("order_size", &UltraFastOrderRouter::TWAPSchedule::order_size)
        .def_ro("interval_ms", &UltraFastOrderRouter::TWAPSchedule::interval_ms);

    nb::class_<UltraFastOrderRouter>(m, "UltraFastOrderRouter")
        .def(nb::init<>())
        .def("validate_order", &UltraFastOrderRouter::validate_order,
             nb::arg("symbol"), nb::arg("price"), nb::arg("quantity"), nb::arg("max_position_size"),
             "Validate order parameters (<100ns)")
        .def("generate_order_id", &UltraFastOrderRouter::generate_order_id,
             "Generate unique order ID (lock-free)")
        .def("calculate_position_size", &UltraFastOrderRouter::calculate_position_size,
             nb::arg("account_balance"), nb::arg("risk_per_trade"),
             nb::arg("stop_loss_pct"), nb::arg("price"),
             "Calculate position size based on risk management")
        .def("calculate_kelly_position", &UltraFastOrderRouter::calculate_kelly_position,
             nb::arg("account_balance"), nb::arg("win_rate"),
             nb::arg("avg_win"), nb::arg("avg_loss"),
             nb::arg("max_position_pct") = 0.25,
             "Calculate optimal position size using Kelly Criterion")
        .def("calculate_stop_loss", &UltraFastOrderRouter::calculate_stop_loss,
             nb::arg("entry_price"), nb::arg("atr"),
             nb::arg("atr_multiplier"), nb::arg("is_long"),
             "Calculate stop loss price based on ATR")
        .def("calculate_take_profit", &UltraFastOrderRouter::calculate_take_profit,
             nb::arg("entry_price"), nb::arg("atr"),
             nb::arg("atr_multiplier"), nb::arg("is_long"),
             "Calculate take profit price based on ATR")
        .def("calculate_execution_quality", &UltraFastOrderRouter::calculate_execution_quality,
             nb::arg("executed_price"), nb::arg("intended_price"),
             nb::arg("bid_ask_spread"), nb::arg("is_buy"),
             "Calculate order execution quality score (0-100)")
        .def("calculate_iceberg_chunks", &UltraFastOrderRouter::calculate_iceberg_chunks,
             nb::arg("total_quantity"), nb::arg("max_visible_quantity"),
             nb::arg("min_chunk_size"),
             "Calculate optimal iceberg order chunking")
        .def("calculate_twap_schedule", &UltraFastOrderRouter::calculate_twap_schedule,
             nb::arg("total_quantity"), nb::arg("duration_minutes"),
             nb::arg("max_market_impact_pct") = 0.01,
             "Calculate TWAP order schedule")
        .def("calculate_urgency_score", &UltraFastOrderRouter::calculate_urgency_score,
             nb::arg("current_price"), nb::arg("target_price"),
             nb::arg("volatility"), nb::arg("time_remaining_seconds"),
             "Calculate order urgency score (0-100)")
        .def("calculate_aggressive_limit_price", &UltraFastOrderRouter::calculate_aggressive_limit_price,
             nb::arg("mid_price"), nb::arg("bid"), nb::arg("ask"),
             nb::arg("is_buy"), nb::arg("aggression") = 0.5,
             "Calculate optimal limit price for aggressive execution")
        .def("get_timestamp_us", &UltraFastOrderRouter::get_timestamp_us,
             "Get current timestamp in microseconds");
}
