#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "order_engine.h"
#include "position_manager.h"

namespace py = pybind11;
using namespace hean;

PYBIND11_MODULE(hean_meta_cpp, m) {
    m.doc() = "HEAN-META C++ high-performance order execution engine";

    // Enums
    py::enum_<Side>(m, "Side")
        .value("BUY", Side::BUY)
        .value("SELL", Side::SELL)
        .export_values();

    py::enum_<OrderType>(m, "OrderType")
        .value("MARKET", OrderType::MARKET)
        .value("LIMIT", OrderType::LIMIT)
        .value("STOP_MARKET", OrderType::STOP_MARKET)
        .value("STOP_LIMIT", OrderType::STOP_LIMIT)
        .export_values();

    py::enum_<OrderStatus>(m, "OrderStatus")
        .value("PENDING", OrderStatus::PENDING)
        .value("SUBMITTED", OrderStatus::SUBMITTED)
        .value("PARTIALLY_FILLED", OrderStatus::PARTIALLY_FILLED)
        .value("FILLED", OrderStatus::FILLED)
        .value("CANCELLED", OrderStatus::CANCELLED)
        .value("REJECTED", OrderStatus::REJECTED)
        .value("EXPIRED", OrderStatus::EXPIRED)
        .export_values();

    // Order struct
    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("order_id", &Order::order_id)
        .def_readwrite("symbol", &Order::symbol)
        .def_readwrite("side", &Order::side)
        .def_readwrite("type", &Order::type)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("price", &Order::price)
        .def_readwrite("filled_quantity", &Order::filled_quantity)
        .def_readwrite("avg_fill_price", &Order::avg_fill_price)
        .def_readwrite("status", &Order::status)
        .def_readwrite("created_at_us", &Order::created_at_us)
        .def_readwrite("updated_at_us", &Order::updated_at_us)
        .def_readwrite("exchange_order_id", &Order::exchange_order_id)
        .def_readwrite("error_message", &Order::error_message)
        .def("is_complete", &Order::is_complete)
        .def("is_active", &Order::is_active)
        .def("remaining_quantity", &Order::remaining_quantity)
        .def("__repr__", [](const Order& o) {
            return "<Order id=" + o.order_id +
                   " symbol=" + o.symbol +
                   " side=" + to_string(o.side) +
                   " type=" + to_string(o.type) +
                   " qty=" + std::to_string(o.quantity) +
                   " status=" + to_string(o.status) + ">";
        });

    // Position struct
    py::class_<Position>(m, "Position")
        .def(py::init<>())
        .def_readwrite("symbol", &Position::symbol)
        .def_readwrite("quantity", &Position::quantity)
        .def_readwrite("entry_price", &Position::entry_price)
        .def_readwrite("unrealized_pnl", &Position::unrealized_pnl)
        .def_readwrite("realized_pnl", &Position::realized_pnl)
        .def_readwrite("updated_at_us", &Position::updated_at_us)
        .def("is_long", &Position::is_long)
        .def("is_short", &Position::is_short)
        .def("is_flat", &Position::is_flat)
        .def("get_side", &Position::get_side)
        .def("__repr__", [](const Position& p) {
            return "<Position symbol=" + p.symbol +
                   " qty=" + std::to_string(p.quantity) +
                   " entry=" + std::to_string(p.entry_price) +
                   " unrealized_pnl=" + std::to_string(p.unrealized_pnl) +
                   " realized_pnl=" + std::to_string(p.realized_pnl) + ">";
        });

    // OrderResult struct
    py::class_<OrderResult>(m, "OrderResult")
        .def(py::init<>())
        .def_readwrite("success", &OrderResult::success)
        .def_readwrite("order_id", &OrderResult::order_id)
        .def_readwrite("error_message", &OrderResult::error_message)
        .def_readwrite("latency_us", &OrderResult::latency_us)
        .def("__repr__", [](const OrderResult& r) {
            return "<OrderResult success=" + std::to_string(r.success) +
                   " order_id=" + r.order_id +
                   " latency=" + std::to_string(r.latency_us) + "us>";
        });

    // OrderEngine class
    py::class_<OrderEngine>(m, "OrderEngine")
        .def(py::init<>())
        .def("place_market_order", &OrderEngine::place_market_order,
             py::arg("symbol"), py::arg("side"), py::arg("quantity"),
             "Place a market order")
        .def("place_limit_order", &OrderEngine::place_limit_order,
             py::arg("symbol"), py::arg("side"), py::arg("quantity"), py::arg("price"),
             "Place a limit order")
        .def("cancel_order", &OrderEngine::cancel_order,
             py::arg("order_id"),
             "Cancel an order")
        .def("update_order", &OrderEngine::update_order,
             py::arg("order_id"), py::arg("new_price"),
             "Update order price")
        .def("get_order", &OrderEngine::get_order,
             py::arg("order_id"),
             "Get order by ID")
        .def("get_active_orders", &OrderEngine::get_active_orders,
             "Get all active orders")
        .def("get_orders_by_symbol", &OrderEngine::get_orders_by_symbol,
             py::arg("symbol"),
             "Get all orders for a symbol")
        .def("get_total_orders", &OrderEngine::get_total_orders,
             "Get total order count")
        .def("get_active_order_count", &OrderEngine::get_active_order_count,
             "Get active order count")
        .def("get_avg_latency_us", &OrderEngine::get_avg_latency_us,
             "Get average latency in microseconds")
        .def("simulate_fill", &OrderEngine::simulate_fill,
             py::arg("order_id"), py::arg("fill_price"),
             "Simulate order fill (for testing)")
        .def("__repr__", [](const OrderEngine& e) {
            return "<OrderEngine total_orders=" + std::to_string(e.get_total_orders()) +
                   " active=" + std::to_string(e.get_active_order_count()) +
                   " avg_latency=" + std::to_string(e.get_avg_latency_us()) + "us>";
        });

    // PositionManager class
    py::class_<PositionManager>(m, "PositionManager")
        .def(py::init<>())
        .def("update_position", &PositionManager::update_position,
             py::arg("symbol"), py::arg("quantity_delta"), py::arg("price"),
             "Update position")
        .def("close_position", &PositionManager::close_position,
             py::arg("symbol"), py::arg("price"),
             "Close position")
        .def("get_position", &PositionManager::get_position,
             py::arg("symbol"),
             "Get position by symbol")
        .def("get_all_positions", &PositionManager::get_all_positions,
             "Get all positions")
        .def("get_open_positions", &PositionManager::get_open_positions,
             "Get open positions")
        .def("get_total_unrealized_pnl", &PositionManager::get_total_unrealized_pnl,
             py::arg("current_prices"),
             "Get total unrealized PnL")
        .def("get_total_realized_pnl", &PositionManager::get_total_realized_pnl,
             "Get total realized PnL")
        .def("update_unrealized_pnl", &PositionManager::update_unrealized_pnl,
             py::arg("symbol"), py::arg("current_price"),
             "Update unrealized PnL")
        .def("get_position_count", &PositionManager::get_position_count,
             "Get total position count")
        .def("get_open_position_count", &PositionManager::get_open_position_count,
             "Get open position count")
        .def("__repr__", [](const PositionManager& pm) {
            return "<PositionManager positions=" + std::to_string(pm.get_position_count()) +
                   " open=" + std::to_string(pm.get_open_position_count()) + ">";
        });

    // Utility functions
    m.def("timestamp_us", &timestamp_us, "Get current timestamp in microseconds");
}
