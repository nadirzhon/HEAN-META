#include "order_engine.h"
#include "position_manager.h"
#include <iostream>
#include <cassert>
#include <iomanip>

using namespace hean;

void test_order_placement() {
    std::cout << "🧪 Testing order placement...\n";

    OrderEngine engine;

    // Test market order
    auto result = engine.place_market_order("BTCUSDT", Side::BUY, 0.1);
    assert(result.success);
    assert(!result.order_id.empty());
    assert(result.latency_us > 0);
    std::cout << "  ✅ Market order: " << result.order_id
              << " (latency: " << result.latency_us << "μs)\n";

    // Test limit order
    result = engine.place_limit_order("ETHUSDT", Side::SELL, 1.0, 3000.0);
    assert(result.success);
    std::cout << "  ✅ Limit order: " << result.order_id
              << " (latency: " << result.latency_us << "μs)\n";

    // Verify order retrieval
    auto order = engine.get_order(result.order_id);
    assert(order.has_value());
    assert(order->symbol == "ETHUSDT");
    assert(order->side == Side::SELL);
    assert(order->quantity == 1.0);
    assert(order->price == 3000.0);
    std::cout << "  ✅ Order retrieval successful\n";
}

void test_order_cancellation() {
    std::cout << "\n🧪 Testing order cancellation...\n";

    OrderEngine engine;

    auto result = engine.place_limit_order("BTCUSDT", Side::BUY, 0.5, 45000.0);
    assert(result.success);

    bool cancelled = engine.cancel_order(result.order_id);
    assert(cancelled);

    auto order = engine.get_order(result.order_id);
    assert(order.has_value());
    assert(order->status == OrderStatus::CANCELLED);
    std::cout << "  ✅ Order cancelled successfully\n";
}

void test_position_management() {
    std::cout << "\n🧪 Testing position management...\n";

    PositionManager pm;

    // Open long position
    pm.update_position("BTCUSDT", 0.5, 45000.0);

    auto pos = pm.get_position("BTCUSDT");
    assert(pos.has_value());
    assert(pos->quantity == 0.5);
    assert(pos->entry_price == 45000.0);
    assert(pos->is_long());
    std::cout << "  ✅ Long position opened\n";

    // Add to position
    pm.update_position("BTCUSDT", 0.3, 46000.0);

    pos = pm.get_position("BTCUSDT");
    assert(pos.has_value());
    assert(pos->quantity == 0.8);
    // Entry price should be weighted average
    double expected_entry = (0.5 * 45000.0 + 0.3 * 46000.0) / 0.8;
    assert(std::abs(pos->entry_price - expected_entry) < 0.01);
    std::cout << "  ✅ Position increased\n";

    // Partial close
    pm.update_position("BTCUSDT", -0.3, 47000.0);

    pos = pm.get_position("BTCUSDT");
    assert(pos.has_value());
    assert(std::abs(pos->quantity - 0.5) < 1e-8);
    assert(pos->realized_pnl > 0);  // Profit from partial close
    std::cout << "  ✅ Position partially closed\n";

    // Full close
    pm.close_position("BTCUSDT", 48000.0);

    pos = pm.get_position("BTCUSDT");
    assert(pos.has_value());
    assert(pos->is_flat());
    std::cout << "  ✅ Position fully closed\n";
}

void test_performance() {
    std::cout << "\n🧪 Testing performance...\n";

    OrderEngine engine;

    const int num_orders = 10000;
    auto start = now();

    for (int i = 0; i < num_orders; ++i) {
        engine.place_market_order("BTCUSDT", Side::BUY, 0.001);
    }

    auto end = now();
    auto elapsed_us = std::chrono::duration_cast<Microseconds>(end - start).count();

    double avg_latency = static_cast<double>(elapsed_us) / num_orders;
    double throughput = (num_orders * 1000000.0) / elapsed_us;

    std::cout << "  📊 Placed " << num_orders << " orders in "
              << elapsed_us << "μs\n";
    std::cout << "  📊 Average latency: " << std::fixed << std::setprecision(2)
              << avg_latency << "μs\n";
    std::cout << "  📊 Throughput: " << std::fixed << std::setprecision(0)
              << throughput << " orders/sec\n";

    assert(avg_latency < 100);  // Target: <100μs
    std::cout << "  ✅ Performance target met (<100μs)\n";
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "HEAN-META C++ Order Engine Tests\n";
    std::cout << "========================================\n\n";

    try {
        test_order_placement();
        test_order_cancellation();
        test_position_management();
        test_performance();

        std::cout << "\n========================================\n";
        std::cout << "✅ All tests passed!\n";
        std::cout << "========================================\n\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << "\n";
        return 1;
    }
}
