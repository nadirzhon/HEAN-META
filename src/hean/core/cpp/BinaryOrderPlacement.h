/**
 * Binary WebSocket Order Placement for Bybit V5
 * Bypasses HTTP overhead using binary WebSocket frames
 */

#ifndef BINARY_ORDER_PLACEMENT_H
#define BINARY_ORDER_PLACEMENT_H

#include <cstdint>
#include <string>

// Binary order request structure (aligned for fast serialization)
#pragma pack(push, 1)
struct BinaryOrderRequest {
    uint8_t msg_type;      // 0x01 = Place Order
    uint16_t symbol_len;   // Symbol string length
    uint8_t side;          // 0 = Buy, 1 = Sell
    uint8_t order_type;    // 0 = Market, 1 = Limit
    double price;          // Price (for limit orders)
    double quantity;       // Quantity
    double stop_loss;      // Stop loss (optional)
    double take_profit;    // Take profit (optional)
    uint64_t timestamp_ns; // Client timestamp
    uint64_t request_id;   // Unique request ID
    char symbol[32];       // Symbol string (padded)
};

struct BinaryOrderResponse {
    uint8_t msg_type;      // 0x81 = Order Response
    uint8_t status;        // 0 = Success, 1 = Rejected, 2 = Error
    uint64_t request_id;   // Matches request
    char order_id[64];     // Exchange order ID
    double fill_price;     // Fill price (if filled)
    double fill_qty;       // Fill quantity
    uint64_t timestamp_ns; // Server timestamp
};
#pragma pack(pop)

/**
 * Serialize order request to binary format
 * @param request: Order request structure
 * @param output: Output buffer (must be at least sizeof(BinaryOrderRequest))
 * @return Number of bytes written
 */
size_t serialize_order_request(const BinaryOrderRequest& request, uint8_t* output);

/**
 * Deserialize binary order response
 * @param data: Binary data
 * @param data_len: Data length
 * @param output: Output structure
 * @return true if successful
 */
bool deserialize_order_response(const uint8_t* data, size_t data_len, BinaryOrderResponse& output);

/**
 * Create binary WebSocket frame for order placement
 * @param order_data: Serialized order data
 * @param order_len: Order data length
 * @param output: Output buffer (must be large enough)
 * @return Number of bytes written
 */
size_t create_binary_websocket_frame(const uint8_t* order_data, size_t order_len, uint8_t* output);

#endif // BINARY_ORDER_PLACEMENT_H
