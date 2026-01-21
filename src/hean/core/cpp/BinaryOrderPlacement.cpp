/**
 * Binary WebSocket Order Placement Implementation
 */

#include "BinaryOrderPlacement.h"
#include <cstring>
#include <algorithm>

size_t serialize_order_request(const BinaryOrderRequest& request, uint8_t* output) {
    // Direct memory copy for maximum speed
    memcpy(output, &request, sizeof(BinaryOrderRequest));
    return sizeof(BinaryOrderRequest);
}

bool deserialize_order_response(const uint8_t* data, size_t data_len, BinaryOrderResponse& output) {
    if (data_len < sizeof(BinaryOrderResponse)) {
        return false;
    }
    
    memcpy(&output, data, sizeof(BinaryOrderResponse));
    return output.msg_type == 0x81; // Validate message type
}

size_t create_binary_websocket_frame(const uint8_t* order_data, size_t order_len, uint8_t* output) {
    size_t offset = 0;
    
    // WebSocket frame header (binary frame)
    output[offset++] = 0x82; // FIN=1, opcode=2 (binary)
    
    // Payload length (assume < 126 for ultra-fast path)
    if (order_len < 126) {
        output[offset++] = static_cast<uint8_t>(order_len);
    } else if (order_len < 65536) {
        output[offset++] = 126;
        output[offset++] = static_cast<uint8_t>((order_len >> 8) & 0xFF);
        output[offset++] = static_cast<uint8_t>(order_len & 0xFF);
    } else {
        output[offset++] = 127;
        for (int i = 7; i >= 0; i--) {
            output[offset++] = static_cast<uint8_t>((order_len >> (i * 8)) & 0xFF);
        }
    }
    
    // Copy payload (unmasked - server to client, or client can mask if needed)
    memcpy(output + offset, order_data, order_len);
    offset += order_len;
    
    return offset;
}
