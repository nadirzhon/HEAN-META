/**
 * Algorithmic Fingerprinting Engine Header
 */

#ifndef HEAN_FINGERPRINTER_H
#define HEAN_FINGERPRINTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the fingerprinting engine
void algo_fingerprinter_init();

// Update order information (called when order book updates)
// order_id: Unique order identifier
// symbol: Trading symbol (e.g., "BTCUSDT")
// price: Order price
// size: Order size
// timestamp_ns: Timestamp in nanoseconds
// is_limit: 1 if limit order, 0 if market order
void algo_fingerprinter_update_order(const char* order_id, const char* symbol,
                                    double price, double size, int64_t timestamp_ns,
                                    int is_limit);

// Remove order from tracking (called when order is cancelled/filled)
void algo_fingerprinter_remove_order(const char* order_id);

// Get predictive alpha signal for a symbol
// Returns: 1 if signal available, 0 otherwise
// alpha_signal: Output signal (-1.0 = bearish, 0.0 = neutral, 1.0 = bullish)
// confidence: Output confidence (0.0 to 1.0)
// bot_id_out: Output buffer for identified bot ID
// max_bot_id_len: Maximum length of bot_id_out buffer
int algo_fingerprinter_get_predictive_alpha(const char* symbol, double* alpha_signal,
                                            double* confidence, char* bot_id_out, int max_bot_id_len);

// Get statistics
int algo_fingerprinter_get_active_orders_count();
int algo_fingerprinter_get_identified_bots_count();

// Cleanup
void algo_fingerprinter_cleanup();

#ifdef __cplusplus
}
#endif

#endif  // HEAN_FINGERPRINTER_H