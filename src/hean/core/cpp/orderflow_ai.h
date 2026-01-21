/**
 * AI Micro-Structure Layer: OrderFlow Intelligence Header
 */

#ifndef ORDERFLOW_AI_H
#define ORDERFLOW_AI_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize OrderFlow AI system
void orderflow_ai_init();

// Process an order flow event
void orderflow_ai_process_event(
    int64_t timestamp_ns,
    const char* symbol,
    double price,
    double size,
    int is_buy,
    int is_trade,
    int is_cancel,
    int is_new_order
);

// Get VPIN (Volume-weighted Probability of Informed Trading)
double orderflow_ai_get_vpin(const char* symbol);

// Check if VPIN is high (indicates informed trading)
int orderflow_ai_is_high_vpin(const char* symbol);

// Get cancel-to-fill ratio for spoofing detection
double orderflow_ai_get_cancel_to_fill_ratio(const char* symbol);

// Get number of detected spoofing patterns
int orderflow_ai_get_spoofing_count(const char* symbol);

// Get number of detected iceberg orders
int orderflow_ai_get_iceberg_count(const char* symbol);

// Get iceberg order details (returns count, fills prices and sizes arrays)
int orderflow_ai_get_iceberg_details(const char* symbol, double* prices, double* sizes, int max_count);

// Cleanup
void orderflow_ai_cleanup();

#ifdef __cplusplus
}
#endif

#endif // ORDERFLOW_AI_H