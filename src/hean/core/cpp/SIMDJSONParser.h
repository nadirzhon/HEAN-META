/**
 * SIMD-Optimized JSON Parser for Bybit V5 WebSocket
 * Uses AVX2/SSE4.2 for ultra-fast JSON field extraction
 */

#ifndef SIMD_JSON_PARSER_H
#define SIMD_JSON_PARSER_H

#include <cstdint>
#include <string>
#include <string_view>

#ifdef __AVX2__
#include <immintrin.h>
#define USE_SIMD_JSON
#endif

#ifdef __SSE4_2__
#include <nmmintrin.h>
#define USE_SIMD_JSON
#endif

struct BybitTickerData {
    double last_price = 0.0;
    double bid1_price = 0.0;
    double ask1_price = 0.0;
    double volume_24h = 0.0;
    int64_t timestamp = 0;
    char symbol[32] = {0};
};

struct BybitOrderbookData {
    double bids[25][2];  // [price, qty]
    double asks[25][2];
    int64_t update_id = 0;
    int64_t timestamp = 0;
    char symbol[32] = {0};
    int bid_count = 0;
    int ask_count = 0;
};

/**
 * Parse Bybit ticker JSON using SIMD
 * @param json: JSON string (must be valid)
 * @param output: Output structure
 * @return true if successful
 */
bool parse_bybit_ticker_simd(const char* json, size_t json_len, BybitTickerData& output);

/**
 * Parse Bybit orderbook JSON using SIMD
 * @param json: JSON string (must be valid)
 * @param output: Output structure
 * @return true if successful
 */
bool parse_bybit_orderbook_simd(const char* json, size_t json_len, BybitOrderbookData& output);

/**
 * Extract numeric value from JSON field using SIMD search
 * @param json: JSON string
 * @param field_name: Field name to search for (e.g., "lastPrice")
 * @param output: Output value
 * @return true if found and parsed
 */
bool extract_json_double_simd(const char* json, size_t json_len, const char* field_name, double& output);

/**
 * Extract string value from JSON field using SIMD search
 * @param json: JSON string
 * @param field_name: Field name to search for
 * @param output: Output buffer
 * @param output_len: Output buffer size
 * @return true if found
 */
bool extract_json_string_simd(const char* json, size_t json_len, const char* field_name, char* output, size_t output_len);

#endif // SIMD_JSON_PARSER_H
